#!/usr/bin/env python3
"""Plan and execute explicit, content-bound DeepSpeed checkpoint pruning."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Optional


SCHEMA = "instavar-cosyvoice-deepspeed-prune-plan/v1"
LOCK_NAME = ".instavar-deepspeed-prune.lock"
MAX_PLAN_BYTES = 4 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
SAFE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class PruneError(RuntimeError):
    """Raised when pruning cannot prove that its targets are authorized."""


@dataclass(frozen=True)
class CheckpointInfo:
    tag: str
    step: Optional[int]
    epoch: Optional[int]
    mtime_ns: int
    metric: Optional[float]
    components: list[dict[str, Any]]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PruneError(f"Expected an integer checkpoint position, got {value!r}")
    if value < 0:
        raise PruneError(f"Checkpoint position cannot be negative: {value}")
    return value


def _strict_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as exc:
            raise PruneError(f"Expected a numeric metric, got {value!r}") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PruneError(f"Expected a numeric metric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise PruneError(f"Checkpoint metric must be finite, got {value!r}")
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    """Parse the scalar mapping subset emitted by CosyVoice log_per_save."""
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise PruneError(f"Checkpoint metadata is too large: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise PruneError(f"Checkpoint metadata is not UTF-8: {path}") from exc
    data: dict[str, Any] = {}
    loss_dict: dict[str, str] | None = None
    current_root: str | None = None
    line_pattern = re.compile(r"^( *)([A-Za-z_][A-Za-z0-9_.-]*):(?: +(.*))?$")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line:
            raise PruneError(f"Tabs are not supported in metadata at line {line_number}")
        match = line_pattern.fullmatch(raw_line.rstrip())
        if match is None:
            raise PruneError(
                f"Unsupported checkpoint metadata syntax at {path}:{line_number}"
            )
        indent, key, value = match.groups()
        if len(indent) not in {0, 2}:
            raise PruneError(
                f"Unsupported checkpoint metadata indentation at {path}:{line_number}"
            )
        if not indent:
            if key in data:
                raise PruneError(f"Duplicate YAML key: {key!r}")
            current_root = key
            if key == "loss_dict":
                if value not in {None, ""}:
                    raise PruneError("loss_dict must be an indented scalar mapping")
                loss_dict = {}
                data[key] = loss_dict
            else:
                data[key] = value
            continue
        if current_root != "loss_dict" or loss_dict is None:
            raise PruneError(
                f"Only loss_dict may contain nested metadata at {path}:{line_number}"
            )
        if key in loss_dict:
            raise PruneError(f"Duplicate YAML key in loss_dict: {key!r}")
        if value in {None, ""}:
            raise PruneError(f"loss_dict value cannot be empty for {key!r}")
        loss_dict[key] = value
    if "step" not in data and "epoch" not in data and "loss_dict" not in data:
        raise PruneError(f"Checkpoint metadata lacks step, epoch, and loss_dict: {path}")
    return data


def _extract_metric(data: dict[str, Any], metric: str) -> Optional[float]:
    if not metric or len(metric) > 256:
        raise PruneError("Metric key must contain between 1 and 256 characters")
    loss_dict = data.get("loss_dict")
    if loss_dict is None:
        return None
    if not isinstance(loss_dict, dict):
        raise PruneError("loss_dict must be a mapping")
    return _strict_float(loss_dict.get(metric))


def _assert_safe_tag(tag: str) -> None:
    if len(tag) > 128 or not SAFE_TAG.fullmatch(tag) or tag in {".", ".."}:
        raise PruneError(f"Unsafe checkpoint tag: {tag!r}")


def _owned_lstat(path: Path, *, expected: str | None = None) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise PruneError(f"Required path does not exist: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise PruneError(f"Symlinks are not valid prune targets: {path}")
    if info.st_uid != os.geteuid():
        raise PruneError(f"Path is not owned by the effective user: {path}")
    if expected == "file" and not stat.S_ISREG(info.st_mode):
        raise PruneError(f"Expected a regular file: {path}")
    if expected == "directory" and not stat.S_ISDIR(info.st_mode):
        raise PruneError(f"Expected a directory: {path}")
    if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
        raise PruneError(f"Hard-linked files are not valid prune targets: {path}")
    return info


def _entry_identity(path: Path, model_dir: Path) -> dict[str, Any]:
    info = _owned_lstat(path)
    relative = path.relative_to(model_dir).as_posix()
    common = {
        "path": relative,
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": stat.S_IMODE(info.st_mode),
        "mtime_ns": info.st_mtime_ns,
    }
    if stat.S_ISREG(info.st_mode):
        return {
            **common,
            "type": "file",
            "size": info.st_size,
            "sha256": _sha256_file(path),
        }
    if stat.S_ISDIR(info.st_mode):
        return {**common, "type": "directory"}
    raise PruneError(f"Unsupported filesystem object in checkpoint: {path}")


def _component_manifest(path: Path, model_dir: Path) -> dict[str, Any]:
    root = _entry_identity(path, model_dir)
    entries = [root]
    if root["type"] == "directory":
        for current, dirnames, filenames in os.walk(path, followlinks=False):
            current_path = Path(current)
            for name in sorted(dirnames + filenames):
                entries.append(_entry_identity(current_path / name, model_dir))
    return {"path": root["path"], "entries": sorted(entries, key=lambda item: item["path"])}


def _normalized_component_manifest(
    path: Path, model_dir: Path, recorded_root: str
) -> dict[str, Any]:
    manifest = _component_manifest(path, model_dir)
    actual_root = manifest["path"]
    for entry in manifest["entries"]:
        entry_path = entry["path"]
        if entry_path == actual_root:
            entry["path"] = recorded_root
        elif entry_path.startswith(f"{actual_root}/"):
            entry["path"] = recorded_root + entry_path[len(actual_root) :]
        else:
            raise PruneError("Component manifest escaped its staging root")
    manifest["path"] = recorded_root
    return manifest


def _checkpoint_info(model_dir: Path, tag: str, metric: str) -> CheckpointInfo:
    _assert_safe_tag(tag)
    yaml_path = model_dir / f"{tag}.yaml"
    yaml_stat = _owned_lstat(yaml_path, expected="file")
    data = _load_yaml(yaml_path)
    components = [_component_manifest(yaml_path, model_dir)]
    for candidate in (model_dir / tag, model_dir / f"{tag}.pt"):
        if candidate.exists() or candidate.is_symlink():
            components.append(_component_manifest(candidate, model_dir))
    if len(components) == 1:
        raise PruneError(f"Checkpoint {tag!r} has metadata but no payload directory or .pt file")
    return CheckpointInfo(
        tag=tag,
        step=_strict_int(data.get("step")),
        epoch=_strict_int(data.get("epoch")),
        mtime_ns=yaml_stat.st_mtime_ns,
        metric=_extract_metric(data, metric),
        components=components,
    )


def _latest_tags(infos: list[CheckpointInfo], keep_latest: int) -> set[str]:
    if keep_latest < 0:
        raise PruneError("--keep-latest cannot be negative")
    ordered = sorted(
        infos,
        key=lambda item: (
            item.step if item.step is not None else -1,
            item.epoch if item.epoch is not None else -1,
            item.mtime_ns,
            item.tag,
        ),
    )
    return {item.tag for item in ordered[-keep_latest:]} if keep_latest else set()


def _best_tags(
    infos: list[CheckpointInfo], keep_best: int, higher_is_better: bool
) -> set[str]:
    if keep_best < 0:
        raise PruneError("--keep-best cannot be negative")
    candidates = [item for item in infos if item.metric is not None]
    ordered = sorted(
        candidates,
        key=lambda item: (item.metric, item.tag),
        reverse=higher_is_better,
    )
    return {item.tag for item in ordered[:keep_best]}


def build_plan(
    model_dir: str | Path,
    owned_tags: list[str],
    *,
    keep_latest: int,
    keep_best: int,
    metric: str,
    higher_is_better: bool,
) -> dict[str, Any]:
    raw_model_dir = Path(model_dir).expanduser()
    model_dir_path = raw_model_dir.resolve(strict=True)
    model_stat = _owned_lstat(model_dir_path, expected="directory")
    if not owned_tags:
        raise PruneError("Planning requires at least one explicit --owned-tag")
    if len(set(owned_tags)) != len(owned_tags):
        raise PruneError("Duplicate --owned-tag values are not allowed")
    infos = [_checkpoint_info(model_dir_path, tag, metric) for tag in sorted(owned_tags)]
    keep = _latest_tags(infos, keep_latest)
    keep.update(_best_tags(infos, keep_best, higher_is_better))
    remove = sorted(item.tag for item in infos if item.tag not in keep)
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "model_dir": {
            "path": str(model_dir_path),
            "device": model_stat.st_dev,
            "inode": model_stat.st_ino,
            "uid": model_stat.st_uid,
        },
        "tool_sha256": _sha256_file(Path(__file__).resolve()),
        "selection": {
            "keep_latest": keep_latest,
            "keep_best": keep_best,
            "metric": metric,
            "higher_is_better": higher_is_better,
            "keep_tags": sorted(keep),
            "remove_tags": remove,
        },
        "checkpoints": [
            {
                "tag": item.tag,
                "step": item.step,
                "epoch": item.epoch,
                "metric": item.metric,
                "components": item.components,
            }
            for item in infos
        ],
    }
    body["plan_sha256"] = _sha256_bytes(_canonical_bytes(body))
    return body


def _write_new_plan(path: Path, plan: dict[str, Any]) -> None:
    path = path.expanduser().absolute()
    parent = path.parent.resolve(strict=True)
    _owned_lstat(parent, expected="directory")
    path = parent / path.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        data = json.dumps(plan, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PruneError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_plan(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PruneError(f"Cannot safely open prune plan {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
            raise PruneError("Prune plan must be a singly linked regular file owned by the effective user")
        if info.st_size > MAX_PLAN_BYTES:
            raise PruneError("Prune plan is too large")
        raw = os.read(descriptor, info.st_size + 1)
        current = path.stat()
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise PruneError("Prune plan path changed while it was open")
    finally:
        os.close(descriptor)
    try:
        data = json.loads(raw, object_pairs_hook=_json_no_duplicates)
    except PruneError:
        raise
    except Exception as exc:
        raise PruneError(f"Cannot parse prune plan: {exc}") from exc
    if not isinstance(data, dict):
        raise PruneError("Prune plan must contain a JSON object")
    return data


def _validated_plan(path: str | Path, confirmation: str) -> dict[str, Any]:
    plan = _read_plan(Path(path).expanduser().absolute())
    recorded = plan.pop("plan_sha256", None)
    actual = _sha256_bytes(_canonical_bytes(plan))
    plan["plan_sha256"] = recorded
    if recorded != actual:
        raise PruneError("Prune plan content digest does not match")
    if confirmation != actual:
        raise PruneError("--confirm-plan-sha256 does not match the reviewed plan")
    if plan.get("schema") != SCHEMA:
        raise PruneError("Unsupported prune plan schema")
    if plan.get("tool_sha256") != _sha256_file(Path(__file__).resolve()):
        raise PruneError("Pruner source changed after plan creation; create and review a new plan")
    return plan


class _ModelDirLock:
    def __init__(self, model_dir: Path) -> None:
        self.path = model_dir / LOCK_NAME
        self.descriptor: int | None = None

    def __enter__(self) -> "_ModelDirLock":
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        self.descriptor = descriptor
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
        ):
            os.close(descriptor)
            self.descriptor = None
            raise PruneError("Unsafe existing model-dir prune lock")
        current = self.path.stat()
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            os.close(descriptor)
            self.descriptor = None
            raise PruneError("Model-dir prune lock path changed while it was open")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            self.descriptor = None
            raise PruneError("Another checkpoint prune operation holds the model-dir lock") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


def _plan_checkpoint(plan: dict[str, Any], tag: str) -> dict[str, Any]:
    matches = [item for item in plan["checkpoints"] if item.get("tag") == tag]
    if len(matches) != 1:
        raise PruneError(f"Plan does not contain exactly one checkpoint record for {tag!r}")
    return matches[0]


def _revalidate_plan(plan: dict[str, Any]) -> Path:
    model_record = plan.get("model_dir")
    selection = plan.get("selection")
    checkpoints = plan.get("checkpoints")
    if not isinstance(model_record, dict) or not isinstance(selection, dict) or not isinstance(checkpoints, list):
        raise PruneError("Prune plan structure is invalid")
    model_dir = Path(str(model_record.get("path", "")))
    info = _owned_lstat(model_dir, expected="directory")
    if [info.st_dev, info.st_ino, info.st_uid] != [
        model_record.get("device"),
        model_record.get("inode"),
        model_record.get("uid"),
    ]:
        raise PruneError("Model directory identity changed after plan creation")
    keep = selection.get("keep_tags")
    remove = selection.get("remove_tags")
    tags = [item.get("tag") for item in checkpoints if isinstance(item, dict)]
    if (
        not isinstance(keep, list)
        or not isinstance(remove, list)
        or len(tags) != len(checkpoints)
        or len(set(tags)) != len(tags)
        or set(keep) & set(remove)
        or set(keep) | set(remove) != set(tags)
    ):
        raise PruneError("Prune plan keep/remove partition is invalid")
    remove_set = set(remove)
    suffix = plan["plan_sha256"][:16]
    for tag in tags:
        _assert_safe_tag(tag)
        record = _plan_checkpoint(plan, tag)
        components = record.get("components")
        if not isinstance(components, list) or not components:
            raise PruneError(f"Checkpoint {tag!r} has no component manifest")
        expected_roots = {
            component.get("path")
            for component in components
            if isinstance(component, dict)
        }
        for candidate_name in (tag, f"{tag}.yaml", f"{tag}.pt"):
            candidate = model_dir / candidate_name
            staged_candidate = model_dir / f".{candidate_name}.prune-{suffix}"
            if candidate_name not in expected_roots and (
                candidate.exists()
                or candidate.is_symlink()
                or staged_candidate.exists()
                or staged_candidate.is_symlink()
            ):
                raise PruneError(f"Checkpoint {tag!r} changed after plan creation")
        actual = []
        for component in components:
            if not isinstance(component, dict) or not isinstance(component.get("path"), str):
                raise PruneError(f"Checkpoint {tag!r} has an invalid component manifest")
            recorded_root = component["path"]
            component_path = model_dir / recorded_root
            staged_path = model_dir / f".{component_path.name}.prune-{suffix}"
            if component_path.parent != model_dir or staged_path.parent != model_dir:
                raise PruneError("Only direct model-dir checkpoint components may be pruned")
            source_exists = component_path.exists() or component_path.is_symlink()
            staged_exists = staged_path.exists() or staged_path.is_symlink()
            if tag not in remove_set:
                if staged_exists:
                    raise PruneError(f"Kept checkpoint has a staged component: {tag!r}")
                actual.append(_component_manifest(component_path, model_dir))
            elif source_exists and staged_exists:
                raise PruneError(f"Both source and staged prune components exist: {recorded_root}")
            elif source_exists:
                actual.append(_component_manifest(component_path, model_dir))
            elif staged_exists:
                actual.append(
                    _normalized_component_manifest(staged_path, model_dir, recorded_root)
                )
            else:
                # A prior execution may already have removed this exact component.
                actual.append(component)
        if actual != components:
            raise PruneError(f"Checkpoint {tag!r} changed after plan creation")
    return model_dir


def execute_plan(path: str | Path, confirmation: str) -> list[str]:
    plan = _validated_plan(path, confirmation)
    model_dir = Path(plan["model_dir"]["path"])
    with _ModelDirLock(model_dir):
        model_dir = _revalidate_plan(plan)
        remove_tags = plan["selection"]["remove_tags"]
        staged: list[tuple[Path, Path]] = []
        suffix = plan["plan_sha256"][:16]
        try:
            for tag in remove_tags:
                record = _plan_checkpoint(plan, tag)
                for component in record["components"]:
                    source = model_dir / component["path"]
                    target = model_dir / f".{source.name}.prune-{suffix}"
                    source_exists = source.exists() or source.is_symlink()
                    target_exists = target.exists() or target.is_symlink()
                    if source_exists:
                        if target_exists:
                            raise PruneError(f"Prune staging target already exists: {target}")
                        source.rename(target)
                        staged.append((source, target))
                    elif target_exists:
                        staged.append((source, target))
        except Exception:
            for source, target in reversed(staged):
                if target.exists() and not source.exists():
                    target.rename(source)
            raise
        for _, target in staged:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        return list(remove_tags)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a reviewed prune plan or execute one exact content-bound plan."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-out", help="Create a new JSON prune plan at this path.")
    mode.add_argument("--execute-plan", help="Execute an existing reviewed JSON plan.")
    parser.add_argument("--model-dir", help="Model directory used when creating a plan.")
    parser.add_argument(
        "--owned-tag",
        action="append",
        default=[],
        help="Explicitly adopt one checkpoint tag into the plan. Repeat for every tag.",
    )
    parser.add_argument("--keep-latest", type=int, default=2)
    parser.add_argument("--keep-best", type=int, default=1)
    parser.add_argument("--metric", default="loss")
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument(
        "--confirm-plan-sha256",
        help="Exact digest printed when the reviewed plan was created.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.plan_out:
        if not args.model_dir:
            raise PruneError("--model-dir is required with --plan-out")
        if args.confirm_plan_sha256:
            raise PruneError("--confirm-plan-sha256 is valid only with --execute-plan")
        plan = build_plan(
            args.model_dir,
            args.owned_tag,
            keep_latest=args.keep_latest,
            keep_best=args.keep_best,
            metric=args.metric,
            higher_is_better=args.higher_is_better,
        )
        _write_new_plan(Path(args.plan_out), plan)
        print(f"Plan written without deleting checkpoints: {args.plan_out}")
        print(f"Keep tags: {plan['selection']['keep_tags']}")
        print(f"Remove tags: {plan['selection']['remove_tags']}")
        print(f"plan_sha256={plan['plan_sha256']}")
        return
    if not args.confirm_plan_sha256:
        raise PruneError("--confirm-plan-sha256 is required with --execute-plan")
    if args.model_dir or args.owned_tag:
        raise PruneError("Execution takes all target authority from the reviewed plan")
    removed = execute_plan(args.execute_plan, args.confirm_plan_sha256)
    print(f"Pruned {len(removed)} explicitly adopted checkpoints: {removed}")


if __name__ == "__main__":
    main()
