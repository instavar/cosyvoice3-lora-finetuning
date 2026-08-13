"""Dependency-free guarded epoch checkpoint contract for CosyVoice3 LoRA."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO


SCHEMA_VERSION = "1.0.0"
SIDECAR_NAME = "resume-contract.json"
STATE_NAME = "training-state.json"
RUNTIME_STATE_NAME = "runtime-state.pt"
LOCK_NAME = ".instavar-training.lock"
_CHECKPOINT_RE = re.compile(r"^resume_epoch_(\d{6})$")


class ResumeContractError(ValueError):
    """Raised when guarded continuation state is unsafe or incompatible."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [canonical_value(item) for item in value]
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ResumeContractError(f"File identity rejects symlinks: {raw}")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ResumeContractError(f"Expected a regular file: {resolved}")
    return resolved


def _safe_directory(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ResumeContractError(f"Directory identity rejects symlinks: {raw}")
    resolved = raw.resolve(strict=True)
    if not resolved.is_dir():
        raise ResumeContractError(f"Expected a directory: {resolved}")
    return resolved


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = _safe_file(path)
    file_stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size": file_stat.st_size,
    }


def tree_identity(path: str | Path) -> dict[str, Any]:
    resolved = _safe_directory(path)
    root_stat = resolved.stat()
    files: list[dict[str, Any]] = []
    for item in sorted(resolved.rglob("*")):
        if item.is_symlink():
            raise ResumeContractError(f"Directory identity rejects symlinks: {item}")
        if item.is_file():
            identity = file_identity(item)
            identity["path"] = item.relative_to(resolved).as_posix()
            files.append(identity)
    return {
        "path": str(resolved),
        "device": root_stat.st_dev,
        "inode": root_stat.st_ino,
        "files": files,
    }


def output_identity(path: str | Path) -> dict[str, Any]:
    resolved = _safe_directory(path)
    root_stat = resolved.stat()
    return {
        "path": str(resolved),
        "device": root_stat.st_dev,
        "inode": root_stat.st_ino,
    }


def build_contract(
    *,
    output_dir: str | Path,
    base_checkpoint: str | Path,
    config_file: str | Path,
    qwen_pretrain: str | Path | None,
    data_files: Mapping[str, Iterable[str | Path]],
    source_files: Iterable[str | Path],
    training_config: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    inputs: dict[str, list[dict[str, Any]]] = {}
    for role, paths in sorted(data_files.items()):
        identities = [file_identity(path) for path in paths]
        unique = {identity["path"]: identity for identity in identities}
        inputs[role] = [unique[key] for key in sorted(unique)]
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "lora",
        "output_dir": output_identity(output_dir),
        "base_checkpoint": file_identity(base_checkpoint),
        "config_file": file_identity(config_file),
        "qwen_pretrain": tree_identity(qwen_pretrain) if qwen_pretrain else None,
        "inputs": inputs,
        "sources": [
            file_identity(path)
            for path in sorted((Path(item) for item in source_files), key=str)
        ],
        "training_config": canonical_value(training_config),
        "runtime": canonical_value(runtime),
    }


def contract_digest(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()


def epoch_checkpoint_name(completed_epoch: int) -> str:
    if completed_epoch < 0:
        raise ResumeContractError("completed_epoch must be non-negative")
    return f"resume_epoch_{completed_epoch:06d}"


def resolve_checkpoint(checkpoint: str | Path, output_dir: str | Path) -> Path:
    raw = Path(checkpoint).expanduser()
    if raw.is_symlink():
        raise ResumeContractError(f"Checkpoint symlinks are not allowed: {raw}")
    resolved = raw.resolve(strict=True)
    output = _safe_directory(output_dir)
    if resolved.parent != output:
        raise ResumeContractError(
            "Resume checkpoint must be a direct child of model_dir"
        )
    if not resolved.is_dir() or not _CHECKPOINT_RE.fullmatch(resolved.name):
        raise ResumeContractError(
            "Resume checkpoint must be an immutable resume_epoch_NNNNNN"
        )
    return resolved


def checkpoint_children(output_dir: str | Path) -> list[Path]:
    output = _safe_directory(output_dir)
    checkpoints: list[tuple[int, Path]] = []
    for item in output.iterdir():
        match = _CHECKPOINT_RE.fullmatch(item.name)
        if not match:
            continue
        if item.is_symlink() or not item.is_dir():
            raise ResumeContractError(f"Unsafe guarded checkpoint child: {item}")
        checkpoints.append((int(match.group(1)), item.resolve(strict=True)))
    return [path for _, path in sorted(checkpoints)]


def require_fresh_output(output_dir: str | Path) -> None:
    output = _safe_directory(output_dir)
    conflicts = [item.name for item in output.iterdir() if item.name != LOCK_NAME]
    if conflicts:
        raise ResumeContractError(
            "Fresh guarded training needs an empty model_dir; use exact --resume-from "
            f"or a fresh output directory ({', '.join(sorted(conflicts)[:5])})"
        )


def _required_manifest(checkpoint: Path) -> list[dict[str, Any]]:
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        STATE_NAME,
        RUNTIME_STATE_NAME,
    }
    files: list[dict[str, Any]] = []
    for item in sorted(checkpoint.iterdir()):
        if item.name == SIDECAR_NAME or item.name.startswith(f".{SIDECAR_NAME}."):
            continue
        if item.is_symlink() or not item.is_file():
            raise ResumeContractError(f"Checkpoint contains an unsafe member: {item}")
        identity = file_identity(item)
        identity["path"] = item.name
        files.append(identity)
    names = {item["path"] for item in files}
    if not required.issubset(names):
        raise ResumeContractError(
            f"Checkpoint omits continuation files: {', '.join(sorted(required - names))}"
        )
    return files


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            canonical_value(value), handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def publish_checkpoint(
    *,
    output_dir: str | Path,
    completed_epoch: int,
    completed_step: int,
    contract: Mapping[str, Any],
    adapter_saver,
    runtime_state_saver,
    monitor_state: Mapping[str, Any],
) -> Path:
    output = _safe_directory(output_dir)
    name = epoch_checkpoint_name(completed_epoch)
    target = output / name
    if target.exists() or target.is_symlink():
        raise ResumeContractError(
            f"Refusing to overwrite or adopt guarded checkpoint: {target}"
        )
    partial = output / f".{name}.{os.getpid()}.partial"
    if partial.exists() or partial.is_symlink():
        raise ResumeContractError(
            f"Guarded checkpoint partial already exists: {partial}"
        )
    partial.mkdir(mode=0o700)
    created_partial = True
    published = False
    try:
        adapter_saver(partial)
        state = {
            "schema_version": SCHEMA_VERSION,
            "completed_epoch": int(completed_epoch),
            "completed_step": int(completed_step),
            "monitor_state": canonical_value(monitor_state),
        }
        _write_json(partial / STATE_NAME, state)
        runtime_state_saver(partial / RUNTIME_STATE_NAME)
        manifest = _required_manifest(partial)
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_name": name,
            "completed_epoch": int(completed_epoch),
            "completed_step": int(completed_step),
            "contract_sha256": contract_digest(contract),
            "contract": canonical_value(contract),
            "files": manifest,
        }
        _write_json(partial / SIDECAR_NAME, sidecar)
        _fsync_directory(partial)
        os.rename(partial, target)
        published = True
        _fsync_directory(output)
    finally:
        if (
            created_partial
            and not published
            and partial.exists()
            and not partial.is_symlink()
        ):
            shutil.rmtree(partial)
    return target


def validate_checkpoint(
    checkpoint: str | Path,
    *,
    output_dir: str | Path,
    expected_contract: Mapping[str, Any],
    trust_resume_state: bool,
    world_size: int,
    train_engine: str,
) -> tuple[Path, dict[str, Any]]:
    if not trust_resume_state:
        raise ResumeContractError(
            "Resume includes pickle-capable optimizer and RNG state; use --trust-resume-state "
            "only for state you trust"
        )
    if world_size != 1 or train_engine != "torch_ddp":
        raise ResumeContractError(
            "Guarded continuation supports one torch_ddp process only; DeepSpeed and "
            "multi-rank state need a collective protocol"
        )
    selected = resolve_checkpoint(checkpoint, output_dir)
    children = checkpoint_children(output_dir)
    for child in children:
        _validate_checkpoint_files(child, expected_contract=expected_contract)
    if not children or children[-1] != selected:
        raise ResumeContractError(
            "Resume checkpoint must be the newest owned guarded checkpoint"
        )
    sidecar = _validate_checkpoint_files(selected, expected_contract=expected_contract)
    state = json.loads((selected / STATE_NAME).read_text(encoding="utf-8"))
    completed_epoch = state.get("completed_epoch")
    completed_step = state.get("completed_step")
    if (
        not isinstance(completed_epoch, int)
        or completed_epoch < 0
        or not isinstance(completed_step, int)
        or completed_step < 0
    ):
        raise ResumeContractError("Guarded training state has invalid progress")
    if selected.name != epoch_checkpoint_name(completed_epoch):
        raise ResumeContractError("Checkpoint name and completed epoch disagree")
    if (
        sidecar.get("completed_epoch") != completed_epoch
        or sidecar.get("completed_step") != completed_step
    ):
        raise ResumeContractError("Checkpoint sidecar and training progress disagree")
    max_epoch = expected_contract.get("training_config", {}).get("max_epoch")
    if isinstance(max_epoch, int) and completed_epoch + 1 >= max_epoch:
        raise ResumeContractError(
            "Checkpoint already reached the configured max_epoch target"
        )
    monitor = state.get("monitor_state", {})
    early_stop = expected_contract.get("training_config", {}).get(
        "early_stop_on_cv_overfit"
    )
    if early_stop and isinstance(monitor, dict) and monitor.get("cv_overfit_flag") == 1:
        raise ResumeContractError(
            "Checkpoint already reached the configured early-stop target"
        )
    return selected, state


def _validate_checkpoint_files(
    checkpoint: Path,
    *,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    sidecar_path = checkpoint / SIDECAR_NAME
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        raise ResumeContractError(
            f"Checkpoint has no safe {SIDECAR_NAME}: {checkpoint}"
        )
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ResumeContractError("Checkpoint sidecar is invalid JSON") from error
    if sidecar.get("schema_version") != SCHEMA_VERSION:
        raise ResumeContractError("Unsupported guarded checkpoint schema")
    if sidecar.get("checkpoint_name") != checkpoint.name:
        raise ResumeContractError("Checkpoint sidecar names a different directory")
    if sidecar.get("contract_sha256") != contract_digest(expected_contract):
        raise ResumeContractError("Guarded resume contract drift detected")
    if sidecar.get("contract") != canonical_value(expected_contract):
        raise ResumeContractError("Guarded resume contract payload does not match")
    manifest = sidecar.get("files")
    if not isinstance(manifest, list) or manifest != _required_manifest(checkpoint):
        raise ResumeContractError("Guarded checkpoint file identity drift detected")
    return sidecar


def prune_owned_checkpoints(
    output_dir: str | Path,
    *,
    keep_last: int | None,
    expected_contract: Mapping[str, Any],
) -> list[Path]:
    if keep_last is None:
        return []
    if keep_last < 1:
        raise ResumeContractError("resume_keep_last must be at least 1")
    checkpoints = checkpoint_children(output_dir)
    identities: dict[Path, tuple[int, int]] = {}
    for checkpoint in checkpoints:
        _validate_checkpoint_files(checkpoint, expected_contract=expected_contract)
        item_stat = checkpoint.stat()
        identities[checkpoint] = (item_stat.st_dev, item_stat.st_ino)
    victims = checkpoints[: max(0, len(checkpoints) - keep_last)]
    for victim in victims:
        item_stat = victim.stat()
        if (item_stat.st_dev, item_stat.st_ino) != identities[victim]:
            raise ResumeContractError(
                f"Checkpoint identity changed before pruning: {victim}"
            )
        shutil.rmtree(victim)
        _fsync_directory(victim.parent)
    return victims


def acquire_output_lock(output_dir: str | Path) -> TextIO:
    raw = Path(output_dir).expanduser()
    if raw.is_symlink():
        raise ResumeContractError(f"model_dir must not be a symlink: {raw}")
    raw.mkdir(parents=True, exist_ok=True)
    output = _safe_directory(raw)
    lock_path = output / LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise ResumeContractError(
            f"Could not open safe training lock: {lock_path}"
        ) from error
    handle = os.fdopen(descriptor, "r+", encoding="utf-8")
    lock_stat = os.fstat(handle.fileno())
    if (
        not stat.S_ISREG(lock_stat.st_mode)
        or lock_stat.st_nlink != 1
        or lock_stat.st_uid != os.geteuid()
    ):
        handle.close()
        raise ResumeContractError(
            f"Training lock has unsafe ownership or link count: {lock_path}"
        )
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise ResumeContractError(
            f"Another guarded writer holds model_dir: {output}"
        ) from error
    path_stat = lock_path.stat(follow_symlinks=False)
    if (path_stat.st_dev, path_stat.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
        handle.close()
        raise ResumeContractError(
            f"Training lock path changed during acquisition: {lock_path}"
        )
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
