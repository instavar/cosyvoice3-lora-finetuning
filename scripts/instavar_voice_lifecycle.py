#!/usr/bin/env python3
"""Execute CosyVoice3 LoRA through the Instavar Voice lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).parents[1]
PATCHES = tuple(sorted((REPO_ROOT / "patches").glob("*.patch")))


def _path(name: str, *, directory: bool = False) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise FileNotFoundError(f"{name} is a symlink: {unresolved}")
    path = unresolved.resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        raise FileNotFoundError(f"{name} is missing: {path}")
    return path


def _work() -> Path:
    return _path("INSTAVAR_VOICE_WORK_DIR", directory=True)


def _persistent_package_root() -> Path:
    root = _path("PERSISTED_PACKAGE_ROOT", directory=True)
    protected = {
        "lifecycle work directory": _work(),
        "repository checkout": REPO_ROOT.resolve(),
        "CosyVoice checkout": _path("COSYVOICE_DIR", directory=True),
        "pretrained model directory": _path("PRETRAINED_DIR", directory=True),
        "Qwen dependency directory": _path("QWEN_PRETRAIN_DIR", directory=True),
        "prepared training tree": _path("PREPARED_TRAIN_ROOT", directory=True),
        "prepared validation tree": _path("PREPARED_VALIDATION_ROOT", directory=True),
        "base LLM checkpoint directory": _path("BASE_LLM_CHECKPOINT").parent,
    }
    for label, path in protected.items():
        if root == path or root.is_relative_to(path):
            raise ValueError(f"PERSISTED_PACKAGE_ROOT must be outside the {label}")
    return root


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _probe_persistent_package_root(root: Path) -> dict[str, Any]:
    probe_path: Path | None = None
    linked_path: Path | None = None
    linked_created = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=root,
            prefix=".instavar-voice-persistence-probe.",
            suffix=".partial",
            delete=False,
        ) as probe:
            probe_path = Path(probe.name)
            probe.write(b"instavar-voice-persistence-probe-v1\n")
            probe.flush()
            os.fsync(probe.fileno())
        linked_path = probe_path.with_suffix(".linked")
        os.link(probe_path, linked_path)
        linked_created = True
        _fsync_directory(root)
        if linked_path.read_bytes() != probe_path.read_bytes():
            raise ValueError(
                "persistent package root failed its atomic publication probe"
            )
        identity = root.stat()
        return {
            "writable": True,
            "atomic_hard_link": True,
            "device": identity.st_dev,
            "inode": identity.st_ino,
        }
    except OSError as error:
        raise ValueError(
            f"PERSISTED_PACKAGE_ROOT cannot publish an atomic package: {error}"
        ) from error
    finally:
        if linked_path is not None and linked_created:
            linked_path.unlink(missing_ok=True)
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)


def _locked_persistent_package_root(preflight: dict[str, Any]) -> Path:
    root = _persistent_package_root()
    probe = preflight.get("persistence_probe", {})
    identity = root.stat()
    if (
        preflight.get("persistent_package_root") != str(root)
        or probe.get("device") != identity.st_dev
        or probe.get("inode") != identity.st_ino
    ):
        raise ValueError("PERSISTED_PACKAGE_ROOT changed after preflight")
    return root


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_persisted_package(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"persisted package is missing, empty, or unsafe: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"persisted package hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )


def _persist_package(source: Path, root: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"package source is missing, empty, or unsafe: {source}")
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"persistent package root is missing or unsafe: {root}")
    package_sha256 = _sha256(source)
    destination = root / f"cosyvoice3-lora-package-sha256-{package_sha256}.tar"
    reused_existing = destination.exists() or destination.is_symlink()
    if reused_existing:
        _verify_persisted_package(destination, package_sha256)
    else:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=root,
                prefix=f".{destination.name}.",
                suffix=".partial",
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
                with source.open("rb") as package:
                    shutil.copyfileobj(package, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            _verify_persisted_package(temporary_path, package_sha256)
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                reused_existing = True
            else:
                _fsync_directory(root)
            _verify_persisted_package(destination, package_sha256)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return {
        "schema_version": "1.0.0",
        "adaptation_mode": "lora",
        "package_sha256": package_sha256,
        "package_bytes": source.stat().st_size,
        "persisted_path": str(destination),
        "reused_existing": reused_existing,
    }


def _safe_name(value: str) -> str:
    path = Path(value)
    if not value or value in {".", ".."} or path.is_absolute() or len(path.parts) != 1:
        raise ValueError("SELECTED_ADAPTER_NAME must be one safe child directory")
    return value


def _run(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    environment: dict[str, str] | None = None,
    capture: bool = False,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=capture,
        text=capture,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() if capture else ""
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {command[0]}: {detail}"
        )
    return (result.stdout or "").strip() if capture else ""


def _git_status_paths(repository: Path) -> set[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip()}")
    output = result.stdout
    records = [record for record in output.split("\0") if record]
    paths: set[str] = set()
    index = 0
    while index < len(records):
        record = records[index]
        if len(record) < 4:
            raise ValueError("unexpected git status record")
        status = record[:2]
        paths.add(record[3:])
        index += 2 if "R" in status or "C" in status else 1
    return paths


def _patch_paths(patch: Path) -> set[str]:
    paths: set[str] = set()
    for line in patch.read_text(encoding="utf-8").splitlines():
        for prefix in ("+++ b/", "--- a/"):
            if line.startswith(prefix):
                paths.add(line[len(prefix) :])
    if not paths:
        raise ValueError(f"patch contains no repository paths: {patch}")
    return paths


def _verify_patched_upstream(
    upstream: Path, patches: tuple[Path, ...]
) -> dict[str, Any]:
    if not patches:
        raise ValueError("companion patch set is empty")
    touched = set().union(*(_patch_paths(patch) for patch in patches))
    dirty = _git_status_paths(upstream)
    unexpected = sorted(dirty - touched)
    if unexpected:
        raise ValueError(
            "CosyVoice checkout has changes outside the pinned patches: "
            + ", ".join(unexpected)
        )
    with tempfile.TemporaryDirectory() as temporary:
        environment = os.environ.copy()
        environment["GIT_INDEX_FILE"] = str(Path(temporary) / "index")
        _run(
            ["git", "read-tree", "HEAD"],
            cwd=upstream,
            environment=environment,
            capture=True,
        )
        for patch in patches:
            _run(
                ["git", "apply", "--cached", str(patch)],
                cwd=upstream,
                environment=environment,
                capture=True,
            )
        for relative in sorted(touched):
            expected = _run(
                ["git", "ls-files", "--stage", "--", relative],
                cwd=upstream,
                environment=environment,
                capture=True,
            )
            path = upstream / relative
            if not expected:
                if path.exists() or path.is_symlink():
                    raise ValueError(f"patched checkout should delete {relative}")
                continue
            expected_blob = expected.split()[1]
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"patched checkout file is missing or unsafe: {relative}"
                )
            observed_blob = _run(
                ["git", "hash-object", "--", relative], cwd=upstream, capture=True
            )
            if observed_blob != expected_blob:
                raise ValueError(
                    f"CosyVoice checkout does not match the pinned patch set: {relative}"
                )
    return {
        "upstream_revision": _run(
            ["git", "rev-parse", "HEAD"], cwd=upstream, capture=True
        ),
        "patches": [
            {
                "path": (
                    patch.relative_to(REPO_ROOT).as_posix()
                    if patch.is_relative_to(REPO_ROOT)
                    else patch.name
                ),
                "sha256": _sha256(patch),
            }
            for patch in patches
        ],
        "patched_paths": sorted(touched),
    }


def _tree_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"model tree contains a symlink: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        elif not path.is_dir():
            raise ValueError(f"model tree contains an unsupported entry: {path}")
    if not files:
        raise ValueError("model tree contains no files")
    digest = hashlib.sha256()
    for record in files:
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "file_count": len(files), "files": files}


def _audit_data_list(path: Path) -> tuple[dict[str, Any], set[Path]]:
    artifacts: set[Path] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        artifact = Path(value).expanduser()
        if not artifact.is_absolute():
            artifact = path.parent / artifact
        if artifact.is_symlink():
            raise ValueError(f"{path}:{line_number}: prepared artifact is a symlink")
        artifact = artifact.resolve()
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(
                f"{path}:{line_number}: prepared artifact is missing or empty"
            )
        if artifact in artifacts:
            raise ValueError(f"{path}:{line_number}: duplicate prepared artifact")
        artifacts.add(artifact)
    if not artifacts:
        raise ValueError(f"prepared data list contains no artifacts: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "artifacts": len(artifacts),
    }, artifacts


def _verify_dataset_lineage() -> dict[str, Any]:
    from instavar_voice_lab.lineage import verify_dataset_lineage

    train_root = _path("PREPARED_TRAIN_ROOT", directory=True)
    validation_root = _path("PREPARED_VALIDATION_ROOT", directory=True)
    for name, root in (("TRAIN_DATA_LIST", train_root), ("CV_DATA_LIST", validation_root)):
        data_list = _path(name)
        if not data_list.is_relative_to(root):
            raise ValueError(f"{name} must be inside its declared prepared-data root")
        _, artifacts = _audit_data_list(data_list)
        escaped = sorted(str(artifact) for artifact in artifacts if not artifact.is_relative_to(root))
        if escaped:
            raise ValueError(f"{name} references artifacts outside its declared prepared-data root: {escaped[0]}")
    document = json.loads(_path("DATASET_LINEAGE").read_text(encoding="utf-8"))
    return verify_dataset_lineage(
        document,
        producer_revision=_run(["git", "rev-parse", "HEAD"], capture=True),
        inputs={
            "raw_train": (_path("RAW_TRAIN_JSONL"), "file"),
            "raw_validation": (_path("RAW_VALIDATION_JSONL"), "file"),
            "raw_test": (_path("RAW_TEST_JSONL"), "file"),
        },
        outputs={
            "prepared_train": (train_root, "tree"),
            "prepared_validation": (validation_root, "tree"),
        },
    )


def _configured_max_epoch(config: Path) -> int:
    matches: list[str] = []
    in_train_conf = False
    train_conf_sections = 0
    for line in config.read_text(encoding="utf-8").splitlines():
        if re.fullmatch(r"train_conf\s*:\s*(?:#.*)?", line):
            in_train_conf = True
            train_conf_sections += 1
            continue
        if (
            in_train_conf
            and line.strip()
            and not line.lstrip().startswith("#")
            and not line[0].isspace()
        ):
            in_train_conf = False
        if in_train_conf:
            match = re.fullmatch(r"\s+max_epoch\s*:\s*([0-9]+)\s*(?:#.*)?", line)
            if match:
                matches.append(match.group(1))
    if train_conf_sections != 1:
        raise ValueError("TRAIN_CONFIG must contain exactly one train_conf section")
    if len(matches) != 1:
        raise ValueError(
            "TRAIN_CONFIG train_conf must contain exactly one integer max_epoch entry"
        )
    return int(matches[0])


def _stage_adapter(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"selected adapter must be a non-symlink directory: {source}")
    required = ("adapter_config.json", "adapter_model.safetensors")
    destination.mkdir(parents=True, exist_ok=False)
    for name in required:
        artifact = source / name
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or artifact.stat().st_size == 0
        ):
            raise ValueError(
                f"selected adapter is missing safe inference artifact: {artifact}"
            )
        shutil.copyfile(artifact, destination / name)
    config = json.loads(
        (destination / "adapter_config.json").read_text(encoding="utf-8")
    )
    if not isinstance(config, dict) or not config:
        raise ValueError("adapter_config.json must be a non-empty object")


def _archive(source: Path, destination: Path, *, arcname: str) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"archive source must be a non-symlink directory: {source}")
    count = 0
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"archive source contains a symlink: {path}")
        if path.is_file():
            count += 1
        elif not path.is_dir():
            raise ValueError(f"archive source contains an unsupported entry: {path}")
    if count == 0:
        raise ValueError("archive source contains no files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w") as archive:
        archive.add(source, arcname=arcname, recursive=True)


def _extract(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(source, "r") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("adapter archive is empty")
        seen: set[str] = set()
        for member in members:
            member_path = PurePosixPath(member.name)
            normalized = member_path.as_posix().rstrip("/")
            if (
                member_path.is_absolute()
                or not member_path.parts
                or member_path.parts[0] != "adapter"
                or any(part in {"", ".", ".."} for part in member_path.parts)
                or normalized in seen
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe adapter archive member: {member.name}")
            seen.add(normalized)
        archive.extractall(destination, members=members, filter="data")
    adapter = destination / "adapter"
    if not adapter.is_dir() or not any(path.is_file() for path in adapter.rglob("*")):
        raise ValueError("adapter archive did not contain a non-empty adapter root")
    return adapter


def _python_environment(cosyvoice: Path) -> dict[str, str]:
    matcha = cosyvoice / "third_party" / "Matcha-TTS"
    if not matcha.is_dir():
        raise FileNotFoundError(f"CosyVoice Matcha-TTS checkout not found: {matcha}")
    environment = os.environ.copy()
    paths = [str(matcha), str(cosyvoice)]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def _training_settings() -> dict[str, str]:
    settings = {
        "TRAIN_PROCESSES": os.environ.get("TRAIN_PROCESSES", "1"),
        "LORA_R": os.environ.get("LORA_R", "16"),
        "LORA_ALPHA": os.environ.get("LORA_ALPHA", "64"),
        "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
        "LORA_TARGET_MODULES": os.environ.get(
            "LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj"
        ),
        "USE_AMP": os.environ.get("USE_AMP", "1"),
        "EARLY_STOP_ON_CV_OVERFIT": os.environ.get("EARLY_STOP_ON_CV_OVERFIT", "1"),
        "FP16": os.environ.get("FP16", "0"),
        "RESUME_FROM": os.environ.get("RESUME_FROM", ""),
        "TRUST_RESUME_STATE": os.environ.get("TRUST_RESUME_STATE", "0"),
    }
    for name in ("TRAIN_PROCESSES", "LORA_R", "LORA_ALPHA"):
        if not settings[name].isdigit() or int(settings[name]) < 1:
            raise ValueError(f"{name} must be a positive integer")
    dropout = float(settings["LORA_DROPOUT"])
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("LORA_DROPOUT must be in [0, 1)")
    if not settings["LORA_TARGET_MODULES"].strip():
        raise ValueError("LORA_TARGET_MODULES must not be empty")
    for name in ("USE_AMP", "EARLY_STOP_ON_CV_OVERFIT", "FP16", "TRUST_RESUME_STATE"):
        if settings[name] not in {"0", "1"}:
            raise ValueError(f"{name} must equal 0 or 1")
    return settings


def _preflight() -> None:
    from instavar_voice_lab.corpus import audit_corpus

    lineage = _verify_dataset_lineage()
    experiment = json.loads(
        _path("INSTAVAR_VOICE_EXPERIMENT_MANIFEST").read_text(encoding="utf-8")
    )
    companion_revision = _run(["git", "rev-parse", "HEAD"], capture=True)
    if _git_status_paths(REPO_ROOT):
        raise ValueError(
            "companion repository must be clean; use a work directory outside the checkout"
        )
    cosyvoice = _path("COSYVOICE_DIR", directory=True)
    source = _verify_patched_upstream(cosyvoice, PATCHES)
    backend = experiment.get("backend", {})
    if backend.get("instavar_revision") != companion_revision:
        raise ValueError(
            "experiment backend.instavar_revision does not match the companion checkout"
        )
    if backend.get("upstream_revision") != source["upstream_revision"]:
        raise ValueError(
            "experiment backend.upstream_revision does not match the CosyVoice checkout"
        )
    engine = os.environ["TRAIN_ENGINE"].strip()
    if engine not in {"torch_ddp", "deepspeed"}:
        raise ValueError("TRAIN_ENGINE must be torch_ddp or deepspeed")
    if engine == "deepspeed":
        _path("DEEPSPEED_CONFIG")
    settings = _training_settings()
    if settings["RESUME_FROM"] and (
        engine != "torch_ddp"
        or settings["TRAIN_PROCESSES"] != "1"
        or settings["TRUST_RESUME_STATE"] != "1"
    ):
        raise ValueError(
            "RESUME_FROM requires single-process torch_ddp and TRUST_RESUME_STATE=1"
        )
    config = _path("TRAIN_CONFIG")
    source_max_epoch = _configured_max_epoch(config)
    effective_max_epoch = int(os.environ["MAX_EPOCH"])
    if effective_max_epoch < 1:
        raise ValueError("MAX_EPOCH must be positive")
    learning_rate = float(os.environ["LEARNING_RATE"])
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("LEARNING_RATE must be a finite positive number")
    deepspeed_config_sha256 = None
    if engine == "deepspeed":
        deepspeed_config = _path("DEEPSPEED_CONFIG")
        deepspeed_config_sha256 = _sha256(deepspeed_config)
        deepspeed = json.loads(deepspeed_config.read_text(encoding="utf-8"))
        deepspeed_lr = deepspeed.get("optimizer", {}).get("params", {}).get("lr")
        if not isinstance(deepspeed_lr, (int, float)) or not math.isclose(
            float(deepspeed_lr), learning_rate, rel_tol=1e-12
        ):
            raise ValueError(
                "DEEPSPEED_CONFIG optimizer learning rate does not match LEARNING_RATE"
            )
    splits = {
        "train": _path("RAW_TRAIN_JSONL"),
        "validation": _path("RAW_VALIDATION_JSONL"),
        "test": _path("RAW_TEST_JSONL"),
    }
    audit = audit_corpus(
        splits, group_field=os.environ.get("CORPUS_GROUP_FIELD") or None
    )
    if audit["status"] != "passed":
        raise ValueError("corpus audit failed: " + "; ".join(audit["errors"]))
    plan = json.loads(_path("GENERATION_PLAN").read_text(encoding="utf-8"))
    rows = [
        row
        for row in plan.get("samples", [])
        if row.get("candidate_id") == os.environ["CANDIDATE_ID"]
    ]
    if plan.get("schema_version") not in {"1.0.0", "1.1.0"} or not rows:
        raise ValueError(
            "GENERATION_PLAN must be schema 1.0.0 or 1.1.0 and contain CANDIDATE_ID rows"
        )
    pretrained = _path("PRETRAINED_DIR", directory=True)
    base_checkpoint = _path("BASE_LLM_CHECKPOINT")
    qwen_pretrain = _path("QWEN_PRETRAIN_DIR", directory=True)
    persistent_package_root = _persistent_package_root()
    persistence_probe = _probe_persistent_package_root(persistent_package_root)
    train_data, train_artifacts = _audit_data_list(_path("TRAIN_DATA_LIST"))
    cv_data, cv_artifacts = _audit_data_list(_path("CV_DATA_LIST"))
    overlap = sorted(str(path) for path in train_artifacts.intersection(cv_artifacts))
    if overlap:
        raise ValueError(
            "prepared train and CV lists overlap: " + ", ".join(overlap[:10])
        )
    _path("REFERENCE_AUDIO")
    _write_json(
        _work() / "preflight" / "preflight.json",
        {
            "schema_version": "1.0.0",
            "status": "passed",
            "companion_revision": companion_revision,
            "source": source,
            "train_engine": engine,
            "source_config_max_epoch": source_max_epoch,
            "effective_max_epoch": effective_max_epoch,
            "learning_rate": learning_rate,
            "training_settings": settings,
            "deepspeed_config_sha256": deepspeed_config_sha256,
            "selected_adapter_name": _safe_name(os.environ["SELECTED_ADAPTER_NAME"]),
            "training_config_sha256": _sha256(config),
            "base_llm_checkpoint_sha256": _sha256(base_checkpoint),
            "persistent_package_root": str(persistent_package_root),
            "persistence_probe": persistence_probe,
            "qwen_pretrain": _tree_manifest(qwen_pretrain),
            "pretrained_model": _tree_manifest(pretrained),
            "prepared_data": {"train": train_data, "cv": cv_data},
            "corpus_audit": audit,
            "generation_rows": len(rows),
            "dataset_lineage": lineage,
        },
    )


def _train() -> None:
    _verify_dataset_lineage()
    work = _work()
    cosyvoice = _path("COSYVOICE_DIR", directory=True)
    output = work / "train" / "output"
    settings = _training_settings()
    command = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={settings['TRAIN_PROCESSES']}",
        str(REPO_ROOT / "tools" / "train_cosyvoice3_lora.py"),
        "--train_engine",
        os.environ["TRAIN_ENGINE"],
        "--model",
        "llm",
        "--config",
        str(_path("TRAIN_CONFIG")),
        "--train_data",
        str(_path("TRAIN_DATA_LIST")),
        "--cv_data",
        str(_path("CV_DATA_LIST")),
        "--qwen_pretrain_path",
        str(_path("QWEN_PRETRAIN_DIR", directory=True)),
        "--checkpoint",
        str(_path("BASE_LLM_CHECKPOINT")),
        "--model_dir",
        str(output),
        "--tensorboard_dir",
        str(work / "train" / "tensorboard"),
        "--max_epoch",
        os.environ["MAX_EPOCH"],
        "--learning_rate",
        os.environ["LEARNING_RATE"],
        "--lora-r",
        settings["LORA_R"],
        "--lora-alpha",
        settings["LORA_ALPHA"],
        "--lora-dropout",
        settings["LORA_DROPOUT"],
        "--lora-target-modules",
        settings["LORA_TARGET_MODULES"],
    ]
    if os.environ["TRAIN_ENGINE"] == "deepspeed":
        command.extend(["--deepspeed_config", str(_path("DEEPSPEED_CONFIG"))])
    if settings["USE_AMP"] == "1":
        command.append("--use_amp")
    if settings["EARLY_STOP_ON_CV_OVERFIT"] == "1":
        command.append("--early-stop-on-cv-overfit")
    if os.environ["TRAIN_ENGINE"] == "torch_ddp" and settings["TRAIN_PROCESSES"] == "1":
        command.extend(["--guarded-checkpoints", "--trust-model-checkpoint"])
        if settings["RESUME_FROM"]:
            command.extend(
                ["--resume-from", settings["RESUME_FROM"], "--trust-resume-state"]
            )
    _run(command, cwd=cosyvoice, environment=_python_environment(cosyvoice))
    selected = output / _safe_name(os.environ["SELECTED_ADAPTER_NAME"])
    staged = work / "train" / "selected-adapter"
    _stage_adapter(selected, staged)
    _archive(staged, work / "train" / "selected-adapter.tar", arcname="adapter")


def _infer() -> None:
    work = _work()
    cosyvoice = _path("COSYVOICE_DIR", directory=True)
    adapter = _extract(
        work / "train" / "selected-adapter.tar", work / "infer" / "reload"
    )
    output = work / "infer" / "candidate.wav"
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "infer_cosyvoice3_lora.py"),
        "--pretrained-dir",
        str(_path("PRETRAINED_DIR", directory=True)),
        "--lora-dir",
        str(adapter),
        "--prompt-wav",
        str(_path("REFERENCE_AUDIO")),
        "--prompt-text",
        os.environ["REFERENCE_TEXT"],
        "--text",
        os.environ.get("SMOKE_TEXT", "A held-out sentence verifies adapter reload."),
        "--out-wav",
        str(output),
    ]
    if _training_settings()["FP16"] == "1":
        command.append("--fp16")
    _run(command, cwd=cosyvoice, environment=_python_environment(cosyvoice))
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("fresh-process adapter inference did not produce audio")


def _evaluate() -> None:
    work = _work()
    adapter = _extract(
        work / "train" / "selected-adapter.tar", work / "evaluate" / "reload"
    )
    output = work / "evaluate" / "output"
    command = [
        sys.executable,
        "tools/run_evaluation_suite.py",
        "--cosyvoice-dir",
        str(_path("COSYVOICE_DIR", directory=True)),
        "--pretrained-dir",
        str(_path("PRETRAINED_DIR", directory=True)),
        "--lora-dir",
        str(adapter),
        "--prompt-wav",
        str(_path("REFERENCE_AUDIO")),
        "--prompt-text",
        os.environ["REFERENCE_TEXT"],
        "--generation-plan",
        os.environ["GENERATION_PLAN"],
        "--candidate-id",
        os.environ["CANDIDATE_ID"],
        "--output-dir",
        str(output),
        "--allow-invalid-output",
    ]
    if _training_settings()["FP16"] == "1":
        command.append("--fp16")
    _run(command)
    raw_observations = output / "generation-observations.json"
    receipt = output / "generation-attempt-receipt.json"
    bound_observations = output / "objective-observations.json"
    plan = _path("GENERATION_PLAN")
    producer_revision = _run(["git", "rev-parse", "HEAD"], capture=True)
    _run([
        sys.executable, "-m", "instavar_voice_lab.cli", "build-generation-attempt-receipt",
        str(raw_observations), "--plan", str(plan), "--audio-base-dir", str(output),
        "--producer-name", "cosyvoice3-evaluation-runner", "--producer-revision", producer_revision,
        "--output", str(receipt),
    ])
    _run([
        sys.executable, "-m", "instavar_voice_lab.cli", "apply-generation-attempt-receipt",
        str(raw_observations), str(receipt), "--plan", str(plan), "--audio-base-dir", str(output),
        "--output", str(bound_observations),
    ])
    _archive(output, work / "evaluate" / "evaluation-bundle.tar", arcname="evaluation")


def _package() -> None:
    work = _work()
    preflight = json.loads(
        (work / "preflight" / "preflight.json").read_text(encoding="utf-8")
    )
    staging = work / "package" / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    sources = {
        "selected-adapter.tar": work / "train" / "selected-adapter.tar",
        "evaluation-bundle.tar": work / "evaluate" / "evaluation-bundle.tar",
        "preflight.json": work / "preflight" / "preflight.json",
        "smoke-candidate.wav": work / "infer" / "candidate.wav",
        "experiment-manifest.json": _path("INSTAVAR_VOICE_EXPERIMENT_MANIFEST"),
        "generation-plan.json": _path("GENERATION_PLAN"),
        "dataset-lineage.json": _path("DATASET_LINEAGE"),
        "training-config.yaml": _path("TRAIN_CONFIG"),
    }
    for name, source in sources.items():
        if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"package source is missing, empty, or unsafe: {source}")
        shutil.copyfile(source, staging / name)
    files = [
        {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(staging.iterdir())
        if path.is_file()
    ]
    _write_json(
        staging / "package-manifest.json",
        {
            "schema_version": "1.0.0",
            "backend_id": "cosyvoice3-lora-pytorch",
            "external_pretrained_model_sha256": preflight["pretrained_model"]["sha256"],
            "external_base_llm_sha256": preflight["base_llm_checkpoint_sha256"],
            "files": files,
            "evidence_boundary": (
                "The adapter and evidence completed the PyTorch lifecycle. Perceptual quality, "
                "merged-vLLM equivalence, and distribution rights remain separate gates."
            ),
        },
    )
    package = work / "package" / "adapter-package.tar"
    _archive(staging, package, arcname="package")
    receipt = _persist_package(package, _locked_persistent_package_root(preflight))
    _write_json(work / "package" / "persisted-package.json", receipt)


def run(stage: str) -> None:
    actions = {
        "preflight": _preflight,
        "train": _train,
        "infer": _infer,
        "evaluate": _evaluate,
        "package": _package,
    }
    if stage not in actions:
        raise ValueError(f"unknown lifecycle stage: {stage}")
    actions[stage]()
    if stage in {"preflight", "train"}:
        _verify_dataset_lineage()
    _write_json(
        Path(os.environ["INSTAVAR_VOICE_STAGE_RESULT"]),
        {"schema_version": "1.0.0", "stage": stage, "status": "passed"},
    )


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if len(values) != 1:
        print("usage: instavar_voice_lifecycle.py STAGE", file=sys.stderr)
        return 2
    try:
        run(values[0])
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        tarfile.TarError,
    ) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
