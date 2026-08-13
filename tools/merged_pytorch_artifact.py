"""Persist and reload a content-bound merged PyTorch LLM without pickle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
MANIFEST_NAME = "instavar-merged-pytorch.json"
WEIGHTS_NAME = "merged_model.safetensors"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_MANIFEST_BYTES = 1024 * 1024


def _identity(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def fingerprint_stable_file(path: Path) -> tuple[str, int]:
    """Hash one regular file and reject identity changes during the read."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact file must be a regular non-symlink: {path}")
    before_path = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        before_file = os.fstat(handle.fileno())
        if _identity(before_file) != _identity(before_path):
            raise RuntimeError(f"artifact path identity changed before reading: {path}")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        after_file = os.fstat(handle.fileno())
    after_path = path.stat()
    if _identity(before_file) != _identity(after_file):
        raise RuntimeError(f"artifact file changed while reading: {path}")
    if _identity(after_file) != _identity(after_path):
        raise RuntimeError(f"artifact path identity changed after reading: {path}")
    return digest.hexdigest(), size


def _read_stable_manifest(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"merged PyTorch manifest is missing: {path}")
    before_path = path.stat()
    if before_path.st_size > MAX_MANIFEST_BYTES:
        raise ValueError("merged PyTorch manifest exceeds the 1 MiB limit")
    with path.open("rb") as handle:
        before_file = os.fstat(handle.fileno())
        if _identity(before_file) != _identity(before_path):
            raise RuntimeError("merged PyTorch manifest identity changed before reading")
        payload = handle.read(MAX_MANIFEST_BYTES + 1)
        after_file = os.fstat(handle.fileno())
    after_path = path.stat()
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("merged PyTorch manifest exceeds the 1 MiB limit")
    if _identity(before_file) != _identity(after_file):
        raise RuntimeError("merged PyTorch manifest changed while reading")
    if _identity(after_file) != _identity(after_path):
        raise RuntimeError("merged PyTorch manifest path changed after reading")
    return payload


def validate_merged_pytorch_artifact(root: Path) -> dict[str, Any]:
    """Validate the persisted artifact and return its manifest."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("merged PyTorch artifact must be a non-symlink directory")
    root_identity = _identity(root.stat())
    manifest_path = root / MANIFEST_NAME
    try:
        manifest = json.loads(_read_stable_manifest(manifest_path).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"merged PyTorch manifest is unreadable: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"merged PyTorch manifest schema_version must equal {SCHEMA_VERSION}")
    if manifest.get("format") != "safetensors_model_state_v1":
        raise ValueError("merged PyTorch manifest format is unsupported")
    if not isinstance(manifest.get("model_class"), str) or not manifest["model_class"]:
        raise ValueError("merged PyTorch manifest model_class must be non-empty")
    if not isinstance(manifest.get("exporter_revision"), str) or not GIT_SHA_RE.fullmatch(
        manifest["exporter_revision"]
    ):
        raise ValueError("merged PyTorch manifest exporter_revision is invalid")
    if not isinstance(manifest.get("source_adapter_sha256"), str) or not SHA256_RE.fullmatch(
        manifest["source_adapter_sha256"]
    ):
        raise ValueError("merged PyTorch manifest source_adapter_sha256 is invalid")
    weights = manifest.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("merged PyTorch manifest weights must be an object")
    if weights.get("path") != WEIGHTS_NAME:
        raise ValueError(f"merged PyTorch manifest weights path must equal {WEIGHTS_NAME}")
    expected_sha = weights.get("sha256")
    expected_bytes = weights.get("bytes")
    if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
        raise ValueError("merged PyTorch weights sha256 is invalid")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes <= 0:
        raise ValueError("merged PyTorch weights bytes must be a positive integer")
    actual_sha, actual_bytes = fingerprint_stable_file(root / WEIGHTS_NAME)
    if actual_sha != expected_sha or actual_bytes != expected_bytes:
        raise ValueError("merged PyTorch weights do not match the manifest")
    if _identity(root.stat()) != root_identity:
        raise RuntimeError("merged PyTorch artifact directory identity changed during validation")
    return manifest


def export_merged_pytorch_artifact(
    model: Any,
    output_dir: Path,
    *,
    exporter_revision: str,
    source_adapter_sha256: str,
) -> dict[str, Any]:
    """Publish a no-overwrite safetensors artifact through a staging directory."""
    if not GIT_SHA_RE.fullmatch(exporter_revision):
        raise ValueError("exporter revision must be a lowercase 40-character Git SHA")
    if not SHA256_RE.fullmatch(source_adapter_sha256):
        raise ValueError("source adapter sha256 must be a lowercase SHA-256 digest")
    if os.path.lexists(output_dir):
        raise FileExistsError(f"merged PyTorch output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.parent.is_symlink():
        raise ValueError("merged PyTorch output parent must not be a symlink")
    output_dir = output_dir.parent.resolve() / output_dir.name
    if os.path.lexists(output_dir):
        raise FileExistsError(f"merged PyTorch output already exists: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)

    from safetensors.torch import save_model

    weights_path = staging / WEIGHTS_NAME
    save_model(
        model,
        str(weights_path),
        metadata={
            "format": "pt",
            "instavar_schema_version": SCHEMA_VERSION,
            "source_adapter_sha256": source_adapter_sha256,
        },
    )
    weights_sha256, weights_bytes = fingerprint_stable_file(weights_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format": "safetensors_model_state_v1",
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "exporter_revision": exporter_revision,
        "source_adapter_sha256": source_adapter_sha256,
        "weights": {
            "path": WEIGHTS_NAME,
            "sha256": weights_sha256,
            "bytes": weights_bytes,
        },
        "evidence_boundary": (
            "The manifest binds the serialized merged model state and declared source "
            "adapter. It does not prove loader behavior, numerical equivalence, TTS "
            "quality, or runtime equivalence."
        ),
    }
    manifest_path = staging / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in (weights_path, manifest_path):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    directory_fd = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.rename(staging, output_dir)
    return validate_merged_pytorch_artifact(output_dir)


def load_merged_pytorch_artifact(model: Any, root: Path) -> dict[str, Any]:
    """Verify and load a persisted merged model into an existing architecture."""
    manifest = validate_merged_pytorch_artifact(root)
    weights_path = root / WEIGHTS_NAME
    root_identity = _identity(root.stat())
    weights_identity = _identity(weights_path.stat())

    from safetensors.torch import load_model

    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        device = "cpu"
    missing, unexpected = load_model(model, weights_path, strict=True, device=device)
    if missing or unexpected:
        raise RuntimeError(
            f"merged PyTorch state mismatch: missing={missing}, unexpected={unexpected}"
        )
    if _identity(root.stat()) != root_identity:
        raise RuntimeError("merged PyTorch artifact directory changed during load")
    if _identity(weights_path.stat()) != weights_identity:
        raise RuntimeError("merged PyTorch weights changed during load")
    model.eval()
    return manifest
