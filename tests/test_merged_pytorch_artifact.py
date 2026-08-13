from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from merged_pytorch_artifact import (
    MANIFEST_NAME,
    MAX_MANIFEST_BYTES,
    WEIGHTS_NAME,
    fingerprint_stable_file,
    validate_merged_pytorch_artifact,
)


class MergedPyTorchArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact = self.root / "artifact"
        self.artifact.mkdir()
        self.weights = self.artifact / WEIGHTS_NAME
        self.weights.write_bytes(b"safe tensor fixture")
        digest = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        self.manifest = {
            "schema_version": "1.0.0",
            "format": "safetensors_model_state_v1",
            "model_class": "fixture.Model",
            "exporter_revision": "a" * 40,
            "source_adapter_sha256": "b" * 64,
            "weights": {
                "path": WEIGHTS_NAME,
                "sha256": digest,
                "bytes": self.weights.stat().st_size,
            },
        }
        (self.artifact / MANIFEST_NAME).write_text(
            json.dumps(self.manifest) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_stable_file_and_manifest_validate(self) -> None:
        digest, size = fingerprint_stable_file(self.weights)
        self.assertEqual(digest, self.manifest["weights"]["sha256"])
        self.assertEqual(size, self.manifest["weights"]["bytes"])
        self.assertEqual(
            validate_merged_pytorch_artifact(self.artifact)["schema_version"],
            "1.0.0",
        )

    def test_tampered_weights_fail_closed(self) -> None:
        self.weights.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_merged_pytorch_artifact(self.artifact)

    def test_symlinked_artifact_inputs_fail_closed(self) -> None:
        weights_target = self.root / "weights-target"
        weights_target.write_bytes(self.weights.read_bytes())
        self.weights.unlink()
        self.weights.symlink_to(weights_target)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            validate_merged_pytorch_artifact(self.artifact)

        artifact_link = self.root / "artifact-link"
        artifact_link.symlink_to(self.artifact, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "non-symlink directory"):
            validate_merged_pytorch_artifact(artifact_link)

    def test_invalid_manifest_fields_fail_closed(self) -> None:
        manifest_path = self.artifact / MANIFEST_NAME
        for patch in (
            {"schema_version": "9.0.0"},
            {"format": "pickle"},
            {"model_class": ""},
            {"exporter_revision": "short"},
            {"source_adapter_sha256": "not-a-hash"},
            {"weights": {"path": "../escape", "sha256": "0" * 64, "bytes": 1}},
            {"weights": {"path": WEIGHTS_NAME, "sha256": "not-a-hash", "bytes": 1}},
            {"weights": {"path": WEIGHTS_NAME, "sha256": "0" * 64, "bytes": 0}},
        ):
            with self.subTest(patch=patch):
                candidate = dict(self.manifest)
                candidate.update(patch)
                manifest_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_merged_pytorch_artifact(self.artifact)

    def test_manifest_size_and_encoding_fail_closed(self) -> None:
        manifest_path = self.artifact / MANIFEST_NAME
        manifest_path.write_bytes(b"x" * (MAX_MANIFEST_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            validate_merged_pytorch_artifact(self.artifact)
        manifest_path.write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "unreadable"):
            validate_merged_pytorch_artifact(self.artifact)


if __name__ == "__main__":
    unittest.main()
