from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cosyvoice_lifecycle", ROOT / "scripts" / "instavar_voice_lifecycle.py"
)
assert SPEC and SPEC.loader
LIFECYCLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFECYCLE)


class LifecycleBackendTests(unittest.TestCase):
    def test_backend_binds_pytorch_lora(self) -> None:
        spec = json.loads((ROOT / "instavar-voice-backend.json").read_text())
        self.assertEqual(spec["schema_version"], "1.2.0")
        self.assertEqual(spec["capability_binding"]["adaptation"], "lora")
        self.assertEqual(spec["capability_binding"]["runtime_ids"], ["pytorch"])
        for stage in ("preflight", "train", "infer", "evaluate", "package"):
            self.assertEqual(spec["commands"][stage][-1], stage)

    def test_selected_adapter_is_one_safe_child(self) -> None:
        self.assertEqual(LIFECYCLE._safe_name("epoch_12_whole"), "epoch_12_whole")
        for unsafe in ("", ".", "..", "../epoch", "nested/epoch", "/epoch"):
            with self.assertRaises(ValueError):
                LIFECYCLE._safe_name(unsafe)

    def test_max_epoch_requires_one_explicit_integer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.yaml"
            config.write_text("train_conf:\n  max_epoch: 20 # lifecycle\n")
            self.assertEqual(LIFECYCLE._configured_max_epoch(config), 20)
            config.write_text("train_conf:\n  max_epoch: 20\nother:\n  max_epoch: 30\n")
            self.assertEqual(LIFECYCLE._configured_max_epoch(config), 20)
            config.write_text("train_conf:\n  max_epoch: 20\n  max_epoch: 30\n")
            with self.assertRaises(ValueError):
                LIFECYCLE._configured_max_epoch(config)

    def test_trainer_applies_explicit_epoch_and_learning_rate_overrides(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text()
        self.assertIn('"--max_epoch"', source)
        self.assertIn('"--learning_rate"', source)
        self.assertIn('configs["train_conf"]["max_epoch"] = args.max_epoch', source)
        self.assertIn(
            'configs["train_conf"]["optim_conf"]["lr"] = args.learning_rate', source
        )

    def test_prepared_data_list_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "data.parquet"
            artifact.write_bytes(b"fixture")
            data_list = root / "data.list"
            data_list.write_text("data.parquet\n")
            report, artifacts = LIFECYCLE._audit_data_list(data_list)
            self.assertEqual(report["artifacts"], 1)
            self.assertEqual(artifacts, {artifact.resolve()})
            data_list.write_text("data.parquet\ndata.parquet\n")
            with self.assertRaises(ValueError):
                LIFECYCLE._audit_data_list(data_list)

    def test_training_settings_reject_invalid_processes_and_flags(self) -> None:
        with (
            patch.dict(os.environ, {"TRAIN_PROCESSES": "0"}, clear=False),
            self.assertRaises(ValueError),
        ):
            LIFECYCLE._training_settings()
        with (
            patch.dict(os.environ, {"USE_AMP": "yes"}, clear=False),
            self.assertRaises(ValueError),
        ):
            LIFECYCLE._training_settings()

    def test_patched_upstream_must_equal_head_plus_exact_patch_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "upstream"
            repository.mkdir()
            for command in (
                ["git", "init", "-q"],
                ["git", "config", "user.name", "Fixture"],
                ["git", "config", "user.email", "fixture@example.invalid"],
            ):
                subprocess.run(command, cwd=repository, check=True)
            tracked = repository / "tracked.txt"
            tracked.write_text("old\n")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "fixture"], cwd=repository, check=True
            )
            patch_file = root / "change.patch"
            patch_file.write_text(
                "diff --git a/tracked.txt b/tracked.txt\n"
                "--- a/tracked.txt\n"
                "+++ b/tracked.txt\n"
                "@@ -1 +1 @@\n"
                "-old\n"
                "+new\n"
            )
            subprocess.run(
                ["git", "apply", str(patch_file)], cwd=repository, check=True
            )
            evidence = LIFECYCLE._verify_patched_upstream(repository, (patch_file,))
            self.assertEqual(evidence["patched_paths"], ["tracked.txt"])
            (repository / "unexpected.txt").write_text("unexpected\n")
            with self.assertRaisesRegex(ValueError, "outside the pinned patches"):
                LIFECYCLE._verify_patched_upstream(repository, (patch_file,))

    def test_adapter_staging_excludes_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "adapter_config.json").write_text('{"r": 16}\n')
            (source / "adapter_model.safetensors").write_bytes(b"adapter")
            (source / "optimizer.pt").write_bytes(b"unsafe-pickle-placeholder")
            destination = root / "destination"
            LIFECYCLE._stage_adapter(source, destination)
            self.assertEqual(
                {path.name for path in destination.iterdir()},
                {"adapter_config.json", "adapter_model.safetensors"},
            )


if __name__ == "__main__":
    unittest.main()
