from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from instavar_voice_lab.lineage import build_dataset_lineage

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
        required = {item["name"] for item in spec["required_environment"]}
        self.assertIn("PERSISTED_PACKAGE_ROOT", required)
        self.assertIn(
            "package/persisted-package.json", spec["expected_artifacts"]["package"]
        )
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

    def test_dataset_lineage_binds_raw_splits_to_both_prepared_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw: dict[str, Path] = {}
            for split in ("train", "validation", "test"):
                audio = root / f"{split}.wav"
                audio.write_bytes(b"audio")
                manifest = root / f"{split}.jsonl"
                manifest.write_text(
                    json.dumps({"audio": str(audio), "text": split}) + "\n"
                )
                raw[split] = manifest
            prepared_roots: dict[str, Path] = {}
            data_lists: dict[str, Path] = {}
            for split in ("train", "validation"):
                prepared = root / f"prepared-{split}"
                prepared.mkdir()
                artifact = prepared / "data.parquet"
                artifact.write_bytes(split.encode())
                data_list = prepared / "data.list"
                data_list.write_text("data.parquet\n")
                prepared_roots[split] = prepared
                data_lists[split] = data_list
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            receipt = root / "dataset-lineage.json"
            receipt.write_text(
                json.dumps(
                    build_dataset_lineage(
                        lineage_id="cosy-fixture-v1",
                        producer_repository="instavar/cosyvoice3-lora-finetuning",
                        producer_revision=revision,
                        inputs={
                            "raw_train": (raw["train"], "file"),
                            "raw_validation": (raw["validation"], "file"),
                            "raw_test": (raw["test"], "file"),
                        },
                        outputs={
                            "prepared_train": (prepared_roots["train"], "tree"),
                            "prepared_validation": (
                                prepared_roots["validation"],
                                "tree",
                            ),
                        },
                    )
                )
            )
            environment = {
                "RAW_TRAIN_JSONL": str(raw["train"]),
                "RAW_VALIDATION_JSONL": str(raw["validation"]),
                "RAW_TEST_JSONL": str(raw["test"]),
                "TRAIN_DATA_LIST": str(data_lists["train"]),
                "CV_DATA_LIST": str(data_lists["validation"]),
                "PREPARED_TRAIN_ROOT": str(prepared_roots["train"]),
                "PREPARED_VALIDATION_ROOT": str(prepared_roots["validation"]),
                "DATASET_LINEAGE": str(receipt),
            }
            with patch.dict(os.environ, environment, clear=False):
                report = LIFECYCLE._verify_dataset_lineage()
            self.assertEqual(report["lineage_id"], "cosy-fixture-v1")
            (prepared_roots["validation"] / "data.parquet").write_bytes(b"changed")
            with (
                patch.dict(os.environ, environment, clear=False),
                self.assertRaisesRegex(ValueError, "prepared_validation"),
            ):
                LIFECYCLE._verify_dataset_lineage()

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

    def test_extract_rejects_traversal_duplicates_and_special_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = {
                "traversal": [("adapter/../escape.bin", b"escape", tarfile.REGTYPE)],
                "duplicate": [
                    ("adapter/model.safetensors", b"first", tarfile.REGTYPE),
                    ("adapter/model.safetensors", b"second", tarfile.REGTYPE),
                ],
                "special": [("adapter/device", b"", tarfile.CHRTYPE)],
                "sibling": [("notes.txt", b"notes", tarfile.REGTYPE)],
            }
            for name, entries in cases.items():
                source = root / f"{name}.tar"
                with tarfile.open(source, "w") as archive:
                    for member_name, payload, member_type in entries:
                        member = tarfile.TarInfo(member_name)
                        member.type = member_type
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "unsafe adapter archive member"),
                ):
                    LIFECYCLE._extract(source, root / f"reload-{name}")

    def test_persist_package_is_content_addressed_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "package.tar"
            source.write_bytes(b"immutable package")
            store = root / "store"
            store.mkdir()
            first = LIFECYCLE._persist_package(source, store)
            destination = Path(first["persisted_path"])
            self.assertTrue(
                destination.name.startswith("cosyvoice3-lora-package-sha256-")
            )
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertFalse(first["reused_existing"])
            self.assertTrue(
                LIFECYCLE._persist_package(source, store)["reused_existing"]
            )
            destination.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                LIFECYCLE._persist_package(source, store)

    def test_package_root_is_bound_to_path_device_and_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, paths = self._persistence_fixture(root)
            store = paths["store"]
            identity = store.stat()
            preflight = {
                "persistent_package_root": str(store.resolve()),
                "persistence_probe": {
                    "device": identity.st_dev,
                    "inode": identity.st_ino,
                },
            }
            with patch.dict(os.environ, environment, clear=False):
                self.assertEqual(
                    LIFECYCLE._locked_persistent_package_root(preflight),
                    store.resolve(),
                )
                store.rename(root / "retired-store")
                store.mkdir()
                self.assertEqual(store.stat().st_dev, identity.st_dev)
                with self.assertRaisesRegex(ValueError, "changed after preflight"):
                    LIFECYCLE._locked_persistent_package_root(preflight)

    def test_persistent_package_root_rejects_each_immutable_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, paths = self._persistence_fixture(root)
            protected = {
                "work": "lifecycle work directory",
                "cosyvoice": "CosyVoice checkout",
                "pretrained": "pretrained model directory",
                "qwen": "Qwen dependency directory",
                "prepared_train": "prepared training tree",
                "prepared_validation": "prepared validation tree",
                "base_dir": "base LLM checkpoint directory",
            }
            for key, message in protected.items():
                candidate = paths[key] / "packages"
                candidate.mkdir()
                with (
                    self.subTest(key=key),
                    patch.dict(
                        os.environ,
                        {**environment, "PERSISTED_PACKAGE_ROOT": str(candidate)},
                        clear=False,
                    ),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    LIFECYCLE._persistent_package_root()

    def test_package_stage_persists_archive_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, paths = self._persistence_fixture(root)
            work = paths["work"]
            for path in (
                work / "preflight",
                work / "train",
                work / "evaluate",
                work / "infer",
                work / "package",
            ):
                path.mkdir(parents=True, exist_ok=True)
            identity = paths["store"].stat()
            (work / "preflight" / "preflight.json").write_text(
                json.dumps(
                    {
                        "persistent_package_root": str(paths["store"].resolve()),
                        "persistence_probe": {
                            "device": identity.st_dev,
                            "inode": identity.st_ino,
                        },
                        "pretrained_model": {"sha256": "a" * 64},
                        "base_llm_checkpoint_sha256": "b" * 64,
                    }
                )
            )
            (work / "train" / "selected-adapter.tar").write_bytes(b"adapter")
            (work / "evaluate" / "evaluation-bundle.tar").write_bytes(b"evaluation")
            (work / "infer" / "candidate.wav").write_bytes(b"wav")
            controls = {}
            for name in ("experiment", "plan", "lineage", "config"):
                path = root / f"{name}.fixture"
                path.write_bytes(name.encode())
                controls[name] = path
            environment.update(
                {
                    "INSTAVAR_VOICE_EXPERIMENT_MANIFEST": str(controls["experiment"]),
                    "GENERATION_PLAN": str(controls["plan"]),
                    "DATASET_LINEAGE": str(controls["lineage"]),
                    "TRAIN_CONFIG": str(controls["config"]),
                }
            )
            with patch.dict(os.environ, environment, clear=False):
                LIFECYCLE._package()
            package = work / "package" / "adapter-package.tar"
            receipt = json.loads(
                (work / "package" / "persisted-package.json").read_text()
            )
            self.assertEqual(
                Path(receipt["persisted_path"]).read_bytes(), package.read_bytes()
            )

    @staticmethod
    def _persistence_fixture(root: Path) -> tuple[dict[str, str], dict[str, Path]]:
        paths = {
            name: root / name
            for name in (
                "work",
                "cosyvoice",
                "pretrained",
                "qwen",
                "prepared_train",
                "prepared_validation",
                "base_dir",
                "store",
            )
        }
        for path in paths.values():
            path.mkdir()
        base_checkpoint = paths["base_dir"] / "llm.pt"
        base_checkpoint.write_bytes(b"base")
        environment = {
            "INSTAVAR_VOICE_WORK_DIR": str(paths["work"]),
            "COSYVOICE_DIR": str(paths["cosyvoice"]),
            "PRETRAINED_DIR": str(paths["pretrained"]),
            "QWEN_PRETRAIN_DIR": str(paths["qwen"]),
            "PREPARED_TRAIN_ROOT": str(paths["prepared_train"]),
            "PREPARED_VALIDATION_ROOT": str(paths["prepared_validation"]),
            "BASE_LLM_CHECKPOINT": str(base_checkpoint),
            "PERSISTED_PACKAGE_ROOT": str(paths["store"]),
        }
        return environment, paths


if __name__ == "__main__":
    unittest.main()
