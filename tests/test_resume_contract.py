from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.cosyvoice_resume_contract import (
    LOCK_NAME,
    OPTIMIZER_STATE_NAME,
    ResumeContractError,
    SCHEDULER_STATE_NAME,
    acquire_output_lock,
    build_contract,
    checkpoint_children,
    epoch_checkpoint_name,
    evaluator_lora_artifact_paths,
    prune_owned_checkpoints,
    publish_checkpoint,
    require_fresh_output,
    resolve_checkpoint,
    validate_checkpoint,
)


class ResumeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.base = self.root / "llm.pt"
        self.base.write_bytes(b"base")
        self.config = self.root / "config.yaml"
        self.config.write_text("train_conf: {}\n", encoding="utf-8")
        self.qwen = self.root / "qwen"
        self.qwen.mkdir()
        (self.qwen / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        self.train_list, self.train_data = self._data("train")
        self.cv_list, self.cv_data = self._data("cv")
        self.source = self.root / "trainer.py"
        self.source.write_text("print('fixture')\n", encoding="utf-8")
        self.contract = self._contract()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _data(self, name: str) -> tuple[Path, Path]:
        artifact = self.root / f"{name}.parquet"
        artifact.write_bytes(name.encode())
        listing = self.root / f"{name}.list"
        listing.write_text(str(artifact) + "\n", encoding="utf-8")
        return listing, artifact

    def _contract(self, *, max_epoch: int = 5) -> dict:
        return build_contract(
            output_dir=self.output,
            base_checkpoint=self.base,
            config_file=self.config,
            qwen_pretrain=self.qwen,
            data_files={
                "train": [self.train_list, self.train_data],
                "cross_validation": [self.cv_list, self.cv_data],
            },
            source_files=[self.source],
            training_config={"max_epoch": max_epoch, "train_engine": "torch_ddp"},
            runtime={"python": "fixture", "world_size": 1},
        )

    @staticmethod
    def _adapter_saver(directory: Path) -> None:
        (directory / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(b"adapter")

    @staticmethod
    def _runtime_saver(path: Path) -> None:
        path.write_bytes(b"trusted-pickle-fixture")
        (path.parent / OPTIMIZER_STATE_NAME).write_bytes(b"optimizer-fixture")
        (path.parent / SCHEDULER_STATE_NAME).write_bytes(b"scheduler-fixture")

    def _checkpoint(
        self,
        epoch: int,
        *,
        step: int | None = None,
        contract: dict | None = None,
    ) -> Path:
        return publish_checkpoint(
            output_dir=self.output,
            completed_epoch=epoch,
            completed_step=epoch * 10 if step is None else step,
            contract=contract or self.contract,
            adapter_saver=self._adapter_saver,
            runtime_state_saver=self._runtime_saver,
            monitor_state={"best_cv_loss": 3.0, "cv_no_improve_epochs": epoch},
        )

    def test_checkpoint_name_uses_completed_epoch(self) -> None:
        self.assertEqual(epoch_checkpoint_name(0), "resume_epoch_000000")
        self.assertEqual(epoch_checkpoint_name(12), "resume_epoch_000012")
        with self.assertRaises(ResumeContractError):
            epoch_checkpoint_name(-1)

    def test_exact_trusted_checkpoint_validates(self) -> None:
        checkpoint = self._checkpoint(1)
        selected, state = validate_checkpoint(
            checkpoint,
            output_dir=self.output,
            expected_contract=self.contract,
            trust_resume_state=True,
            world_size=1,
            train_engine="torch_ddp",
        )
        self.assertEqual(selected, checkpoint)
        self.assertEqual(state["completed_epoch"], 1)
        self.assertEqual(state["monitor_state"]["cv_no_improve_epochs"], 1)

    def test_evaluator_lora_artifact_roles_are_independent(self) -> None:
        checkpoint = self._checkpoint(1)
        artifacts = evaluator_lora_artifact_paths(checkpoint)
        self.assertEqual(
            {role: path.name for role, path in artifacts.items()},
            {
                "model_state": "adapter_model.safetensors",
                "optimizer_state": OPTIMIZER_STATE_NAME,
                "scheduler_state": SCHEDULER_STATE_NAME,
                "trainer_state": "training-state.json",
                "rng_state": "runtime-state.pt",
            },
        )

    def test_legacy_combined_runtime_state_remains_resumable(self) -> None:
        checkpoint = publish_checkpoint(
            output_dir=self.output,
            completed_epoch=1,
            completed_step=10,
            contract=self.contract,
            adapter_saver=self._adapter_saver,
            runtime_state_saver=lambda path: path.write_bytes(b"legacy-combined-state"),
            monitor_state={"best_cv_loss": 3.0},
        )
        selected, _ = validate_checkpoint(
            checkpoint,
            output_dir=self.output,
            expected_contract=self.contract,
            trust_resume_state=True,
            world_size=1,
            train_engine="torch_ddp",
        )
        self.assertEqual(selected, checkpoint)
        with self.assertRaisesRegex(ResumeContractError, "omits decomposed state"):
            evaluator_lora_artifact_paths(checkpoint)

    def test_evaluator_mapping_rejects_ambiguous_model_and_hardlinks(self) -> None:
        def ambiguous_adapter(directory: Path) -> None:
            self._adapter_saver(directory)
            (directory / "adapter_model.bin").write_bytes(b"second-adapter")

        ambiguous = publish_checkpoint(
            output_dir=self.output,
            completed_epoch=1,
            completed_step=10,
            contract=self.contract,
            adapter_saver=ambiguous_adapter,
            runtime_state_saver=self._runtime_saver,
            monitor_state={},
        )
        with self.assertRaisesRegex(ResumeContractError, "exactly one adapter"):
            evaluator_lora_artifact_paths(ambiguous)

        other_output = self.root / "hardlink-output"
        other_output.mkdir()
        hardlink_contract = build_contract(
            output_dir=other_output,
            base_checkpoint=self.base,
            config_file=self.config,
            qwen_pretrain=self.qwen,
            data_files={"train": [self.train_list], "cross_validation": [self.cv_list]},
            source_files=[self.source],
            training_config={"max_epoch": 5, "train_engine": "torch_ddp"},
            runtime={"python": "fixture", "world_size": 1},
        )

        def hardlinked_runtime(path: Path) -> None:
            path.write_bytes(b"runtime")
            optimizer = path.parent / OPTIMIZER_STATE_NAME
            optimizer.write_bytes(b"shared-state")
            os.link(optimizer, path.parent / SCHEDULER_STATE_NAME)

        hardlinked = publish_checkpoint(
            output_dir=other_output,
            completed_epoch=1,
            completed_step=10,
            contract=hardlink_contract,
            adapter_saver=self._adapter_saver,
            runtime_state_saver=hardlinked_runtime,
            monitor_state={},
        )
        with self.assertRaisesRegex(ResumeContractError, "must not share hardlinks"):
            evaluator_lora_artifact_paths(hardlinked)

    def test_trainer_writes_decomposed_state_before_publication(self) -> None:
        source = (
            Path(__file__).parents[1] / "tools" / "train_cosyvoice3_lora.py"
        ).read_text(encoding="utf-8")
        self.assertIn("path.parent / OPTIMIZER_STATE_NAME", source)
        self.assertIn("path.parent / SCHEDULER_STATE_NAME", source)

    def test_resume_requires_explicit_trust(self) -> None:
        checkpoint = self._checkpoint(1)
        with self.assertRaisesRegex(ResumeContractError, "pickle-capable"):
            validate_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=False,
                world_size=1,
                train_engine="torch_ddp",
            )

    def test_deepspeed_and_multi_rank_resume_are_rejected(self) -> None:
        checkpoint = self._checkpoint(1)
        for world_size, engine in ((2, "torch_ddp"), (1, "deepspeed")):
            with self.assertRaisesRegex(ResumeContractError, "collective protocol"):
                validate_checkpoint(
                    checkpoint,
                    output_dir=self.output,
                    expected_contract=self.contract,
                    trust_resume_state=True,
                    world_size=world_size,
                    train_engine=engine,
                )

    def test_checkpoint_must_be_a_direct_immutable_child(self) -> None:
        checkpoint = self._checkpoint(1)
        nested = self.output / "nested"
        nested.mkdir()
        moved = nested / checkpoint.name
        checkpoint.rename(moved)
        with self.assertRaisesRegex(ResumeContractError, "direct child"):
            resolve_checkpoint(moved, self.output)

    def test_checkpoint_symlink_is_rejected(self) -> None:
        checkpoint = self._checkpoint(1)
        link = self.output / "resume_epoch_000002"
        link.symlink_to(checkpoint, target_is_directory=True)
        with self.assertRaisesRegex(ResumeContractError, "symlinks"):
            resolve_checkpoint(link, self.output)

    def test_checkpoint_is_published_without_partial_directory(self) -> None:
        checkpoint = self._checkpoint(1)
        self.assertTrue((checkpoint / "resume-contract.json").is_file())
        self.assertFalse(
            any(
                item.name.startswith(".resume_epoch_") for item in self.output.iterdir()
            )
        )

    def test_failed_publication_cleans_only_its_partial(self) -> None:
        protected = self.output / "keep.txt"
        protected.write_text("keep\n", encoding="utf-8")

        def fail(_: Path) -> None:
            raise RuntimeError("fixture failure")

        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            publish_checkpoint(
                output_dir=self.output,
                completed_epoch=1,
                completed_step=10,
                contract=self.contract,
                adapter_saver=fail,
                runtime_state_saver=self._runtime_saver,
                monitor_state={},
            )
        self.assertEqual(protected.read_text(encoding="utf-8"), "keep\n")
        self.assertFalse(
            any(
                item.name.startswith(".resume_epoch_") for item in self.output.iterdir()
            )
        )

    def test_existing_destination_is_never_adopted(self) -> None:
        checkpoint = self._checkpoint(1)
        marker = checkpoint / "marker.txt"
        marker.write_text("owned\n", encoding="utf-8")
        with self.assertRaisesRegex(ResumeContractError, "overwrite or adopt"):
            self._checkpoint(1)
        self.assertEqual(marker.read_text(encoding="utf-8"), "owned\n")

    def test_checkpoint_byte_drift_is_rejected(self) -> None:
        checkpoint = self._checkpoint(1)
        (checkpoint / "runtime-state.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ResumeContractError, "file identity drift"):
            validate_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=True,
                world_size=1,
                train_engine="torch_ddp",
            )

    def test_data_drift_changes_the_contract(self) -> None:
        checkpoint = self._checkpoint(1)
        self.train_data.write_bytes(b"changed")
        changed = self._contract()
        with self.assertRaisesRegex(ResumeContractError, "contract drift"):
            validate_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=changed,
                trust_resume_state=True,
                world_size=1,
                train_engine="torch_ddp",
            )

    def test_resume_requires_newest_owned_checkpoint(self) -> None:
        older = self._checkpoint(1)
        self._checkpoint(2)
        with self.assertRaisesRegex(ResumeContractError, "newest owned"):
            validate_checkpoint(
                older,
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=True,
                world_size=1,
                train_engine="torch_ddp",
            )

    def test_unowned_numeric_sibling_fails_before_resume(self) -> None:
        unowned = self.output / "resume_epoch_000001"
        unowned.mkdir()
        selected = self._checkpoint(2)
        with self.assertRaisesRegex(ResumeContractError, "no safe"):
            validate_checkpoint(
                selected,
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=True,
                world_size=1,
                train_engine="torch_ddp",
            )

    def test_completed_epoch_target_is_rejected(self) -> None:
        final_contract = self._contract(max_epoch=2)
        checkpoint = self._checkpoint(1, contract=final_contract)
        with self.assertRaisesRegex(ResumeContractError, "reached"):
            validate_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=final_contract,
                trust_resume_state=True,
                world_size=1,
                train_engine="torch_ddp",
            )

    def test_completed_early_stop_target_is_rejected(self) -> None:
        contract = self._contract()
        contract["training_config"]["early_stop_on_cv_overfit"] = True
        checkpoint = publish_checkpoint(
            output_dir=self.output,
            completed_epoch=1,
            completed_step=10,
            contract=contract,
            adapter_saver=self._adapter_saver,
            runtime_state_saver=self._runtime_saver,
            monitor_state={"cv_overfit_flag": 1},
        )
        with self.assertRaisesRegex(ResumeContractError, "early-stop target"):
            validate_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=contract,
                trust_resume_state=True,
                world_size=1,
                train_engine="torch_ddp",
            )

    def test_retention_deletes_only_oldest_owned_checkpoints(self) -> None:
        first = self._checkpoint(0)
        self._checkpoint(1)
        self._checkpoint(2)
        (self.output / "epoch_2_whole").mkdir()
        victims = prune_owned_checkpoints(
            self.output,
            keep_last=2,
            expected_contract=self.contract,
        )
        self.assertEqual(victims, [first])
        self.assertFalse(first.exists())
        self.assertTrue((self.output / "epoch_2_whole").is_dir())

    def test_retention_fails_closed_on_unowned_numeric_child(self) -> None:
        (self.output / "resume_epoch_000000").mkdir()
        self._checkpoint(1)
        with self.assertRaisesRegex(ResumeContractError, "no safe"):
            prune_owned_checkpoints(
                self.output,
                keep_last=1,
                expected_contract=self.contract,
            )
        self.assertTrue((self.output / "resume_epoch_000000").is_dir())

    def test_fresh_output_requires_an_empty_namespace(self) -> None:
        (self.output / LOCK_NAME).write_text("pid=1\n", encoding="utf-8")
        require_fresh_output(self.output)
        (self.output / "epoch_0_whole").mkdir()
        with self.assertRaisesRegex(ResumeContractError, "empty model_dir"):
            require_fresh_output(self.output)

    def test_output_lock_rejects_second_writer(self) -> None:
        first = acquire_output_lock(self.output)
        try:
            with self.assertRaisesRegex(ResumeContractError, "Another guarded writer"):
                acquire_output_lock(self.output)
        finally:
            first.close()

    def test_output_lock_rejects_hardlink_without_truncation(self) -> None:
        protected = self.root / "protected.txt"
        protected.write_text("keep me\n", encoding="utf-8")
        (self.output / LOCK_NAME).hardlink_to(protected)
        with self.assertRaisesRegex(
            ResumeContractError, "unsafe ownership or link count"
        ):
            acquire_output_lock(self.output)
        self.assertEqual(protected.read_text(encoding="utf-8"), "keep me\n")

    def test_checkpoint_children_ignore_inference_exports(self) -> None:
        checkpoint = self._checkpoint(1)
        (self.output / "epoch_1_whole").mkdir()
        (self.output / "epoch_1_whole.yaml").write_text("epoch: 1\n")
        self.assertEqual(checkpoint_children(self.output), [checkpoint])

    def test_state_and_sidecar_progress_must_agree(self) -> None:
        checkpoint = self._checkpoint(1)
        state_path = checkpoint / "training-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["completed_epoch"] = 0
        state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ResumeContractError, "file identity drift"):
            validate_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=True,
                world_size=1,
                train_engine="torch_ddp",
            )


if __name__ == "__main__":
    unittest.main()
