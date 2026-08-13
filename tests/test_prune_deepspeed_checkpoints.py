import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.prune_deepspeed_checkpoints import (
    LOCK_NAME,
    MAX_METADATA_BYTES,
    PruneError,
    _validated_plan,
    build_plan,
    execute_plan,
)


class PruneContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _checkpoint(self, tag: str, step: int, loss: float) -> None:
        (self.model_dir / f"{tag}.yaml").write_text(
            f"step: {step}\nloss_dict:\n  loss: {loss}\n", encoding="utf-8"
        )
        payload = self.model_dir / tag
        payload.mkdir()
        (payload / "model.bin").write_bytes(f"weights-{tag}".encode())

    def _plan(self, *, keep_latest: int = 1, keep_best: int = 1) -> dict:
        return build_plan(
            self.model_dir,
            ["epoch_1", "epoch_2", "epoch_3"],
            keep_latest=keep_latest,
            keep_best=keep_best,
            metric="loss",
            higher_is_better=False,
        )

    def _write_plan(self, plan: dict) -> Path:
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        return path

    def test_plan_requires_explicit_tags(self) -> None:
        with self.assertRaisesRegex(PruneError, "explicit --owned-tag"):
            build_plan(
                self.model_dir,
                [],
                keep_latest=1,
                keep_best=1,
                metric="loss",
                higher_is_better=False,
            )

    def test_keep_best_honors_counts_above_one(self) -> None:
        for tag, step, loss in (("epoch_1", 1, 3.0), ("epoch_2", 2, 1.0), ("epoch_3", 3, 2.0)):
            self._checkpoint(tag, step, loss)
        plan = self._plan(keep_latest=0, keep_best=2)
        self.assertEqual(plan["selection"]["keep_tags"], ["epoch_2", "epoch_3"])
        self.assertEqual(plan["selection"]["remove_tags"], ["epoch_1"])

    def test_metadata_only_tag_is_rejected(self) -> None:
        (self.model_dir / "foreign.yaml").write_text("step: 1\n", encoding="utf-8")
        with self.assertRaisesRegex(PruneError, "no payload"):
            build_plan(
                self.model_dir,
                ["foreign"],
                keep_latest=0,
                keep_best=0,
                metric="loss",
                higher_is_better=False,
            )

    def test_malformed_and_duplicate_yaml_fail_closed(self) -> None:
        for text in ("step: [\n", "step: 1\nstep: 2\n"):
            (self.model_dir / "bad.yaml").write_text(text, encoding="utf-8")
            (self.model_dir / "bad").mkdir(exist_ok=True)
            with self.assertRaises(PruneError):
                build_plan(
                    self.model_dir,
                    ["bad"],
                    keep_latest=0,
                    keep_best=0,
                    metric="loss",
                    higher_is_better=False,
                )

    def test_oversized_metadata_fails_before_yaml_parse(self) -> None:
        (self.model_dir / "huge.yaml").write_bytes(b"x" * (MAX_METADATA_BYTES + 1))
        (self.model_dir / "huge").mkdir()
        with self.assertRaisesRegex(PruneError, "too large"):
            build_plan(
                self.model_dir,
                ["huge"],
                keep_latest=0,
                keep_best=0,
                metric="loss",
                higher_is_better=False,
            )

    def test_unsafe_tag_and_symlink_payload_are_rejected(self) -> None:
        for tag in ("../escape", "x" * 129):
            with self.assertRaisesRegex(PruneError, "Unsafe checkpoint tag"):
                build_plan(
                    self.model_dir,
                    [tag],
                    keep_latest=0,
                    keep_best=0,
                    metric="loss",
                    higher_is_better=False,
                )
        (self.model_dir / "linked.yaml").write_text("step: 1\n", encoding="utf-8")
        outside = self.root / "outside"
        outside.mkdir()
        (self.model_dir / "linked").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(PruneError, "Symlinks"):
            build_plan(
                self.model_dir,
                ["linked"],
                keep_latest=0,
                keep_best=0,
                metric="loss",
                higher_is_better=False,
            )

    def test_hard_linked_file_is_rejected_without_changing_bytes(self) -> None:
        self._checkpoint("epoch_1", 1, 1.0)
        model = self.model_dir / "epoch_1" / "model.bin"
        alias = self.root / "alias.bin"
        os.link(model, alias)
        before = alias.read_bytes()
        with self.assertRaisesRegex(PruneError, "Hard-linked"):
            build_plan(
                self.model_dir,
                ["epoch_1"],
                keep_latest=0,
                keep_best=0,
                metric="loss",
                higher_is_better=False,
            )
        self.assertEqual(alias.read_bytes(), before)

    def test_execution_requires_exact_confirmation(self) -> None:
        self._checkpoint("epoch_1", 1, 1.0)
        plan = build_plan(
            self.model_dir,
            ["epoch_1"],
            keep_latest=0,
            keep_best=0,
            metric="loss",
            higher_is_better=False,
        )
        path = self._write_plan(plan)
        with self.assertRaisesRegex(PruneError, "does not match"):
            execute_plan(path, "0" * 64)
        self.assertTrue((self.model_dir / "epoch_1").exists())

    def test_plan_tampering_is_rejected(self) -> None:
        self._checkpoint("epoch_1", 1, 1.0)
        plan = build_plan(
            self.model_dir,
            ["epoch_1"],
            keep_latest=0,
            keep_best=0,
            metric="loss",
            higher_is_better=False,
        )
        digest = plan["plan_sha256"]
        plan["selection"]["remove_tags"] = []
        path = self._write_plan(plan)
        with self.assertRaisesRegex(PruneError, "digest"):
            execute_plan(path, digest)

    def test_byte_drift_is_rejected_before_any_rename(self) -> None:
        for tag, step, loss in (("epoch_1", 1, 2.0), ("epoch_2", 2, 1.0)):
            self._checkpoint(tag, step, loss)
        plan = build_plan(
            self.model_dir,
            ["epoch_1", "epoch_2"],
            keep_latest=1,
            keep_best=0,
            metric="loss",
            higher_is_better=False,
        )
        path = self._write_plan(plan)
        (self.model_dir / "epoch_1" / "model.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(PruneError, "changed after plan"):
            execute_plan(path, plan["plan_sha256"])
        self.assertTrue((self.model_dir / "epoch_1.yaml").exists())
        self.assertTrue((self.model_dir / "epoch_1").exists())

    def test_execute_removes_only_reviewed_victim(self) -> None:
        for tag, step, loss in (("epoch_1", 1, 3.0), ("epoch_2", 2, 1.0), ("epoch_3", 3, 2.0)):
            self._checkpoint(tag, step, loss)
        foreign = self.model_dir / "foreign"
        foreign.mkdir()
        (foreign / "precious.bin").write_bytes(b"keep")
        plan = self._plan()
        path = self._write_plan(plan)
        removed = execute_plan(path, plan["plan_sha256"])
        self.assertEqual(removed, ["epoch_1"])
        self.assertFalse((self.model_dir / "epoch_1").exists())
        self.assertFalse((self.model_dir / "epoch_1.yaml").exists())
        self.assertTrue((self.model_dir / "epoch_2").exists())
        self.assertTrue((self.model_dir / "epoch_3").exists())
        self.assertEqual((foreign / "precious.bin").read_bytes(), b"keep")
        self.assertTrue((self.model_dir / LOCK_NAME).exists())

    def test_execution_resumes_exact_partial_staging(self) -> None:
        self._checkpoint("epoch_1", 1, 1.0)
        plan = build_plan(
            self.model_dir,
            ["epoch_1"],
            keep_latest=0,
            keep_best=0,
            metric="loss",
            higher_is_better=False,
        )
        path = self._write_plan(plan)
        suffix = plan["plan_sha256"][:16]
        source = self.model_dir / "epoch_1"
        staged = self.model_dir / f".epoch_1.prune-{suffix}"
        source.rename(staged)
        (self.model_dir / "epoch_1.pt").write_bytes(b"not-in-plan")
        with self.assertRaisesRegex(PruneError, "changed after plan"):
            execute_plan(path, plan["plan_sha256"])
        (self.model_dir / "epoch_1.pt").unlink()
        removed = execute_plan(path, plan["plan_sha256"])
        self.assertEqual(removed, ["epoch_1"])
        self.assertFalse(staged.exists())
        self.assertFalse((self.model_dir / "epoch_1.yaml").exists())

    def test_execution_resumes_after_one_component_was_deleted(self) -> None:
        self._checkpoint("epoch_1", 1, 1.0)
        plan = build_plan(
            self.model_dir,
            ["epoch_1"],
            keep_latest=0,
            keep_best=0,
            metric="loss",
            higher_is_better=False,
        )
        path = self._write_plan(plan)
        (self.model_dir / "epoch_1.yaml").unlink()
        removed = execute_plan(path, plan["plan_sha256"])
        self.assertEqual(removed, ["epoch_1"])
        self.assertFalse((self.model_dir / "epoch_1").exists())

    def test_lock_hard_link_is_rejected_without_changing_alias(self) -> None:
        self._checkpoint("epoch_1", 1, 1.0)
        lock = self.model_dir / LOCK_NAME
        lock.write_bytes(b"do-not-touch")
        alias = self.root / "lock-alias"
        os.link(lock, alias)
        plan = build_plan(
            self.model_dir,
            ["epoch_1"],
            keep_latest=0,
            keep_best=0,
            metric="loss",
            higher_is_better=False,
        )
        path = self._write_plan(plan)
        with self.assertRaisesRegex(PruneError, "Unsafe existing"):
            execute_plan(path, plan["plan_sha256"])
        self.assertEqual(alias.read_bytes(), b"do-not-touch")
        self.assertTrue((self.model_dir / "epoch_1").exists())

    def test_plan_file_symlink_is_rejected(self) -> None:
        self._checkpoint("epoch_1", 1, 1.0)
        plan = build_plan(
            self.model_dir,
            ["epoch_1"],
            keep_latest=1,
            keep_best=0,
            metric="loss",
            higher_is_better=False,
        )
        real = self._write_plan(plan)
        link = self.root / "linked-plan.json"
        link.symlink_to(real)
        with self.assertRaisesRegex(PruneError, "safely open"):
            _validated_plan(link, plan["plan_sha256"])

    def test_cli_requires_separate_plan_and_exact_execution(self) -> None:
        self._checkpoint("epoch_1", 1, 2.0)
        self._checkpoint("epoch_2", 2, 1.0)
        script = Path(__file__).parents[1] / "tools" / "prune_deepspeed_checkpoints.py"
        plan_path = self.root / "cli-plan.json"
        planned = subprocess.run(
            [
                sys.executable,
                str(script),
                "--plan-out",
                str(plan_path),
                "--model-dir",
                str(self.model_dir),
                "--owned-tag",
                "epoch_1",
                "--owned-tag",
                "epoch_2",
                "--keep-latest",
                "1",
                "--keep-best",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("without deleting checkpoints", planned.stdout)
        self.assertTrue((self.model_dir / "epoch_1").exists())
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        executed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--execute-plan",
                str(plan_path),
                "--confirm-plan-sha256",
                plan["plan_sha256"],
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Pruned 1 explicitly adopted checkpoints", executed.stdout)
        self.assertFalse((self.model_dir / "epoch_1").exists())
        self.assertTrue((self.model_dir / "epoch_2").exists())


if __name__ == "__main__":
    unittest.main()
