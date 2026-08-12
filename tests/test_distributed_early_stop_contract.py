from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class DistributedEarlyStopContractTests(unittest.TestCase):
    def test_trainer_synchronizes_before_breaking(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(encoding="utf-8")
        ast.parse(source)
        sync_index = source.index("should_stop = synchronize_early_stop")
        break_index = source.index("            break", sync_index)
        self.assertLess(sync_index, break_index)

    def test_sync_uses_max_across_ranks(self) -> None:
        source = (ROOT / "tools" / "distributed_early_stop.py").read_text(encoding="utf-8")
        self.assertIn("dist.all_reduce", source)
        self.assertIn("dist.ReduceOp.MAX", source)

    def test_evaluation_runner_uses_exact_plan_seed_without_retry(self) -> None:
        source = (ROOT / "tools" / "run_evaluation_suite.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('matcha_dir = cosyvoice_dir / "third_party" / "Matcha-TTS"', source)
        self.assertIn('seed_everything(int(row["seed"]))', source)
        self.assertIn('"implausible_audio_output"', source)
        self.assertIn("duration >= args.min_audio_seconds", source)
        self.assertNotIn("max_seed_tries", source)
        self.assertIn("generation-observations.json", source)
        self.assertIn("artifact set id and sha256 must be provided together", source)
        self.assertIn('"runtime_id": runtime_id', source)
        self.assertIn('"artifact_set_sha256": args.artifact_set_sha256', source)
        self.assertIn('"observation_schema_version": "1.0.0"', source)


if __name__ == "__main__":
    unittest.main()
