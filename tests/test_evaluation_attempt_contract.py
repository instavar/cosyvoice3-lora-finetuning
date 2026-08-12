from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class EvaluationAttemptContractTests(unittest.TestCase):
    def test_invalid_outputs_are_preserved_for_attempt_binding(self) -> None:
        runner = (ROOT / "tools" / "run_evaluation_suite.py").read_text(encoding="utf-8")
        lifecycle = (ROOT / "scripts" / "instavar_voice_lifecycle.py").read_text(encoding="utf-8")
        ast.parse(runner)
        ast.parse(lifecycle)
        self.assertIn("allow-invalid-output", runner)
        self.assertIn('not in {"1.0.0", "1.1.0"}', runner)
        self.assertNotIn("max_memory_allocated()) if torch.cuda.is_available() else 0", runner)
        self.assertIn("build-generation-attempt-receipt", lifecycle)
        self.assertIn("apply-generation-attempt-receipt", lifecycle)
        self.assertIn("objective-observations.json", lifecycle)


if __name__ == "__main__":
    unittest.main()
