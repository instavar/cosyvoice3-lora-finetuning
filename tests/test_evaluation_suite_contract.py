from __future__ import annotations

import argparse
import ast
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from evaluation_contract import ADAPTER_ASSETS, BASE_ASSETS, resolve_inference_mode


class EvaluationSuiteContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.adapter = self.root / "adapter"
        self.base.mkdir()
        self.adapter.mkdir()
        for asset in BASE_ASSETS:
            (self.base / asset).write_bytes(b"base")
        for asset in ADAPTER_ASSETS:
            (self.adapter / asset).write_bytes(b"adapter")
        self.prompt = self.root / "prompt.wav"
        self.prompt.write_bytes(b"wav")
        self.parser = argparse.ArgumentParser()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def args(self, mode: str | None, **overrides: object) -> argparse.Namespace:
        values = {
            "inference_mode": mode,
            "pretrained_dir": str(self.base),
            "lora_dir": None,
            "prompt_wav": str(self.prompt),
            "vllm_dir": None,
            "reuse_vllm_dir": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_base_is_explicit_and_forbids_adapted_artifacts(self) -> None:
        self.assertEqual(resolve_inference_mode(self.args("base"), self.parser), "base")
        with self.assertRaises(SystemExit):
            resolve_inference_mode(self.args(None), self.parser)
        with self.assertRaises(SystemExit):
            resolve_inference_mode(
                self.args("base", lora_dir=str(self.adapter)), self.parser
            )
        with self.assertRaises(SystemExit):
            resolve_inference_mode(
                self.args("base", vllm_dir=str(self.root / "merged")), self.parser
            )

    def test_adapter_and_legacy_adapter_require_exact_adapter_assets(self) -> None:
        explicit = self.args("adapter", lora_dir=str(self.adapter))
        legacy = self.args(None, lora_dir=str(self.adapter))
        self.assertEqual(resolve_inference_mode(explicit, self.parser), "adapter")
        self.assertEqual(resolve_inference_mode(legacy, self.parser), "adapter")
        (self.adapter / "adapter_model.safetensors").unlink()
        with self.assertRaises(SystemExit):
            resolve_inference_mode(explicit, self.parser)

    def test_merged_vllm_requires_adapter_and_export_path(self) -> None:
        merged = self.args(
            "merged-vllm",
            lora_dir=str(self.adapter),
            vllm_dir=str(self.root / "merged"),
        )
        self.assertEqual(resolve_inference_mode(merged, self.parser), "merged-vllm")
        with self.assertRaises(SystemExit):
            resolve_inference_mode(
                self.args("merged-vllm", lora_dir=str(self.adapter)), self.parser
            )
        existing = self.root / "existing-merged"
        existing.mkdir()
        with self.assertRaises(SystemExit):
            resolve_inference_mode(
                self.args(
                    "merged-vllm",
                    lora_dir=str(self.adapter),
                    vllm_dir=str(existing),
                    reuse_vllm_dir=True,
                ),
                self.parser,
            )
        (existing / "config.json").write_text("{}\n", encoding="utf-8")
        self.assertEqual(
            resolve_inference_mode(
                self.args(
                    "merged-vllm",
                    lora_dir=str(self.adapter),
                    vllm_dir=str(existing),
                    reuse_vllm_dir=True,
                ),
                self.parser,
            ),
            "merged-vllm",
        )

    def test_merged_pytorch_requires_adapter_and_forbids_vllm_paths(self) -> None:
        merged = self.args("merged-pytorch", lora_dir=str(self.adapter))
        self.assertEqual(
            resolve_inference_mode(merged, self.parser), "merged-pytorch"
        )
        with self.assertRaises(SystemExit):
            resolve_inference_mode(self.args("merged-pytorch"), self.parser)
        with self.assertRaises(SystemExit):
            resolve_inference_mode(
                self.args(
                    "merged-pytorch",
                    lora_dir=str(self.adapter),
                    vllm_dir=str(self.root / "merged"),
                ),
                self.parser,
            )

    def test_missing_base_asset_and_symlinked_prompt_fail_closed(self) -> None:
        (self.base / "llm.pt").unlink()
        with self.assertRaises(SystemExit):
            resolve_inference_mode(self.args("base"), self.parser)
        (self.base / "llm.pt").write_bytes(b"base")
        link = self.root / "prompt-link.wav"
        link.symlink_to(self.prompt)
        with self.assertRaises(SystemExit):
            resolve_inference_mode(
                self.args("base", prompt_wav=str(link)), self.parser
            )

    def test_runner_records_mode_and_loads_lora_only_for_adapted_modes(self) -> None:
        source = (ROOT / "tools" / "run_evaluation_suite.py").read_text(
            encoding="utf-8"
        )
        ast.parse(source)
        self.assertIn(
            'choices=("base", "adapter", "merged-pytorch", "merged-vllm")',
            source,
        )
        self.assertIn(
            'if args.inference_mode != "base":\n'
            "        from infer_cosyvoice3_lora import",
            source,
        )
        self.assertIn('if args.inference_mode != "base"', source)
        self.assertIn('"artifact_mode": artifact_mode', source)
        self.assertIn('f"cosyvoice3_pytorch_{device_family}_{artifact_mode}"', source)
        self.assertIn('args.inference_mode == "merged-pytorch"', source)
        self.assertIn("merge_lora_into_cosyvoice3(cosyvoice, peft_model)", source)
        self.assertIn("BackgroundThreadFailureCapture", source)
        self.assertIn('"background_thread_failures"', source)
        self.assertIn('"background_thread_exception"', source)

    def test_inference_helper_fails_after_preserving_background_failure_audio(self) -> None:
        source = (ROOT / "tools" / "infer_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        ast.parse(source)
        self.assertIn("BackgroundThreadFailureCapture", source)
        self.assertIn("preserved invalid output", source)


if __name__ == "__main__":
    unittest.main()
