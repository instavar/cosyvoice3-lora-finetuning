from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from evaluation_contract import ADAPTER_ASSETS, BASE_ASSETS, resolve_inference_mode
from evaluation_plan import select_plan_rows


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
            "merged_pytorch_dir": None,
            "vllm_sampling_profile": "upstream",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_sample_selector_requires_one_exact_candidate_row(self) -> None:
        plan = {
            "samples": [
                {"candidate_id": "upstream", "sample_id": "prefix-seed-42"},
                {"candidate_id": "upstream", "sample_id": "tail-seed-42"},
                {"candidate_id": "seeded", "sample_id": "prefix-seed-42"},
            ]
        }
        self.assertEqual(
            select_plan_rows(plan, "upstream", "tail-seed-42"),
            [{"candidate_id": "upstream", "sample_id": "tail-seed-42"}],
        )
        self.assertEqual(len(select_plan_rows(plan, "upstream")), 2)
        with self.assertRaisesRegex(ValueError, "found 0"):
            select_plan_rows(plan, "upstream", "missing")
        duplicate = {
            "samples": [
                {"candidate_id": "upstream", "sample_id": "duplicate"},
                {"candidate_id": "upstream", "sample_id": "duplicate"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "found 2"):
            select_plan_rows(duplicate, "upstream", "duplicate")

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

    def test_vllm_sampling_profiles_are_confined_to_vllm_mode(self) -> None:
        with self.assertRaises(SystemExit):
            resolve_inference_mode(
                self.args("base", vllm_sampling_profile="request-seeded"),
                self.parser,
            )
        merged = self.args(
            "merged-vllm",
            lora_dir=str(self.adapter),
            vllm_dir=str(self.root / "merged-sampling"),
            vllm_sampling_profile="request-seeded-top-p-0.8",
        )
        self.assertEqual(resolve_inference_mode(merged, self.parser), "merged-vllm")
        with self.assertRaises(SystemExit):
            resolve_inference_mode(
                self.args(
                    "merged-vllm",
                    lora_dir=str(self.adapter),
                    vllm_dir=str(self.root / "other-merged"),
                    vllm_sampling_profile="unknown",
                ),
                self.parser,
            )

    def test_reloaded_merged_pytorch_requires_a_bound_artifact(self) -> None:
        artifact = self.root / "merged-pytorch"
        artifact.mkdir()
        weights = artifact / "merged_model.safetensors"
        weights.write_bytes(b"merged weights")
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        (artifact / "instavar-merged-pytorch.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "format": "safetensors_model_state_v1",
                    "model_class": "fixture.Model",
                    "exporter_revision": "a" * 40,
                    "source_adapter_sha256": "b" * 64,
                    "weights": {
                        "path": "merged_model.safetensors",
                        "sha256": digest,
                        "bytes": weights.stat().st_size,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        reloaded = self.args(
            "reloaded-merged-pytorch",
            merged_pytorch_dir=str(artifact),
        )
        self.assertEqual(
            resolve_inference_mode(reloaded, self.parser),
            "reloaded-merged-pytorch",
        )
        with self.assertRaises(SystemExit):
            resolve_inference_mode(
                self.args("reloaded-merged-pytorch"),
                self.parser,
            )
        with self.assertRaises(SystemExit):
            resolve_inference_mode(
                self.args(
                    "reloaded-merged-pytorch",
                    lora_dir=str(self.adapter),
                    merged_pytorch_dir=str(artifact),
                ),
                self.parser,
            )
        weights.write_bytes(b"tampered")
        with self.assertRaises(SystemExit):
            resolve_inference_mode(reloaded, self.parser)

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
        self.assertIn('"reloaded-merged-pytorch"', source)
        self.assertIn(
            'if args.inference_mode in {"adapter", "merged-pytorch", "merged-vllm"}:\n'
            "        from infer_cosyvoice3_lora import",
            source,
        )
        self.assertIn("load_merged_pytorch_artifact", source)
        self.assertIn('"artifact_mode": artifact_mode', source)
        self.assertIn('f"cosyvoice3_pytorch_{device_family}_{artifact_mode}"', source)
        self.assertIn('args.inference_mode == "merged-pytorch"', source)
        self.assertIn("merge_lora_into_cosyvoice3(cosyvoice, peft_model)", source)
        self.assertIn("BackgroundThreadFailureCapture", source)
        self.assertIn('"background_thread_failures"', source)
        self.assertIn('"background_thread_exception"', source)
        self.assertIn("begin_vllm_observation", source)
        self.assertIn("build_frontend_segmentation_receipt", source)
        self.assertIn('"frontend_segmentation": frontend_segmentation', source)
        self.assertIn('"request_count_matches"', source)
        generation_end = source.index(
            'if args.inference_mode == "merged-vllm":\n'
            '            observation["vllm_sampling"] = vllm_sampling_evidence(cosyvoice)'
        )
        generation_try = source.index("        background_capture =")
        self.assertGreater(generation_end, generation_try)

    def test_inference_helper_fails_after_preserving_background_failure_audio(self) -> None:
        source = (ROOT / "tools" / "infer_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        ast.parse(source)
        self.assertIn("BackgroundThreadFailureCapture", source)
        self.assertIn("preserved invalid output", source)

    def test_persisted_merged_export_uses_safetensors_and_no_overwrite(self) -> None:
        artifact_source = (ROOT / "tools" / "merged_pytorch_artifact.py").read_text(
            encoding="utf-8"
        )
        exporter_source = (
            ROOT / "tools" / "export_cosyvoice3_merged_pytorch.py"
        ).read_text(encoding="utf-8")
        ast.parse(artifact_source)
        ast.parse(exporter_source)
        self.assertIn("from safetensors.torch import save_model", artifact_source)
        self.assertIn("from safetensors.torch import load_model", artifact_source)
        self.assertIn("os.path.lexists(output_dir)", artifact_source)
        self.assertIn("validate_merged_pytorch_artifact", artifact_source)
        self.assertIn("merge_lora_into_cosyvoice3", exporter_source)


if __name__ == "__main__":
    unittest.main()
