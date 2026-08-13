from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from vllm_sampling_controls import (  # noqa: E402
    install_vllm_sampling_control,
    set_vllm_request_seed,
    validate_standalone_sampling_request,
    vllm_sampling_evidence,
)


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLlm:
    def __init__(self):
        self.vllm = object()
        self.captured = None

    def inference_wrapper(self, lm_input, sampling, min_len, max_len, uuid):
        from vllm import SamplingParams

        self.captured = SamplingParams(
            top_k=sampling,
            min_tokens=min_len,
            max_tokens=max_len,
        )
        yield 7
        yield 8


class FakeCosyVoice:
    def __init__(self):
        self.model = types.SimpleNamespace(llm=FakeLlm())


class VllmSamplingControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_vllm = sys.modules.get("vllm")
        sys.modules["vllm"] = types.SimpleNamespace(SamplingParams=FakeSamplingParams)

    def tearDown(self) -> None:
        if self.previous_vllm is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = self.previous_vllm

    def test_upstream_profile_preserves_existing_parameters(self) -> None:
        cosyvoice = FakeCosyVoice()
        install_vllm_sampling_control(cosyvoice, "upstream")
        set_vllm_request_seed(cosyvoice, 42)
        self.assertEqual(
            list(cosyvoice.model.llm.inference_wrapper(None, 25, 2, 20, "x")),
            [7, 8],
        )
        self.assertEqual(
            cosyvoice.model.llm.captured.kwargs,
            {"top_k": 25, "min_tokens": 2, "max_tokens": 20},
        )
        self.assertIsNone(vllm_sampling_evidence(cosyvoice)["parameters"]["seed"])

    def test_request_seed_is_injected_and_patch_is_restored(self) -> None:
        cosyvoice = FakeCosyVoice()
        module = sys.modules["vllm"]
        original = module.SamplingParams
        install_vllm_sampling_control(cosyvoice, "request-seeded")
        set_vllm_request_seed(cosyvoice, 314159)
        stream = cosyvoice.model.llm.inference_wrapper(None, 25, 2, 20, "x")
        self.assertEqual(next(stream), 7)
        self.assertIs(module.SamplingParams, original)
        self.assertEqual(list(stream), [8])
        self.assertEqual(cosyvoice.model.llm.captured.kwargs["seed"], 314159)

    def test_seeded_top_p_profile_changes_one_additional_parameter(self) -> None:
        cosyvoice = FakeCosyVoice()
        install_vllm_sampling_control(cosyvoice, "request-seeded-top-p-0.8")
        set_vllm_request_seed(cosyvoice, 20260812)
        list(cosyvoice.model.llm.inference_wrapper(None, 25, 2, 20, "x"))
        self.assertEqual(cosyvoice.model.llm.captured.kwargs["seed"], 20260812)
        self.assertEqual(cosyvoice.model.llm.captured.kwargs["top_p"], 0.8)
        evidence = vllm_sampling_evidence(cosyvoice)
        self.assertFalse(evidence["parameters"]["pytorch_ras_equivalent"])

    def test_seed_is_required_and_bounded(self) -> None:
        cosyvoice = FakeCosyVoice()
        install_vllm_sampling_control(cosyvoice, "request-seeded")
        with self.assertRaisesRegex(RuntimeError, "must be set"):
            list(cosyvoice.model.llm.inference_wrapper(None, 25, 2, 20, "x"))
        with self.assertRaises(ValueError):
            set_vllm_request_seed(cosyvoice, -1)
        with self.assertRaises(TypeError):
            set_vllm_request_seed(cosyvoice, True)

    def test_duplicate_install_fails_closed(self) -> None:
        cosyvoice = FakeCosyVoice()
        install_vllm_sampling_control(cosyvoice, "upstream")
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            install_vllm_sampling_control(cosyvoice, "upstream")

    def test_seeded_standalone_batch_is_rejected_as_ambiguous(self) -> None:
        validate_standalone_sampling_request("request-seeded", 42, 1)
        with self.assertRaisesRegex(ValueError, "exactly one text"):
            validate_standalone_sampling_request("request-seeded", 42, 2)
        with self.assertRaisesRegex(ValueError, "require --seed"):
            validate_standalone_sampling_request("request-seeded", None, 1)


if __name__ == "__main__":
    unittest.main()
