from __future__ import annotations

import sys
import types
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from vllm_sampling_controls import (  # noqa: E402
    begin_vllm_observation,
    install_vllm_sampling_control,
    set_vllm_request_seed,
    validate_standalone_sampling_request,
    vllm_sampling_evidence,
)


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_p = kwargs.get("top_p", 1.0)
        self.top_k = kwargs.get("top_k", 0)
        self.min_p = kwargs.get("min_p", 0.0)
        self.seed = kwargs.get("seed")
        self.min_tokens = kwargs.get("min_tokens", 0)
        self.max_tokens = kwargs.get("max_tokens", 16)


class FakeSamplingStates:
    def __init__(self):
        self.vocab_size = 4096
        self.temperature = types.SimpleNamespace(np=[0.0])
        self.top_p = types.SimpleNamespace(np=[0.0])
        self.top_k = types.SimpleNamespace(np=[0])
        self.min_p = types.SimpleNamespace(np=[0.0])
        self.seeds = types.SimpleNamespace(np=[0])

    def add_request(self, req_idx, sampling_params):
        self.temperature.np[req_idx] = sampling_params.temperature
        self.top_p.np[req_idx] = sampling_params.top_p
        self.top_k.np[req_idx] = sampling_params.top_k
        self.min_p.np[req_idx] = sampling_params.min_p
        self.seeds.np[req_idx] = (
            sampling_params.seed if sampling_params.seed is not None else -777
        )


class FakeLlm:
    def __init__(self):
        self.vllm = object()
        self.captured = None

    def inference_wrapper(self, lm_input, sampling, min_len, max_len, uuid):
        from vllm import SamplingParams
        from vllm.v1.worker.gpu.sample.states import SamplingStates

        self.captured = SamplingParams(
            top_k=sampling,
            min_tokens=min_len,
            max_tokens=max_len,
        )
        self.states = SamplingStates()
        self.states.add_request(0, self.captured)
        yield 7
        yield 8


class FakeCosyVoice:
    def __init__(self):
        self.model = types.SimpleNamespace(llm=FakeLlm())


class FakeNoStateLlm(FakeLlm):
    def inference_wrapper(self, lm_input, sampling, min_len, max_len, uuid):
        from vllm import SamplingParams

        self.captured = SamplingParams(
            top_k=sampling,
            min_tokens=min_len,
            max_tokens=max_len,
        )
        yield 7


class FakeNoStateCosyVoice:
    def __init__(self):
        self.model = types.SimpleNamespace(llm=FakeNoStateLlm())


class VllmSamplingControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module_names = (
            "vllm",
            "vllm.v1",
            "vllm.v1.worker",
            "vllm.v1.worker.gpu",
            "vllm.v1.worker.gpu.sample",
            "vllm.v1.worker.gpu.sample.states",
        )
        self.previous_modules = {
            name: sys.modules.get(name) for name in self.module_names
        }
        for name in self.module_names:
            sys.modules[name] = types.ModuleType(name)
        sys.modules["vllm"].__version__ = "0.15.1"
        sys.modules["vllm"].SamplingParams = FakeSamplingParams
        sys.modules["vllm.v1.worker.gpu.sample.states"].SamplingStates = (
            FakeSamplingStates
        )

    def tearDown(self) -> None:
        for name, previous in self.previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def test_upstream_profile_preserves_existing_parameters(self) -> None:
        cosyvoice = FakeCosyVoice()
        install_vllm_sampling_control(cosyvoice, "upstream")
        begin_vllm_observation(cosyvoice, "sample-1", 42)
        self.assertEqual(
            list(cosyvoice.model.llm.inference_wrapper(None, 25, 2, 20, "x")),
            [7, 8],
        )
        self.assertEqual(
            cosyvoice.model.llm.captured.kwargs,
            {"top_k": 25, "min_tokens": 2, "max_tokens": 20},
        )
        self.assertIsNone(vllm_sampling_evidence(cosyvoice)["parameters"]["seed"])
        request = vllm_sampling_evidence(cosyvoice)["requests"][0]
        self.assertEqual(request["schema_version"], "1.0.0")
        self.assertEqual(request["vllm_version"], "0.15.1")
        self.assertEqual(request["applied_sampling"]["runtime_seed"], -777)
        self.assertEqual(request["applied_sampling"]["seed_source"], "vllm_runtime_generated")

    def test_request_seed_is_injected_and_patch_is_restored(self) -> None:
        cosyvoice = FakeCosyVoice()
        module = sys.modules["vllm"]
        original = module.SamplingParams
        original_add_request = FakeSamplingStates.add_request
        install_vllm_sampling_control(cosyvoice, "request-seeded")
        begin_vllm_observation(cosyvoice, "sample-2", 314159)
        stream = cosyvoice.model.llm.inference_wrapper(None, 25, 2, 20, "x")
        self.assertEqual(next(stream), 7)
        self.assertIs(module.SamplingParams, original)
        self.assertIs(FakeSamplingStates.add_request, original_add_request)
        self.assertEqual(list(stream), [8])
        self.assertEqual(cosyvoice.model.llm.captured.kwargs["seed"], 314159)
        evidence = vllm_sampling_evidence(cosyvoice)
        request = evidence["requests"][0]
        expected_hash = sha256(
            (7).to_bytes(8, "big") + (8).to_bytes(8, "big")
        ).hexdigest()
        self.assertEqual(request["output_token_count"], 2)
        self.assertEqual(request["output_token_sha256"], expected_hash)
        self.assertNotIn("token_ids", request)

    def test_seeded_top_p_profile_changes_one_additional_parameter(self) -> None:
        cosyvoice = FakeCosyVoice()
        install_vllm_sampling_control(cosyvoice, "request-seeded-top-p-0.8")
        begin_vllm_observation(cosyvoice, "sample-3", 20260812)
        list(cosyvoice.model.llm.inference_wrapper(None, 25, 2, 20, "x"))
        self.assertEqual(cosyvoice.model.llm.captured.kwargs["seed"], 20260812)
        self.assertEqual(cosyvoice.model.llm.captured.kwargs["top_p"], 0.8)
        evidence = vllm_sampling_evidence(cosyvoice)
        self.assertFalse(evidence["parameters"]["pytorch_ras_equivalent"])
        self.assertEqual(evidence["request_count"], 1)
        self.assertAlmostEqual(
            evidence["requests"][0]["applied_sampling"]["top_p"],
            0.8,
        )

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

    def test_multiple_frontend_requests_receive_ordinals(self) -> None:
        cosyvoice = FakeCosyVoice()
        install_vllm_sampling_control(cosyvoice, "request-seeded")
        begin_vllm_observation(cosyvoice, "sample-long", 42)
        for request_id in ("first", "second"):
            list(cosyvoice.model.llm.inference_wrapper(None, 25, 2, 20, request_id))
        evidence = vllm_sampling_evidence(cosyvoice)
        self.assertEqual(evidence["request_count"], 2)
        self.assertEqual(
            [request["request_ordinal"] for request in evidence["requests"]],
            [1, 2],
        )
        self.assertEqual(
            [request["applied_sampling"]["runtime_seed"] for request in evidence["requests"]],
            [42, 42],
        )

    def test_missing_sampling_state_capture_fails_closed_with_receipt(self) -> None:
        cosyvoice = FakeNoStateCosyVoice()
        install_vllm_sampling_control(cosyvoice, "upstream")
        begin_vllm_observation(cosyvoice, "sample-no-state", 42)
        with self.assertRaisesRegex(RuntimeError, "observed 0"):
            list(cosyvoice.model.llm.inference_wrapper(None, 25, 2, 20, "x"))
        evidence = vllm_sampling_evidence(cosyvoice)
        self.assertEqual(evidence["request_count"], 1)
        self.assertEqual(evidence["requests"][0]["status"], "interrupted")
        self.assertNotIn("applied_sampling", evidence["requests"][0])

    def test_unpinned_vllm_version_fails_closed(self) -> None:
        cosyvoice = FakeCosyVoice()
        install_vllm_sampling_control(cosyvoice, "upstream")
        begin_vllm_observation(cosyvoice, "sample-version", 42)
        sys.modules["vllm"].__version__ = "0.15.2"
        with self.assertRaisesRegex(RuntimeError, "require version 0.15.1"):
            list(cosyvoice.model.llm.inference_wrapper(None, 25, 2, 20, "x"))

    def test_seeded_standalone_batch_is_rejected_as_ambiguous(self) -> None:
        validate_standalone_sampling_request("request-seeded", 42, 1)
        with self.assertRaisesRegex(ValueError, "exactly one text"):
            validate_standalone_sampling_request("request-seeded", 42, 2)
        with self.assertRaisesRegex(ValueError, "require --seed"):
            validate_standalone_sampling_request("request-seeded", None, 1)


if __name__ == "__main__":
    unittest.main()
