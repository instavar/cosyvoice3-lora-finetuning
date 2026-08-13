"""Explicit, bounded vLLM request-sampling controls for CosyVoice diagnosis."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Iterator


PROFILE_UPSTREAM = "upstream"
PROFILE_REQUEST_SEEDED = "request-seeded"
PROFILE_REQUEST_SEEDED_TOP_P_0_8 = "request-seeded-top-p-0.8"
VLLM_SAMPLING_PROFILES = (
    PROFILE_UPSTREAM,
    PROFILE_REQUEST_SEEDED,
    PROFILE_REQUEST_SEEDED_TOP_P_0_8,
)


def validate_standalone_sampling_request(
    profile: str,
    seed: int | None,
    text_count: int,
) -> None:
    """Reject ambiguous seeded batches in the standalone inference helper."""
    if profile not in VLLM_SAMPLING_PROFILES:
        raise ValueError(f"unsupported vLLM sampling profile: {profile}")
    if profile == PROFILE_UPSTREAM:
        return
    if seed is None:
        raise ValueError("request-seeded vLLM profiles require --seed")
    _validate_seed(seed)
    if text_count != 1:
        raise ValueError(
            "request-seeded vLLM profiles require exactly one text in the "
            "standalone helper; use the evaluation runner for a recorded "
            "per-row seed schedule"
        )


@dataclass
class VllmSamplingControl:
    profile: str
    request_seed: int | None = None

    @property
    def requires_request_seed(self) -> bool:
        return self.profile != PROFILE_UPSTREAM

    def effective_parameters(self, top_k: int) -> dict[str, Any]:
        return {
            "temperature": 1.0,
            "top_p": 0.8
            if self.profile == PROFILE_REQUEST_SEEDED_TOP_P_0_8
            else 1.0,
            "top_k": top_k,
            "seed": self.request_seed if self.requires_request_seed else None,
            "pytorch_ras_equivalent": False,
        }

    def overrides(self) -> dict[str, Any]:
        if not self.requires_request_seed:
            return {}
        if self.request_seed is None:
            raise RuntimeError("vLLM request seed must be set before generation")
        overrides: dict[str, Any] = {"seed": self.request_seed}
        if self.profile == PROFILE_REQUEST_SEEDED_TOP_P_0_8:
            overrides["top_p"] = 0.8
        return overrides


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("vLLM request seed must be an integer")
    if seed < 0 or seed > 2**63 - 1:
        raise ValueError("vLLM request seed must be between 0 and 2^63 - 1")
    return seed


def install_vllm_sampling_control(cosyvoice: Any, profile: str) -> VllmSamplingControl:
    """Wrap one CosyVoice vLLM instance without permanently patching vLLM."""
    if profile not in VLLM_SAMPLING_PROFILES:
        raise ValueError(f"unsupported vLLM sampling profile: {profile}")
    llm = cosyvoice.model.llm
    if not hasattr(llm, "vllm"):
        raise RuntimeError("vLLM must be enabled before sampling controls are installed")
    if hasattr(llm, "_instavar_vllm_sampling_control"):
        raise RuntimeError("vLLM sampling controls are already installed")

    control = VllmSamplingControl(profile=profile)
    original_wrapper = llm.inference_wrapper

    def controlled_wrapper(
        self: Any,
        lm_input: Any,
        sampling: int,
        min_len: int,
        max_len: int,
        uuid: str,
    ) -> Iterator[Any]:
        overrides = control.overrides()
        if not overrides:
            yield from original_wrapper(lm_input, sampling, min_len, max_len, uuid)
            return

        import vllm

        original_sampling_params = vllm.SamplingParams

        def controlled_sampling_params(*args: Any, **kwargs: Any) -> Any:
            for key, value in overrides.items():
                if key in kwargs and kwargs[key] != value:
                    raise RuntimeError(
                        f"upstream vLLM {key}={kwargs[key]!r} conflicts with "
                        f"the selected profile value {value!r}"
                    )
                kwargs[key] = value
            return original_sampling_params(*args, **kwargs)

        vllm.SamplingParams = controlled_sampling_params
        stream = original_wrapper(lm_input, sampling, min_len, max_len, uuid)
        try:
            try:
                first = next(stream)
            except StopIteration:
                return
        finally:
            vllm.SamplingParams = original_sampling_params
        yield first
        yield from stream

    llm.inference_wrapper = types.MethodType(controlled_wrapper, llm)
    llm._instavar_vllm_sampling_control = control
    return control


def set_vllm_request_seed(cosyvoice: Any, seed: int) -> None:
    llm = cosyvoice.model.llm
    control = getattr(llm, "_instavar_vllm_sampling_control", None)
    if not isinstance(control, VllmSamplingControl):
        raise RuntimeError("vLLM sampling controls are not installed")
    control.request_seed = _validate_seed(seed)


def vllm_sampling_evidence(cosyvoice: Any, top_k: int = 25) -> dict[str, Any]:
    llm = cosyvoice.model.llm
    control = getattr(llm, "_instavar_vllm_sampling_control", None)
    if not isinstance(control, VllmSamplingControl):
        raise RuntimeError("vLLM sampling controls are not installed")
    return {
        "profile": control.profile,
        "parameters": control.effective_parameters(top_k),
        "evidence_boundary": (
            "These are the request-sampling parameters supplied to vLLM. "
            "They do not reproduce CosyVoice PyTorch repetition-aware sampling, "
            "attest internal runtime behavior, or prove deterministic output."
        ),
    }
