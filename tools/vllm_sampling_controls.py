"""Explicit, bounded vLLM request-sampling controls for CosyVoice diagnosis."""

from __future__ import annotations

import copy
import hashlib
import operator
import types
from dataclasses import dataclass, field
from typing import Any, Iterator


PROFILE_UPSTREAM = "upstream"
PROFILE_REQUEST_SEEDED = "request-seeded"
PROFILE_REQUEST_SEEDED_TOP_P_0_8 = "request-seeded-top-p-0.8"
VLLM_SAMPLING_PROFILES = (
    PROFILE_UPSTREAM,
    PROFILE_REQUEST_SEEDED,
    PROFILE_REQUEST_SEEDED_TOP_P_0_8,
)
REQUEST_RECEIPT_SCHEMA_VERSION = "1.0.0"
REQUEST_RECEIPT_VLLM_VERSION = "0.15.1"
SAMPLING_STATE_CAPTURE = (
    "vllm.v1.worker.gpu_input_batch.InputBatch.add_request"
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
    active_sample_id: str | None = None
    request_receipts: list[dict[str, Any]] = field(default_factory=list)

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


def _engine_config_seed(llm: Any) -> int | None:
    vllm_config = getattr(getattr(llm, "vllm", None), "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    seed = getattr(model_config, "seed", None)
    return int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else None


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
        import vllm
        from vllm.v1.worker.gpu_input_batch import InputBatch

        vllm_version = getattr(vllm, "__version__", None)
        if vllm_version != REQUEST_RECEIPT_VLLM_VERSION:
            raise RuntimeError(
                "vLLM request receipts require version "
                f"{REQUEST_RECEIPT_VLLM_VERSION}, observed {vllm_version!r}"
            )

        original_sampling_params = vllm.SamplingParams
        original_add_request = InputBatch.add_request
        applied_parameters: list[dict[str, Any]] = []
        token_digest = hashlib.sha256()
        token_count = 0
        completed = False
        receipt: dict[str, Any] = {
            "schema_version": REQUEST_RECEIPT_SCHEMA_VERSION,
            "request_ordinal": len(control.request_receipts) + 1,
            "sample_id": control.active_sample_id,
            "vllm_version": vllm_version,
            "sampling_state_capture": SAMPLING_STATE_CAPTURE,
            "token_hash_encoding": "unsigned-64-bit-big-endian-token-ids-v1",
        }

        def controlled_sampling_params(*args: Any, **kwargs: Any) -> Any:
            for key, value in overrides.items():
                if key in kwargs and kwargs[key] != value:
                    raise RuntimeError(
                        f"upstream vLLM {key}={kwargs[key]!r} conflicts with "
                        f"the selected profile value {value!r}"
                    )
                kwargs[key] = value
            return original_sampling_params(*args, **kwargs)

        def controlled_add_request(input_batch: Any, request: Any) -> int:
            req_idx = original_add_request(input_batch, request)
            sampling_params = request.sampling_params
            if sampling_params is None:
                raise RuntimeError("vLLM request has no sampling parameters")
            generator = input_batch.generators.get(req_idx)
            request_seed = (
                int(generator.initial_seed()) if generator is not None else None
            )
            applied_parameters.append(
                {
                    "temperature": float(input_batch.temperature_cpu[req_idx]),
                    "top_p": float(input_batch.top_p_cpu[req_idx]),
                    "top_k": int(input_batch.top_k_cpu[req_idx]),
                    "request_generator_seed": request_seed,
                    "seed_source": (
                        "supplied_request_seed"
                        if generator is not None
                        else "global_engine_generator"
                    ),
                    "engine_config_seed": _engine_config_seed(llm),
                    "request_generator_state_captured": generator is not None,
                    "request_id_matches_wrapper": request.req_id == uuid,
                }
            )
            receipt["registered_request_parameters"] = {
                "min_p": float(sampling_params.min_p),
                "min_tokens": int(sampling_params.min_tokens),
                "max_tokens": int(sampling_params.max_tokens),
                "supplied_request_seed": (
                    int(sampling_params.seed)
                    if sampling_params.seed is not None
                    else None
                ),
            }
            return req_idx

        def record_token(token: Any) -> None:
            nonlocal token_count
            token_id = operator.index(token)
            if token_id < 0 or token_id > 2**64 - 1:
                raise ValueError("vLLM output token id is outside unsigned 64-bit range")
            token_digest.update(token_id.to_bytes(8, byteorder="big", signed=False))
            token_count += 1

        vllm.SamplingParams = controlled_sampling_params
        InputBatch.add_request = controlled_add_request
        stream = original_wrapper(lm_input, sampling, min_len, max_len, uuid)
        first: Any | None = None
        has_first = False
        startup_error: BaseException | None = None
        try:
            try:
                first = next(stream)
                has_first = True
            except StopIteration:
                completed = True
            except BaseException as error:
                startup_error = error
        finally:
            vllm.SamplingParams = original_sampling_params
            InputBatch.add_request = original_add_request
        try:
            if startup_error is not None:
                raise startup_error
            if len(applied_parameters) != 1:
                raise RuntimeError(
                    "expected exactly one vLLM input-batch capture per request, "
                    f"observed {len(applied_parameters)}"
                )
            receipt["applied_sampling"] = applied_parameters[0]
            if has_first:
                record_token(first)
                yield first
                for token in stream:
                    record_token(token)
                    yield token
                completed = True
        finally:
            receipt.update(
                {
                    "status": "complete" if completed else "interrupted",
                    "output_token_count": token_count,
                    "output_token_sha256": token_digest.hexdigest(),
                }
            )
            control.request_receipts.append(receipt)

    llm.inference_wrapper = types.MethodType(controlled_wrapper, llm)
    llm._instavar_vllm_sampling_control = control
    return control


def set_vllm_request_seed(cosyvoice: Any, seed: int) -> None:
    llm = cosyvoice.model.llm
    control = getattr(llm, "_instavar_vllm_sampling_control", None)
    if not isinstance(control, VllmSamplingControl):
        raise RuntimeError("vLLM sampling controls are not installed")
    control.request_seed = _validate_seed(seed)


def begin_vllm_observation(cosyvoice: Any, sample_id: str, seed: int) -> None:
    """Start one row-scoped request receipt and apply its declared seed."""
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("vLLM observation sample id must be non-empty")
    llm = cosyvoice.model.llm
    control = getattr(llm, "_instavar_vllm_sampling_control", None)
    if not isinstance(control, VllmSamplingControl):
        raise RuntimeError("vLLM sampling controls are not installed")
    control.active_sample_id = sample_id.strip()
    control.request_receipts = []
    control.request_seed = _validate_seed(seed)


def vllm_sampling_evidence(cosyvoice: Any, top_k: int = 25) -> dict[str, Any]:
    llm = cosyvoice.model.llm
    control = getattr(llm, "_instavar_vllm_sampling_control", None)
    if not isinstance(control, VllmSamplingControl):
        raise RuntimeError("vLLM sampling controls are not installed")
    return {
        "request_receipt_schema_version": REQUEST_RECEIPT_SCHEMA_VERSION,
        "vllm_version": REQUEST_RECEIPT_VLLM_VERSION,
        "sampling_state_capture": SAMPLING_STATE_CAPTURE,
        "profile": control.profile,
        "parameters": control.effective_parameters(top_k),
        "sample_id": control.active_sample_id,
        "request_count": len(control.request_receipts),
        "requests": copy.deepcopy(control.request_receipts),
        "evidence_boundary": (
            "Applied temperature, top-p, top-k, and request generator state are "
            "read from vLLM's persistent GPU input batch after request "
            "registration under the pinned version. Request limits and min-p "
            "are read from the registered SamplingParams. An upstream request "
            "without its own generator uses vLLM's process-global generator. "
            "Token hashes cover the token IDs yielded by the CosyVoice wrapper "
            "without retaining token content. "
            "They do not reproduce CosyVoice PyTorch repetition-aware sampling, "
            "attest kernel behavior, prove deterministic output, or establish "
            "content or perceptual quality."
        ),
    }
