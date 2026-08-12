"""Route CosyVoice3 generation through the control mode requested by a plan."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

END_OF_PROMPT = "<|endofprompt|>"
ZERO_SHOT_ROUTE = "inference_zero_shot"
INSTRUCT_ROUTE = "inference_instruct2"


def normalize_instruction(instruction: object) -> str | None:
    """Validate a plan instruction before any model is loaded."""
    if instruction is None:
        return None
    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string when present")
    normalized = instruction.strip()
    if not normalized:
        raise ValueError("instruction must be non-empty when present")
    if END_OF_PROMPT in normalized:
        raise ValueError(
            "instruction must not contain <|endofprompt|>; "
            "CosyVoice inference_instruct2 adds the delimiter internally"
        )
    return normalized


def generation_route(instruction: object) -> str:
    return (
        INSTRUCT_ROUTE
        if normalize_instruction(instruction) is not None
        else ZERO_SHOT_ROUTE
    )


def validate_generation_inputs(
    *,
    text: object,
    instruction: object,
    prompt_text: object,
    prompt_wav: object,
    speed: object,
    text_frontend: object,
) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    normalized_instruction = normalize_instruction(instruction)
    if normalized_instruction is None and (
        not isinstance(prompt_text, str) or not prompt_text.strip()
    ):
        raise ValueError("prompt_text must be non-empty for inference_zero_shot")
    if not isinstance(prompt_wav, str) or not prompt_wav.strip():
        raise ValueError("prompt_wav must be a non-empty string")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise TypeError("speed must be a finite positive number")
    if not math.isfinite(float(speed)) or float(speed) <= 0:
        raise ValueError("speed must be a finite positive number")
    if not isinstance(text_frontend, bool):
        raise TypeError("text_frontend must be a boolean")


def invoke_generation(
    cosyvoice: Any,
    *,
    text: str,
    instruction: object,
    prompt_text: str,
    prompt_wav: str,
    speed: float,
    text_frontend: bool,
) -> tuple[Iterable[dict[str, Any]], str, str | None]:
    """Invoke exactly one upstream route without silently dropping control text."""
    validate_generation_inputs(
        text=text,
        instruction=instruction,
        prompt_text=prompt_text,
        prompt_wav=prompt_wav,
        speed=speed,
        text_frontend=text_frontend,
    )
    normalized_instruction = normalize_instruction(instruction)
    if normalized_instruction is None:
        method = getattr(cosyvoice, ZERO_SHOT_ROUTE, None)
        if not callable(method):
            raise AttributeError(f"CosyVoice runtime does not expose {ZERO_SHOT_ROUTE}")
        output = method(
            text,
            prompt_text,
            prompt_wav,
            stream=False,
            speed=speed,
            text_frontend=text_frontend,
        )
        return output, ZERO_SHOT_ROUTE, None

    method = getattr(cosyvoice, INSTRUCT_ROUTE, None)
    if not callable(method):
        raise TypeError(
            "generation plan requests an instruction, but the CosyVoice runtime "
            f"does not expose {INSTRUCT_ROUTE}"
        )
    output = method(
        text,
        normalized_instruction,
        prompt_wav,
        stream=False,
        speed=speed,
        text_frontend=text_frontend,
    )
    return output, INSTRUCT_ROUTE, normalized_instruction
