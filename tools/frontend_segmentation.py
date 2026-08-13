"""Hash the CosyVoice frontend segmentation preview used by an evaluation row."""

from __future__ import annotations

import hashlib
from typing import Any


FRONTEND_SEGMENTATION_SCHEMA_VERSION = "1.0.0"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_frontend_segmentation_receipt(
    frontend: Any,
    text: str,
    *,
    text_frontend: bool,
) -> dict[str, Any]:
    """Preview deterministic frontend chunks without retaining normalized text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("frontend segmentation text must be non-empty")
    segments = frontend.text_normalize(
        text,
        split=True,
        text_frontend=text_frontend,
    )
    if not isinstance(segments, (list, tuple)) or not segments:
        raise RuntimeError("frontend segmentation preview must return a non-empty list")
    if any(not isinstance(segment, str) or not segment for segment in segments):
        raise RuntimeError("frontend segmentation preview contains a non-text segment")

    tokenizer = getattr(frontend, "tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("frontend tokenizer does not expose encode")
    allowed_special = getattr(frontend, "allowed_special", "all")
    rows = []
    for ordinal, segment in enumerate(segments, start=1):
        token_ids = encode(segment, allowed_special=allowed_special)
        rows.append(
            {
                "segment_ordinal": ordinal,
                "normalized_text_sha256": _sha256_text(segment),
                "normalized_character_count": len(segment),
                "text_token_count": len(token_ids),
            }
        )

    backend = getattr(frontend, "text_frontend", None)
    return {
        "schema_version": FRONTEND_SEGMENTATION_SCHEMA_VERSION,
        "source_text_sha256": _sha256_text(text),
        "text_frontend_requested": bool(text_frontend),
        "normalizer_enabled": bool(
            getattr(frontend, "text_normalizer_enabled", False)
        ),
        "normalizer_backend": backend if isinstance(backend, str) else None,
        "segment_count": len(rows),
        "segments": rows,
        "normalized_text_retained": False,
        "evidence_boundary": (
            "This is a deterministic preview from the same frontend method and "
            "input before generation. It binds normalized segment hashes and "
            "token counts without retaining normalized text, but it does not "
            "intercept or prove the later generation call."
        ),
    }
