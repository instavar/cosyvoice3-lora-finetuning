from __future__ import annotations

import hashlib
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from frontend_segmentation import build_frontend_segmentation_receipt  # noqa: E402


class FakeTokenizer:
    def encode(self, text, *, allowed_special):
        if allowed_special != "all":
            raise AssertionError("unexpected allowed_special")
        return text.split()


class FrontendSegmentationTests(unittest.TestCase):
    def frontend(self):
        return types.SimpleNamespace(
            tokenizer=FakeTokenizer(),
            allowed_special="all",
            text_frontend="",
            text_normalizer_enabled=True,
            text_normalize=lambda text, split, text_frontend: [
                "First segment.",
                "Second segment here.",
            ],
        )

    def test_hashes_segments_without_retaining_text(self) -> None:
        receipt = build_frontend_segmentation_receipt(
            self.frontend(),
            "Original source.",
            text_frontend=True,
        )
        self.assertEqual(receipt["schema_version"], "1.0.0")
        self.assertEqual(receipt["segment_count"], 2)
        self.assertEqual(
            receipt["segments"][0]["normalized_text_sha256"],
            hashlib.sha256(b"First segment.").hexdigest(),
        )
        self.assertEqual(
            [row["text_token_count"] for row in receipt["segments"]],
            [2, 3],
        )
        self.assertFalse(receipt["normalized_text_retained"])
        self.assertNotIn("First segment.", repr(receipt))

    def test_rejects_non_list_or_empty_segments(self) -> None:
        frontend = self.frontend()
        frontend.text_normalize = lambda text, split, text_frontend: iter([text])
        with self.assertRaisesRegex(RuntimeError, "non-empty list"):
            build_frontend_segmentation_receipt(
                frontend,
                "Original source.",
                text_frontend=True,
            )
        frontend.text_normalize = lambda text, split, text_frontend: [""]
        with self.assertRaisesRegex(RuntimeError, "non-text segment"):
            build_frontend_segmentation_receipt(
                frontend,
                "Original source.",
                text_frontend=True,
            )


if __name__ == "__main__":
    unittest.main()
