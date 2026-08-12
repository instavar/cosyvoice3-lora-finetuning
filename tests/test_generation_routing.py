from __future__ import annotations

import math
import unittest

from tools.cosyvoice_generation import (
    INSTRUCT_ROUTE,
    ZERO_SHOT_ROUTE,
    invoke_generation,
    normalize_instruction,
    validate_generation_inputs,
)


class FakeCosyVoice:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def inference_zero_shot(self, *args, **kwargs):
        self.calls.append((ZERO_SHOT_ROUTE, args, kwargs))
        return iter([{"tts_speech": "neutral"}])

    def inference_instruct2(self, *args, **kwargs):
        self.calls.append((INSTRUCT_ROUTE, args, kwargs))
        return iter([{"tts_speech": "instructed"}])


class GenerationRoutingTests(unittest.TestCase):
    def invoke(self, model: object, instruction: object):
        return invoke_generation(
            model,
            text="Target text.",
            instruction=instruction,
            prompt_text="Reference transcript.",
            prompt_wav="reference.wav",
            speed=1.0,
            text_frontend=True,
        )

    def test_missing_instruction_uses_zero_shot_route(self) -> None:
        model = FakeCosyVoice()
        output, route, applied = self.invoke(model, None)
        self.assertEqual(list(output), [{"tts_speech": "neutral"}])
        self.assertEqual((route, applied), (ZERO_SHOT_ROUTE, None))
        self.assertEqual(model.calls[0][0], ZERO_SHOT_ROUTE)
        self.assertEqual(
            model.calls[0][1][:3],
            ("Target text.", "Reference transcript.", "reference.wav"),
        )

    def test_instruction_uses_instruct2_without_prompt_transcript(self) -> None:
        model = FakeCosyVoice()
        output, route, applied = self.invoke(model, "  Read with calm confidence.  ")
        self.assertEqual(list(output), [{"tts_speech": "instructed"}])
        self.assertEqual(route, INSTRUCT_ROUTE)
        self.assertEqual(applied, "Read with calm confidence.")
        self.assertEqual(
            model.calls[0][1][:3],
            ("Target text.", "Read with calm confidence.", "reference.wav"),
        )

    def test_delimiter_is_rejected_instead_of_being_doubled(self) -> None:
        with self.assertRaisesRegex(ValueError, "adds the delimiter internally"):
            normalize_instruction("Speak calmly.<|endofprompt|>")

    def test_instruction_never_falls_back_when_runtime_lacks_instruct2(self) -> None:
        class ZeroShotOnly:
            def inference_zero_shot(self, *args, **kwargs):
                raise AssertionError("must not silently fall back")

        with self.assertRaisesRegex(
            TypeError, "does not expose inference_instruct2"
        ):
            self.invoke(ZeroShotOnly(), "Read with concern.")

    def test_zero_shot_requires_a_prompt_transcript(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt_text must be non-empty"):
            validate_generation_inputs(
                text="Target text.",
                instruction=None,
                prompt_text=" ",
                prompt_wav="reference.wav",
                speed=1.0,
                text_frontend=True,
            )

    def test_instruct_route_does_not_require_unused_prompt_transcript(self) -> None:
        validate_generation_inputs(
            text="Target text.",
            instruction="Read calmly.",
            prompt_text="",
            prompt_wav="reference.wav",
            speed=1.0,
            text_frontend=True,
        )

    def test_nonpositive_and_nonfinite_speeds_fail_closed(self) -> None:
        for speed in (0, -1, math.nan, math.inf):
            with self.subTest(speed=speed), self.assertRaisesRegex(
                ValueError, "finite positive number"
            ):
                validate_generation_inputs(
                    text="Target text.",
                    instruction=None,
                    prompt_text="Reference transcript.",
                    prompt_wav="reference.wav",
                    speed=speed,
                    text_frontend=True,
                )


if __name__ == "__main__":
    unittest.main()
