#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party" / "Matcha-TTS"
DEFAULT_PROMPT_WAV = ""
DEFAULT_PROMPT_TEXT = ""
RETRYABLE_ERROR_HINTS = (
    "eos",
    "no audio",
    "no audio generated",
    "empty output",
    "empty audio",
    "too short",
    "kernel size",
    "padded input size",
    "input size",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_texts(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def parse_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def load_llm_state(checkpoint_path: Path) -> dict:
    state = torch.load(checkpoint_path, map_location="cpu")
    module_state = state.get("module")
    if module_state is None:
        raise ValueError(f"Missing 'module' in {checkpoint_path}")
    return module_state


def is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in RETRYABLE_ERROR_HINTS)


def generate_sample(
    cosyvoice,
    text: str,
    prompt_text: str,
    prompt_wav: str,
    speed: float,
    text_frontend: bool,
    sample_rate: int,
    min_seconds: float,
    base_seed: int,
    max_seed_tries: int,
    seed_step: int,
    tag: str,
    idx: int,
) -> tuple[torch.Tensor, float, int, list[int]]:
    attempted_seeds: list[int] = []
    last_exc: Exception | None = None
    for attempt in range(max_seed_tries):
        seed = base_seed + attempt * seed_step
        attempted_seeds.append(seed)
        seed_everything(seed)
        try:
            chunks = []
            for model_output in cosyvoice.inference_zero_shot(
                text,
                prompt_text,
                prompt_wav,
                stream=False,
                speed=speed,
                text_frontend=text_frontend,
            ):
                chunks.append(model_output["tts_speech"].cpu())
            if not chunks:
                raise RuntimeError("No audio generated (empty output).")
            speech = torch.cat(chunks, dim=1).to(torch.float32)
            duration = speech.shape[-1] / sample_rate
            if duration < min_seconds:
                raise RuntimeError(
                    f"Generated audio too short: {duration:.2f}s < {min_seconds:.2f}s"
                )
            return speech, duration, seed, attempted_seeds
        except Exception as exc:
            if not is_retryable_error(exc):
                raise
            last_exc = exc
            print(
                f"Retrying tag={tag} sample={idx} after error: {exc} "
                f"(attempt {attempt + 1}/{max_seed_tries}, seed={seed})",
                file=sys.stderr,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(
        f"Failed to generate audio for tag={tag} sample={idx} after "
        f"{max_seed_tries} attempts. Last error: {last_exc}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate consistent samples from CosyVoice3 LLM checkpoints.")
    parser.add_argument("--pretrained-dir", default="pretrained_models/Fun-CosyVoice3-0.5B", help="CosyVoice3 pretrained model directory.")
    parser.add_argument("--exp-root", default="exp/female01/cosyvoice3/llm/deepspeed", help="Checkpoint root containing epoch tags.")
    parser.add_argument("--tags", required=True, help="Comma-separated checkpoint tags (e.g., epoch_3_whole,epoch_4_whole).")
    parser.add_argument(
        "--prompt-wav",
        required=True,
        help="Prompt wav path for zero-shot.",
    )
    parser.add_argument(
        "--prompt-text",
        required=True,
        help="Prompt text that matches the prompt wav.",
    )
    parser.add_argument("--texts-file", default="samples/cosyvoice3_long_texts.txt", help="Text file with one target sentence per line.")
    parser.add_argument("--out-dir", default="samples/cosyvoice3_long", help="Output directory for generated wavs.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for deterministic sampling.")
    parser.add_argument("--max-seed-tries", type=int, default=20, help="Max seed attempts per sample before failing.")
    parser.add_argument("--seed-step", type=int, default=1, help="Increment applied between retry seeds.")
    parser.add_argument("--min-seconds", type=float, default=10.0, help="Minimum sample duration in seconds.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 inference.")
    parser.add_argument("--no-text-frontend", dest="text_frontend", action="store_false", help="Disable text frontend normalization.")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Relax checkpoint loading strictness.")
    parser.set_defaults(text_frontend=True, strict=True)
    args = parser.parse_args()

    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(THIRD_PARTY))

    from cosyvoice.cli.cosyvoice import CosyVoice3

    pretrained_dir = Path(args.pretrained_dir)
    exp_root = Path(args.exp_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    texts = read_texts(Path(args.texts_file))
    if not texts:
        raise ValueError(f"No valid lines in {args.texts_file}")

    prompt_wav_path = Path(args.prompt_wav)
    if not prompt_wav_path.exists():
        raise FileNotFoundError(f"Prompt wav not found: {prompt_wav_path}")
    cosyvoice = CosyVoice3(str(pretrained_dir), fp16=args.fp16)

    metadata: dict[str, object] = {
        "pretrained_dir": str(pretrained_dir),
        "exp_root": str(exp_root),
        "prompt_wav": args.prompt_wav,
        "prompt_text": args.prompt_text,
        "texts_file": args.texts_file,
        "texts": texts,
        "sample_rate": cosyvoice.sample_rate,
        "seed": args.seed,
        "max_seed_tries": args.max_seed_tries,
        "seed_step": args.seed_step,
        "min_seconds": args.min_seconds,
        "fp16": args.fp16,
        "speed": args.speed,
        "text_frontend": args.text_frontend,
        "strict": args.strict,
        "runs": [],
    }

    with torch.no_grad():
        for tag in parse_tags(args.tags):
            tag_dir = exp_root / tag
            ckpt_path = tag_dir / "mp_rank_00_model_states.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            llm_state = load_llm_state(ckpt_path)
            cosyvoice.model.llm.load_state_dict(llm_state, strict=args.strict)
            cosyvoice.model.llm.eval()

            tag_out = out_root / tag
            tag_out.mkdir(parents=True, exist_ok=True)
            tag_record = {"tag": tag, "checkpoint": str(ckpt_path), "samples": []}

            for idx, text in enumerate(texts, start=1):
                base_seed = args.seed + idx - 1
                speech, duration, seed, attempted_seeds = generate_sample(
                    cosyvoice,
                    text,
                    args.prompt_text,
                    str(prompt_wav_path),
                    args.speed,
                    args.text_frontend,
                    cosyvoice.sample_rate,
                    args.min_seconds,
                    base_seed,
                    args.max_seed_tries,
                    args.seed_step,
                    tag,
                    idx,
                )
                out_path = tag_out / f"sample_{idx:02d}.wav"
                torchaudio.save(str(out_path), speech, cosyvoice.sample_rate)
                tag_record["samples"].append(
                    {
                        "text": text,
                        "path": str(out_path),
                        "duration_seconds": round(float(duration), 3),
                        "seed": seed,
                        "attempts": len(attempted_seeds),
                        "attempted_seeds": attempted_seeds,
                    }
                )

            metadata["runs"].append(tag_record)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metadata_path = out_root / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
