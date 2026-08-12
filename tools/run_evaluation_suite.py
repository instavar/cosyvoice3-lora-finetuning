#!/usr/bin/env python3
"""Run a frozen Instavar Voice plan through one loaded CosyVoice3 adapter runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosyvoice-dir", type=Path, required=True)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--generation-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--min-audio-seconds", type=float, default=0.5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-text-frontend", dest="text_frontend", action="store_false")
    parser.add_argument("--vllm-dir")
    parser.add_argument("--reuse-vllm-dir", action="store_true")
    parser.set_defaults(text_frontend=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_observations(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    cosyvoice_dir = args.cosyvoice_dir.resolve()
    matcha_dir = cosyvoice_dir / "third_party" / "Matcha-TTS"
    if not matcha_dir.is_dir():
        raise FileNotFoundError(f"CosyVoice Matcha-TTS checkout not found: {matcha_dir}")
    sys.path.insert(0, str(matcha_dir))
    sys.path.insert(0, str(cosyvoice_dir))

    from cosyvoice.cli.cosyvoice import CosyVoice3
    from generate_cosyvoice3_samples import seed_everything
    from infer_cosyvoice3_lora import apply_lora_to_cosyvoice3, enable_vllm_with_merged_lora

    plan = json.loads(args.generation_plan.read_text(encoding="utf-8"))
    rows = [row for row in plan.get("samples", []) if row.get("candidate_id") == args.candidate_id]
    if not rows:
        raise ValueError(f"generation plan has no rows for candidate {args.candidate_id!r}")

    cosyvoice = CosyVoice3(args.pretrained_dir, fp16=args.fp16)
    peft_model = None
    if not (args.vllm_dir and args.reuse_vllm_dir):
        peft_model = apply_lora_to_cosyvoice3(cosyvoice, args.lora_dir)
    if args.vllm_dir:
        enable_vllm_with_merged_lora(
            cosyvoice,
            peft_model,
            args.vllm_dir,
            reuse_vllm_dir=args.reuse_vllm_dir,
        )

    observations: list[dict] = []
    for row in rows:
        output = args.output_dir / row["expected_audio_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        seed_everything(int(row["seed"]))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        observation = {
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "seed": row["seed"],
            "requested_text": row["text"],
            "valid": False,
            "runtime": "cosyvoice3_vllm_merged" if args.vllm_dir else "cosyvoice3_pytorch_adapter",
            "instruction_applied": False,
        }
        try:
            chunks = [
                value["tts_speech"].cpu()
                for value in cosyvoice.inference_zero_shot(
                    row["text"],
                    args.prompt_text,
                    args.prompt_wav,
                    stream=False,
                    speed=args.speed,
                    text_frontend=args.text_frontend,
                )
            ]
            if not chunks:
                raise RuntimeError("No audio generated")
            speech = torch.cat(chunks, dim=1).to(torch.float32)
            torchaudio.save(str(output), speech, cosyvoice.sample_rate)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            duration = float(speech.shape[-1] / cosyvoice.sample_rate)
            peak = float(speech.abs().max().item())
            rms = float(torch.sqrt(torch.mean(torch.square(speech))).item())
            valid = duration >= args.min_audio_seconds and peak > 1e-4 and rms > 1e-5
            observation.update(
                {
                    "valid": valid,
                    "audio_path": str(output),
                    "audio_sha256": sha256(output),
                    "audio_duration_seconds": duration,
                    "audio_peak": peak,
                    "audio_rms": rms,
                    "generation_seconds": elapsed,
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                    "instruction_note": (
                        "The zero-shot path has no separate style-instruction input."
                        if row.get("instruction")
                        else None
                    ),
                }
            )
            if not valid:
                observation.update(
                    {
                        "error_type": "implausible_audio_output",
                        "error": (
                            "CosyVoice can hide a background LLM exception behind a short WAV; "
                            "the output failed duration or signal-level validity checks."
                        ),
                    }
                )
        except Exception as error:
            observation.update(
                {
                    "generation_seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        observations.append(observation)
        write_observations(args.output_dir / "generation-observations.json", observations)
    return 0 if all(row["valid"] for row in observations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
