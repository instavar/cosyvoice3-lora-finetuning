#!/usr/bin/env python3
"""Run a frozen Instavar Voice plan through one explicit CosyVoice3 condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch
import torchaudio
from thread_failures import BackgroundThreadFailureCapture

IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosyvoice-dir", type=Path, required=True)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument(
        "--inference-mode",
        choices=(
            "base",
            "adapter",
            "merged-pytorch",
            "reloaded-merged-pytorch",
            "merged-vllm",
        ),
        help="Explicit artifact condition. Legacy LoRA invocations are inferred.",
    )
    parser.add_argument("--lora-dir")
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--generation-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--runtime-id")
    parser.add_argument("--artifact-set-id")
    parser.add_argument("--artifact-set-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-invalid-output",
        action="store_true",
        help="return success after recording every planned attempt even when an output is invalid",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--min-audio-seconds", type=float, default=0.5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-text-frontend", dest="text_frontend", action="store_false")
    parser.add_argument("--vllm-dir")
    parser.add_argument("--reuse-vllm-dir", action="store_true")
    parser.add_argument(
        "--vllm-sampling-profile",
        choices=("upstream", "request-seeded", "request-seeded-top-p-0.8"),
        default="upstream",
    )
    parser.add_argument("--merged-pytorch-dir")
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


def runtime_artifact_fields(args: argparse.Namespace) -> dict[str, str]:
    runtime_id = args.runtime_id or ("vllm" if args.vllm_dir else "pytorch")
    if not IDENTIFIER_RE.fullmatch(runtime_id):
        raise ValueError("runtime id must be a lowercase machine-readable identifier")
    if bool(args.artifact_set_id) != bool(args.artifact_set_sha256):
        raise ValueError("artifact set id and sha256 must be provided together")
    fields = {"runtime_id": runtime_id}
    if args.artifact_set_id:
        if not IDENTIFIER_RE.fullmatch(args.artifact_set_id):
            raise ValueError("artifact set id must be a lowercase machine-readable identifier")
        if not re.fullmatch(r"[0-9a-f]{64}", args.artifact_set_sha256):
            raise ValueError("artifact set sha256 must be a lowercase SHA-256 digest")
        fields.update(
            {
                "artifact_set_id": args.artifact_set_id,
                "artifact_set_sha256": args.artifact_set_sha256,
            }
        )
    return fields


def main() -> int:
    args = parse_args()
    from evaluation_contract import resolve_inference_mode

    args.inference_mode = resolve_inference_mode(args, argparse.ArgumentParser())
    artifact_fields = runtime_artifact_fields(args)
    cosyvoice_dir = args.cosyvoice_dir.resolve()
    matcha_dir = cosyvoice_dir / "third_party" / "Matcha-TTS"
    if not matcha_dir.is_dir():
        raise FileNotFoundError(f"CosyVoice Matcha-TTS checkout not found: {matcha_dir}")
    sys.path.insert(0, str(matcha_dir))
    sys.path.insert(0, str(cosyvoice_dir))

    from cosyvoice.cli.cosyvoice import CosyVoice3
    from cosyvoice_generation import (
        generation_route,
        invoke_generation,
        validate_generation_inputs,
    )
    from generate_cosyvoice3_samples import seed_everything
    if args.inference_mode in {"adapter", "merged-pytorch", "merged-vllm"}:
        from infer_cosyvoice3_lora import (
            apply_lora_to_cosyvoice3,
            enable_vllm_with_merged_lora,
            merge_lora_into_cosyvoice3,
        )
    if args.inference_mode == "reloaded-merged-pytorch":
        from merged_pytorch_artifact import load_merged_pytorch_artifact

    plan = json.loads(args.generation_plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") not in {"1.0.0", "1.1.0"}:
        raise ValueError("generation plan schema_version must equal 1.0.0 or 1.1.0")
    rows = [row for row in plan.get("samples", []) if row.get("candidate_id") == args.candidate_id]
    if not rows:
        raise ValueError(f"generation plan has no rows for candidate {args.candidate_id!r}")
    for index, row in enumerate(rows):
        try:
            validate_generation_inputs(
                text=row.get("text"),
                instruction=row.get("instruction"),
                prompt_text=args.prompt_text,
                prompt_wav=args.prompt_wav,
                speed=args.speed,
                text_frontend=args.text_frontend,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"generation plan sample {index} has invalid generation input: {error}"
            ) from error

    cosyvoice = CosyVoice3(args.pretrained_dir, fp16=args.fp16)
    peft_model = None
    if args.inference_mode in {"adapter", "merged-pytorch"} or (
        args.inference_mode == "merged-vllm" and not args.reuse_vllm_dir
    ):
        peft_model = apply_lora_to_cosyvoice3(cosyvoice, args.lora_dir)
    if args.inference_mode == "merged-vllm":
        enable_vllm_with_merged_lora(
            cosyvoice,
            peft_model,
            args.vllm_dir,
            reuse_vllm_dir=args.reuse_vllm_dir,
            sampling_profile=args.vllm_sampling_profile,
        )
    elif args.inference_mode == "merged-pytorch":
        if peft_model is None:
            raise RuntimeError("merged-pytorch mode requires a loaded PEFT model")
        merge_lora_into_cosyvoice3(cosyvoice, peft_model)
    elif args.inference_mode == "reloaded-merged-pytorch":
        load_merged_pytorch_artifact(
            cosyvoice.model.llm.llm.model,
            Path(args.merged_pytorch_dir),
        )

    observations: list[dict] = []
    for row in rows:
        output = args.output_dir / row["expected_audio_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        seed_everything(int(row["seed"]))
        if args.inference_mode == "merged-vllm":
            from vllm_sampling_controls import (
                set_vllm_request_seed,
                vllm_sampling_evidence,
            )

            set_vllm_request_seed(cosyvoice, int(row["seed"]))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        planned_instruction_route = generation_route(row.get("instruction"))
        device_family = "cuda" if torch.cuda.is_available() else "cpu"
        artifact_mode = (
            "merged"
            if args.inference_mode
            in {"merged-pytorch", "reloaded-merged-pytorch", "merged-vllm"}
            else args.inference_mode
        )
        runtime = (
            f"cosyvoice3_vllm_{device_family}_merged"
            if args.inference_mode == "merged-vllm"
            else f"cosyvoice3_pytorch_{device_family}_{artifact_mode}"
        )
        observation = {
            "observation_schema_version": "1.0.0",
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "seed": row["seed"],
            "requested_text": row["text"],
            "valid": False,
            "runtime": runtime,
            "artifact_mode": artifact_mode,
            **artifact_fields,
            "requested_instruction": row.get("instruction"),
            "instruction_requested": bool(row.get("instruction")),
            "instruction_route": planned_instruction_route,
            "instruction_applied": False,
            "background_thread_check": "threading_excepthook_during_stream_consumption",
        }
        if args.inference_mode == "merged-vllm":
            observation["vllm_sampling"] = vllm_sampling_evidence(cosyvoice)
        background_capture = BackgroundThreadFailureCapture()
        try:
            with background_capture:
                output_stream, instruction_route, applied_instruction = invoke_generation(
                    cosyvoice,
                    text=row["text"],
                    instruction=row.get("instruction"),
                    prompt_text=args.prompt_text,
                    prompt_wav=args.prompt_wav,
                    speed=args.speed,
                    text_frontend=args.text_frontend,
                )
                observation.update(
                    {
                        "instruction_route": instruction_route,
                        "applied_instruction": applied_instruction,
                        "instruction_applied": applied_instruction is not None,
                    }
                )
                chunks = [value["tts_speech"].cpu() for value in output_stream]
            if background_capture.failures:
                observation["background_thread_failures"] = background_capture.failures
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
            valid = (
                not background_capture.failures
                and duration >= args.min_audio_seconds
                and peak > 1e-4
                and rms > 1e-5
            )
            observation.update(
                {
                    "valid": valid,
                    "audio_path": str(output),
                    "audio_sha256": sha256(output),
                    "audio_duration_seconds": duration,
                    "audio_peak": peak,
                    "audio_rms": rms,
                    "generation_seconds": elapsed,
                    **(
                        {"peak_memory_bytes": int(torch.cuda.max_memory_allocated())}
                        if torch.cuda.is_available()
                        else {}
                    ),
                    "instruction_note": (
                        "Submitted through inference_instruct2. Valid audio does not by itself prove instruction obedience."
                        if applied_instruction
                        else "No instruction was requested; inference_zero_shot was used."
                    ),
                }
            )
            if not valid:
                background_failure = bool(background_capture.failures)
                observation.update(
                    {
                        "error_type": (
                            "background_thread_exception"
                            if background_failure
                            else "implausible_audio_output"
                        ),
                        "error": (
                            "An uncaught background-thread exception occurred during stream consumption; "
                            "the preserved WAV is invalid evidence."
                            if background_failure
                            else "The output failed duration or signal-level validity checks."
                        ),
                    }
                )
        except Exception as error:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if background_capture.failures:
                observation["background_thread_failures"] = background_capture.failures
            observation.update(
                {
                    "generation_seconds": time.perf_counter() - started,
                    **(
                        {"peak_memory_bytes": int(torch.cuda.max_memory_allocated())}
                        if torch.cuda.is_available()
                        else {}
                    ),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        observations.append(observation)
        write_observations(args.output_dir / "generation-observations.json", observations)
    return 0 if args.allow_invalid_output or all(row["valid"] for row in observations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
