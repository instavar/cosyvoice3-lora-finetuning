#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torchaudio

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover
    raise ImportError("peft is required for LoRA inference. Install with: pip install peft") from exc

from cosyvoice.cli.cosyvoice import CosyVoice3


DEFAULT_PROMPT_WAV = ""
DEFAULT_PROMPT_TEXT = ""


def read_texts(text: str | None, texts_file: str | None) -> list[str]:
    if text:
        return [text]
    if not texts_file:
        raise ValueError("Provide --text or --texts-file")
    path = Path(texts_file)
    if not path.exists():
        raise FileNotFoundError(f"Texts file not found: {texts_file}")
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    if not lines:
        raise ValueError(f"No valid lines in {texts_file}")
    return lines


def expose_embed_tokens_for_cosyvoice(peft_model: PeftModel) -> None:
    """Restore the attribute path used by CosyVoice after PEFT wrapping."""
    qwen2_causal = peft_model.model
    if not hasattr(qwen2_causal, "embed_tokens") and hasattr(qwen2_causal, "model"):
        qwen2_causal.embed_tokens = qwen2_causal.model.embed_tokens


def apply_lora_to_cosyvoice3(cosyvoice: CosyVoice3, lora_dir: str) -> PeftModel:
    model = cosyvoice.model
    if not hasattr(model, "llm") or not hasattr(model.llm, "llm"):
        raise RuntimeError("Unexpected CosyVoice3 model structure; cannot locate Qwen2 encoder.")
    encoder = model.llm.llm
    if not hasattr(encoder, "model"):
        raise RuntimeError("Expected Qwen2Encoder with .model attribute.")
    base = encoder.model
    peft_model = PeftModel.from_pretrained(base, lora_dir, is_trainable=False)
    peft_model.eval()
    expose_embed_tokens_for_cosyvoice(peft_model)
    encoder.model = peft_model
    return peft_model


def enable_vllm_with_merged_lora(
    cosyvoice: CosyVoice3,
    peft_model: PeftModel,
    vllm_dir: str,
) -> None:
    """Merge the adapter, export the merged LLM, and enable CosyVoice vLLM."""
    export_dir = Path(vllm_dir)
    if export_dir.exists():
        raise FileExistsError(
            f"vLLM export directory already exists: {export_dir}. "
            "Use a new directory so a stale export cannot be loaded for this adapter."
        )

    try:
        merged_model = peft_model.merge_and_unload(safe_merge=True)
    except TypeError:
        merged_model = peft_model.merge_and_unload()

    encoder = cosyvoice.model.llm.llm
    encoder.model = merged_model

    try:
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    except ImportError as exc:
        raise ImportError(
            "vLLM inference requires a CosyVoice-supported vLLM installation. "
            "See the repository README for supported version pairs."
        ) from exc

    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
    cosyvoice.model.load_vllm(str(export_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice3 LoRA inference helper.")
    parser.add_argument("--pretrained-dir", required=True, help="CosyVoice3 pretrained model directory")
    parser.add_argument("--lora-dir", required=True, help="LoRA adapter directory")
    parser.add_argument("--prompt-wav", required=True, help="Prompt wav path")
    parser.add_argument("--prompt-text", required=True, help="Prompt text matching the prompt audio")
    parser.add_argument("--text", default="", help="Single input text")
    parser.add_argument("--texts-file", default="", help="Text file with one sentence per line")
    parser.add_argument("--out-wav", default="", help="Output wav path (single text only)")
    parser.add_argument("--out-dir", default="", help="Output directory (multiple texts)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 inference")
    parser.add_argument(
        "--vllm-dir",
        default="",
        help="Export the merged base plus LoRA model here and use vLLM for LLM decoding",
    )
    parser.add_argument("--no-text-frontend", dest="text_frontend", action="store_false", help="Disable text frontend")
    parser.set_defaults(text_frontend=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    texts = read_texts(args.text or None, args.texts_file or None)
    if args.out_wav and len(texts) != 1:
        raise ValueError("--out-wav only supports a single --text")
    if not args.out_wav and not args.out_dir:
        raise ValueError("Provide --out-wav or --out-dir")

    prompt_wav = Path(args.prompt_wav)
    if not prompt_wav.exists():
        raise FileNotFoundError(f"Prompt wav not found: {prompt_wav}")

    cosyvoice = CosyVoice3(args.pretrained_dir, fp16=args.fp16)
    peft_model = apply_lora_to_cosyvoice3(cosyvoice, args.lora_dir)
    if args.vllm_dir:
        enable_vllm_with_merged_lora(cosyvoice, peft_model, args.vllm_dir)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    for idx, text in enumerate(texts, start=1):
        chunks = []
        for model_output in cosyvoice.inference_zero_shot(
            text,
            args.prompt_text,
            str(prompt_wav),
            stream=False,
            speed=args.speed,
            text_frontend=args.text_frontend,
        ):
            chunks.append(model_output["tts_speech"].cpu())
        if not chunks:
            raise RuntimeError(f"No audio generated for text: {text}")
        speech = torch.cat(chunks, dim=1).to(torch.float32)

        if args.out_wav:
            out_path = Path(args.out_wav)
        else:
            out_path = out_dir / f"sample_{idx:02d}.wav"
        torchaudio.save(str(out_path), speech, cosyvoice.sample_rate)
        logging.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
