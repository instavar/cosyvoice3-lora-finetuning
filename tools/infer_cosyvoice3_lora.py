#!/usr/bin/env python3
from __future__ import annotations

import argparse
import builtins
import json
import logging
import os
import threading
from pathlib import Path
from typing import Iterable, Union

import torch
import torchaudio

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover
    raise ImportError("peft is required for LoRA inference. Install with: pip install peft") from exc

from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice_generation import invoke_generation, validate_generation_inputs
from thread_failures import BackgroundThreadFailureCapture

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
        # Bypass nn.Module.__setattr__ so this compatibility alias is not
        # registered as a duplicate submodule in state_dict exports.
        object.__setattr__(qwen2_causal, "embed_tokens", qwen2_causal.model.embed_tokens)


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


def merge_lora_into_cosyvoice3(
    cosyvoice: CosyVoice3, peft_model: PeftModel
) -> torch.nn.Module:
    """Merge a loaded adapter and keep the merged LLM on the PyTorch route."""
    try:
        merged_model = peft_model.merge_and_unload(safe_merge=True)
    except TypeError:
        merged_model = peft_model.merge_and_unload()
    cosyvoice.model.llm.llm.model = merged_model
    return merged_model


def enable_vllm_with_merged_lora(
    cosyvoice: CosyVoice3,
    peft_model: PeftModel | None,
    vllm_dir: str,
    reuse_vllm_dir: bool = False,
    sampling_profile: str = "upstream",
) -> None:
    """Merge the adapter, export the merged LLM, and enable CosyVoice vLLM."""
    # Dynamic custom-model registration is process-local. Keep the V1 engine
    # in-process so a spawned worker does not lose the CosyVoice registry entry.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    export_dir = Path(vllm_dir)
    if export_dir.exists() and not reuse_vllm_dir:
        raise FileExistsError(
            f"vLLM export directory already exists: {export_dir}. "
            "Use a new directory or pass --reuse-vllm-dir after verifying the export."
        )

    if not export_dir.exists():
        if peft_model is None:
            raise ValueError("A loaded PEFT model is required to create a new vLLM export")
        merge_lora_into_cosyvoice3(cosyvoice, peft_model)

    # Older CosyVoice adapters relied on these names leaking from vLLM's
    # wildcard Qwen2 import. Current vLLM releases no longer export them.
    missing = object()
    previous_union = getattr(builtins, "Union", missing)
    previous_iterable = getattr(builtins, "Iterable", missing)
    builtins.Union = Union
    builtins.Iterable = Iterable
    try:
        from cosyvoice.utils.file_utils import export_cosyvoice2_vllm
        from cosyvoice.vllm import cosyvoice2 as cosyvoice2_module
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
        from vllm import EngineArgs, LLMEngine, ModelRegistry
    except ImportError as exc:
        raise ImportError(
            "vLLM inference requires a CosyVoice-supported vLLM installation. "
            "See the repository README for supported version pairs."
        ) from exc
    finally:
        if previous_union is missing:
            del builtins.Union
        else:
            builtins.Union = previous_union
        if previous_iterable is missing:
            del builtins.Iterable
        else:
            builtins.Iterable = previous_iterable

    def embed_input_ids(model, input_ids):
        return model.model.embed_tokens(input_ids)

    # vLLM 0.15 requires this interface and its Qwen2Model no longer exposes
    # the older get_input_embeddings helper used by CosyVoice.
    CosyVoice2ForCausalLM.embed_input_ids = embed_input_ids

    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

    # Upstream only rewrites this architecture when the LM head uses a bias.
    # Export first and normalize it for bias-free CosyVoice3 checkpoints too.
    if not export_dir.exists():
        export_cosyvoice2_vllm(cosyvoice.model.llm, str(export_dir), cosyvoice.model.device)
    config_path = export_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["architectures"] = ["CosyVoice2ForCausalLM"]
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    original_parallel_lm_head = cosyvoice2_module.ParallelLMHead

    def parallel_lm_head_with_exported_bias(
        num_embeddings: int,
        embedding_dim: int,
        _legacy_bias: bool = False,
        **kwargs,
    ):
        return original_parallel_lm_head(
            num_embeddings,
            embedding_dim,
            bias=bool(config.get("use_bias", False)),
            **kwargs,
        )

    cosyvoice2_module.ParallelLMHead = parallel_lm_head_with_exported_bias

    # Release the duplicate PyTorch LLM before vLLM profiles free memory.
    # Upstream releases it after engine creation, which invalidates V1's
    # in-process memory snapshot as GPU memory rises during profiling.
    del cosyvoice.model.llm.llm.model.model.layers
    torch.cuda.empty_cache()
    engine_args = EngineArgs(
        model=str(export_dir),
        skip_tokenizer_init=True,
        enable_prompt_embeds=True,
        gpu_memory_utilization=0.2,
    )
    cosyvoice.model.llm.vllm = LLMEngine.from_engine_args(engine_args)
    cosyvoice.model.llm.lock = threading.Lock()
    from vllm_sampling_controls import install_vllm_sampling_control

    install_vllm_sampling_control(cosyvoice, sampling_profile)


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice3 LoRA inference helper.")
    parser.add_argument("--pretrained-dir", required=True, help="CosyVoice3 pretrained model directory")
    parser.add_argument("--lora-dir", required=True, help="LoRA adapter directory")
    parser.add_argument("--prompt-wav", required=True, help="Prompt wav path")
    parser.add_argument(
        "--prompt-text",
        default="",
        help="Prompt text matching the prompt audio; required for zero-shot",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Optional style instruction routed through CosyVoice inference_instruct2",
    )
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
    parser.add_argument(
        "--reuse-vllm-dir",
        action="store_true",
        help="Reuse an existing --vllm-dir instead of creating a fresh merged export",
    )
    parser.add_argument(
        "--vllm-sampling-profile",
        choices=("upstream", "request-seeded", "request-seeded-top-p-0.8"),
        default="upstream",
        help="Explicit vLLM request-sampling profile for bounded diagnosis",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Per-request seed required by request-seeded vLLM profiles",
    )
    parser.add_argument("--no-text-frontend", dest="text_frontend", action="store_false", help="Disable text frontend")
    parser.set_defaults(text_frontend=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    texts = read_texts(args.text or None, args.texts_file or None)
    for text in texts:
        validate_generation_inputs(
            text=text,
            instruction=args.instruction,
            prompt_text=args.prompt_text,
            prompt_wav=args.prompt_wav,
            speed=args.speed,
            text_frontend=args.text_frontend,
        )
    if args.out_wav and len(texts) != 1:
        raise ValueError("--out-wav only supports a single --text")
    if not args.out_wav and not args.out_dir:
        raise ValueError("Provide --out-wav or --out-dir")
    if not args.vllm_dir and args.vllm_sampling_profile != "upstream":
        raise ValueError("vLLM sampling profiles require --vllm-dir")
    from vllm_sampling_controls import validate_standalone_sampling_request

    validate_standalone_sampling_request(
        args.vllm_sampling_profile,
        args.seed,
        len(texts),
    )

    prompt_wav = Path(args.prompt_wav)
    if not prompt_wav.exists():
        raise FileNotFoundError(f"Prompt wav not found: {prompt_wav}")

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
            sampling_profile=args.vllm_sampling_profile,
        )
        if args.vllm_sampling_profile != "upstream":
            from vllm_sampling_controls import set_vllm_request_seed

            set_vllm_request_seed(cosyvoice, args.seed)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    for idx, text in enumerate(texts, start=1):
        chunks = []
        with BackgroundThreadFailureCapture() as background_capture:
            output_stream, route, _ = invoke_generation(
                cosyvoice,
                text=text,
                instruction=args.instruction,
                prompt_text=args.prompt_text,
                prompt_wav=str(prompt_wav),
                speed=args.speed,
                text_frontend=args.text_frontend,
            )
            logging.info("Generation route: %s", route)
            for model_output in output_stream:
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
        if background_capture.failures:
            raise RuntimeError(
                "Uncaught background-thread failure during generation; "
                f"preserved invalid output at {out_path}"
            )


if __name__ == "__main__":
    main()
