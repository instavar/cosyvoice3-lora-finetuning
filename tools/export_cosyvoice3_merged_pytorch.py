#!/usr/bin/env python3
"""Export one merged CosyVoice3 LLM as a reloadable safetensors artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosyvoice-dir", type=Path, required=True)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exporter-revision", required=True)
    parser.add_argument("--source-adapter-sha256", required=True)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cosyvoice_dir = args.cosyvoice_dir.resolve()
    matcha_dir = cosyvoice_dir / "third_party" / "Matcha-TTS"
    if not matcha_dir.is_dir():
        raise FileNotFoundError(f"CosyVoice Matcha-TTS checkout not found: {matcha_dir}")
    sys.path.insert(0, str(matcha_dir))
    sys.path.insert(0, str(cosyvoice_dir))

    from cosyvoice.cli.cosyvoice import CosyVoice3
    from infer_cosyvoice3_lora import (
        apply_lora_to_cosyvoice3,
        merge_lora_into_cosyvoice3,
    )
    from merged_pytorch_artifact import export_merged_pytorch_artifact

    cosyvoice = CosyVoice3(args.pretrained_dir, fp16=args.fp16)
    peft_model = apply_lora_to_cosyvoice3(cosyvoice, args.lora_dir)
    merged_model = merge_lora_into_cosyvoice3(cosyvoice, peft_model)
    manifest = export_merged_pytorch_artifact(
        merged_model,
        args.output_dir,
        exporter_revision=args.exporter_revision,
        source_adapter_sha256=args.source_adapter_sha256,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
