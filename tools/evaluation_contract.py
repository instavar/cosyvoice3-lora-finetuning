"""Dependency-free artifact-mode validation for CosyVoice3 evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from merged_pytorch_artifact import validate_merged_pytorch_artifact


BASE_ASSETS = ("cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt")
ADAPTER_ASSETS = ("adapter_config.json", "adapter_model.safetensors")


def _require_assets(root: Path, name: str, assets: tuple[str, ...]) -> None:
    if not root.is_dir():
        raise ValueError(f"{name} must be an existing directory")
    missing = [asset for asset in assets if not (root / asset).is_file()]
    if missing:
        raise ValueError(f"{name} is missing required assets: {', '.join(missing)}")


def resolve_inference_mode(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> str:
    """Resolve exactly one unchanged-base, adapter, or merged condition."""
    mode = args.inference_mode
    vllm_sampling_profile = getattr(args, "vllm_sampling_profile", "upstream")
    if vllm_sampling_profile not in {
        "upstream",
        "request-seeded",
        "request-seeded-top-p-0.8",
    }:
        parser.error("unsupported vLLM sampling profile")
    if mode is None:
        if args.lora_dir and args.vllm_dir:
            mode = "merged-vllm"
        elif args.lora_dir:
            mode = "adapter"
        else:
            parser.error("base evaluation requires explicit --inference-mode base")
    if mode not in {
        "base",
        "adapter",
        "merged-pytorch",
        "reloaded-merged-pytorch",
        "merged-vllm",
    }:
        parser.error(
            "inference mode must be base, adapter, merged-pytorch, "
            "reloaded-merged-pytorch, or merged-vllm"
        )

    pretrained = Path(args.pretrained_dir)
    try:
        _require_assets(pretrained, "pretrained model", BASE_ASSETS)
    except ValueError as error:
        parser.error(str(error))

    prompt_wav = Path(args.prompt_wav)
    if prompt_wav.is_symlink():
        parser.error("prompt WAV must not be a symlink")
    if not prompt_wav.is_file():
        parser.error("prompt WAV must be an existing file")

    if mode == "base":
        if (
            args.lora_dir
            or args.vllm_dir
            or args.reuse_vllm_dir
            or args.merged_pytorch_dir
            or vllm_sampling_profile != "upstream"
        ):
            parser.error(
                "base mode forbids adapted and merged artifact arguments"
            )
        return mode

    if mode == "reloaded-merged-pytorch":
        if (
            args.lora_dir
            or args.vllm_dir
            or args.reuse_vllm_dir
            or vllm_sampling_profile != "upstream"
        ):
            parser.error(
                "reloaded-merged-pytorch mode accepts only --merged-pytorch-dir"
            )
        if not args.merged_pytorch_dir:
            parser.error(
                "reloaded-merged-pytorch mode requires --merged-pytorch-dir"
            )
        try:
            validate_merged_pytorch_artifact(Path(args.merged_pytorch_dir))
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        return mode

    if not args.lora_dir:
        parser.error(f"{mode} mode requires --lora-dir")
    try:
        _require_assets(Path(args.lora_dir), "LoRA adapter", ADAPTER_ASSETS)
    except ValueError as error:
        parser.error(str(error))

    if mode in {"adapter", "merged-pytorch"}:
        if (
            args.vllm_dir
            or args.reuse_vllm_dir
            or args.merged_pytorch_dir
            or vllm_sampling_profile != "upstream"
        ):
            parser.error(f"{mode} mode forbids other merged artifact arguments")
        return mode

    if args.merged_pytorch_dir:
        parser.error("merged-vllm mode forbids --merged-pytorch-dir")

    if not args.vllm_dir:
        parser.error("merged-vllm mode requires --vllm-dir")
    export = Path(args.vllm_dir)
    if args.reuse_vllm_dir:
        if not export.is_dir() or not (export / "config.json").is_file():
            parser.error(
                "merged-vllm reuse requires an existing export with config.json"
            )
    elif export.exists():
        parser.error(
            "new merged-vllm export path must not exist; use --reuse-vllm-dir "
            "only after verifying an existing export"
        )
    return mode
