#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional

import yaml


@dataclass
class CheckpointInfo:
    tag: str
    yaml_path: Path
    step: Optional[int]
    epoch: Optional[int]
    mtime: float
    metric: Optional[float]


def _coerce_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _extract_metric(data: dict, metric: str) -> Optional[float]:
    loss_dict = data.get("loss_dict")
    if not isinstance(loss_dict, dict):
        return None
    if metric not in loss_dict:
        return None
    return _coerce_float(loss_dict[metric])


def _collect_checkpoints(model_dir: Path, metric: str) -> list[CheckpointInfo]:
    infos: list[CheckpointInfo] = []
    for yaml_path in sorted(model_dir.glob("*.yaml")):
        data = _load_yaml(yaml_path)
        if not data:
            continue
        if "step" not in data and "epoch" not in data and "loss_dict" not in data:
            continue
        tag = yaml_path.stem
        infos.append(
            CheckpointInfo(
                tag=tag,
                yaml_path=yaml_path,
                step=_coerce_int(data.get("step")),
                epoch=_coerce_int(data.get("epoch")),
                mtime=yaml_path.stat().st_mtime,
                metric=_extract_metric(data, metric),
            )
        )
    return infos


def _latest_tags(infos: list[CheckpointInfo], keep_latest: int) -> set[str]:
    if keep_latest <= 0 or not infos:
        return set()

    def sort_key(info: CheckpointInfo) -> tuple[int, int, float]:
        step = info.step if info.step is not None else -1
        epoch = info.epoch if info.epoch is not None else -1
        return (step, epoch, info.mtime)

    ordered = sorted(infos, key=sort_key)
    return {info.tag for info in ordered[-keep_latest:]}


def _best_tag(infos: list[CheckpointInfo], higher_is_better: bool) -> Optional[str]:
    candidates = [info for info in infos if info.metric is not None]
    if not candidates:
        return None
    if higher_is_better:
        return max(candidates, key=lambda info: info.metric).tag
    return min(candidates, key=lambda info: info.metric).tag


def _remove_path(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    if dry_run:
        print(f"[dry-run] remove {path}")
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune DeepSpeed checkpoints, keeping {latest N + best} by CV metric."
    )
    parser.add_argument("--model-dir", required=True, help="Model dir with checkpoint tags and YAML metadata.")
    parser.add_argument("--keep-latest", type=int, default=2, help="Number of most recent checkpoints to keep.")
    parser.add_argument("--keep-best", type=int, default=1, help="Number of best checkpoints to keep (0 to disable).")
    parser.add_argument("--metric", default="loss", help="CV metric key used to pick the best checkpoint.")
    parser.add_argument("--higher-is-better", action="store_true", help="Use when higher metric values are better.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without deleting files.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    infos = _collect_checkpoints(model_dir, args.metric)
    if not infos:
        print(f"No checkpoint YAMLs found in {model_dir}")
        return

    keep_tags = _latest_tags(infos, args.keep_latest)
    best_tag = _best_tag(infos, args.higher_is_better) if args.keep_best > 0 else None
    if best_tag is not None:
        keep_tags.add(best_tag)

    to_remove = [info for info in infos if info.tag not in keep_tags]
    print(f"Keeping tags: {sorted(keep_tags)}")
    if best_tag is None and args.keep_best > 0:
        print(f"No best checkpoint found for metric={args.metric}; keeping only latest.")

    if not to_remove:
        print("Nothing to prune.")
        return

    for info in to_remove:
        tag = info.tag
        _remove_path(model_dir / tag, args.dry_run)
        _remove_path(model_dir / f"{tag}.yaml", args.dry_run)
        _remove_path(model_dir / f"{tag}.pt", args.dry_run)

    print(f"Pruned {len(to_remove)} checkpoints from {model_dir}")


if __name__ == "__main__":
    main()
