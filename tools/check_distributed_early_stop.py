#!/usr/bin/env python3
"""Two-rank smoke for the synchronized CosyVoice early-stop decision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch.distributed as dist

from distributed_early_stop import synchronize_early_stop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_stop = rank == 0
    synchronized = synchronize_early_stop(local_stop)
    result = {
        "rank": rank,
        "world_size": dist.get_world_size(),
        "local_stop": local_stop,
        "synchronized_stop": synchronized,
        "pid": os.getpid(),
    }
    (args.output_dir / f"rank-{rank}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dist.barrier()
    dist.destroy_process_group()
    return 0 if synchronized else 1


if __name__ == "__main__":
    raise SystemExit(main())
