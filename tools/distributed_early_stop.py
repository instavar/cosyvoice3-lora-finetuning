from __future__ import annotations

import torch
import torch.distributed as dist


def synchronize_early_stop(local_stop: bool, *, device: torch.device | None = None) -> bool:
    """Return one stop decision shared by every initialized training rank."""
    if not dist.is_available() or not dist.is_initialized():
        return bool(local_stop)
    if device is None:
        if dist.get_backend() == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
    flag = torch.tensor([1 if local_stop else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())
