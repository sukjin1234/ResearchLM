"""Minimal helpers around torch.distributed for single-node DDP.

Launch with:  torchrun --standalone --nproc_per_node=2 scripts/train.py ...
torchrun sets RANK / LOCAL_RANK / WORLD_SIZE in the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    enabled: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        info = DistInfo(
            rank=dist.get_rank(),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=dist.get_world_size(),
            enabled=True,
        )
        torch.cuda.set_device(info.local_rank)
        return info
    return DistInfo()


def cleanup_distributed(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_mean(t: torch.Tensor, info: DistInfo) -> torch.Tensor:
    if not info.enabled:
        return t
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / info.world_size
