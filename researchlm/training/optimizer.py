"""AdamW with the standard two parameter groups.

Weight decay is applied only to 2-D weight matrices (Linear + Embedding).
Norm gains (1-D) are *not* decayed: decaying them toward zero would shrink
every activation for no modelling benefit.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from .config import TrainConfig


def build_optimizer(model: nn.Module, cfg: TrainConfig, device_type: str) -> torch.optim.AdamW:
    decay, no_decay = [], []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:  # tied weights appear twice
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # fused AdamW is noticeably faster on CUDA
    kwargs = dict(lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    if device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    opt = torch.optim.AdamW(groups, **kwargs)
    opt.n_decay = sum(p.numel() for p in decay)      # type: ignore[attr-defined]
    opt.n_no_decay = sum(p.numel() for p in no_decay)  # type: ignore[attr-defined]
    return opt
