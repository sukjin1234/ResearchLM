"""Learning-rate schedules, expressed as pure functions of the step index.

    warmup : linear 0 -> lr over warmup_steps (protects Adam's early moments)
    cosine : lr -> min_lr following a half cosine (the classic choice)
    wsd    : Warmup-Stable-Decay, constant then linear decay in the last X %
             (lets you extend training without re-planning the whole curve)
"""

from __future__ import annotations

import math

from .config import TrainConfig


def lr_at(step: int, cfg: TrainConfig) -> float:
    lr, min_lr = cfg.lr, cfg.lr * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return min_lr

    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)

    if cfg.scheduler == "cosine":
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    if cfg.scheduler == "linear":
        return lr - (lr - min_lr) * progress
    if cfg.scheduler == "constant":
        return lr
    if cfg.scheduler == "wsd":
        decay_start = 1.0 - cfg.wsd_decay_fraction
        if progress < decay_start:
            return lr
        frac = (progress - decay_start) / max(1e-9, cfg.wsd_decay_fraction)
        return lr - (lr - min_lr) * frac
    raise ValueError(f"unknown scheduler {cfg.scheduler}")
