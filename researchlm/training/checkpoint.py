"""Checkpoint save / load.

A checkpoint is a *directory*:

    <checkpoint_dir>/<run_name>/step_000500/
        model.pt        # raw (un-wrapped, un-compiled) state_dict
        optimizer.pt
        trainer_state.pt  # step, scaler, data sampler state, RNG
        model_config.yaml
        train_config.yaml
    <checkpoint_dir>/<run_name>/latest -> step_000500  (text file, not symlink,
                                                       so it works on any FS)
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from ..model import ModelConfig
from .config import TrainConfig


def unwrap(model: nn.Module) -> nn.Module:
    """Strip DDP and torch.compile wrappers."""
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def save_checkpoint(
    run_dir: str | Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    trainer_state: dict[str, Any],
    keep_last: int = 3,
) -> Path:
    run_dir = Path(run_dir)
    ckpt = run_dir / f"step_{step:06d}"
    tmp = run_dir / f".tmp_step_{step:06d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    torch.save(unwrap(model).state_dict(), tmp / "model.pt")
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    trainer_state = dict(trainer_state)
    trainer_state["rng"] = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    torch.save(trainer_state, tmp / "trainer_state.pt")
    with open(tmp / "model_config.yaml", "w") as f:
        yaml.safe_dump(model_cfg.to_dict(), f, sort_keys=False)
    with open(tmp / "train_config.yaml", "w") as f:
        yaml.safe_dump(train_cfg.to_dict(), f, sort_keys=False)

    if ckpt.exists():
        shutil.rmtree(ckpt)
    tmp.rename(ckpt)  # atomic on the same filesystem
    (run_dir / "latest").write_text(ckpt.name)

    # prune
    steps = sorted(p for p in run_dir.glob("step_*") if p.is_dir())
    for old in steps[:-keep_last] if keep_last > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
    return ckpt


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept a step dir, a run dir (-> latest), or a run dir + 'latest'."""
    p = Path(path)
    if (p / "model.pt").exists():
        return p
    latest = p / "latest"
    if latest.exists():
        return p / latest.read_text().strip()
    raise FileNotFoundError(f"no checkpoint found at {p}")


def load_model_from_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, ModelConfig]:
    from ..model import ResearchLM

    ckpt = resolve_checkpoint(path)
    model_cfg = ModelConfig.from_yaml(ckpt / "model_config.yaml")
    model = ResearchLM(model_cfg)
    state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device), model_cfg


def load_training_state(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    ckpt = resolve_checkpoint(path)
    state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    unwrap(model).load_state_dict(state, strict=True)
    opt_state = torch.load(ckpt / "optimizer.pt", map_location=device, weights_only=True)
    optimizer.load_state_dict(opt_state)
    trainer_state = torch.load(ckpt / "trainer_state.pt", map_location="cpu", weights_only=False)
    rng = trainer_state.pop("rng", None)
    if rng:
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    return trainer_state
