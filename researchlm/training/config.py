"""Training hyper-parameters (everything that is *not* model shape)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    run_name: str = "researchlm-50m"

    # --- paths (defaults overridden by configs/server/*.yaml) ---
    tokenizer_path: str = "/storage/researchlm/tokenizer/researchlm_32k.model"
    tokenized_dir: str = "/storage/researchlm/tokenized"
    checkpoint_dir: str = "/storage/researchlm/checkpoints"
    log_dir: str = "logs"

    # --- data ---
    mixture: dict[str, float] = field(default_factory=lambda: {"english": 0.5, "korean": 0.25, "code": 0.15, "math": 0.10})
    seq_len: int = 2048
    seed: int = 1234

    # --- batch ---
    micro_batch_size: int = 8          # sequences per GPU per forward pass
    grad_accum_steps: int = 16         # micro-batches per optimizer step
    # global_batch = micro_batch_size * grad_accum_steps * world_size

    # --- schedule ---
    max_steps: int = 2000
    warmup_steps: int = 100
    lr: float = 1.5e-3
    min_lr_ratio: float = 0.1          # final LR = lr * min_lr_ratio
    scheduler: str = "cosine"          # cosine | wsd | linear | constant
    wsd_decay_fraction: float = 0.2    # only for scheduler == wsd

    # --- optimizer (AdamW) ---
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # --- precision / speed ---
    precision: str = "auto"            # auto | bf16 | fp16 | fp32
    compile: bool = False
    gradient_checkpointing: bool = False
    fp16_init_scale: float = 2.0**14

    # --- eval / logging / checkpoint ---
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    save_interval: int = 500
    keep_last_checkpoints: int = 3
    wandb: bool = False
    wandb_project: str = "researchlm"

    # --- resume ---
    resume: str | None = None          # path to checkpoint dir or "latest"

    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        if extra:
            kwargs.setdefault("extra", {}).update(extra)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path, server_yaml: str | Path | None = None, overrides: dict | None = None) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if server_yaml is not None:
            with open(server_yaml, "r", encoding="utf-8") as f:
                server = yaml.safe_load(f) or {}
            # server config supplies paths + precision; train yaml wins if explicit
            for k in ("tokenized_dir", "checkpoint_dir", "tokenizer_path", "precision"):
                if k in server and k not in data:
                    data[k] = server[k]
        if overrides:
            data.update(overrides)
        return cls.from_dict(data)

    def global_batch_tokens(self, world_size: int) -> int:
        return self.micro_batch_size * self.grad_accum_steps * world_size * self.seq_len
