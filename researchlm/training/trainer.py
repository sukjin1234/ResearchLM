"""The training loop.

One optimizer step =
    for grad_accum_steps:
        x, y = next(batch)                       # [micro_batch, seq_len]
        with autocast(fp16/bf16):
            logits, loss = model(x, y)
        scaler.scale(loss / grad_accum_steps).backward()
    unscale -> clip_grad_norm -> optimizer.step -> scaler.update -> zero_grad

Precision policy
    bf16 : no loss scaling needed (same exponent range as fp32).  Ampere+.
    fp16 : needs GradScaler - gradients are multiplied by a large factor so
           tiny values do not underflow to zero, then un-scaled before the
           optimizer step.  Required on Turing (RTX 8000).
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import MixedTokenDataset, build_domains
from ..model import ModelConfig, ResearchLM
from .checkpoint import load_training_state, resolve_checkpoint, save_checkpoint, unwrap
from .config import TrainConfig
from .distributed import DistInfo, all_reduce_mean
from .optimizer import build_optimizer
from .scheduler import lr_at


def pick_precision(requested: str, device: torch.device) -> str:
    if requested != "auto":
        if requested == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            print("[warn] bf16 requested but not supported on this GPU -> falling back to fp16")
            return "fp16"
        return requested
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


class Trainer:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig, dist: DistInfo) -> None:
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.dist = dist
        self.device = torch.device(f"cuda:{dist.local_rank}" if torch.cuda.is_available() else "cpu")
        self.precision = pick_precision(train_cfg.precision, self.device)
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.precision)

        torch.manual_seed(train_cfg.seed + dist.rank)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # ---- model --------------------------------------------------------
        self.raw_model = ResearchLM(model_cfg).to(self.device)
        self.raw_model.gradient_checkpointing = train_cfg.gradient_checkpointing
        model: nn.Module = self.raw_model
        if train_cfg.compile:
            model = torch.compile(model)
        if dist.enabled:
            model = DDP(model, device_ids=[dist.local_rank])
        self.model = model

        # ---- optimizer / scaler ------------------------------------------
        self.optimizer = build_optimizer(self.raw_model, train_cfg, self.device.type)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.precision == "fp16"), init_scale=train_cfg.fp16_init_scale
        )

        # ---- data ----------------------------------------------------------
        domains = build_domains(train_cfg.tokenized_dir, train_cfg.mixture)
        self.train_data = MixedTokenDataset(
            domains, "train", train_cfg.seq_len, train_cfg.micro_batch_size,
            rank=dist.rank, world_size=dist.world_size, seed=train_cfg.seed,
        )
        self.val_data = MixedTokenDataset(
            domains, "val", train_cfg.seq_len, train_cfg.micro_batch_size,
            rank=dist.rank, world_size=dist.world_size, seed=train_cfg.seed + 999,
        )

        # ---- bookkeeping ---------------------------------------------------
        self.step = 0
        self.tokens_seen = 0
        self.run_dir = Path(train_cfg.checkpoint_dir) / train_cfg.run_name
        self.log_path = Path(train_cfg.log_dir) / f"{train_cfg.run_name}.jsonl"
        if dist.is_main:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.wandb = None
        if train_cfg.wandb and dist.is_main:
            import wandb
            self.wandb = wandb.init(project=train_cfg.wandb_project, name=train_cfg.run_name,
                                    config={"model": model_cfg.to_dict(), "train": train_cfg.to_dict()})

        if train_cfg.resume:
            self._resume(train_cfg.resume)

    # ------------------------------------------------------------------ #
    def _resume(self, where: str) -> None:
        path = self.run_dir if where == "latest" else where
        ckpt = resolve_checkpoint(path)
        state = load_training_state(ckpt, self.model, self.optimizer, self.device)
        self.step = state["step"]
        self.tokens_seen = state["tokens_seen"]
        if state.get("scaler") is not None and self.precision == "fp16":
            self.scaler.load_state_dict(state["scaler"])
        self.train_data.load_state_dict(state["train_data"])
        self.log(f"resumed from {ckpt} at step {self.step}")

    # ------------------------------------------------------------------ #
    def log(self, msg: str) -> None:
        if self.dist.is_main:
            print(msg, flush=True)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if not self.dist.is_main:
            return
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=metrics["step"])

    def autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    # ------------------------------------------------------------------ #
    def describe(self) -> None:
        cfg, m = self.cfg, self.raw_model
        gb = cfg.global_batch_tokens(self.dist.world_size)
        self.log("=" * 72)
        self.log(f"run            : {cfg.run_name}")
        self.log(f"device         : {self.device} x{self.dist.world_size}  precision={self.precision}  compile={cfg.compile}  grad_ckpt={cfg.gradient_checkpointing}")
        self.log(f"params         : {m.num_params():,} total | {m.num_params(non_embedding=True):,} non-embedding")
        self.log(f"model          : L={self.model_cfg.n_layers} d={self.model_cfg.d_model} H={self.model_cfg.n_heads} "
                 f"KV={self.model_cfg.n_kv_heads} ffn={self.model_cfg.ffn_hidden} ctx={self.model_cfg.max_seq_len}")
        self.log(f"batch          : {cfg.micro_batch_size} x {cfg.grad_accum_steps} accum x {self.dist.world_size} gpu "
                 f"x {cfg.seq_len} seq = {gb:,} tokens/step")
        self.log(f"schedule       : {cfg.scheduler} lr={cfg.lr} warmup={cfg.warmup_steps} steps={cfg.max_steps} "
                 f"-> {gb * cfg.max_steps / 1e9:.2f}B tokens total")
        self.log(f"optimizer      : AdamW b=({cfg.beta1},{cfg.beta2}) wd={cfg.weight_decay} "
                 f"decay={self.optimizer.n_decay:,} no_decay={self.optimizer.n_no_decay:,}")
        for name, info in self.train_data.summary().items():
            epochs = gb * cfg.max_steps * info["weight"] / max(1, info["tokens"])
            self.log(f"data[{name:8s}] : {info['tokens']:>13,} train tokens  weight={info['weight']:.2f}  "
                     f"-> ~{epochs:.1f} epochs over the run")
        self.log("=" * 72)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        for _ in range(self.cfg.eval_batches):
            x, y = next(self.val_data)
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with self.autocast():
                _, loss = self.model(x, y)
            losses.append(loss.float())
        self.model.train()
        mean = torch.stack(losses).mean()
        return all_reduce_mean(mean, self.dist).item()

    # ------------------------------------------------------------------ #
    def train(self) -> None:
        cfg = self.cfg
        self.describe()
        self.model.train()
        gb_tokens = cfg.global_batch_tokens(self.dist.world_size)

        t0 = time.time()
        tokens_window = 0
        while self.step < cfg.max_steps:
            lr = lr_at(self.step, cfg)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            loss_accum = torch.zeros((), device=self.device)
            for micro in range(cfg.grad_accum_steps):
                x, y = next(self.train_data)
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                # Only sync gradients across GPUs on the last micro step.
                if self.dist.enabled:
                    self.model.require_backward_grad_sync = micro == cfg.grad_accum_steps - 1
                with self.autocast():
                    _, loss = self.model(x, y)
                    loss = loss / cfg.grad_accum_steps
                loss_accum += loss.detach()
                self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.step += 1
            self.tokens_seen += gb_tokens
            tokens_window += gb_tokens

            # ---- logging --------------------------------------------------
            if self.step % cfg.log_interval == 0 or self.step == 1:
                loss_val = all_reduce_mean(loss_accum, self.dist).item()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t0
                tok_s = tokens_window / max(dt, 1e-6)
                mem = torch.cuda.max_memory_allocated() / 2**30 if self.device.type == "cuda" else 0.0
                gn = float(grad_norm) if math.isfinite(float(grad_norm)) else float("nan")
                self.log(
                    f"step {self.step:6d} | loss {loss_val:.4f} | ppl {math.exp(min(loss_val, 20)):9.2f} | "
                    f"lr {lr:.2e} | gnorm {gn:6.3f} | {tok_s/1e3:7.1f}k tok/s | "
                    f"{self.tokens_seen/1e9:.3f}B tokens | mem {mem:.1f}GB"
                    + (f" | scale {self.scaler.get_scale():.0f}" if self.precision == "fp16" else "")
                )
                self.log_metrics({
                    "step": self.step, "loss": loss_val, "lr": lr, "grad_norm": gn,
                    "tokens_seen": self.tokens_seen, "tok_per_s": tok_s, "mem_gb": mem,
                    "loss_scale": self.scaler.get_scale() if self.precision == "fp16" else None,
                })
                t0, tokens_window = time.time(), 0
                if not math.isfinite(loss_val):
                    raise RuntimeError("loss is NaN/inf - see docs section 'Learning failure diagnosis'")

            # ---- eval -------------------------------------------------------
            if self.step % cfg.eval_interval == 0 or self.step == cfg.max_steps:
                val = self.evaluate()
                self.log(f"[eval] step {self.step} | val_loss {val:.4f} | val_ppl {math.exp(min(val, 20)):.2f}")
                self.log_metrics({"step": self.step, "val_loss": val})
                t0 = time.time()

            # ---- checkpoint ------------------------------------------------
            if self.step % cfg.save_interval == 0 or self.step == cfg.max_steps:
                self.save()
                t0 = time.time()

        self.log("training finished")
        if self.wandb is not None:
            self.wandb.finish()

    # ------------------------------------------------------------------ #
    def save(self) -> None:
        if not self.dist.is_main:
            if self.dist.enabled:
                torch.distributed.barrier()
            return
        state = {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "scaler": self.scaler.state_dict() if self.precision == "fp16" else None,
            "train_data": self.train_data.state_dict(),
            "precision": self.precision,
        }
        path = save_checkpoint(self.run_dir, self.step, self.model, self.optimizer,
                               self.model_cfg, self.cfg, state, keep_last=self.cfg.keep_last_checkpoints)
        self.log(f"[ckpt] saved {path}")
        if self.dist.enabled:
            torch.distributed.barrier()
