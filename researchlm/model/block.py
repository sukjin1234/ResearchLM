"""One pre-norm Transformer decoder block.

    h = x + Attn(RMSNorm(x))
    y = h + FFN(RMSNorm(h))

"Pre-norm" (normalise *before* each sub-layer) keeps the residual stream
un-normalised, which makes deep stacks train stably without warm-up tricks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import CausalSelfAttention, KVCache
from .config import ModelConfig
from .mlp import SwiGLU
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x
