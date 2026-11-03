"""SwiGLU feed-forward network.

    FFN(x) = W_down( SiLU(W_gate x) * W_up x )

Three matrices instead of the classic two (GELU MLP), so for an equal
parameter budget the hidden size is set to ~2/3 of the usual 4*d_model
(here 1408 ~= 2.75 * 512 for the 50M model).  Used by Llama, Mistral, Qwen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
