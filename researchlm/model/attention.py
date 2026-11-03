"""Causal self-attention with Grouped-Query Attention (GQA), QK-Norm and RoPE.

Shapes (B = batch, T = sequence, H = query heads, Hkv = kv heads, D = head_dim)

    x            : [B, T, d_model]
    q            : [B, H,   T, D]
    k, v         : [B, Hkv, T, D]
    output       : [B, T, d_model]

The actual softmax(QK^T / sqrt(D)) V is delegated to
``torch.nn.functional.scaled_dot_product_attention`` (SDPA), which picks the
fastest fused kernel available for the GPU (FlashAttention on Ampere+,
memory-efficient attention on Turing, plain math otherwise).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .norm import RMSNorm
from .rope import apply_rope


class KVCache:
    """Per-layer key/value cache used only at inference time."""

    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None  # [B, Hkv, T_past, D]
        self.v: Optional[torch.Tensor] = None

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep
        self.dropout = cfg.dropout

        d = cfg.d_model
        # One fused projection for q, k, v is slightly faster than three.
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)

        # QK-Norm: normalise every query / key head vector before RoPE.
        # Keeps the attention logits bounded -> no fp16 overflow, allows higher LR.
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # GQA: every KV head serves n_rep query heads.  We materialise the
        # repeat so that *every* SDPA backend works (the enable_gqa flag is
        # not supported by the memory-efficient kernel used on Turing GPUs).
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # With a KV cache the query block is the *tail* of the key block, so a
        # square causal mask would be wrong.  Only use is_causal when q and k
        # cover the same positions (training / prefill).
        if kv_cache is None or k.shape[2] == T:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            T_k = k.shape[2]
            # query i (absolute pos T_k - T + i) may attend to keys j <= its pos
            mask = torch.ones(T, T_k, dtype=torch.bool, device=x.device).tril(diagonal=T_k - T)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)
