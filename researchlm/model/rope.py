"""Rotary Position Embedding (RoPE).

Instead of *adding* a position vector to the token embedding, RoPE *rotates*
each pair of query/key dimensions by an angle proportional to the token's
position.  The dot product q·k then depends only on the *relative* distance
between the two positions, which is what attention actually needs.

We use the "rotate-half" layout (same as Llama / GPT-NeoX):
    x = [x_1 ... x_{d/2} | x_{d/2+1} ... x_d]
    rotate_half(x) = [-x_{d/2+1} ... -x_d | x_1 ... x_{d/2}]
    x_rot = x * cos + rotate_half(x) * sin
"""

from __future__ import annotations

import torch


def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each of shape [max_seq_len, head_dim], in fp32."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )  # [head_dim/2]
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)   # [T, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to x of shape [B, n_heads, T, head_dim].

    cos/sin must already be sliced to the positions of x: [T, head_dim].
    The rotation is done in fp32 for numerical safety under fp16 autocast.
    """
    in_dtype = x.dtype
    x32 = x.float()
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    out = x32 * cos + rotate_half(x32) * sin
    return out.to(in_dtype)
