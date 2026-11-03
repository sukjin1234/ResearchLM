"""RMSNorm (Root Mean Square Layer Normalisation).

    y = x / sqrt(mean(x^2) + eps) * weight

Compared with LayerNorm it has no mean-subtraction and no bias, so it is
cheaper and (empirically) just as good for Transformers.  The statistics are
computed in fp32 even under fp16/bf16 autocast, because the sum of squares of
a 512-dim vector can overflow fp16.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rms).to(in_dtype) * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"
