"""Model hyper-parameters for ResearchLM.

Everything that defines the *shape* of the network lives here so that a
checkpoint can be re-built from its saved config alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    # --- vocabulary / sequence ---
    vocab_size: int = 32_768
    max_seq_len: int = 2_048

    # --- width / depth ---
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8          # query heads
    n_kv_heads: int = 2       # key/value heads (GQA). == n_heads -> MHA, 1 -> MQA
    ffn_hidden: int = 1_408   # SwiGLU hidden size (per branch)

    # --- normalisation / positions ---
    rms_norm_eps: float = 1e-5
    qk_norm: bool = True
    rope_theta: float = 10_000.0

    # --- regularisation ---
    dropout: float = 0.0

    # --- weights ---
    tie_embeddings: bool = True
    init_std: float = 0.02

    # --- special token ids (must match the tokenizer) ---
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 3

    name: str = "researchlm"
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """How many query heads share one KV head."""
        return self.n_heads // self.n_kv_heads

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        if extra:
            kwargs.setdefault("extra", {}).update(extra)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    # ------------------------------------------------------------------ #
    def estimate_params(self) -> dict[str, int]:
        """Analytical parameter count (useful before building the model)."""
        d, h, kvh, hd, f = self.d_model, self.n_heads, self.n_kv_heads, self.head_dim, self.ffn_hidden
        attn = d * (h * hd) + 2 * d * (kvh * hd) + (h * hd) * d
        qk = 2 * hd if self.qk_norm else 0
        ffn = 3 * d * f
        norms = 2 * d
        per_layer = attn + qk + ffn + norms
        embed = self.vocab_size * d
        head = 0 if self.tie_embeddings else self.vocab_size * d
        final_norm = d
        total = per_layer * self.n_layers + embed + head + final_norm
        return {
            "embedding": embed,
            "lm_head": head,
            "per_layer": per_layer,
            "layers_total": per_layer * self.n_layers,
            "total": total,
            "non_embedding": total - embed - head,
        }
