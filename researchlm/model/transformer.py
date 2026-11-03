"""ResearchLM: a decoder-only Transformer language model.

    tokens --embedding--> [block x n_layers] --RMSNorm--> lm_head --> logits

The lm_head shares its weight matrix with the input embedding (weight tying),
which saves vocab_size * d_model parameters (16.8M for the 50M model, a third
of the whole network).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import KVCache
from .block import TransformerBlock
from .config import ModelConfig
from .norm import RMSNorm
from .rope import precompute_rope_cache


class ResearchLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Recompute block activations in backward instead of storing them.
        # Costs ~33% more compute, saves most activation memory (needed for 1B).
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, i) for i in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        # persistent=False -> not saved in checkpoints (they are deterministic)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style: shrink the residual-branch output projections so the
        # residual stream variance does not grow with depth.
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=cfg.init_std * scale)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=cfg.init_std * scale)

    # ------------------------------------------------------------------ #
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    # ------------------------------------------------------------------ #
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[KVCache]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (logits, loss).

        input_ids : [B, T] token ids
        targets   : [B, T] next-token ids (input shifted left by one) or None
        kv_caches : one KVCache per layer when generating incrementally
        start_pos : absolute position of input_ids[:, 0] (needed for RoPE with cache)
        """
        B, T = input_ids.shape
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {start_pos + T} exceeds max_seq_len {self.cfg.max_seq_len}"
            )

        x = self.embed_dropout(self.embed_tokens(input_ids))
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            if self.gradient_checkpointing and self.training and cache is None:
                x = checkpoint(layer, x, cos, sin, None, use_reentrant=False)
            else:
                x = layer(x, cos, sin, cache)

        x = self.final_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            # cross_entropy runs in fp32 under autocast; ignore_index lets a
            # data loader mask out padding if it ever needs to.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss

        return self.lm_head(x), None

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive sampling with a KV cache.  input_ids: [B, T]."""
        self.eval()
        B, T = input_ids.shape
        kv_caches = [KVCache() for _ in self.layers]
        eos = self.cfg.eos_token_id if eos_token_id is None else eos_token_id

        out = input_ids
        logits, _ = self(input_ids, kv_caches=kv_caches, start_pos=0)
        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                for b in range(B):
                    seen = torch.unique(out[b])
                    pen = next_logits[b, seen]
                    next_logits[b, seen] = torch.where(pen > 0, pen / repetition_penalty, pen * repetition_penalty)

            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None and top_k > 0:
                    kth = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values[:, -1:]
                    next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum = probs.cumsum(dim=-1)
                    remove = cum - probs > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    next_logits = torch.full_like(next_logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, eos), next_token)
            out = torch.cat([out, next_token], dim=1)
            finished |= next_token.squeeze(1) == eos
            if finished.all() or out.shape[1] >= self.cfg.max_seq_len:
                break

            logits, _ = self(next_token, kv_caches=kv_caches, start_pos=out.shape[1] - 1)

        return out
