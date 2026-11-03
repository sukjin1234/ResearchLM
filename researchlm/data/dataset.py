"""Streaming dataset over token shards with domain mixing.

For every domain we concatenate its shards (memory-mapped) into one long token
stream and cut it into windows of ``seq_len + 1`` tokens.  Every training step
picks a domain according to the mixture weights, then takes the next window of
that domain from a seeded random permutation.  When a domain runs out, a new
epoch starts with a new permutation.

The sampler is **resumable**: ``state_dict()`` captures per-domain epoch and
cursor, so a restarted job sees exactly the batches it would have seen.

Under DDP, rank r only consumes windows perm[r::world_size].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .shards import count_tokens, iter_shard_files, load_shard


@dataclass
class DomainSpec:
    name: str
    path: str  # <tokenized_dir>/<domain>
    weight: float


@dataclass
class _DomainState:
    epoch: int = 0
    cursor: int = 0  # index into the current permutation (rank-local)
    perm: np.ndarray | None = field(default=None, repr=False)


class _DomainStream:
    def __init__(self, spec: DomainSpec, split: str, seq_len: int, rank: int, world_size: int, seed: int):
        self.spec = spec
        self.split = split
        self.seq_len = seq_len
        self.rank, self.world_size = rank, world_size
        self.seed = seed

        files = list(iter_shard_files(spec.path, split))
        if not files:
            raise FileNotFoundError(f"no {split} shards in {spec.path}")
        self.shards = [load_shard(f) for f in files]
        self.lengths = np.array([len(s) for s in self.shards], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.total_tokens = int(self.offsets[-1])
        self.n_windows = (self.total_tokens - 1) // seq_len  # need seq_len + 1 tokens per window
        if self.n_windows < world_size:
            raise ValueError(f"{spec.name}/{split}: only {self.n_windows} windows for {world_size} ranks")
        self.state = _DomainState()
        self._new_epoch()

    # ---- permutation handling ------------------------------------------
    def _new_epoch(self) -> None:
        rng = np.random.default_rng(self.seed + 1_000_003 * self.state.epoch)
        perm = rng.permutation(self.n_windows)
        self.state.perm = perm[self.rank :: self.world_size]
        self.state.cursor = 0

    def _read(self, start: int, length: int) -> np.ndarray:
        """Read `length` tokens starting at global position `start` (may span shards)."""
        out = np.empty(length, dtype=np.int64)
        filled = 0
        shard_idx = int(np.searchsorted(self.offsets, start, side="right") - 1)
        pos = start - self.offsets[shard_idx]
        while filled < length:
            shard = self.shards[shard_idx]
            take = min(length - filled, len(shard) - pos)
            out[filled : filled + take] = shard[pos : pos + take]
            filled += take
            shard_idx += 1
            pos = 0
        return out

    def next_window(self) -> np.ndarray:
        st = self.state
        if st.cursor >= len(st.perm):
            st.epoch += 1
            self._new_epoch()
        w = int(st.perm[st.cursor])
        st.cursor += 1
        return self._read(w * self.seq_len, self.seq_len + 1)

    # ---- state ---------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.state.epoch, "cursor": self.state.cursor}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.state.epoch = int(d["epoch"])
        self._new_epoch()
        self.state.cursor = int(d["cursor"])

    @property
    def tokens_consumed_fraction(self) -> float:
        return self.state.epoch + self.state.cursor / max(1, len(self.state.perm))


class MixedTokenDataset:
    """Infinite iterator producing (input_ids, targets) batches on CPU."""

    def __init__(
        self,
        domains: list[DomainSpec],
        split: str,
        seq_len: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1234,
    ) -> None:
        if not domains:
            raise ValueError("at least one domain required")
        self.domains = domains
        self.split = split
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank, self.world_size = rank, world_size
        self.seed = seed

        weights = np.array([d.weight for d in domains], dtype=np.float64)
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("mixture weights must be non-negative and sum > 0")
        self.weights = weights / weights.sum()

        self.streams = [
            _DomainStream(d, split, seq_len, rank, world_size, seed + 7919 * i) for i, d in enumerate(domains)
        ]
        self.mix_rng = np.random.default_rng(seed + 31 * rank)
        self.mix_draws = 0  # for exact resume of the mixture RNG

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        picks = self.mix_rng.choice(len(self.streams), size=self.batch_size, p=self.weights)
        self.mix_draws += 1
        windows = np.stack([self.streams[i].next_window() for i in picks])
        x = torch.from_numpy(windows[:, :-1])
        y = torch.from_numpy(windows[:, 1:])
        return x, y

    # ---- state ---------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "mix_draws": self.mix_draws,
            "streams": {s.spec.name: s.state_dict() for s in self.streams},
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        # Re-create the mixture RNG and fast-forward it.
        self.mix_rng = np.random.default_rng(self.seed + 31 * self.rank)
        for _ in range(int(d["mix_draws"])):
            self.mix_rng.choice(len(self.streams), size=self.batch_size, p=self.weights)
        self.mix_draws = int(d["mix_draws"])
        for s in self.streams:
            if s.spec.name in d["streams"]:
                s.load_state_dict(d["streams"][s.spec.name])

    def summary(self) -> dict[str, Any]:
        return {
            s.spec.name: {
                "tokens": s.total_tokens,
                "windows": s.n_windows,
                "weight": float(w),
                "epochs_seen": round(s.tokens_consumed_fraction, 3),
            }
            for s, w in zip(self.streams, self.weights)
        }


def build_domains(tokenized_dir: str | Path, mixture: dict[str, float]) -> list[DomainSpec]:
    tokenized_dir = Path(tokenized_dir)
    specs = []
    for name, w in mixture.items():
        if w <= 0:
            continue
        p = tokenized_dir / name
        if not p.exists():
            raise FileNotFoundError(f"tokenized domain dir not found: {p}")
        specs.append(DomainSpec(name=name, path=str(p), weight=float(w)))
    return specs


def domain_token_counts(tokenized_dir: str | Path, mixture: dict[str, float], split: str) -> dict[str, int]:
    return {name: count_tokens(Path(tokenized_dir) / name, split) for name in mixture}
