"""Binary token shards.

Each shard is a flat little-endian uint16 array of token ids where documents
are concatenated and separated by EOS.  A sidecar ``<name>.json`` records the
number of tokens and documents.  uint16 is enough because vocab_size <= 65535.

Layout on disk:

    <tokenized_dir>/<domain>/train_00000.bin
    <tokenized_dir>/<domain>/train_00000.json
    <tokenized_dir>/<domain>/val_00000.bin
    ...
    <tokenized_dir>/manifest.json     (written by scripts/tokenize_data.py)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

TOKEN_DTYPE = np.uint16


class ShardWriter:
    def __init__(self, out_dir: str | Path, split: str, shard_size_tokens: int = 100_000_000) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = shard_size_tokens
        self.buf = np.empty(shard_size_tokens, dtype=TOKEN_DTYPE)
        self.pos = 0
        self.shard_idx = 0
        self.docs_in_shard = 0
        self.written: list[dict] = []

    def add_document(self, ids: list[int]) -> None:
        arr = np.asarray(ids, dtype=TOKEN_DTYPE)
        n = len(arr)
        if n == 0:
            return
        start = 0
        self.docs_in_shard += 1
        while start < n:
            space = self.shard_size - self.pos
            take = min(space, n - start)
            self.buf[self.pos : self.pos + take] = arr[start : start + take]
            self.pos += take
            start += take
            if self.pos == self.shard_size:
                self._flush()

    def _flush(self) -> None:
        if self.pos == 0:
            return
        name = f"{self.split}_{self.shard_idx:05d}"
        bin_path = self.out_dir / f"{name}.bin"
        self.buf[: self.pos].tofile(bin_path)
        meta = {"tokens": int(self.pos), "documents": int(self.docs_in_shard), "dtype": "uint16"}
        with open(self.out_dir / f"{name}.json", "w") as f:
            json.dump(meta, f)
        self.written.append({"file": str(bin_path), **meta})
        self.shard_idx += 1
        self.pos = 0
        self.docs_in_shard = 0

    def close(self) -> list[dict]:
        self._flush()
        return self.written


def iter_shard_files(domain_dir: str | Path, split: str) -> Iterator[Path]:
    domain_dir = Path(domain_dir)
    yield from sorted(domain_dir.glob(f"{split}_*.bin"))


def load_shard(path: str | Path) -> np.ndarray:
    """Memory-map a shard (no RAM copy)."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def count_tokens(domain_dir: str | Path, split: str) -> int:
    total = 0
    for p in iter_shard_files(domain_dir, split):
        with open(p.with_suffix(".json")) as f:
            total += json.load(f)["tokens"]
    return total
