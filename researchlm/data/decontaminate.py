"""Benchmark decontamination by n-gram overlap.

We build the set of all word 13-grams (hashed) appearing in benchmark
questions / answers.  A training document is flagged if it shares at least
``min_hits`` such 13-grams.  13 was used by GPT-3 and Llama; it is long enough
that natural coincidences are rare but short enough to catch paraphrase-free
copies of test questions.  Benchmark items shorter than 13 words contribute
their full text as a single gram.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import xxhash

from .dedup import normalize_text

NGRAM = 13
MIN_UNIQUE = 8  # a 13-gram must contain >= 8 distinct words; "1 0 0 0 1 0 0 ..." style runs are ignored


def ngram_hashes(text: str, n: int = NGRAM, min_unique: int = MIN_UNIQUE) -> list[int]:
    """Hashes of word n-grams.  Degenerate grams (few distinct words: number
    tables, separators, repeated tokens) are skipped on *both* the benchmark
    and the document side, otherwise integer-sequence pages match MATH answers."""
    words = normalize_text(text).split()
    if not words:
        return []
    if len(words) < n:
        return [xxhash.xxh64_intdigest(" ".join(words).encode("utf-8"))] if len(set(words)) >= min(min_unique, len(words)) and len(words) >= 6 else []
    out = []
    for i in range(len(words) - n + 1):
        gram = words[i : i + n]
        if len(set(gram)) < min_unique:
            continue
        out.append(xxhash.xxh64_intdigest(" ".join(gram).encode("utf-8")))
    return out


class ContaminationIndex:
    def __init__(self, hashes: Iterable[int] | None = None, n: int = NGRAM, min_hits: int = 1):
        self.n = n
        self.min_hits = min_hits
        self._set: set[int] = set(hashes) if hashes is not None else set()

    def add_text(self, text: str) -> None:
        self._set.update(ngram_hashes(text, self.n))

    def __len__(self) -> int:
        return len(self._set)

    def hits(self, text: str) -> int:
        if not self._set:
            return 0
        return sum(1 for h in ngram_hashes(text, self.n) if h in self._set)

    def is_contaminated(self, text: str) -> bool:
        return self.hits(text) >= self.min_hits

    def save(self, path: str | Path) -> None:
        arr = np.fromiter(self._set, dtype=np.uint64, count=len(self._set))
        np.save(path, arr)

    @classmethod
    def load(cls, path: str | Path, n: int = NGRAM, min_hits: int = 1) -> "ContaminationIndex":
        arr = np.load(path)
        return cls(hashes=(int(x) for x in arr), n=n, min_hits=min_hits)
