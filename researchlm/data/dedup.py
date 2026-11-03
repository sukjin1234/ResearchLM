"""Exact and near-duplicate detection.

Exact  : sha1 of whitespace/case-normalised text.
Near   : MinHash over word 5-gram shingles (128 permutations) + LSH banding
         (16 bands x 8 rows -> ~0.7 Jaccard threshold) + union-find clusters.

Why own implementation instead of `datasketch`: the maths is 40 lines, the
numpy version is faster for batch signatures, and the sort-based LSH below
needs ~16 bytes per document per band instead of a Python dict, so 25M
documents fit comfortably in RAM.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np
import xxhash

_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def exact_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def shingles(text: str, n: int = 5) -> np.ndarray:
    """64-bit hashes of word n-grams.  Short texts fall back to the whole text."""
    words = normalize_text(text).split()
    if len(words) < n:
        grams = [" ".join(words)] if words else [""]
    else:
        grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    h = np.fromiter((xxhash.xxh64_intdigest(g.encode("utf-8")) for g in grams), dtype=np.uint64, count=len(grams))
    return np.unique(h)


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.RandomState(seed)
        self.num_perm = num_perm
        # universal hashing h(x) = (a*x + b) mod p, then mod 2^32
        self.a = rng.randint(1, _MERSENNE, size=num_perm, dtype=np.uint64)
        self.b = rng.randint(0, _MERSENNE, size=num_perm, dtype=np.uint64)

    def signature(self, text: str, n: int = 5) -> np.ndarray:
        sh = shingles(text, n).astype(np.uint64)
        if len(sh) == 0:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint32)
        # use python ints via object? too slow. Work in uint64 with wraparound:
        # (a*x + b) mod p where p = 2^61-1 can be approximated; we use the
        # classic trick of computing modulo with 128-bit intermediate via
        # splitting x into 32-bit halves is overkill here — collisions from
        # 64-bit wraparound are negligible for dedup purposes.
        x = sh.reshape(-1, 1)                                   # [S, 1]
        hv = (x * self.a.reshape(1, -1) + self.b.reshape(1, -1))  # [S, P] uint64 wrap
        hv = (hv >> np.uint64(32)).astype(np.uint32)              # take high 32 bits
        return hv.min(axis=0)

    @staticmethod
    def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float((sig_a == sig_b).mean())


def lsh_band_hashes(signatures: np.ndarray, num_bands: int = 16) -> np.ndarray:
    """signatures [N, P] uint32 -> band hashes [num_bands, N] uint64."""
    N, P = signatures.shape
    rows = P // num_bands
    out = np.empty((num_bands, N), dtype=np.uint64)
    for b in range(num_bands):
        band = np.ascontiguousarray(signatures[:, b * rows : (b + 1) * rows])
        # hash each row of the band to 64 bits
        view = band.view(np.uint8).reshape(N, -1)
        out[b] = np.fromiter((xxhash.xxh64_intdigest(view[i].tobytes()) for i in range(N)), dtype=np.uint64, count=N)
    return out


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:  # path compression
            p[x], x = root, p[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def cluster_by_lsh(signatures: np.ndarray, num_bands: int = 16) -> np.ndarray:
    """Return cluster root id per document (root = smallest index in cluster)."""
    N = signatures.shape[0]
    uf = UnionFind(N)
    bands = lsh_band_hashes(signatures, num_bands)
    for b in range(num_bands):
        order = np.argsort(bands[b], kind="stable")
        sorted_vals = bands[b][order]
        # boundaries of equal runs
        change = np.flatnonzero(np.diff(sorted_vals)) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [N]])
        for s, e in zip(starts, ends):
            if e - s > 1:
                first = int(order[s])
                for j in order[s + 1 : e]:
                    uf.union(first, int(j))
    roots = np.fromiter((uf.find(i) for i in range(N)), dtype=np.int64, count=N)
    return roots


def duplicates_to_drop(roots: np.ndarray) -> np.ndarray:
    """Boolean mask: True for every document that is not its cluster's representative."""
    idx = np.arange(len(roots))
    return roots != idx
