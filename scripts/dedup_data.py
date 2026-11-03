"""Exact + near deduplication within a domain (goal.md Step 6).

    python scripts/dedup_data.py --domain math \
        --input /storage/researchlm/cleaned/math \
        --out /storage/researchlm/cleaned/math_dedup

Stage 1 (parallel, per file) : sha1 exact hash + MinHash signature for every doc
Stage 2 (single process)     : exact-hash dedup, then LSH clustering of
                               signatures, union-find -> drop mask
Stage 3 (parallel, per file) : rewrite shards without dropped docs
Memory: 128 x 4 bytes per doc for signatures (25M docs = 12.8 GB).
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from researchlm.data.dedup import MinHasher, cluster_by_lsh, exact_hash
from researchlm.data.io import JsonlShardWriter, expand_inputs, iter_chunk, make_chunks, write_json

_HASHER: MinHasher | None = None


def _init(num_perm: int, ngram: int):
    global _HASHER, _NGRAM
    _HASHER = MinHasher(num_perm=num_perm)
    _NGRAM = ngram


def _signatures(job):
    file_idx, chunk = job
    sigs, exact, n = [], [], 0
    for doc in iter_chunk(chunk):
        text = doc.get("text") or ""
        exact.append(exact_hash(text))
        sigs.append(_HASHER.signature(text, _NGRAM))
        n += 1
    sig_arr = np.stack(sigs) if sigs else np.zeros((0, _HASHER.num_perm), dtype=np.uint32)
    return file_idx, sig_arr, exact, n


def _rewrite(job):
    file_idx, chunk, out_dir, drop = job
    writer = JsonlShardWriter(out_dir, f"dedup_f{file_idx:04d}", max_docs=100_000)
    kept = 0
    for i, doc in enumerate(iter_chunk(chunk)):
        if not drop[i]:
            writer.write(doc); kept += 1
    writer.close()
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--bands", type=int, default=16, help="16 bands x 8 rows ~ Jaccard 0.7 threshold")
    ap.add_argument("--ngram", type=int, default=5)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--exact-only", action="store_true")
    ap.add_argument("--chunk-rows", type=int, default=25_000)
    args = ap.parse_args()

    files = expand_inputs(args.input)
    chunks = make_chunks(files, args.chunk_rows)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{args.domain}] dedup {len(files)} files / {len(chunks)} chunks -> {out_dir}")
    t0 = time.time()

    # ---- stage 1
    with Pool(min(args.workers, len(chunks)), initializer=_init, initargs=(args.num_perm, args.ngram)) as pool:
        results = pool.map(_signatures, list(enumerate(chunks)), chunksize=1)
    results.sort(key=lambda r: r[0])
    counts = [r[3] for r in results]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    N = int(offsets[-1])
    sigs = np.concatenate([r[1] for r in results]) if N else np.zeros((0, args.num_perm), dtype=np.uint32)
    exact = [h for r in results for h in r[2]]
    print(f"   signatures for {N:,} docs in {time.time()-t0:.0f}s")

    # ---- stage 2: exact
    drop = np.zeros(N, dtype=bool)
    seen: dict[str, int] = {}
    for i, h in enumerate(exact):
        if h in seen:
            drop[i] = True
        else:
            seen[h] = i
    n_exact = int(drop.sum())
    print(f"   exact duplicates: {n_exact:,} ({100*n_exact/max(1,N):.2f}%)")

    # ---- stage 2: near
    n_near = 0
    if not args.exact_only and N > 1:
        t1 = time.time()
        survivors = np.flatnonzero(~drop)
        roots = cluster_by_lsh(sigs[survivors], num_bands=args.bands)
        near_drop = roots != np.arange(len(survivors))
        drop[survivors[near_drop]] = True
        n_near = int(near_drop.sum())
        n_clusters = int((~near_drop).sum())
        print(f"   near duplicates: {n_near:,} ({100*n_near/max(1,N):.2f}%), clusters {n_clusters:,}, {time.time()-t1:.0f}s")

    # ---- stage 3
    jobs = [(i, c, str(out_dir), drop[offsets[i]:offsets[i+1]]) for i, c in enumerate(chunks)]
    with Pool(min(args.workers, len(chunks))) as pool:
        kept = sum(pool.map(_rewrite, jobs, chunksize=1))
    report = {"domain": args.domain, "documents": N, "exact_duplicates": n_exact, "near_duplicates": n_near,
              "kept": kept, "kept_frac": kept / max(1, N), "num_perm": args.num_perm, "bands": args.bands,
              "ngram": args.ngram, "seconds": time.time() - t0}
    write_json(out_dir / "dedup_stats.json", report)
    print(f"[{args.domain}] kept {kept:,}/{N:,} ({100*kept/max(1,N):.1f}%) in {report['seconds']:.0f}s")


if __name__ == "__main__":
    main()
