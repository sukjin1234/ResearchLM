"""Remove training documents that overlap benchmark test sets (goal.md Step 7).

Build the index once:
    python scripts/decontaminate_data.py --build-index \
        --benchmarks /storage/researchlm/metadata/benchmarks --index /storage/researchlm/metadata/benchmarks/ngram13.npy
Then per domain:
    python scripts/decontaminate_data.py --domain math --index /storage/researchlm/metadata/benchmarks/ngram13.npy \
        --input /storage/researchlm/cleaned/math_dedup --out /storage/researchlm/cleaned/math_final
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import _bootstrap  # noqa: F401
from researchlm.data.decontaminate import ContaminationIndex
from researchlm.data.io import JsonlShardWriter, expand_inputs, iter_chunk, iter_jsonl, make_chunks, write_json

_INDEX: ContaminationIndex | None = None


def _init(index_path: str, ngram: int, min_hits: int):
    global _INDEX
    _INDEX = ContaminationIndex.load(index_path, n=ngram, min_hits=min_hits)


def _work(job):
    file_idx, chunk, out_dir, keep_samples = job
    writer = JsonlShardWriter(out_dir, f"final_f{file_idx:04d}", max_docs=100_000)
    n = removed = 0
    samples = []
    for doc in iter_chunk(chunk):
        n += 1
        hits = _INDEX.hits(doc.get("text") or "")
        if hits >= _INDEX.min_hits:
            removed += 1
            if len(samples) < keep_samples:
                samples.append({"hits": hits, "source": doc.get("source"), "url": doc.get("url"), "text": (doc.get("text") or "")[:800]})
            continue
        writer.write(doc)
    writer.close()
    return n, removed, samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--benchmarks", default="/storage/researchlm/metadata/benchmarks")
    ap.add_argument("--index", required=True)
    ap.add_argument("--ngram", type=int, default=13)
    ap.add_argument("--min-hits", type=int, default=1)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--input", action="append", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    if args.build_index:
        idx = ContaminationIndex(n=args.ngram)
        per_bench = {}
        for f in sorted(Path(args.benchmarks).glob("*.jsonl")):
            before = len(idx)
            for r in iter_jsonl(f):
                idx.add_text(r["text"])
            per_bench[f.stem] = len(idx) - before
        idx.save(args.index)
        write_json(Path(args.index).with_suffix(".json"), {"ngram": args.ngram, "total_grams": len(idx), "per_benchmark_new_grams": per_bench})
        print(f"index: {len(idx):,} {args.ngram}-grams from {len(per_bench)} benchmarks -> {args.index}")
        for k, v in per_bench.items():
            print(f"   {k:12s} {v:>10,}")
        if not args.domain:
            return

    files = expand_inputs(args.input)
    if not files or not args.out:
        raise SystemExit("--input and --out required")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    chunks = make_chunks(files)
    jobs = [(i, c, str(out_dir), 5) for i, c in enumerate(chunks)]
    with Pool(min(args.workers, len(chunks)), initializer=_init, initargs=(args.index, args.ngram, args.min_hits)) as pool:
        res = pool.map(_work, jobs, chunksize=1)
    n = sum(r[0] for r in res); removed = sum(r[1] for r in res)
    samples = [s for r in res for s in r[2]][:50]
    report = {"domain": args.domain, "documents": n, "removed": removed, "removed_frac": removed / max(1, n),
              "ngram": args.ngram, "min_hits": args.min_hits, "seconds": time.time() - t0}
    write_json(out_dir / "decontam_stats.json", report)
    with open(out_dir / "contaminated_samples.txt", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[{args.domain}] removed {removed:,}/{n:,} ({100*removed/max(1,n):.3f}%) in {report['seconds']:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
