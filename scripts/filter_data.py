"""Apply Data Filter v1 to raw documents (goal.md Step 5).

    python scripts/filter_data.py --domain math \
        --input /storage/researchlm/raw/math \
        --out /storage/researchlm/cleaned/math \
        --filters configs/data/filters_v1.yaml --sample-rejected 20

Output: <out>/filtered_XXXXX.jsonl.zst  (text + source + a slim metadata set)
        <out>/filter_stats.json          (kept / rejected counts per rule)
        <out>/rejected_samples.txt       (jsonl content; .txt so later stages never read it as data; N examples per rejection reason, for manual review)
One worker per input file; each worker writes its own shards, so no locking.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import _bootstrap  # noqa: F401
from researchlm.data.filters import filter_config_version, load_filters
from researchlm.data.io import JsonlShardWriter, expand_inputs, iter_chunk, make_chunks, write_json

SLIM_META = ["id", "url", "score", "int_score", "language_score", "token_count", "minhash_cluster_size", "wordlist_ratio",
             "blob_id", "repo_name", "path", "language", "license_type", "length_bytes", "date", "crawl"]


def _work(job):
    file_idx, chunk, domain, filters_path, out_dir, sample_rejected, seed = job
    path = chunk[0]
    flt = load_filters(filters_path)[domain]
    rng = random.Random(seed + file_idx)
    writer = JsonlShardWriter(out_dir, f"filtered_f{file_idx:04d}", max_docs=100_000)
    stats = Counter(); kept_chars = rej_chars = 0
    rejected = defaultdict(list)
    for i, doc in enumerate(iter_chunk(chunk)):
        text = doc.get("text") or ""
        meta = {k: doc[k] for k in SLIM_META if k in doc}
        keep, reason, _ = flt.apply(text, meta)
        if keep:
            stats["kept"] += 1; kept_chars += len(text)
            out = {"text": text, "source": f"{domain}:{Path(path).name}", **meta}
            writer.write(out)
        else:
            stats[f"rej:{reason}"] += 1; rej_chars += len(text)
            bucket = rejected[reason]
            if len(bucket) < sample_rejected:
                bucket.append({"reason": reason, "text": text[:1500], **meta})
            elif rng.random() < 0.01:
                bucket[rng.randrange(sample_rejected)] = {"reason": reason, "text": text[:1500], **meta}
        stats["total"] += 1
    files = writer.close()
    return {"file": str(path), "stats": dict(stats), "kept_chars": kept_chars, "rej_chars": rej_chars,
            "outputs": [str(f) for f in files], "rejected": {k: v for k, v in rejected.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["english", "korean", "math", "code"])
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--filters", default="configs/data/filters_v1.yaml")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--sample-rejected", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-rows", type=int, default=25_000)
    args = ap.parse_args()

    files = expand_inputs(args.input)
    if not files:
        raise SystemExit("no input files")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{args.domain}] filtering {len(files)} files with {args.filters} ({filter_config_version(args.filters)}) -> {out_dir}")
    t0 = time.time()
    chunks = make_chunks(files, args.chunk_rows)
    print(f"[{args.domain}] {len(chunks)} work chunks")
    jobs = [(i, c, args.domain, args.filters, str(out_dir), args.sample_rejected, args.seed) for i, c in enumerate(chunks)]
    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(_work, jobs, chunksize=1)

    total = Counter(); kept_chars = rej_chars = 0
    rejected = defaultdict(list)
    for r in results:
        total.update(r["stats"]); kept_chars += r["kept_chars"]; rej_chars += r["rej_chars"]
        for k, v in r["rejected"].items():
            rejected[k].extend(v)
    n = total["total"]; k = total["kept"]
    report = {
        "domain": args.domain, "filter_version": filter_config_version(args.filters), "input_files": len(files),
        "documents": n, "kept": k, "kept_frac": k / max(1, n),
        "kept_chars": kept_chars, "rejected_chars": rej_chars, "kept_char_frac": kept_chars / max(1, kept_chars + rej_chars),
        "rejections": {kk[4:]: v for kk, v in sorted(total.items(), key=lambda x: -x[1]) if kk.startswith("rej:")},
        "seconds": time.time() - t0,
    }
    write_json(out_dir / "filter_stats.json", report)
    with open(out_dir / "rejected_samples.txt", "w", encoding="utf-8") as f:
        for reason, items in rejected.items():
            for it in items[: args.sample_rejected]:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[{args.domain}] {n:,} docs -> kept {k:,} ({100*k/max(1,n):.1f}% docs, {100*report['kept_char_frac']:.1f}% chars) in {report['seconds']:.0f}s")
    for reason, cnt in list(report["rejections"].items())[:15]:
        print(f"   {reason:28s} {cnt:>10,} ({100*cnt/max(1,n):.2f}%)")


if __name__ == "__main__":
    main()
