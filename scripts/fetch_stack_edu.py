"""Fetch Stack-Edu source files from Software Heritage (S3, anonymous).

Stack-Edu on Hugging Face contains only metadata (blob_id, language, score,
licence...).  The file contents live in the public bucket
s3://softwareheritage/content/<blob_id> (gzip).  This script:

  1. reads the metadata parquet(s) of one language
  2. keeps rows with int_score >= min_int_score, permissive licence, size bounds
  3. samples rows (seeded) until the language's token budget is met
  4. downloads blobs with a thread pool, writes .jsonl.zst shards with text +
     metadata to <raw_dir>/code/<Language>/
  5. is resumable: blob_ids already written are skipped

    python scripts/fetch_stack_edu.py --config configs/data/download_v1.yaml --language Rust --max-docs 2000
    python scripts/fetch_stack_edu.py --config configs/data/download_v1.yaml --language all
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import yaml

import _bootstrap  # noqa: F401
from researchlm.data.io import JsonlShardWriter, expand_inputs, iter_jsonl

META_COLS = ["blob_id", "language", "repo_name", "path", "src_encoding", "length_bytes", "score", "int_score", "detected_licenses", "license_type"]


def make_s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED, region_name="us-east-1",
                                            retries={"max_attempts": 5, "mode": "adaptive"}, max_pool_connections=128))


def fetch_blob(s3, row: dict, retries: int = 4) -> tuple[dict, str | None, str | None]:
    for attempt in range(retries):
        try:
            obj = s3.get_object(Bucket="softwareheritage", Key=f"content/{row['blob_id']}")
            raw = gzip.decompress(obj["Body"].read())
            text = raw.decode(row.get("src_encoding") or "utf-8", errors="replace")
            return row, text, None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            if "NoSuchKey" in err:
                return row, None, err
            time.sleep(0.5 * (2 ** attempt))
    return row, None, err


def select_rows(meta_files: list[Path], budget_tokens: float, bytes_per_token: float, min_int_score: int,
                license_types: list[str], min_bytes: int, max_bytes: int, seed: int, already: set[str], max_docs: int | None) -> list[dict]:
    rng = random.Random(seed)
    candidates = []
    for f in meta_files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=65536, columns=[c for c in META_COLS if c in pf.schema_arrow.names]):
            for r in batch.to_pylist():
                if r.get("int_score", 0) < min_int_score:
                    continue
                if license_types and r.get("license_type") not in license_types:
                    continue
                lb = r.get("length_bytes") or 0
                if lb < min_bytes or lb > max_bytes:
                    continue
                if r["blob_id"] in already:
                    continue
                candidates.append(r)
    rng.shuffle(candidates)
    out, tokens = [], 0
    have_tokens = 0  # tokens from already-fetched docs are accounted by caller
    for r in candidates:
        if tokens >= budget_tokens or (max_docs is not None and len(out) >= max_docs):
            break
        out.append(r)
        tokens += r["length_bytes"] / bytes_per_token
    return out


def existing_blob_ids(out_dir: Path) -> tuple[set[str], int]:
    ids, nbytes = set(), 0
    for f in expand_inputs([out_dir]):
        for d in iter_jsonl(f):
            if "blob_id" in d:
                ids.add(d["blob_id"]); nbytes += d.get("length_bytes", 0)
    return ids, nbytes


def run_language(lang: str, cfg: dict, args) -> None:
    se = cfg["stack_edu"]
    raw_dir = Path(cfg["raw_dir"])
    meta_files = expand_inputs([raw_dir / "code_meta" / lang])
    if not meta_files:
        raise SystemExit(f"no metadata for {lang}: run download_data.py --domain code_meta first")
    out_dir = raw_dir / "code" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    already, have_bytes = existing_blob_ids(out_dir)
    bpt = float(se["bytes_per_token"])
    budget = float(se["token_budget_total"]) * float(se["language_weights"][lang]) - have_bytes / bpt
    if args.max_docs is None and budget <= 0:
        print(f"[{lang}] budget already met ({have_bytes/bpt/1e9:.2f}B tokens)"); return
    rows = select_rows(meta_files, max(budget, 0) if args.max_docs is None else float("inf"), bpt, se["min_int_score"],
                       se.get("license_types", []), se["min_bytes"], se["max_bytes"], args.seed, already, args.max_docs)
    est = sum(r["length_bytes"] for r in rows) / bpt
    print(f"[{lang}] {len(meta_files)} meta files | already {len(already):,} docs | selecting {len(rows):,} docs (~{est/1e9:.3f}B tokens)")

    s3 = make_s3()
    start_idx = len(expand_inputs([out_dir]))
    writer = JsonlShardWriter(out_dir, "code", max_docs=args.shard_docs, start_index=start_idx)
    stats = {"ok": 0, "missing": 0, "error": 0, "bytes": 0}
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch_blob, s3, r) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            row, text, err = fut.result()
            if text is None:
                stats["missing" if err and "NoSuchKey" in err else "error"] += 1
            else:
                doc = {k: row.get(k) for k in META_COLS}
                doc["text"] = text
                doc["source"] = "stack-edu"
                writer.write(doc)
                stats["ok"] += 1; stats["bytes"] += len(text.encode("utf-8", errors="replace"))
            if i % 2000 == 0 or i == len(rows):
                rate = i / max(1e-6, time.time() - t0)
                print(f"  [{lang}] {i:,}/{len(rows):,} | ok {stats['ok']:,} missing {stats['missing']} err {stats['error']} | {rate:.0f} files/s | ~{stats['bytes']/bpt/1e6:.0f}M tok")
    writer.close()
    with open(out_dir / "fetch_stats.log", "a") as f:
        f.write(json.dumps({"time": time.time(), **stats, "selected": len(rows)}) + "\n")
    print(f"[{lang}] done in {time.time()-t0:.0f}s: {stats}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/download_v1.yaml")
    ap.add_argument("--language", required=True, help="e.g. Python | all")
    ap.add_argument("--max-docs", type=int, default=None, help="testing: fetch only N docs")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--shard-docs", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    args.workers = args.workers or cfg["stack_edu"].get("workers", 32)
    langs = list(cfg["stack_edu"]["language_weights"]) if args.language == "all" else [args.language]
    for lang in langs:
        run_language(lang, cfg, args)


if __name__ == "__main__":
    main()
