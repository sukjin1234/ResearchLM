"""Tokenize JSONL documents into uint16 binary shards (train/val split).

    python scripts/tokenize_data.py \
        --tokenizer /storage/researchlm/tokenizer/researchlm_32k.model \
        --input english=data_samples/english.jsonl \
        --input korean=/storage/researchlm/cleaned/korean_final \
        --input "math=/storage/researchlm/cleaned/math_final/*.jsonl.zst" \
        --out-dir /storage/researchlm/tokenized/samples_v0 \
        --val-fraction 0.02

Each document becomes  [tokens...] + EOS  (no BOS; EOS doubles as the
document separator inside the concatenated stream).
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import _bootstrap  # noqa: F401
from researchlm.data.io import expand_inputs, iter_documents
from researchlm.data.shards import ShardWriter
from researchlm.data.tokenizer import Tokenizer

_TOK: Tokenizer | None = None


def _init(tokenizer_path: str) -> None:
    global _TOK
    _TOK = Tokenizer(tokenizer_path)


def _encode(text: str) -> list[int]:
    assert _TOK is not None
    return _TOK.encode(text, add_bos=False, add_eos=True)


def read_texts(spec: str):
    """spec = file / directory / glob of .parquet, .jsonl, .jsonl.zst (streamed)."""
    files = expand_inputs([spec])
    if not files:
        raise SystemExit(f"no input files for {spec}")
    for doc in iter_documents(files, columns=["text"]):
        t = doc.get("text")
        if t:
            yield t


def process_domain(domain: str, path: str, out_dir: Path, tokenizer_path: str,
                   val_fraction: float, shard_size: int, workers: int, chunksize: int) -> dict:
    t0 = time.time()
    # deterministic split without knowing the total: every k-th document is val
    k = int(round(1.0 / val_fraction)) if val_fraction > 0 else 0

    writers = {
        "train": ShardWriter(out_dir / domain, "train", shard_size),
        "val": ShardWriter(out_dir / domain, "val", shard_size),
    }
    tokens = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    with Pool(workers, initializer=_init, initargs=(tokenizer_path,)) as pool:
        for i, ids in enumerate(pool.imap(_encode, read_texts(str(path)), chunksize=chunksize)):
            split = "val" if (k and i % k == 0) else "train"
            writers[split].add_document(ids)
            tokens[split] += len(ids)
            docs[split] += 1
    shards = {s: w.close() for s, w in writers.items()}
    dt = time.time() - t0
    print(f"[{domain:8s}] train {tokens['train']:>12,} tok / {docs['train']:,} docs | "
          f"val {tokens['val']:>10,} tok / {docs['val']:,} docs | {dt:.1f}s")
    return {"domain": domain, "source": str(path), "tokens": tokens, "documents": docs,
            "shards": {s: len(v) for s, v in shards.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input", action="append", required=True, help="domain=path.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--shard-size", type=int, default=100_000_000, help="tokens per shard")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunksize", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"tokenizer": args.tokenizer, "val_fraction": args.val_fraction, "domains": []}
    for item in args.input:
        domain, path = item.split("=", 1)
        manifest["domains"].append(process_domain(domain, path, out_dir, args.tokenizer,
                                                  args.val_fraction, args.shard_size, args.workers, args.chunksize))
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    total = sum(d["tokens"]["train"] for d in manifest["domains"])
    print(f"\n[done] {total:,} train tokens -> {out_dir}")


if __name__ == "__main__":
    main()
