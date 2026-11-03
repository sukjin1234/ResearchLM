"""Train the 32K SentencePiece BPE tokenizer.

Step 1: build a tokenizer corpus by sampling documents per domain according
        to the mixture (English 50 / Korean 25 / Code 15 / Math 10).
Step 2: train SentencePiece on it.

Example (current 30K-doc samples):
    python scripts/train_tokenizer.py \
        --input english=data_samples/english.jsonl \
        --input korean=data_samples/korean.jsonl \
        --input math=data_samples/math.jsonl \
        --weights english=0.5,korean=0.25,math=0.10 \
        --out-prefix /storage/researchlm/tokenizer/researchlm_32k
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401
from researchlm.data.io import expand_inputs, iter_documents
from researchlm.data.tokenizer import train_tokenizer


def parse_kv(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


args_keep_prob: dict[str, float] = {}


def build_corpus(inputs: dict[str, Path], weights: dict[str, float], corpus_path: Path,
                 target_chars: int, seed: int) -> dict[str, int]:
    """Write one paragraph per line; SentencePiece treats a line as a sentence."""
    rng = random.Random(seed)
    total_w = sum(weights[d] for d in inputs)
    stats = {}
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as out:
        for domain, path in inputs.items():
            budget = int(target_chars * weights[domain] / total_w)
            files = expand_inputs([str(path)])
            if not files:
                raise SystemExit(f"no input files for {path}")
            rng.shuffle(files)
            # stream documents; take each with probability `keep_prob` so the
            # corpus is a spread-out sample rather than the first N docs
            keep_prob = float(args_keep_prob.get(domain, 1.0))
            used = 0
            n_avail = 0
            for doc in iter_documents(files, columns=["text"]):
                text = doc.get("text") or ""
                n_avail += 1
                if used >= budget:
                    break
                if keep_prob < 1.0 and rng.random() > keep_prob:
                    continue
                for para in text.split("\n"):
                    para = para.strip("\r")
                    if para.strip():
                        out.write(para + "\n")
                used += len(text)
            stats[domain] = used
            print(f"[corpus] {domain:8s} budget={budget:>12,} chars  used={used:>12,} chars  docs_scanned={n_avail:,}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="domain=path|dir|glob (jsonl, jsonl.zst, parquet)")
    ap.add_argument("--weights", required=True, help="english=0.5,korean=0.25,...")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--vocab-size", type=int, default=32_768)
    ap.add_argument("--target-chars", type=int, default=400_000_000,
                    help="total characters of tokenizer corpus (capped by available data)")
    ap.add_argument("--input-sentence-size", type=int, default=4_000_000)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-prob", default="", help="domain=p,... sampling probability per doc (spread the sample over big corpora)")
    args = ap.parse_args()
    if args.keep_prob:
        args_keep_prob.update({k: float(v) for k, v in parse_kv(args.keep_prob.split(",")).items()})

    inputs = {k: Path(v) for k, v in parse_kv(args.input).items()}
    weights = {k: float(v) for k, v in parse_kv(args.weights.split(",")).items()}
    for d in inputs:
        if d not in weights:
            raise SystemExit(f"missing weight for domain {d}")

    out_prefix = Path(args.out_prefix)
    corpus_path = out_prefix.parent / f"{out_prefix.name}_corpus.txt"
    stats = build_corpus(inputs, weights, corpus_path, args.target_chars, args.seed)

    model_path = train_tokenizer(
        [corpus_path], out_prefix,
        vocab_size=args.vocab_size,
        input_sentence_size=args.input_sentence_size,
        num_threads=args.threads,
    )
    with open(out_prefix.with_suffix(".json"), "r+", encoding="utf-8") as f:
        meta = json.load(f)
        meta["corpus_chars_per_domain"] = stats
        meta["weights"] = weights
        f.seek(0); json.dump(meta, f, ensure_ascii=False, indent=2); f.truncate()
    print(f"\n[done] tokenizer -> {model_path}")


if __name__ == "__main__":
    main()
