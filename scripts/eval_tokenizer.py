"""Report compression (chars/token, bytes/token) per domain and show examples.

    python scripts/eval_tokenizer.py --tokenizer /storage/researchlm/tokenizer/researchlm_32k.model \
        --input english=data_samples/english.jsonl --input korean=data_samples/korean.jsonl \
        --input math=data_samples/math.jsonl --limit 2000
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from researchlm.data.tokenizer import Tokenizer
from researchlm.eval.tokenizer_metrics import compression_stats, iter_texts

EXAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "대한민국의 수도는 서울이며, 인구는 약 950만 명이다.",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Solve: 3x + 7 = 2024, so x = (2024 - 7) / 3 = 672.33",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input", action="append", default=[], help="domain=file|dir|glob")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = Tokenizer(args.tokenizer)
    print(f"vocab={tok.vocab_size} unk={tok.unk_id} bos={tok.bos_id} eos={tok.eos_id} pad={tok.pad_id}\n")

    for ex in EXAMPLES:
        pieces = tok.tokenize(ex)
        print(f"{len(pieces):3d} tokens | {ex!r}")
        print("    ", " ".join(pieces))
    print()

    results = {}
    header = f"{'domain':10s} {'docs':>6s} {'tokens':>10s} {'chars/tok':>9s} {'bytes/tok':>9s} {'tok/doc':>8s} {'unk':>6s} {'bytefb':>7s}"
    print(header); print("-" * len(header))
    for item in args.input:
        domain, path = item.split("=", 1)
        s = compression_stats(tok, iter_texts(path, args.limit))
        results[domain] = s
        print(f"{domain:10s} {s['documents']:>6d} {s['tokens']:>10,d} {s['chars_per_token']:>9.3f} "
              f"{s['bytes_per_token']:>9.3f} {s['tokens_per_doc']:>8.0f} {s['unk_rate']:>6.4f} {s['byte_fallback_rate']:>7.4f}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
