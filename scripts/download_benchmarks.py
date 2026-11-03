"""Download evaluation benchmarks and store their text for decontamination.

    python scripts/download_benchmarks.py --out /storage/researchlm/metadata/benchmarks

Each benchmark becomes <out>/<name>.jsonl with {"benchmark", "split", "text"} where
text = question + choices + answer concatenated (anything a training doc could leak).
Then build the 13-gram index:
    python scripts/decontaminate_data.py --build-index --benchmarks /storage/researchlm/metadata/benchmarks \
        --index /storage/researchlm/metadata/benchmarks/ngram13.npy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401


def rows_to_text(name: str, r: dict) -> str:
    if name == "mmlu" or name == "kmmlu":
        choices = r.get("choices") or [r.get(k) for k in ("A", "B", "C", "D") if r.get(k) is not None]
        return " ".join([str(r.get("question", ""))] + [str(c) for c in choices])
    if name == "gsm8k":
        return f"{r['question']} {r['answer']}"
    if name == "math":
        return f"{r['problem']} {r['solution']}"
    if name == "humaneval":
        return f"{r['prompt']} {r['canonical_solution']} {r['test']}"
    if name == "mbpp":
        return f"{r['text']} {r['code']} {' '.join(r.get('test_list') or [])}"
    if name == "arc":
        return f"{r['question']} {' '.join(r['choices']['text'])}"
    if name == "hellaswag":
        return f"{r['ctx']} {' '.join(r['endings'])}"
    return json.dumps(r, ensure_ascii=False)


SPECS = {
    # name: (repo, configs or None for all, split)
    "mmlu": ("cais/mmlu", ["all"], "test"),
    "kmmlu": ("HAERAE-HUB/KMMLU", None, "test"),
    "gsm8k": ("openai/gsm8k", ["main"], "test"),
    "math": ("EleutherAI/hendrycks_math", None, "test"),
    "humaneval": ("openai/openai_humaneval", None, "test"),
    "mbpp": ("google-research-datasets/mbpp", ["full"], "test"),
    "arc": ("allenai/ai2_arc", ["ARC-Challenge", "ARC-Easy"], "test"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/storage/researchlm/metadata/benchmarks")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/storage/researchlm/hf_cache")
    from datasets import get_dataset_config_names, load_dataset

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name, (repo, configs, split) in SPECS.items():
        if args.only and name not in args.only:
            continue
        if configs is None:
            try:
                configs = get_dataset_config_names(repo)
            except Exception:  # noqa: BLE001
                configs = [None]
        n = 0
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for cfg in configs:
                try:
                    ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
                except Exception as e:  # noqa: BLE001
                    print(f"  [{name}/{cfg}] skipped: {str(e)[:120]}"); continue
                for r in ds:
                    f.write(json.dumps({"benchmark": name, "config": cfg, "split": split, "text": rows_to_text(name, r)}, ensure_ascii=False) + "\n")
                    n += 1
        print(f"[{name}] {n:,} items -> {out / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
