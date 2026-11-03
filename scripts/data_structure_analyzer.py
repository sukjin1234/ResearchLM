"""Analyse document structure + metadata per domain and draw plots (goal.md Step 3).

    python scripts/data_structure_analyzer.py --domain math \
        --input /storage/researchlm/raw/math --sample 20000 \
        --out /storage/researchlm/metadata/analysis

Outputs (per domain):
    <out>/<domain>_features.parquet   one row per sampled doc (features + metadata)
    <out>/<domain>_summary.json       percentiles for every numeric column
    <out>/<domain>_*.png              histograms / scatter plots
    <out>/<domain>_examples.jsonl     a few docs at low / mid / high quality score for manual review
"""

from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from researchlm.data.io import expand_inputs, iter_chunk, make_chunks, write_json
from researchlm.data.quality import code_features, text_features

META_KEEP = ["id", "url", "score", "int_score", "language", "language_score", "token_count", "minhash_cluster_size",
             "wordlist_ratio", "top_langs", "license_type", "detected_licenses", "repo_name", "path", "length_bytes", "date", "crawl"]


def _features(args):
    text, meta, feature_set = args
    f = code_features(text) if feature_set == "code" else text_features(text)
    if isinstance(meta.get("detected_licenses"), list):
        meta["detected_licenses"] = ",".join(meta["detected_licenses"])
    if isinstance(meta.get("top_langs"), (dict, list)):
        meta["top_langs"] = json.dumps(meta["top_langs"])[:200]
    f.update({f"meta_{k}": v for k, v in meta.items()})
    return f


def reservoir_sample(files, k: int, seed: int, scan_factor: int = 5):
    """Uniform sample from randomly chosen chunks.

    Scanning all of a 14M-document domain would take most of an hour.  We visit
    row groups in random order and stop after scan_factor * k documents, which
    still spreads the sample across files and crawl dates.
    """
    rng = random.Random(seed)
    chunks = make_chunks(files, target_rows=1000)
    rng.shuffle(chunks)
    sample, n = [], 0
    for chunk in chunks:
        for d in iter_chunk(chunk):
            n += 1
            if len(sample) < k:
                sample.append(d)
            else:
                j = rng.randrange(n)
                if j < k:
                    sample[j] = d
        if n >= scan_factor * k:
            break
    return sample, n


def plot(df: pd.DataFrame, domain: str, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []

    def hist(col, log=False, bins=60, title=None):
        if col not in df or df[col].dropna().empty:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = df[col].dropna().astype(float)
        if log:
            vals = vals[vals > 0]
            ax.hist(np.log10(vals), bins=bins, color="#4C72B0")
            ax.set_xlabel(f"log10({col})")
        else:
            ax.hist(vals, bins=bins, color="#4C72B0")
            ax.set_xlabel(col)
        ax.set_ylabel("documents"); ax.set_title(title or f"{domain}: {col}")
        p = out / f"{domain}_{col}.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)

    hist("n_chars", log=True)
    hist("meta_token_count", log=True)
    for c in ["meta_score", "meta_language_score", "hangul_ratio", "latin_ratio", "alnum_ratio", "repeated_char_ratio",
              "dup_line_frac", "dup_10gram_frac", "url_char_ratio", "en_stopword_frac", "ko_function_per_100chars",
              "mean_line_len", "alpha_ratio", "meta_minhash_cluster_size", "meta_wordlist_ratio"]:
        hist(c, log=c in ("meta_minhash_cluster_size",))

    # quality score vs heuristics
    if "meta_score" in df:
        for c in ["n_chars", "dup_line_frac", "repeated_char_ratio", "alnum_ratio", "url_char_ratio", "en_stopword_frac"]:
            if c in df:
                fig, ax = plt.subplots(figsize=(6, 4))
                x = df["meta_score"].astype(float); y = df[c].astype(float)
                if c == "n_chars":
                    y = np.log10(y.clip(lower=1)); ylabel = "log10(n_chars)"
                else:
                    ylabel = c
                ax.scatter(x, y, s=3, alpha=0.3, color="#DD8452")
                ax.set_xlabel("metadata quality score"); ax.set_ylabel(ylabel); ax.set_title(f"{domain}: score vs {c}")
                p = out / f"{domain}_score_vs_{c}.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)

    for cat in ["meta_license_type", "meta_int_score", "meta_language"]:
        if cat in df and df[cat].notna().any():
            counts = df[cat].astype(str).value_counts().head(20)
            fig, ax = plt.subplots(figsize=(7, 4))
            counts.plot.bar(ax=ax, color="#55A868"); ax.set_title(f"{domain}: {cat}"); ax.set_ylabel("documents")
            p = out / f"{domain}_{cat}.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--input", action="append", required=True, help="file / dir / glob (parquet or jsonl[.zst])")
    ap.add_argument("--out", default="/storage/researchlm/metadata/analysis")
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--feature-set", default=None, help="text | code (default: code for domain 'code')")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    feature_set = args.feature_set or ("code" if args.domain == "code" else "text")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = expand_inputs(args.input)
    if not files:
        raise SystemExit(f"no input files for {args.input}")
    print(f"[{args.domain}] {len(files)} files; sampling {args.sample:,} docs")

    docs, n_total = reservoir_sample(files, args.sample, args.seed)
    print(f"[{args.domain}] scanned {n_total:,} docs, kept {len(docs):,}")
    jobs = [(d.get("text") or "", {k: d.get(k) for k in META_KEEP if k in d}, feature_set) for d in docs]
    with Pool(args.workers) as pool:
        feats = pool.map(_features, jobs, chunksize=64)
    df = pd.DataFrame(feats)
    df.to_parquet(out / f"{args.domain}_features.parquet", index=False)

    num = df.select_dtypes(include=[np.number])
    pct = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    summary = {"domain": args.domain, "documents_scanned": n_total, "documents_sampled": len(df), "feature_set": feature_set,
               "percentiles": {c: {f"p{int(p*100):02d}": float(num[c].quantile(p)) for p in pct} | {"mean": float(num[c].mean())}
                               for c in num.columns if num[c].notna().any()}}
    for cat in ["meta_license_type", "meta_int_score", "meta_language"]:
        if cat in df:
            summary[cat] = df[cat].astype(str).value_counts().head(30).to_dict()
    if "meta_score" in df:
        summary["corr_with_meta_score"] = {c: float(num[c].corr(num["meta_score"])) for c in num.columns if c != "meta_score" and num[c].notna().sum() > 10}
    write_json(out / f"{args.domain}_summary.json", summary)

    # examples for manual review (Step 4): low / mid / high by metadata score or by length
    key = "score" if any("score" in d for d in docs) else None
    ranked = sorted(docs, key=lambda d: (d.get(key) if key else len(d.get("text", ""))) or 0)
    picks = {"low": ranked[:5], "mid": ranked[len(ranked)//2 - 2: len(ranked)//2 + 3], "high": ranked[-5:]}
    with open(out / f"{args.domain}_examples.jsonl", "w", encoding="utf-8") as f:
        for bucket, ds in picks.items():
            for d in ds:
                f.write(json.dumps({"bucket": bucket, **{k: d.get(k) for k in META_KEEP if k in d}, "text": (d.get("text") or "")[:3000]}, ensure_ascii=False, default=str) + "\n")

    made = [] if args.no_plots else plot(df, args.domain, out)
    print(f"[{args.domain}] wrote features/summary/examples + {len(made)} plots -> {out}")
    for c in ["n_chars", "meta_score", "hangul_ratio", "repeated_char_ratio", "dup_line_frac", "mean_line_len"]:
        if c in summary["percentiles"]:
            p = summary["percentiles"][c]
            print(f"   {c:22s} p10 {p['p10']:>10.3f}  p50 {p['p50']:>10.3f}  p90 {p['p90']:>10.3f}  p99 {p['p99']:>10.3f}")


if __name__ == "__main__":
    main()
