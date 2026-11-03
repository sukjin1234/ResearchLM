"""Download raw parquet shards from Hugging Face until a token budget is met.

    python scripts/download_data.py --config configs/data/download_v1.yaml --domain english
    python scripts/download_data.py --config configs/data/download_v1.yaml --domain all
    python scripts/download_data.py ... --domain korean --max-files 1     # quick test

How the budget is met: files are taken in order; after each download the
tokens in that file are counted (token_count column, or chars / chars_per_token)
and we stop when the running total reaches token_budget.  Re-running skips
files that already exist (hf_hub_download is resumable) and only recounts.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401


def list_parquet_files(repo: str, prefix: str) -> list[tuple[str, int]]:
    from huggingface_hub import HfApi
    api = HfApi()
    files = [
        (f.path, f.size)
        for f in api.list_repo_tree(repo, path_in_repo=prefix.rstrip("/"), repo_type="dataset", recursive=True)
        if hasattr(f, "size") and f.path.endswith(".parquet")
    ]
    return sorted(files)


def count_tokens_in_file(path: Path, token_field: str | None, text_field: str, chars_per_token: float) -> tuple[int, int]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    rows = pf.metadata.num_rows
    if token_field and token_field in names:
        import pyarrow.compute as pc
        total = 0
        for batch in pf.iter_batches(batch_size=65536, columns=[token_field]):
            total += int(pc.sum(batch.column(0)).as_py() or 0)
        return total, rows
    if token_field and "metadata" in names:
        # FineWeb-Edu stores token_count inside the metadata struct.
        import pyarrow as pa
        import pyarrow.compute as pc
        mtype = pf.schema_arrow.field("metadata").type
        if pa.types.is_struct(mtype) and token_field in [f.name for f in mtype]:
            total = 0
            for batch in pf.iter_batches(batch_size=65536, columns=["metadata"]):
                total += int(pc.sum(batch.column(0).field(token_field)).as_py() or 0)
            return total, rows
    # estimate from characters
    import pyarrow.compute as pc
    chars = 0
    for batch in pf.iter_batches(batch_size=16384, columns=[text_field]):
        chars += int(pc.sum(pc.utf8_length(batch.column(0))).as_py() or 0)
    return int(chars / chars_per_token), rows


def download_one(repo: str, path_in_repo: str, local_dir: Path, retries: int = 8) -> Path:
    """Unauthenticated requests hit rate limits on long runs, so back off and retry."""
    from huggingface_hub import hf_hub_download
    for attempt in range(retries):
        try:
            return Path(hf_hub_download(repo, path_in_repo, repo_type="dataset", local_dir=str(local_dir)))
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            wait = min(600, 30 * 2 ** attempt)
            print(f"  retry {attempt + 1}/{retries - 1} for {path_in_repo} in {wait}s: {type(e).__name__}: {str(e)[:160]}", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def run_domain(name: str, spec: dict, raw_dir: Path, max_files: int | None, workers: int) -> dict:
    repo = spec["repo"]
    local_dir = raw_dir / name
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = local_dir / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {"repo": repo, "files": {}}

    if spec.get("all_files"):  # stack-edu metadata: whole language directories
        todo = []
        for lang in spec["languages"]:
            todo += list_parquet_files(repo, f"{lang}/")
        print(f"[{name}] {len(todo)} metadata files, {sum(s for _, s in todo)/2**30:.1f} GiB")
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(download_one, repo, p, local_dir): (p, s) for p, s in todo}
            for fut in as_completed(futs):
                p, s = futs[fut]
                local = fut.result()
                manifest["files"][p] = {"bytes": s, "local": str(local)}
                print(f"  ok {p} ({s/2**20:.0f} MiB)")
        json.dump(manifest, open(manifest_path, "w"), indent=2)
        return manifest

    files = list_parquet_files(repo, spec["prefix"])
    budget = float(spec["token_budget"])
    token_field = spec.get("token_field")
    text_field = spec.get("text_field", "text")
    cpt = float(spec.get("chars_per_token", 4.0))
    print(f"[{name}] {len(files)} files available ({sum(s for _, s in files)/2**30:.0f} GiB). budget {budget/1e9:.2f}B tokens")

    total_tokens = sum(f["tokens"] for f in manifest["files"].values())
    n_done = 0
    for path_in_repo, size in files:
        if total_tokens >= budget or (max_files is not None and n_done >= max_files):
            break
        entry = manifest["files"].get(path_in_repo)
        if entry and Path(entry["local"]).exists():
            n_done += 1
            continue
        t0 = time.time()
        local = download_one(repo, path_in_repo, local_dir)
        tokens, rows = count_tokens_in_file(local, token_field, text_field, cpt)
        total_tokens += tokens
        n_done += 1
        manifest["files"][path_in_repo] = {"bytes": size, "local": str(local), "tokens": tokens, "rows": rows}
        json.dump(manifest, open(manifest_path, "w"), indent=2)
        print(f"  {path_in_repo}: {size/2**20:.0f} MiB, {rows:,} docs, {tokens/1e9:.3f}B tok "
              f"({time.time()-t0:.0f}s) | total {total_tokens/1e9:.2f}B / {budget/1e9:.2f}B")
    manifest["total_tokens"] = total_tokens
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"[{name}] done: {len(manifest['files'])} files, {total_tokens/1e9:.2f}B tokens -> {local_dir}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/download_v1.yaml")
    ap.add_argument("--domain", required=True, help="english | korean | math | code_meta | all")
    ap.add_argument("--max-files", type=int, default=None, help="stop after N files (testing)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.environ.setdefault("HF_HOME", cfg.get("hf_home", "/storage/researchlm/hf_cache"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    raw_dir = Path(cfg["raw_dir"])
    domains = list(cfg["domains"]) if args.domain == "all" else [args.domain]
    for d in domains:
        run_domain(d, cfg["domains"][d], raw_dir, args.max_files, args.workers)


if __name__ == "__main__":
    main()
