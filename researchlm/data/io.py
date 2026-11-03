"""Document I/O shared by the whole data pipeline.

Supported inputs : .parquet, .jsonl, .jsonl.zst, .jsonl.gz, directories and
                   globs of those.
Standard output  : .jsonl.zst shards (one JSON object per line, zstd level 3,
                   ~3-4x smaller than raw text, streamable).

A "document" is a plain dict with at least a ``text`` field.
"""

from __future__ import annotations

import glob
import gzip
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow.parquet as pq
import zstandard as zstd

JSONL_SUFFIXES = (".jsonl", ".jsonl.zst", ".jsonl.gz")


def expand_inputs(specs: Sequence[str | Path]) -> list[Path]:
    """Expand files / directories / globs into a sorted list of data files."""
    out: list[Path] = []
    for spec in specs:
        spec = str(spec)
        if any(ch in spec for ch in "*?["):
            matches = sorted(glob.glob(spec, recursive=True))
        else:
            matches = [spec]
        for m in matches:
            p = Path(m)
            if p.is_dir():
                out.extend(sorted(q for q in p.rglob("*") if q.is_file() and _is_data_file(q)))
            elif p.is_file() and _is_data_file(p):
                out.append(p)
    return out


def _is_data_file(p: Path) -> bool:
    name = p.name
    return name.endswith(".parquet") or name.endswith(JSONL_SUFFIXES)


def open_text(path: str | Path, mode: str = "rt"):
    path = str(path)
    if path.endswith(".zst"):
        if "r" in mode:
            fh = open(path, "rb")
            return io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        fh = open(path, "wb")
        return io.TextIOWrapper(zstd.ZstdCompressor(level=3, threads=1).stream_writer(fh), encoding="utf-8")
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def open_text_for_write(path: Path, ext: str):
    """open_text picks the codec from the suffix, which a .tmp name hides."""
    if ext.endswith(".zst"):
        fh = open(path, "wb")
        return io.TextIOWrapper(zstd.ZstdCompressor(level=3, threads=1).stream_writer(fh), encoding="utf-8")
    return open(path, "wt", encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def flatten_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    """Promote fields of a nested ``metadata`` struct to the top level.

    FineWeb-Edu keeps language_score, score, int_score, token_count and url inside
    a struct, while FineWeb2 and FineMath have them as plain columns.  Without this
    the filters' ``meta.*`` rules silently do not apply to English.  FineMath's
    ``metadata`` is a JSON string and is left untouched.
    """
    nested = doc.get("metadata")
    if isinstance(nested, dict):
        for k, v in nested.items():
            doc.setdefault(k, v)
        del doc["metadata"]
    return doc


def iter_parquet(path: str | Path, columns: Sequence[str] | None = None, batch_size: int = 2048) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    cols = None
    if columns is not None:
        have = set(pf.schema_arrow.names)
        cols = [c for c in columns if c in have]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        for row in batch.to_pylist():
            yield flatten_metadata(row)


def iter_documents(paths: Iterable[str | Path], columns: Sequence[str] | None = None) -> Iterator[dict[str, Any]]:
    for p in paths:
        p = Path(p)
        if p.suffix == ".parquet":
            yield from iter_parquet(p, columns)
        else:
            yield from iter_jsonl(p)


def count_parquet_rows(path: str | Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


class JsonlShardWriter:
    """Write documents to <out_dir>/<prefix>_00000.jsonl.zst, rolling over every max_docs.

    Each shard is written as ``<name>.tmp`` and renamed when complete.  A killed
    process therefore never leaves a truncated zstd frame that would crash every
    later stage; it only loses the unfinished shard, which a re-run regenerates.
    """

    def __init__(self, out_dir: str | Path, prefix: str, max_docs: int = 100_000, compress: bool = True, start_index: int = 0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_docs = max_docs
        self.ext = ".jsonl.zst" if compress else ".jsonl"
        self.shard_idx = start_index
        self.n_in_shard = 0
        self.total = 0
        self._fh = None
        self.files: list[Path] = []

    def _open(self) -> None:
        path = self.out_dir / f"{self.prefix}_{self.shard_idx:05d}{self.ext}"
        self._final_path = path
        self._tmp_path = path.with_name(path.name + ".tmp")
        self._fh = open_text_for_write(self._tmp_path, self.ext)
        self.n_in_shard = 0

    def write(self, doc: dict[str, Any]) -> None:
        if self._fh is None:
            self._open()
        self._fh.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
        self.n_in_shard += 1
        self.total += 1
        if self.n_in_shard >= self.max_docs:
            self._close_shard()

    def _close_shard(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            os.replace(self._tmp_path, self._final_path)
            self.files.append(self._final_path)
            self.shard_idx += 1

    def close(self) -> list[Path]:
        self._close_shard()
        return self.files

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cpu_count(default: int = 8) -> int:
    return os.cpu_count() or default


# ---------------------------------------------------------------------------
# Work splitting: a "chunk" is (path, row_group_start, row_group_end).  Parquet
# files are split by row groups so that one 2M-document file can be processed
# by many workers; jsonl files are a single chunk (they are written as ~100K
# document shards anyway).
# ---------------------------------------------------------------------------

def make_chunks(paths: Sequence[str | Path], target_rows: int = 25_000) -> list[tuple[str, int, int]]:
    chunks: list[tuple[str, int, int]] = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".parquet":
            md = pq.ParquetFile(p).metadata
            n_rg = md.num_row_groups
            rows = [md.row_group(i).num_rows for i in range(n_rg)]
            start, acc = 0, 0
            for i, r in enumerate(rows):
                acc += r
                if acc >= target_rows or i == n_rg - 1:
                    chunks.append((str(p), start, i + 1))
                    start, acc = i + 1, 0
        else:
            chunks.append((str(p), 0, -1))
    return chunks


def iter_chunk(chunk: tuple[str, int, int], columns: Sequence[str] | None = None) -> Iterator[dict[str, Any]]:
    path, rg_start, rg_end = chunk
    p = Path(path)
    if p.suffix != ".parquet":
        yield from iter_jsonl(p)
        return
    pf = pq.ParquetFile(p)
    cols = None
    if columns is not None:
        have = set(pf.schema_arrow.names)
        cols = [c for c in columns if c in have]
    for rg in range(rg_start, rg_end):
        for row in pf.read_row_group(rg, columns=cols).to_pylist():
            yield flatten_metadata(row)
