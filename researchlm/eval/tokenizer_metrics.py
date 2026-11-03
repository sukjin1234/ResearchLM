"""Tokenizer compression metrics per domain.

    chars/token  : higher = fewer tokens per character (good for that language)
    bytes/token  : same idea but fair across scripts (Hangul is 3 bytes/char)
    unk rate     : should be 0 with byte fallback
    byte-fallback rate : how often we had to fall back to raw bytes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..data.tokenizer import Tokenizer


def iter_texts(jsonl_path: str | Path, limit: int | None = None) -> Iterable[str]:
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)["text"]


def compression_stats(tok: Tokenizer, texts: Iterable[str]) -> dict[str, float]:
    n_chars = n_bytes = n_tokens = n_unk = n_byte = n_docs = 0
    for text in texts:
        ids = tok.encode(text)
        n_docs += 1
        n_chars += len(text)
        n_bytes += len(text.encode("utf-8"))
        n_tokens += len(ids)
        n_unk += sum(1 for i in ids if i == tok.unk_id)
        n_byte += sum(1 for i in ids if tok.id_to_piece(i).startswith("<0x"))
    return {
        "documents": n_docs,
        "tokens": n_tokens,
        "chars_per_token": n_chars / max(1, n_tokens),
        "bytes_per_token": n_bytes / max(1, n_tokens),
        "tokens_per_doc": n_tokens / max(1, n_docs),
        "unk_rate": n_unk / max(1, n_tokens),
        "byte_fallback_rate": n_byte / max(1, n_tokens),
    }
