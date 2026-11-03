"""Tokenizer compression metrics per domain.

    chars/token  : higher = fewer tokens per character (good for that language)
    bytes/token  : same idea but fair across scripts (Hangul is 3 bytes/char)
    unk rate     : should be 0 with byte fallback
    byte-fallback rate : how often we had to fall back to raw bytes
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..data.io import expand_inputs, iter_documents
from ..data.tokenizer import Tokenizer


def iter_texts(spec: str | Path, limit: int | None = None) -> Iterable[str]:
    """spec = file / directory / glob. `limit` is split evenly over the files, because
    a code directory is sorted by language and would otherwise yield only its first one."""
    files = expand_inputs([spec])
    per_file = None if limit is None else max(1, limit // max(1, len(files)))
    for path in files:
        for i, doc in enumerate(iter_documents([path], columns=["text"])):
            if per_file is not None and i >= per_file:
                break
            yield doc["text"]


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
