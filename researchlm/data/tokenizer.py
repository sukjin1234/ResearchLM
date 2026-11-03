"""SentencePiece BPE tokenizer for ResearchLM.

Design choices (see docs/researchlm_50m_guide.md, section "Tokenizer"):
  * BPE, vocab 32,768 (= 2^15, friendly for GPU matmul tiling)
  * byte_fallback=True  -> no <unk> ever; unseen characters become byte tokens
  * split_digits=True   -> "2024" -> 2 0 2 4, helps arithmetic / math data
  * normalization "identity" -> we do not NFKC-normalise (keeps code intact)
  * fixed special ids: <unk>=0 <s>=1 </s>=2 <pad>=3, then reserved tokens
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import sentencepiece as spm

UNK_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3

# Reserved for post-training (chat template, fill-in-the-middle for code, ...).
# Reserving them *now* means the vocabulary never has to change later.
DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|repo_name|>",
    "<|file_sep|>",
    *[f"<|reserved_{i}|>" for i in range(6)],
)


def train_tokenizer(
    corpus_files: Sequence[str | Path],
    model_prefix: str | Path,
    vocab_size: int = 32_768,
    special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    input_sentence_size: int = 4_000_000,
    character_coverage: float = 0.9999,
    num_threads: int | None = None,
    max_sentence_length: int = 16_384,
) -> Path:
    """Train a SentencePiece BPE model. Returns the path to the .model file."""
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    num_threads = num_threads or min(64, os.cpu_count() or 8)

    spm.SentencePieceTrainer.train(
        input=",".join(str(p) for p in corpus_files),
        model_prefix=str(model_prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        user_defined_symbols=list(special_tokens),
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        max_sentence_length=max_sentence_length,
        num_threads=num_threads,
        train_extremely_large_corpus=False,
    )
    model_path = model_prefix.with_suffix(".model")

    meta = {
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "unk_id": UNK_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "pad_id": PAD_ID,
        "special_tokens": list(special_tokens),
        "byte_fallback": True,
        "split_digits": True,
        "normalization": "identity",
        "corpus_files": [str(p) for p in corpus_files],
    }
    with open(model_prefix.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return model_path


class Tokenizer:
    """Thin wrapper around a trained SentencePiece model."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: Iterable[str], add_bos: bool = False, add_eos: bool = False) -> list[list[int]]:
        out = self.sp.encode(list(texts), out_type=int)
        if add_bos or add_eos:
            out = [
                ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])
                for ids in out
            ]
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return self.sp.decode([int(i) for i in ids])

    def id_to_piece(self, i: int) -> str:
        return self.sp.id_to_piece(int(i))

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)

    def tokenize(self, text: str) -> list[str]:
        return self.sp.encode(text, out_type=str)
