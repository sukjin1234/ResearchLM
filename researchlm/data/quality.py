"""Text quality features used by the structure analyzer and the filters.

Most heuristics follow Gopher (Rae et al. 2021), C4 (Raffel et al. 2020),
RefinedWeb and FineWeb.  Every feature is a plain float so that thresholds can
live in YAML.  ``text_features`` is the single entry point.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_RE = re.compile(r"<(?:/?)(?:html|body|div|span|script|style|a|p|br|table|tr|td|img|meta|link|ul|li|h[1-6])\b[^>]*>", re.IGNORECASE)
REPEATED_CHAR_RE = re.compile(r"(.)\1{5,}")
LATEX_RE = re.compile(r"\\(?:frac|sqrt|sum|int|begin|end|alpha|beta|cdot|left|right|mathbb|text)|\$[^$\n]{1,200}\$")
CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)
MD_HEADER_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
BULLET_RE = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)])\s")
WORD_RE = re.compile(r"[A-Za-z]+|[\uac00-\ud7a3]+|\d+")
LOREM_RE = re.compile(r"lorem ipsum", re.IGNORECASE)
CURLY_RE = re.compile(r"[{}]")
ELLIPSIS_ENDINGS = ("...", "…")

EN_STOPWORDS = frozenset("the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me".split())
# Common Korean particles / endings that almost every natural Korean sentence contains.
KO_FUNCTION_TOKENS = ("은", "는", "이", "가", "을", "를", "의", "에", "에서", "로", "으로", "와", "과", "도", "다", "요", "니다", "습니다", "했다", "있다", "없다", "하는", "것")


def _is_hangul(ch: str) -> bool:
    o = ord(ch)
    return 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F


def _is_latin(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF


def _top_ngram_fraction(words: list[str], n: int) -> float:
    """Share of characters covered by the single most frequent word n-gram (Gopher, n=2..4)."""
    if len(words) < n * 2:
        return 0.0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return 0.0
    top, cnt = grams.most_common(1)[0]
    return min(1.0, cnt * sum(len(w) for w in top) / total_chars)


def _dup_ngram_fraction(words: list[str], n: int) -> float:
    """Share of characters inside *any* word n-gram that occurs more than once
    (Gopher, n=5..10).  Each word is counted at most once even when it sits in
    several overlapping duplicated n-grams."""
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return 0.0
    covered = [False] * len(words)
    for i, g in enumerate(grams):
        if counts[g] > 1:
            for j in range(i, i + n):
                covered[j] = True
    dup_chars = sum(len(w) for w, c in zip(words, covered) if c)
    return dup_chars / total_chars


def text_features(text: str) -> dict[str, Any]:
    n_chars = len(text)
    if n_chars == 0:
        return {"n_chars": 0, "empty": 1}

    n_hangul = n_latin = n_digit = n_space = n_cjk = n_symbol = n_alnum = 0
    for ch in text:
        if _is_hangul(ch):
            n_hangul += 1; n_alnum += 1
        elif _is_latin(ch):
            n_latin += 1; n_alnum += 1
        elif ch.isdigit():
            n_digit += 1; n_alnum += 1
        elif ch.isspace():
            n_space += 1
        elif _is_cjk(ch):
            n_cjk += 1; n_alnum += 1
        elif ch in "#$%^&*{}[]<>|\\~`@=+":
            n_symbol += 1

    lines = text.split("\n")
    nonempty = [l for l in lines if l.strip()]
    n_lines = max(1, len(nonempty))
    line_lens = [len(l) for l in nonempty] or [0]
    line_counter = Counter(l.strip() for l in nonempty)
    dup_lines = sum(c - 1 for c in line_counter.values() if c > 1)
    dup_line_chars = sum(len(l) * (c - 1) for l, c in line_counter.items() if c > 1)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    para_counter = Counter(paras)
    dup_paras = sum(c - 1 for c in para_counter.values() if c > 1)

    words = WORD_RE.findall(text)
    n_words = len(words)
    word_lens = [len(w) for w in words] or [0]
    lower_words = [w.lower() for w in words]
    en_stop = sum(1 for w in lower_words if w in EN_STOPWORDS)
    ko_func = sum(text.count(t) for t in KO_FUNCTION_TOKENS)

    urls = URL_RE.findall(text)
    url_chars = sum(len(u) for u in urls)
    repeated = REPEATED_CHAR_RE.findall(text)
    repeated_chars = sum(len(m.group(0)) for m in REPEATED_CHAR_RE.finditer(text))

    ellipsis_lines = sum(1 for l in nonempty if l.rstrip().endswith(ELLIPSIS_ENDINGS))
    bullet_lines = sum(1 for l in nonempty if BULLET_RE.match(l))

    return {
        "empty": 0,
        "n_chars": n_chars,
        "n_words": n_words,
        "n_lines": n_lines,
        "n_paragraphs": len(paras),
        "mean_word_len": sum(word_lens) / max(1, n_words),
        "mean_line_len": sum(line_lens) / n_lines,
        "max_line_len": max(line_lens),
        "hangul_ratio": n_hangul / n_chars,
        "latin_ratio": n_latin / n_chars,
        "digit_ratio": n_digit / n_chars,
        "cjk_ratio": n_cjk / n_chars,
        "space_ratio": n_space / n_chars,
        "symbol_ratio": n_symbol / n_chars,
        "alnum_ratio": n_alnum / n_chars,
        "url_count": len(urls),
        "url_char_ratio": url_chars / n_chars,
        "html_tag_count": len(HTML_RE.findall(text)),
        "repeated_char_runs": len(repeated),
        "repeated_char_ratio": repeated_chars / n_chars,
        "dup_line_frac": dup_lines / n_lines,
        "dup_line_char_frac": dup_line_chars / n_chars,
        "dup_para_frac": dup_paras / max(1, len(paras)),
        "top_2gram_frac": _top_ngram_fraction(lower_words, 2),
        "top_3gram_frac": _top_ngram_fraction(lower_words, 3),
        "top_4gram_frac": _top_ngram_fraction(lower_words, 4),
        "dup_5gram_frac": _dup_ngram_fraction(lower_words, 5),
        "dup_10gram_frac": _dup_ngram_fraction(lower_words, 10),
        "ellipsis_line_frac": ellipsis_lines / n_lines,
        "bullet_line_frac": bullet_lines / n_lines,
        "en_stopword_frac": en_stop / max(1, n_words),
        "ko_function_per_100chars": 100.0 * ko_func / n_chars,
        "latex_count": len(LATEX_RE.findall(text)),
        "code_fence_count": len(CODE_FENCE_RE.findall(text)),
        "md_header_count": len(MD_HEADER_RE.findall(text)),
        "curly_count": len(CURLY_RE.findall(text)),
        "lorem_ipsum": int(bool(LOREM_RE.search(text))),
    }


def code_features(text: str) -> dict[str, Any]:
    """Cheaper feature set tuned for source code (RedPajama / StarCoder filters)."""
    n_chars = len(text)
    if n_chars == 0:
        return {"n_chars": 0, "empty": 1}
    lines = text.split("\n")
    nonempty = [l for l in lines if l.strip()]
    line_lens = [len(l) for l in nonempty] or [0]
    n_alnum = sum(ch.isalnum() for ch in text)
    n_alpha = sum(ch.isalpha() for ch in text)
    n_digit = sum(ch.isdigit() for ch in text)
    n_hex_like = len(re.findall(r"\b[0-9a-fA-F]{32,}\b", text))
    n_b64_like = len(re.findall(r"[A-Za-z0-9+/=]{64,}", text))
    comment_lines = sum(1 for l in nonempty if l.lstrip().startswith(("#", "//", "/*", "*", "--", "<!--")))
    autogen = int(bool(re.search(r"(?i)auto-?generated|do not edit|generated by", text[:2000])))
    return {
        "empty": 0,
        "n_chars": n_chars,
        "n_lines": len(nonempty),
        "mean_line_len": sum(line_lens) / max(1, len(line_lens)),
        "max_line_len": max(line_lens),
        "alnum_ratio": n_alnum / n_chars,
        "alpha_ratio": n_alpha / n_chars,
        "digit_ratio": n_digit / n_chars,
        "hex_blob_count": n_hex_like,
        "base64_blob_count": n_b64_like,
        "comment_line_frac": comment_lines / max(1, len(nonempty)),
        "autogenerated": autogen,
    }


def entropy_of_chars(text: str, limit: int = 20_000) -> float:
    sample = text[:limit]
    if not sample:
        return 0.0
    c = Counter(sample)
    n = len(sample)
    return -sum((v / n) * math.log2(v / n) for v in c.values())
