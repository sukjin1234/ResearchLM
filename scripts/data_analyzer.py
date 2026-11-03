import json
import re
import statistics
from collections import Counter
from pathlib import Path


DATA_DIR = Path("data_samples")


URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)

HTML_RE = re.compile(
    r"<(?:html|body|div|span|script|style|a|p|br|table|img|meta|link)\b",
    re.IGNORECASE,
)

REPEATED_CHAR_RE = re.compile(r"(.)\1{5,}")


def is_hangul(char):
    code = ord(char)

    return (
        0xAC00 <= code <= 0xD7A3 or
        0x1100 <= code <= 0x11FF or
        0x3130 <= code <= 0x318F
    )


def is_latin(char):
    return (
        "a" <= char <= "z"
        or "A" <= char <= "Z"
    )


def calculate_character_ratios(text):
    if not text:
        return {
            "hangul": 0,
            "latin": 0,
            "digit": 0,
            "whitespace": 0,
            "other": 0,
        }

    total = len(text)

    hangul = 0
    latin = 0
    digit = 0
    whitespace = 0

    for char in text:

        if is_hangul(char):
            hangul += 1

        elif is_latin(char):
            latin += 1

        elif char.isdigit():
            digit += 1

        elif char.isspace():
            whitespace += 1

    other = (
        total
        - hangul
        - latin
        - digit
        - whitespace
    )

    return {
        "hangul": hangul / total,
        "latin": latin / total,
        "digit": digit / total,
        "whitespace": whitespace / total,
        "other": other / total,
    }


def duplicate_line_ratio(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) <= 1:
        return 0.0

    unique_lines = len(set(lines))

    duplicates = len(lines) - unique_lines

    return duplicates / len(lines)


def percentile(values, p):
    if not values:
        return 0

    values = sorted(values)

    index = int(
        (len(values) - 1) * p
    )

    return values[index]


def analyze_file(path):

    char_lengths = []
    line_counts = []

    hangul_ratios = []
    latin_ratios = []
    digit_ratios = []
    other_ratios = []

    duplicate_line_ratios = []

    url_docs = 0
    html_docs = 0
    repeated_char_docs = 0

    short_100 = 0
    short_200 = 0

    very_long = 0

    empty_docs = 0

    documents = 0

    suspicious_examples = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            row = json.loads(line)

            text = row.get(
                "text",
                ""
            )

            documents += 1

            if not text.strip():
                empty_docs += 1
                continue

            length = len(text)

            char_lengths.append(length)

            line_count = len(
                text.splitlines()
            )

            line_counts.append(
                line_count
            )

            if length < 100:
                short_100 += 1

            if length < 200:
                short_200 += 1

            if length > 100_000:
                very_long += 1

            if URL_RE.search(text):
                url_docs += 1

            if HTML_RE.search(text):
                html_docs += 1

            if REPEATED_CHAR_RE.search(text):
                repeated_char_docs += 1

            ratios = (
                calculate_character_ratios(
                    text
                )
            )

            hangul_ratios.append(
                ratios["hangul"]
            )

            latin_ratios.append(
                ratios["latin"]
            )

            digit_ratios.append(
                ratios["digit"]
            )

            other_ratios.append(
                ratios["other"]
            )

            dup_ratio = (
                duplicate_line_ratio(
                    text
                )
            )

            duplicate_line_ratios.append(
                dup_ratio
            )

            if (
                dup_ratio > 0.30
                or ratios["other"] > 0.30
            ):
                if len(
                    suspicious_examples
                ) < 10:

                    suspicious_examples.append({
                        "text": text[:500],
                        "duplicate_ratio": dup_ratio,
                        "other_ratio": ratios["other"],
                    })

    def avg(values):
        if not values:
            return 0

        return statistics.mean(
            values
        )

    stats = {

        "documents": documents,

        "empty_documents": empty_docs,

        "length": {
            "mean": avg(char_lengths),
            "median": statistics.median(
                char_lengths
            ) if char_lengths else 0,

            "p10": percentile(
                char_lengths,
                0.10
            ),

            "p90": percentile(
                char_lengths,
                0.90
            ),

            "p99": percentile(
                char_lengths,
                0.99
            ),

            "max": max(
                char_lengths,
                default=0
            ),
        },

        "average_lines": avg(
            line_counts
        ),

        "character_ratios": {
            "hangul": avg(
                hangul_ratios
            ),

            "latin": avg(
                latin_ratios
            ),

            "digit": avg(
                digit_ratios
            ),

            "other": avg(
                other_ratios
            ),
        },

        "shorter_than_100": (
            short_100 / documents
            if documents else 0
        ),

        "shorter_than_200": (
            short_200 / documents
            if documents else 0
        ),

        "longer_than_100k": (
            very_long / documents
            if documents else 0
        ),

        "contains_url": (
            url_docs / documents
            if documents else 0
        ),

        "contains_html": (
            html_docs / documents
            if documents else 0
        ),

        "repeated_characters": (
            repeated_char_docs
            / documents
            if documents else 0
        ),

        "average_duplicate_line_ratio":
            avg(
                duplicate_line_ratios
            ),

        "suspicious_examples":
            suspicious_examples,
    }

    return stats


def print_stats(
    name,
    stats
):

    print()
    print("=" * 60)

    print(
        f"{name.upper()}"
    )

    print("=" * 60)

    print(
        f"Documents        : "
        f"{stats['documents']:,}"
    )

    print(
        f"Empty            : "
        f"{stats['empty_documents']:,}"
    )

    length = stats["length"]

    print()
    print("[Document length]")

    print(
        f"Mean             : "
        f"{length['mean']:,.0f}"
    )

    print(
        f"Median           : "
        f"{length['median']:,.0f}"
    )

    print(
        f"P10              : "
        f"{length['p10']:,}"
    )

    print(
        f"P90              : "
        f"{length['p90']:,}"
    )

    print(
        f"P99              : "
        f"{length['p99']:,}"
    )

    print(
        f"Max              : "
        f"{length['max']:,}"
    )

    ratios = (
        stats["character_ratios"]
    )

    print()
    print("[Character ratios]")

    print(
        f"Hangul           : "
        f"{ratios['hangul']:.2%}"
    )

    print(
        f"Latin            : "
        f"{ratios['latin']:.2%}"
    )

    print(
        f"Digit            : "
        f"{ratios['digit']:.2%}"
    )

    print(
        f"Other            : "
        f"{ratios['other']:.2%}"
    )

    print()
    print("[Quality signals]")

    print(
        f"< 100 chars       : "
        f"{stats['shorter_than_100']:.2%}"
    )

    print(
        f"< 200 chars       : "
        f"{stats['shorter_than_200']:.2%}"
    )

    print(
        f"> 100k chars      : "
        f"{stats['longer_than_100k']:.2%}"
    )

    print(
        f"Contains URL      : "
        f"{stats['contains_url']:.2%}"
    )

    print(
        f"Contains HTML     : "
        f"{stats['contains_html']:.2%}"
    )

    print(
        f"Repeated chars    : "
        f"{stats['repeated_characters']:.2%}"
    )

    print(
        f"Duplicate lines   : "
        f"{stats['average_duplicate_line_ratio']:.2%}"
    )


def main():

    output = {}

    for path in sorted(
        DATA_DIR.glob("*.jsonl")
    ):

        name = path.stem

        stats = analyze_file(
            path
        )

        output[name] = stats

        print_stats(
            name,
            stats
        )

    output_path = (
        DATA_DIR /
        "analysis.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        f"Saved analysis -> "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
