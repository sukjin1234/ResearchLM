import json
from pathlib import Path

from datasets import load_dataset


OUTPUT_DIR = Path("data_samples")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DATASETS = {
    "english": {
        "path": "HuggingFaceTB/smollm-corpus",
        "name": "fineweb-edu-dedup",
    },

    "korean": {
        "path": "HuggingFaceFW/fineweb-2",
        "name": "kor_Hang",
    },

    "math": {
        "path": "HuggingFaceTB/finemath",
        "name": "finemath-4plus",
    },
}


def download_sample(
    domain,
    path,
    name,
    limit=10_000,
):

    print(
        f"\n===== {domain.upper()} ====="
    )

    dataset = load_dataset(
        path,
        name,
        split="train",
        streaming=True,
    )

    output_file = (
        OUTPUT_DIR /
        f"{domain}.jsonl"
    )

    count = 0

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in dataset:

            text = row.get("text")

            if not text:
                continue

            data = {
                "domain": domain,
                "text": text,
            }

            f.write(
                json.dumps(
                    data,
                    ensure_ascii=False
                )
                + "\n"
            )

            count += 1

            if count >= limit:
                break

    print(
        f"Saved: {count:,}"
    )

    print(
        f"Path: {output_file}"
    )


def main():

    for domain, info in DATASETS.items():

        download_sample(
            domain=domain,
            path=info["path"],
            name=info["name"],
        )


if __name__ == "__main__":
    main()
