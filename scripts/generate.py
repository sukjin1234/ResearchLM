"""Sample text from a checkpoint.

    python scripts/generate.py --ckpt /storage/researchlm/checkpoints/researchlm-50m-pilot \
        --tokenizer /storage/researchlm/tokenizer/researchlm_32k.model \
        --prompt "대한민국의 수도는" --max-new-tokens 100 --temperature 0.8 --top-p 0.95
"""

from __future__ import annotations

import argparse

import torch

import _bootstrap  # noqa: F401
from researchlm.data.tokenizer import Tokenizer
from researchlm.training.checkpoint import load_model_from_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="step dir or run dir (uses 'latest')")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = Tokenizer(args.tokenizer)
    model, cfg = load_model_from_checkpoint(args.ckpt, args.device)
    model.eval()
    print(f"loaded {cfg.name}: {model.num_params():,} params on {args.device}\n")

    prompts = args.prompt or [
        "The history of mathematics",
        "대한민국의 수도는",
        "In this tutorial, we will learn how to",
        "Theorem. Let x be a real number such that",
    ]
    for p in prompts:
        ids = torch.tensor([tok.encode(p, add_bos=True)], device=args.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                 top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty)
        text = tok.decode(out[0, ids.shape[1]:].tolist())
        print("=" * 72)
        print(f"PROMPT : {p}")
        print(f"OUTPUT : {text}")
    print("=" * 72)


if __name__ == "__main__":
    main()
