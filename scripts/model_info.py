"""Print parameter counts, memory estimates and FLOPs for a model config.

    python scripts/model_info.py --model configs/model/researchlm_50m.yaml --tokens 1e9
"""

from __future__ import annotations

import argparse

import torch

import _bootstrap  # noqa: F401
from researchlm.model import ModelConfig, ResearchLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=float, default=1e9, help="planned training tokens")
    ap.add_argument("--build", action="store_true", help="instantiate the model to verify the analytic count")
    args = ap.parse_args()

    cfg = ModelConfig.from_yaml(args.model)
    est = cfg.estimate_params()
    print(f"config: {cfg.name}")
    for k, v in cfg.to_dict().items():
        if k != "extra":
            print(f"  {k:16s}: {v}")
    print()
    print(f"analytic params   : {est['total']:,}  (non-embedding {est['non_embedding']:,})")
    print(f"  embedding       : {est['embedding']:,}  (tied lm_head -> {est['lm_head']:,} extra)")
    print(f"  per layer       : {est['per_layer']:,}  x {cfg.n_layers}")
    if args.build:
        m = ResearchLM(cfg)
        print(f"built params      : {m.num_params():,}")

    N = est["total"]
    D = args.tokens
    flops = 6 * N * D
    print()
    print(f"tokens planned    : {D:.3g}  ({D / N:.1f} tokens/param; Chinchilla-optimal ~20)")
    print(f"training FLOPs    : 6*N*D = {flops:.3g}")
    # RTX 8000 fp16 tensor-core peak ~130 TFLOPs; realistic MFU for a tiny model ~25-35%
    for name, peak_tflops, mfu, n_gpu in (("2x RTX 8000 fp16", 130, 0.30, 2), ("1x A10G bf16", 125, 0.30, 1)):
        secs = flops / (peak_tflops * 1e12 * mfu * n_gpu)
        print(f"  {name:18s}: ~{secs / 3600:.1f} h at {int(mfu*100)}% MFU")
    print()
    bytes_per_param = 2 + 4 + 4 + 4 + 4  # fp16 grad + fp32 master(weights) + adam m,v + fp32 grad copy worst case
    print(f"optimizer-state memory (upper bound) : {N * bytes_per_param / 2**30:.2f} GB")
    act = cfg.max_seq_len * cfg.d_model * 2  # fp16 per token per layer (rough: x, attn in/out, ffn)
    print(f"logits for micro_batch=8, seq={cfg.max_seq_len}: {8 * cfg.max_seq_len * cfg.vocab_size * 2 / 2**30:.2f} GB (fp16) "
          f"+ {8 * cfg.max_seq_len * cfg.vocab_size * 4 / 2**30:.2f} GB (fp32 for CE)")


if __name__ == "__main__":
    main()
