"""Train ResearchLM.

Single GPU:
    python scripts/train.py --model configs/model/tiny_debug.yaml --train configs/train/tiny_debug.yaml

Two GPUs (DDP):
    torchrun --standalone --nproc_per_node=2 scripts/train.py \
        --model configs/model/researchlm_50m.yaml --train configs/train/50m_pilot_samples.yaml

Resume:
    ... --resume latest          (latest checkpoint of this run_name)
    ... --resume /storage/researchlm/checkpoints/researchlm-50m-v1/step_000500

Any TrainConfig field can be overridden:  --set lr=1e-3 --set max_steps=500
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from researchlm.model import ModelConfig
from researchlm.training import TrainConfig, Trainer, cleanup_distributed, init_distributed


def parse_override(s: str):
    """key=value where value is parsed as YAML (1e-3, true, {a: 1}, [x, y], plain strings)."""
    import yaml
    k, v = s.split("=", 1)
    try:
        val = yaml.safe_load(v)
    except yaml.YAMLError:
        return k, v
    if isinstance(val, str):  # YAML 1.1 reads "1e-5" as a string; accept it as a float
        try:
            return k, float(val)
        except ValueError:
            pass
    return k, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="configs/model/*.yaml")
    ap.add_argument("--train", required=True, help="configs/train/*.yaml")
    ap.add_argument("--server", default=None, help="configs/server/*.yaml (paths + precision defaults)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--set", action="append", default=[], help="override train field: key=value")
    args = ap.parse_args()

    overrides = dict(parse_override(s) for s in args.set)
    if args.resume:
        overrides["resume"] = args.resume

    model_cfg = ModelConfig.from_yaml(args.model)
    train_cfg = TrainConfig.from_yaml(args.train, server_yaml=args.server, overrides=overrides)

    dist = init_distributed()
    try:
        trainer = Trainer(model_cfg, train_cfg, dist)
        trainer.train()
    finally:
        cleanup_distributed(dist)


if __name__ == "__main__":
    main()
