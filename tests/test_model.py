"""Unit tests for the model.  Run:  python -m pytest tests -q   (or python tests/test_model.py)"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from researchlm.model import KVCache, ModelConfig, ResearchLM, RMSNorm, apply_rope, precompute_rope_cache  # noqa: E402
from researchlm.training.config import TrainConfig  # noqa: E402
from researchlm.training.scheduler import lr_at  # noqa: E402


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=256, max_seq_len=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64)
    base.update(kw)
    return ModelConfig(**base)


def test_param_count_matches_analytic():
    for path in ("configs/model/researchlm_50m.yaml", "configs/model/tiny_debug.yaml"):
        cfg = ModelConfig.from_yaml(Path(__file__).resolve().parent.parent / path)
        m = ResearchLM(cfg)
        assert m.num_params() == cfg.estimate_params()["total"], path


def test_50m_is_about_50m():
    cfg = ModelConfig.from_yaml(Path(__file__).resolve().parent.parent / "configs/model/researchlm_50m.yaml")
    n = cfg.estimate_params()["total"]
    assert 45e6 < n < 56e6, n


def test_forward_shapes_and_initial_loss():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = ResearchLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    y = torch.randint(0, cfg.vocab_size, (3, 16))  # independent targets (tied
    # embeddings make "predict the input token itself" artificially easy)
    logits, loss = m(x, y)
    assert logits.shape == (3, 16, cfg.vocab_size)
    # a fresh model should be close to uniform: loss ~ ln(vocab)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.3, loss.item()


def test_causality():
    """Changing a future token must not change earlier logits."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = ResearchLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    y = x.clone()
    y[0, 8:] = (y[0, 8:] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(y)
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-5)
    assert not torch.allclose(a[0, 8:], b[0, 8:], atol=1e-5)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = ResearchLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    with torch.no_grad():
        full, _ = m(x)
        caches = [KVCache() for _ in range(cfg.n_layers)]
        step_logits = []
        pre, _ = m(x[:, :4], kv_caches=caches, start_pos=0)
        step_logits.append(pre)
        for t in range(4, 10):
            lg, _ = m(x[:, t : t + 1], kv_caches=caches, start_pos=t)
            step_logits.append(lg)
        inc = torch.cat(step_logits, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


def test_generate_runs_and_respects_length():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = ResearchLM(cfg)
    x = torch.randint(4, cfg.vocab_size, (2, 5))
    out = m.generate(x, max_new_tokens=7, temperature=1.0, top_k=10, eos_token_id=-1)
    assert out.shape == (2, 12)
    assert torch.equal(out[:, :5], x)


def test_gqa_vs_mha_configs_build():
    for kv in (1, 2, 4):
        m = ResearchLM(tiny_cfg(n_kv_heads=kv))
        x = torch.randint(0, 256, (1, 8))
        m(x)


def test_weight_tying():
    m = ResearchLM(tiny_cfg(tie_embeddings=True))
    assert m.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr()
    m2 = ResearchLM(tiny_cfg(tie_embeddings=False))
    assert m2.lm_head.weight.data_ptr() != m2.embed_tokens.weight.data_ptr()


def test_rmsnorm_unit_rms():
    n = RMSNorm(16)
    x = torch.randn(4, 16) * 10
    y = n(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_preserves_norm_and_is_relative():
    cos, sin = precompute_rope_cache(8, 32)
    q = torch.randn(1, 1, 32, 8)
    r = apply_rope(q, cos, sin)
    assert torch.allclose(q.norm(dim=-1), r.norm(dim=-1), atol=1e-5)
    # dot(q_m, k_n) depends only on (m - n): shift both by 5 and compare
    k = torch.randn(1, 1, 32, 8)
    rq, rk = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    s1 = (rq[0, 0, 3] * rk[0, 0, 1]).sum()
    q2 = torch.zeros_like(q); k2 = torch.zeros_like(k)
    q2[0, 0, 8] = q[0, 0, 3]; k2[0, 0, 6] = k[0, 0, 1]
    rq2, rk2 = apply_rope(q2, cos, sin), apply_rope(k2, cos, sin)
    s2 = (rq2[0, 0, 8] * rk2[0, 0, 6]).sum()
    assert torch.allclose(s1, s2, atol=1e-4)


def test_lr_schedule():
    cfg = TrainConfig(lr=1.0, min_lr_ratio=0.1, warmup_steps=10, max_steps=110, scheduler="cosine")
    assert lr_at(0, cfg) == 0.1  # (0+1)/10
    assert abs(lr_at(9, cfg) - 1.0) < 1e-9
    assert abs(lr_at(10, cfg) - 1.0) < 1e-9
    assert abs(lr_at(60, cfg) - 0.55) < 1e-6  # midpoint of cosine
    assert abs(lr_at(109, cfg) - 0.1) < 1e-3
    assert lr_at(500, cfg) == 0.1
    wsd = TrainConfig(lr=1.0, min_lr_ratio=0.0, warmup_steps=0, max_steps=100, scheduler="wsd", wsd_decay_fraction=0.2)
    assert lr_at(50, wsd) == 1.0 and abs(lr_at(90, wsd) - 0.5) < 1e-6


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f(); print(f"ok  {f.__name__}")
    print(f"\n{len(fns)} tests passed")
