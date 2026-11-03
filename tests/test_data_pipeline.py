"""Tests for the data pipeline (quality features, filters, dedup, decontamination, io).
Run: python tests/test_data_pipeline.py"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from researchlm.data.decontaminate import ContaminationIndex  # noqa: E402
from researchlm.data.dedup import MinHasher, cluster_by_lsh, duplicates_to_drop, exact_hash, normalize_text  # noqa: E402
from researchlm.data.filters import load_filters  # noqa: E402
from researchlm.data.io import JsonlShardWriter, expand_inputs, iter_documents, make_chunks, iter_chunk  # noqa: E402
from researchlm.data.quality import code_features, text_features  # noqa: E402

FILTERS = ROOT / "configs/data/filters_v1.yaml"
GOOD_EN = ("The mitochondrion is an organelle found in most eukaryotic cells. It generates most of the cell's supply of "
           "adenosine triphosphate, which is used as a source of chemical energy. Mitochondria have their own DNA, "
           "which is inherited from the mother in most species. The number of mitochondria in a cell varies widely "
           "by organism and tissue type; many cells have a single large mitochondrion, whereas others can contain "
           "several thousand. The organelle is composed of compartments that carry out specialised functions. "
           "Mitochondria were first described by Richard Altmann in 1890, and the term was coined by Carl Benda in 1898. "
           "Their role in respiration was established in the twentieth century through the work of many biochemists.")
GOOD_KO = ("미토콘드리아는 대부분의 진핵세포에서 발견되는 세포소기관이다. 이것은 세포가 사용하는 화학 에너지의 원천인 "
           "아데노신 삼인산의 대부분을 생성한다. 미토콘드리아는 자체 DNA를 가지고 있으며, 대부분의 종에서 모계로부터 "
           "유전된다. 세포 내 미토콘드리아의 수는 생물과 조직의 종류에 따라 크게 다르다. 많은 세포는 하나의 큰 미토콘드리아를 "
           "가지는 반면, 다른 세포는 수천 개를 포함할 수 있다. 이 소기관은 특수한 기능을 수행하는 여러 구획으로 이루어져 있다.")
GOOD_MATH = ("Problem. Let $x$ be a real number such that $x^2 - 5x + 6 = 0$. Find all possible values of $x$.\n\n"
             "----------------\nSolution\n----------------\nWe factor: $(x-2)(x-3) = 0$, so $x = 2$ or $x = 3$. "
             "Check: $2^2 - 10 + 6 = 0$ and $3^2 - 15 + 6 = 0$. Therefore the solutions are $x_1 = 2$ and $x_2 = 3$. "
             "In general, for $ax^2 + bx + c = 0$ the roots are given by the quadratic formula $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$. "
             "Here $a = 1$, $b = -5$ and $c = 6$, so the discriminant is $25 - 24 = 1$ and the roots are $(5 \\pm 1)/2$.")
GOOD_CODE = "\n".join([
    "import os", "import sys", "", "def read_lines(path):", "    \"\"\"Return the non-empty lines of a file.\"\"\"",
    "    with open(path) as f:", "        return [l.rstrip() for l in f if l.strip()]", "",
    "def main():", "    for p in sys.argv[1:]:", "        for line in read_lines(p):", "            print(line)", "",
    "if __name__ == '__main__':", "    main()"])


def test_text_features_basic():
    f = text_features(GOOD_EN)
    assert f["n_chars"] > 500 and f["latin_ratio"] > 0.7 and f["en_stopword_frac"] > 0.15
    assert f["hangul_ratio"] == 0.0 and f["url_count"] == 0
    k = text_features(GOOD_KO)
    assert k["hangul_ratio"] > 0.5 and k["ko_function_per_100chars"] > 1.0
    assert text_features("")["empty"] == 1


def test_filters_keep_good_documents():
    flt = load_filters(FILTERS)
    for domain, text, meta in [("english", GOOD_EN, {"language_score": 0.95, "int_score": 3}),
                               ("korean", GOOD_KO, {"language_score": 0.99}),
                               ("math", GOOD_MATH, {"language_score": 0.9, "int_score": 4}),
                               ("code", GOOD_CODE, {"int_score": 3, "license_type": "permissive"})]:
        keep, reason, _ = flt[domain].apply(text, meta)
        assert keep, (domain, reason)


def test_filters_reject_bad_documents():
    flt = load_filters(FILTERS)
    keep, reason, _ = flt["english"].apply("short", {})
    assert not keep and reason == "too_short"
    keep, reason, _ = flt["english"].apply(GOOD_EN, {"int_score": 1})
    assert not keep and reason == "meta_int_score"
    menu = " ".join(["홈 로그인 회원가입 장바구니 공지사항 이벤트 고객센터"] * 20)
    keep, reason, _ = flt["korean"].apply(menu, {})
    assert not keep, reason  # no particles -> no_korean_grammar (or repetition)
    spam = ("buy now " * 300)
    keep, reason, _ = flt["english"].apply(spam, {})
    assert not keep
    keep, reason, _ = flt["code"].apply(GOOD_CODE, {"license_type": "no_license"})
    assert not keep and reason == "meta_license"
    minified = "var a=1;" * 400
    keep, reason, _ = flt["code"].apply(minified, {})
    assert not keep


def test_math_filter_tolerates_separators():
    """Repeated ---- and duplicated 'Solution' lines must NOT reject math."""
    flt = load_filters(FILTERS)
    text = GOOD_MATH + "\n" + "\n".join(["----------------", "Solution", "----------------", "x = 1"])
    keep, reason, feats = flt["math"].apply(text, {})
    assert keep, (reason, feats["repeated_char_ratio"], feats["dup_line_frac"])
    keep_en, reason_en, _ = flt["english"].apply(text, {})
    assert not keep_en  # the same text fails the stricter English rules


def test_exact_hash_normalisation():
    assert exact_hash("Hello,  World!") == exact_hash("hello world")
    assert normalize_text("A  b\n c") == "a b c"


def test_minhash_similarity_and_clustering():
    m = MinHasher(num_perm=128)
    base = " ".join(f"word{i}" for i in range(200))
    near = base.replace("word50", "changed").replace("word150", "other")
    far = " ".join(f"token{i}" for i in range(200))
    s_base, s_near, s_far = m.signature(base), m.signature(near), m.signature(far)
    assert m.jaccard_estimate(s_base, s_near) > 0.85
    assert m.jaccard_estimate(s_base, s_far) < 0.1
    roots = cluster_by_lsh(np.stack([s_base, s_far, s_near, s_base]), num_bands=16)
    drop = duplicates_to_drop(roots)
    assert roots[2] == 0 and roots[3] == 0 and roots[1] == 1
    assert drop.tolist() == [False, False, True, True]


def test_contamination_index_roundtrip():
    q = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with four."
    idx = ContaminationIndex(n=13)
    idx.add_text(q)
    assert idx.is_contaminated("intro text " + q + " trailing text")
    assert not idx.is_contaminated("Janet has ducks that lay eggs and she bakes muffins sometimes with eggs.")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "idx.npy"
        idx.save(p)
        idx2 = ContaminationIndex.load(p, n=13)
        assert len(idx2) == len(idx) and idx2.is_contaminated(q)


def test_jsonl_zst_roundtrip_and_chunks():
    with tempfile.TemporaryDirectory() as d:
        w = JsonlShardWriter(d, "t", max_docs=3)
        for i in range(7):
            w.write({"text": f"doc {i} 한국어", "i": i})
        files = w.close()
        assert len(files) == 3 and all(f.suffix == ".zst" for f in files)
        docs = list(iter_documents(expand_inputs([d])))
        assert [x["i"] for x in docs] == list(range(7))
        chunks = make_chunks(expand_inputs([d]))
        assert len(chunks) == 3 and sum(1 for c in chunks for _ in iter_chunk(c)) == 7


def test_nested_metadata_is_flattened_and_json_safe():
    import datetime
    import pyarrow as pa
    import pyarrow.parquet as pq
    from researchlm.data.io import flatten_metadata
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.parquet"
        table = pa.table({
            "text": ["hello world"],
            "metadata": [{"language_score": 0.9, "int_score": 3, "token_count": 7, "date": datetime.datetime(2020, 1, 1)}],
        })
        pq.write_table(table, path)
        doc = next(iter(iter_documents([path])))
        assert doc["language_score"] == 0.9 and doc["int_score"] == 3 and "metadata" not in doc
        # the English meta rule must now see the value
        flt = load_filters(FILTERS)["english"]
        keep, reason, _ = flt.apply(GOOD_EN, {"language_score": doc["language_score"], "int_score": 1})
        assert not keep and reason == "meta_int_score"
        # a datetime value must not break the shard writer
        w = JsonlShardWriter(d, "t")
        w.write(doc)
        w.close()
        assert next(iter_documents(expand_inputs([Path(d) / "t_00000.jsonl.zst"])))["date"].startswith("2020-01-01")
        # FineMath stores metadata as a JSON string; it must stay untouched
        assert flatten_metadata({"metadata": "{}"})["metadata"] == "{}"


def test_interrupted_shard_writer_leaves_no_visible_file():
    with tempfile.TemporaryDirectory() as d:
        w = JsonlShardWriter(d, "t", max_docs=2)
        for i in range(3):
            w.write({"text": str(i)})
        # one complete shard, one still open (simulates a killed process)
        assert [p.name for p in expand_inputs([d])] == ["t_00000.jsonl.zst"]
        w.close()
        assert len(expand_inputs([d])) == 2


def test_mixed_dataset_resume_is_exact_for_each_rank():
    from researchlm.data.dataset import DomainSpec, MixedTokenDataset
    from researchlm.data.shards import ShardWriter
    with tempfile.TemporaryDirectory() as d:
        for name, base in (("a", 10), ("b", 20_000)):
            w = ShardWriter(Path(d) / name, "train", shard_size_tokens=5000)
            w.add_document(list(range(base, base + 12_000)))
            w.close()
        specs = [DomainSpec("a", str(Path(d) / "a"), 0.7), DomainSpec("b", str(Path(d) / "b"), 0.3)]
        for rank in (0, 1):
            ds = MixedTokenDataset(specs, "train", 16, 4, rank=rank, world_size=2, seed=5)
            for _ in range(7):
                next(ds)
            state = ds.state_dict()
            expected = [next(ds)[0].clone() for _ in range(3)]
            resumed = MixedTokenDataset(specs, "train", 16, 4, rank=rank, world_size=2, seed=5)
            resumed.load_state_dict(state)
            got = [next(resumed)[0] for _ in range(3)]
            assert all((a == b).all() for a, b in zip(expected, got)), f"rank {rank}"


def test_code_features():
    f = code_features(GOOD_CODE)
    assert f["n_lines"] >= 10 and f["alpha_ratio"] > 0.5 and f["autogenerated"] == 0
    assert code_features("// Auto-generated file. DO NOT EDIT\nx")["autogenerated"] == 1


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f(); print(f"ok  {f.__name__}")
    print(f"\n{len(fns)} tests passed")
