"""Bootstrap ve kol karşılaştırma metrikleri (PROTOCOL §7.1, §7.4)."""

import math

import numpy as np
import pytest

from src.analysis.bootstrap import bootstrap
from src.analysis.compare import flores_comparison, pairs_comparison

BS = {"n_resamples": 2000, "seed": 0, "ci": 0.95}


def flores(nll_tr, bytes_tr, nll_en, bytes_en):
    return {"tr": {"nll": np.asarray(nll_tr, float), "n_bytes": np.asarray(bytes_tr)},
            "en": {"nll": np.asarray(nll_en, float), "n_bytes": np.asarray(bytes_en)}}


def test_bootstrap_deterministic_and_contains_point():
    x = np.random.default_rng(1).normal(size=200)
    stats = {"mean": lambda idx: x[idx].mean(axis=1)}
    a = bootstrap(stats, len(x), **BS)
    b = bootstrap(stats, len(x), **BS)
    assert a == b
    m = a["mean"]
    assert m["value"] == pytest.approx(x.mean())
    assert m["lo"] < m["value"] < m["hi"]


def test_bootstrap_ci_width_shrinks_with_n():
    rng = np.random.default_rng(2)
    small, large = rng.normal(size=50), rng.normal(size=5000)
    w = {}
    for name, x in (("small", small), ("large", large)):
        r = bootstrap({"m": lambda idx, x=x: x[idx].mean(axis=1)}, len(x), **BS)["m"]
        w[name] = r["hi"] - r["lo"]
    assert w["large"] < w["small"] / 5


def test_flores_identical_arm_is_zero():
    rng = np.random.default_rng(3)
    ref = flores(rng.uniform(10, 50, 100), rng.integers(50, 200, 100),
                 rng.uniform(10, 50, 100), rng.integers(50, 200, 100))
    res = flores_comparison(ref, ref, **BS)
    for k in ("dbpb_rel_tr", "dbpb_rel_en", "did"):
        assert res[k] == {"value": 0.0, "lo": 0.0, "hi": 0.0}


def test_flores_values_match_definitions():
    ref = flores([10, 20], [100, 100], [10, 10], [50, 50])
    arm = flores([12, 24], [100, 100], [11, 11], [50, 50])  # TR +%20, EN +%10
    res = flores_comparison(ref, arm, **BS)
    assert res["bpb_tr"]["value"] == pytest.approx(36 / (math.log(2) * 200))
    assert res["dbpb_rel_tr"]["value"] == pytest.approx(0.20)
    assert res["dbpb_rel_en"]["value"] == pytest.approx(0.10)
    assert res["did"]["value"] == pytest.approx(0.10)
    # Her cümlede oran aynı olduğundan yeniden örnekleme değeri değiştirmez.
    assert res["did"]["lo"] == pytest.approx(0.10)
    assert res["did"]["hi"] == pytest.approx(0.10)


def test_pairs_dacc():
    ref = np.array([True] * 90 + [False] * 10)
    arm = ref.copy()
    arm[:20] = False  # 20 doğru çift yanlışa döner
    res = pairs_comparison(ref, arm, **BS)
    assert res["acc_ref"]["value"] == pytest.approx(0.90)
    assert res["acc"]["value"] == pytest.approx(0.70)
    assert res["dacc"]["value"] == pytest.approx(0.20)
    assert res["dacc"]["lo"] > 0
