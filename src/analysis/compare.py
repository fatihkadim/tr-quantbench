"""Bir kolun referansa (fp16) göre birincil metrikleri (PROTOCOL §7.1–7.2), bootstrap GA'lı.

- FLORES: kol BPB'si, `ΔBPB_rel = (BPB_kol − BPB_fp16) / BPB_fp16` (her dil için, toplamların
  oranı) ve `DiD = ΔBPB_rel(TR) − ΔBPB_rel(EN)`.
- Minimal çiftler: kol doğruluğu ve `Δacc = acc_fp16 − acc_kol`.
"""

from __future__ import annotations

import math

import numpy as np

from src.analysis.bootstrap import Stat, bootstrap

LANGS = ("tr", "en")


def _bpb(nll: np.ndarray, n_bytes: np.ndarray) -> Stat:
    return lambda idx: nll[idx].sum(axis=1) / (math.log(2) * n_bytes[idx].sum(axis=1))


def _rel(arm: Stat, ref: Stat) -> Stat:
    def f(idx):
        r = ref(idx)
        return (arm(idx) - r) / r
    return f


def flores_comparison(ref: dict[str, dict[str, np.ndarray]],
                      arm: dict[str, dict[str, np.ndarray]],
                      n_resamples: int, seed: int, ci: float) -> dict:
    """`ref[lang]` / `arm[lang]`: `nll` ve `n_bytes` dizileri, aynı cümle sırasıyla.

    FLORES paralel olduğundan TR ve EN dizileri de indeksle hizalıdır; aynı indeksler iki dile
    birlikte uygulanır.
    """
    n = len(ref["tr"]["nll"])
    for d in (ref, arm):
        assert all(len(d[lang]["nll"]) == n for lang in LANGS)
    stats: dict[str, Stat] = {}
    for lang in LANGS:
        b_ref = _bpb(ref[lang]["nll"], ref[lang]["n_bytes"])
        b_arm = _bpb(arm[lang]["nll"], arm[lang]["n_bytes"])
        stats[f"bpb_ref_{lang}"] = b_ref
        stats[f"bpb_{lang}"] = b_arm
        stats[f"dbpb_rel_{lang}"] = _rel(b_arm, b_ref)
    d_tr, d_en = stats["dbpb_rel_tr"], stats["dbpb_rel_en"]
    stats["did"] = lambda idx: d_tr(idx) - d_en(idx)
    return bootstrap(stats, n, n_resamples, seed, ci)


def pairs_comparison(correct_ref: np.ndarray, correct_arm: np.ndarray,
                     n_resamples: int, seed: int, ci: float) -> dict:
    """Aynı çift sırasıyla hizalanmış doğru/yanlış dizileri üzerinden doğruluk ve Δacc."""
    ref = correct_ref.astype(float)
    arm = correct_arm.astype(float)
    assert len(ref) == len(arm)
    stats: dict[str, Stat] = {
        "acc_ref": lambda idx: ref[idx].mean(axis=1),
        "acc": lambda idx: arm[idx].mean(axis=1),
        "dacc": lambda idx: ref[idx].mean(axis=1) - arm[idx].mean(axis=1),
    }
    return bootstrap(stats, len(ref), n_resamples, seed, ci)
