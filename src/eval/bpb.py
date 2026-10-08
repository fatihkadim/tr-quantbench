"""Bayt başına bit (PROTOCOL §7.3): BPB = toplam_NLL_nat / (ln 2 × toplam_UTF8_bayt)."""

from __future__ import annotations

import math


def bpb(nll_nat: float, n_bytes: int) -> float:
    if n_bytes <= 0:
        raise ValueError("n_bytes pozitif olmalı")
    return nll_nat / (math.log(2) * n_bytes)


def corpus_bpb(rows: list[dict]) -> float:
    """Cümle bazlı satırlardan toplam BPB (oranların ortalaması değil, toplamların oranı)."""
    return bpb(sum(r["nll_nat"] for r in rows), sum(r["n_bytes"] for r in rows))
