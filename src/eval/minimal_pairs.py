"""Minimal çift doğruluğu (README §7.1): good cümlesine bad'den yüksek toplam log-olasılık."""

from __future__ import annotations

from src.eval.nll import score_texts


def score_pairs(model, tokenizer, pairs: list[dict], batch_size: int = 8) -> list[dict]:
    good = score_texts(model, tokenizer, [p["good"] for p in pairs], batch_size)
    bad = score_texts(model, tokenizer, [p["bad"] for p in pairs], batch_size)
    rows = []
    for p, g, b in zip(pairs, good, bad):
        lp_good, lp_bad = -g.nll_nat, -b.nll_nat
        rows.append({
            "item_id": p["id"],
            "lang": p["lang"],
            "n_suffix": p.get("n_suffix"),
            "category": p.get("category"),
            "lp_good": lp_good,
            "lp_bad": lp_bad,
            "n_tokens_good": g.n_tokens,
            "n_tokens_bad": b.n_tokens,
            "correct": lp_good > lp_bad,
        })
    return rows


def accuracy(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows)
