"""Tekrar üretilebilirlik denetimi: iki sonuç dosyasını öğe öğe karşılaştırır.

Ölçülen alanlar (NLL, log-olasılıklar, token sayıları, doğru/yanlış) birebir aynı olmalıdır;
ortam bilgisi karşılaştırılmaz. Bir dizin verilirse `{ad}{ek}.jsonl` dosyaları `{ad}.jsonl` ile
eşleştirilir.

Örnek:
    uv run python -m scripts.compare_runs results/raw --suffix __rerun
    uv run python -m scripts.compare_runs a.jsonl b.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.io import read_jsonl

FIELDS = ("nll_nat", "n_tokens", "lp_good", "lp_bad", "n_tokens_good", "n_tokens_bad", "correct")


def compare(a: Path, b: Path) -> tuple[int, int, float]:
    """(öğe sayısı, farklı alan sayısı, en büyük mutlak sayısal fark)."""
    ra = {r["item_id"]: r for r in read_jsonl(a)}
    rb = {r["item_id"]: r for r in read_jsonl(b)}
    if set(ra) != set(rb):
        raise ValueError(f"{a} ile {b} öğe kümeleri farklı")
    n_diff, max_abs = 0, 0.0
    for i, x in ra.items():
        for k in FIELDS:
            if k in x and x[k] != rb[i][k]:
                n_diff += 1
                if isinstance(x[k], float):
                    max_abs = max(max_abs, abs(x[k] - rb[i][k]))
    return len(ra), n_diff, max_abs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="iki JSONL dosyası ya da bir dizin")
    ap.add_argument("--suffix", default="__rerun")
    args = ap.parse_args()

    if len(args.paths) == 2:
        pairs = [tuple(args.paths)]
    else:
        pairs = [(p.with_name(p.name.replace(f"{args.suffix}.jsonl", ".jsonl")), p)
                 for p in sorted(args.paths[0].glob(f"*{args.suffix}.jsonl"))]
    if not pairs:
        sys.exit("Karşılaştırılacak dosya bulunamadı")
    ok = True
    for a, b in pairs:
        n, n_diff, max_abs = compare(a, b)
        ok &= n_diff == 0
        print(f"{'AYNI ' if n_diff == 0 else 'FARKLI'} {a.name} ↔ {b.name}: {n} öğe, "
              f"{n_diff} farklı alan, en büyük fark {max_abs:g}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
