"""Türkçe minimal çiftleri ve İngilizce kontrol setini üretir (PROTOCOL §6.3).

Çıktılar:
    data/probes/tr_minimal_pairs.jsonl
    data/probes/en_controls.jsonl
    data/probes/VERSION.md  (girdi ve çıktı hash'leri)
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data import en_controls
from src.data.tr_minimal_pairs import PAIRS_PER_LEVEL, generate, read_roots
from src.utils.env_info import file_sha256
from src.utils.io import read_yaml, write_jsonl

ROOTS = Path("data/lexicon/tr_roots.tsv")
OUT = Path("data/probes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-level", type=int, default=PAIRS_PER_LEVEL)
    args = ap.parse_args()
    seed = read_yaml("configs/eval.yaml")["seed"]

    roots = [r["root"] for r in read_roots(ROOTS)]
    tr_pairs = generate(roots, seed=seed, per_level=args.per_level)
    tr_path = OUT / "tr_minimal_pairs.jsonl"
    write_jsonl(tr_path, tr_pairs)

    raw = en_controls.download(OUT / "raw" / "blimp")
    en_pairs = en_controls.to_pairs(raw)
    en_path = OUT / "en_controls.jsonl"
    write_jsonl(en_path, en_pairs)

    lines = [
        "# Prob setleri sürüm notu",
        "",
        (f"- Üretim: `uv run python -m scripts.make_probes` (tohum {seed}, kademe başına "
         f"{args.per_level} çift)"),
        f"- Kök sözlüğü: `{ROOTS.as_posix()}` ({len(roots)} kök), SHA-256 `{file_sha256(ROOTS)}`",
        f"- `tr_minimal_pairs.jsonl`: {len(tr_pairs)} çift, SHA-256 `{file_sha256(tr_path)}`",
        f"- `en_controls.jsonl`: {len(en_pairs)} çift, SHA-256 `{file_sha256(en_path)}`",
        f"- BLiMP kaynağı: `alexwarstadt/blimp` commit `{en_controls.COMMIT}` (CC-BY 4.0)",
        "",
        "| BLiMP dosyası | SHA-256 |",
        "|---|---|",
        *[f"| `{name}.jsonl` | `{en_controls.sha256(p)}` |" for name, p in raw.items()],
        "",
    ]
    (OUT / "VERSION.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("TR:", len(tr_pairs), dict(sorted(Counter(p["n_suffix"] for p in tr_pairs).items())))
    print("TR kategori:", dict(Counter(p["category"] for p in tr_pairs)))
    print("EN:", len(en_pairs), dict(Counter(p["category"] for p in en_pairs)))


if __name__ == "__main__":
    main()
