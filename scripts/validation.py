"""Minimal çiftlerin insan doğrulaması (PROTOCOL §6.3).

    make   : kök listesi ve kademe başına eşit, rastgele %10'luk çift örneklemi için Excel'de
             açılabilen CSV'ler üretir.
    check  : doldurulmuş CSV'lerden hata oranlarını hesaplar (eşik: hatalı çift < %2).

Örnek:
    uv run python -m scripts.validation make
    uv run python -m scripts.validation check
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from src.utils.io import read_jsonl, read_yaml

PAIRS = Path("data/probes/tr_minimal_pairs.jsonl")
ROOTS = Path("data/lexicon/tr_roots.tsv")
OUT = Path("data/probes/validation")
PAIR_SHEET = OUT / "tr_pairs_sample.csv"
ROOT_SHEET = OUT / "tr_roots_review.csv"
FRACTION = 0.10
MAX_ERROR_RATE = 0.02

PAIR_COLS = ["id", "n_suffix", "category", "good", "bad", "fark", "good_dogru (E/H)",
             "bad_tek_uyum_hatasi (E/H)", "not"]
JUDGE_COLS = ["good_dogru (E/H)", "bad_tek_uyum_hatasi (E/H)"]
ROOT_COLS = ["root", "last_vowel_class", "n_syllables", "uygun (E/H)", "not"]


def _write_csv(path: Path, cols: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Excel Türkçe karakterleri doğru göstersin diye BOM ile yazılır.
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(cols)
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    # Excel bölge ayarına göre ";" ya da "," ile kaydedebilir; ayırıcı başlık satırından anlaşılır.
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    header = text.splitlines()[0]
    delimiter = ";" if header.count(";") >= header.count(",") else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def diff_label(good: str, bad: str) -> str:
    """Değişen harfi gösterir: `kitapl[a→e]r`."""
    for i, (a, b) in enumerate(zip(good, bad)):
        if a != b:
            start = good.rfind(" ", 0, i) + 1
            end = good.find(" ", i)
            word = good[start:end if end != -1 else len(good)].rstrip(".")
            j = i - start
            return f"{word[:j]}[{a}→{b}]{word[j + 1:]}"
    return ""


def make(seed: int) -> None:
    pairs = list(read_jsonl(PAIRS))
    by_level = defaultdict(list)
    for p in pairs:
        by_level[p["n_suffix"]].append(p)
    rng = random.Random(seed)
    sample = []
    for level in sorted(by_level):
        k = round(len(by_level[level]) * FRACTION)
        sample.extend(rng.sample(by_level[level], k))
    rng.shuffle(sample)
    _write_csv(PAIR_SHEET, PAIR_COLS,
               [[p["id"], p["n_suffix"], p["category"], p["good"], p["bad"],
                 diff_label(p["good"], p["bad"]), "", "", ""] for p in sample])

    with open(ROOTS, encoding="utf-8") as f:
        roots = list(csv.DictReader(f, delimiter="\t"))
    _write_csv(ROOT_SHEET, ROOT_COLS,
               [[r["root"], r["last_vowel_class"], r["n_syllables"], "", ""] for r in roots])
    print(f"{PAIR_SHEET.as_posix()}: {len(sample)} çift")
    print(f"{ROOT_SHEET.as_posix()}: {len(roots)} kök")


def _is_no(value: str) -> bool:
    return value.strip().upper() in {"H", "HAYIR", "N", "NO"}


def _is_filled(value: str) -> bool:
    return value.strip().upper() in {"E", "EVET", "Y", "YES", "H", "HAYIR", "N", "NO"}


def check() -> None:
    rows = _read_csv(PAIR_SHEET)
    cols = JUDGE_COLS
    empty = [r["id"] for r in rows if not all(_is_filled(r[c]) for c in cols)]
    bad = [r for r in rows if any(_is_no(r[c]) for c in cols)]
    rate = len(bad) / len(rows)
    print(f"Çift örneklemi: {len(rows)} satır, {len(empty)} boş/geçersiz, {len(bad)} hatalı "
          f"(%{100 * rate:.2f}; eşik %{100 * MAX_ERROR_RATE:.0f})")
    for r in bad:
        print(f"  {r['id']} [{r['category']}] {r['good']} | {r['bad']} | not: {r['not']}")

    roots = _read_csv(ROOT_SHEET)
    rejected = [r for r in roots if _is_no(r["uygun (E/H)"])]
    unfilled = [r for r in roots if not _is_filled(r["uygun (E/H)"])]
    print(f"Kök listesi: {len(roots)} kök, {len(unfilled)} boş, {len(rejected)} reddedildi")
    for r in rejected:
        print(f"  {r['root']} | not: {r['not']}")

    passed = not empty and not unfilled and rate < MAX_ERROR_RATE
    print("SONUÇ:", "GEÇTİ" if passed else "GEÇMEDİ")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["make", "check"])
    args = ap.parse_args()
    if args.command == "make":
        make(read_yaml("configs/eval.yaml")["seed"])
    else:
        check()


if __name__ == "__main__":
    main()
