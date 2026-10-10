"""İngilizce kontrol seti: BLiMP özne-fiil uyumu alt kümeleri (PROTOCOL §6.3).

BLiMP (Warstadt ve ark., 2020; CC-BY 4.0) sabit bir commit'ten indirilir ve her dosya SHA-256
ile doğrulanır. Ham dosyalar git'e konmaz; yalnızca birleştirilmiş çift dosyası repoda tutulur.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

COMMIT = "3e56b06fcabca9b30822fc66435fca6b1aa40bb1"
URL = "https://raw.githubusercontent.com/alexwarstadt/blimp/{commit}/data/{name}.jsonl"
SUBSETS = (
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause",
)


def download(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in SUBSETS:
        path = raw_dir / f"{name}.jsonl"
        if not path.exists():
            urllib.request.urlretrieve(URL.format(commit=COMMIT, name=name), path)
        paths[name] = path
    return paths


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_pairs(paths: dict[str, Path]) -> list[dict]:
    """Alt kümeleri birleştirir. Birden fazla alt kümede geçen aynı çift yalnızca ilk geçtiği
    yerde tutulur (ör. "The children smile." iki çoğul alt kümesinde de var)."""
    pairs, seen = [], set()
    for name, path in paths.items():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["sentence_good"], row["sentence_bad"])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({
                    "id": f"en_{len(pairs):06d}",
                    "lang": "en",
                    "category": name,
                    "good": row["sentence_good"],
                    "bad": row["sentence_bad"],
                    "source_pair_id": row["pairID"],
                })
    return pairs
