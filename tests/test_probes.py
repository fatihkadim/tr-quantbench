"""Ünlü uyumu ve minimal çift testleri (PROTOCOL §6.3). Beklenen çıktılar elle yazılmıştır."""

import pytest

from src.data.tr_minimal_pairs import (
    CATEGORIES,
    CHAIN,
    SUFFIXES,
    TEMPLATES,
    ablative,
    build,
    dative,
    generate,
    generate_prototype,
    genitive,
    locative,
    plural,
    poss_1pl,
    poss_1sg,
    violate,
)

CASES = [
    # -lAr
    (plural, "masa", "masalar"), (plural, "ev", "evler"), (plural, "göz", "gözler"),
    (plural, "kız", "kızlar"), (plural, "okul", "okullar"), (plural, "gül", "güller"),
    (plural, "köprü", "köprüler"), (plural, "kapı", "kapılar"),
    # -(I)mIz
    (poss_1pl, "kitaplar", "kitaplarımız"), (poss_1pl, "evler", "evlerimiz"),
    (poss_1pl, "okul", "okulumuz"), (poss_1pl, "göz", "gözümüz"), (poss_1pl, "kedi", "kedimiz"),
    (poss_1pl, "kutu", "kutumuz"), (poss_1pl, "ütü", "ütümüz"), (poss_1pl, "kapı", "kapımız"),
    # -DA
    (locative, "kitap", "kitapta"), (locative, "ev", "evde"), (locative, "göz", "gözde"),
    (locative, "kuş", "kuşta"), (locative, "masa", "masada"), (locative, "çiçek", "çiçekte"),
    (locative, "kitaplarımız", "kitaplarımızda"),
    # -DAn
    (ablative, "kitap", "kitaptan"), (ablative, "ev", "evden"), (ablative, "ağaç", "ağaçtan"),
    (ablative, "okul", "okuldan"),
    # -(y)A
    (dative, "masa", "masaya"), (dative, "ev", "eve"), (dative, "kedi", "kediye"),
    (dative, "okul", "okula"),
    # -(n)In
    (genitive, "masa", "masanın"), (genitive, "ev", "evin"), (genitive, "okul", "okulun"),
    (genitive, "göz", "gözün"), (genitive, "kapı", "kapının"), (genitive, "köprü", "köprünün"),
    (genitive, "kız", "kızın"), (genitive, "kutu", "kutunun"),
    # -(I)m
    (poss_1sg, "masa", "masam"), (poss_1sg, "ev", "evim"), (poss_1sg, "okul", "okulum"),
    (poss_1sg, "göz", "gözüm"), (poss_1sg, "kız", "kızım"), (poss_1sg, "ütü", "ütüm"),
]


@pytest.mark.parametrize("fn,stem,expected", CASES)
def test_suffix(fn, stem, expected):
    assert stem + fn(stem) == expected


@pytest.mark.parametrize("root,expected", [
    ("kitap", "kitaplarımızdaki"), ("ev", "evlerimizdeki"), ("göz", "gözlerimizdeki"),
    ("okul", "okullarımızdaki"), ("köprü", "köprülerimizdeki"), ("kuş", "kuşlarımızdaki"),
])
def test_full_chain(root, expected):
    assert "".join(build(root, CHAIN)) == expected


@pytest.mark.parametrize("which,expected", [
    (0, "kitaplerımızda"), (1, "kitaplarimızda"), (2, "kitaplarımızde"),
])
def test_violation_changes_one_vowel(which, expected):
    chain = CHAIN[:3]
    good = "".join(build("kitap", chain))
    bad = "".join(violate(build("kitap", chain), chain, which))
    assert bad == expected
    assert len(good) == len(bad)
    assert sum(a != b for a, b in zip(good, bad)) == 1


def test_ki_does_not_harmonize():
    with pytest.raises(ValueError):
        violate(build("ev", CHAIN), CHAIN, 3)


@pytest.mark.parametrize("root,cat,expected", [
    ("kitap", ("abl",), "kitaptan"),
    ("ev", ("loc",), "evde"),
    ("masa", ("abl",), "masadan"),
    ("kitap", ("plural", "dat"), "kitaplara"),
    ("ev", ("plural", "gen"), "evlerin"),
    ("göz", ("plural", "abl"), "gözlerden"),
    ("kuş", ("plural", "poss1pl", "dat"), "kuşlarımıza"),
    ("köprü", ("plural", "poss1pl", "gen"), "köprülerimizin"),
    ("okul", ("plural", "poss1pl", "abl"), "okullarımızdan"),
    ("kitap", ("plural", "poss1pl", "gen", "ki"), "kitaplarımızınki"),
    ("ev", ("plural", "poss1pl", "gen", "ki"), "evlerimizinki"),
    ("kapı", ("plural", "poss1pl", "loc", "ki"), "kapılarımızdaki"),
])
def test_categories(root, cat, expected):
    assert "".join(build(root, tuple(SUFFIXES[n] for n in cat))) == expected


def test_every_category_has_templates():
    for cats in CATEGORIES.values():
        for cat in cats:
            assert len(TEMPLATES[cat[-1]]) >= 5
            assert all(t.count("{}") == 1 for t in TEMPLATES[cat[-1]])


ROOTS = ["kitap", "ev", "göz", "kapı", "okul", "köprü", "masa", "kuş", "deniz", "yol"]


def test_generate_no_duplicates_and_single_diff():
    pairs = generate(ROOTS, seed=0, per_level=60)
    assert len({(p["good"], p["bad"]) for p in pairs}) == len(pairs)
    for level in CATEGORIES:
        assert sum(p["n_suffix"] == level for p in pairs) == 60
    for p in pairs:
        assert p["good"] != p["bad"]
        assert len(p["good"]) == len(p["bad"])
        assert sum(a != b for a, b in zip(p["good"], p["bad"])) == 1
        assert p["violated"] != "ki"


def test_generate_is_deterministic():
    assert generate(ROOTS, seed=1, per_level=60) == generate(ROOTS, seed=1, per_level=60)


def test_generate_raises_when_quota_impossible():
    with pytest.raises(RuntimeError):
        generate(["kitap"], seed=0, per_level=1200)
    with pytest.raises(ValueError):
        generate(ROOTS, seed=0, per_level=24)


def test_prototype_file_is_reproducible():
    import json
    from pathlib import Path

    from src.data.tr_minimal_pairs import read_roots

    root = Path(__file__).resolve().parents[1]
    roots = [r["root"] for r in read_roots(root / "data/lexicon/tr_roots_prototype.tsv")]
    with open(root / "data/probes/tr_minimal_pairs_prototype.jsonl", encoding="utf-8") as f:
        stored = [json.loads(line) for line in f]
    assert generate_prototype(roots, seed=0) == stored
