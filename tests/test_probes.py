"""Ünlü uyumu ve minimal çift testleri (README §6.3). Beklenen çıktılar elle yazılmıştır."""

import pytest

from src.data.tr_minimal_pairs import (
    CHAIN,
    ablative,
    build,
    dative,
    generate,
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


def test_generate_no_duplicates_and_single_diff():
    pairs = generate(["kitap", "ev", "göz", "kapı"], seed=0)
    assert len({(p["good"], p["bad"]) for p in pairs}) == len(pairs)
    for p in pairs:
        assert p["good"] != p["bad"]
        assert len(p["good"]) == len(p["bad"])
        assert sum(a != b for a, b in zip(p["good"], p["bad"])) == 1


def test_generate_is_deterministic():
    assert generate(["kitap", "ev"], seed=1) == generate(["kitap", "ev"], seed=1)
