"""Türkçe minimal çift üreteci (PROTOCOL §6.3).

Her kademe (1–4 ek) için birkaç ek dizisi tanımlıdır (`CATEGORIES`); taşıyıcı cümle dizinin son
ekinin işlevine göre seçilir (`TEMPLATES`).

Yanlış varyant: uyuma giren eklerden tam BİR tanesinin ünlüsü kalın↔ince karşıtıyla değiştirilir
(a↔e, ı↔i, u↔ü). Yuvarlaklık ve diğer tüm harfler korunur; -ki uyuma girmediği için değişmez.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

BACK = set("aıou")
FRONT = set("eiöü")
ROUNDED = set("ouöü")
VOWELS = BACK | FRONT
VOICELESS = set("fstkçşhp")  # "fıstıkçı şahap": -DA bunlardan sonra -tA olur
FLIP = {"a": "e", "e": "a", "ı": "i", "i": "ı", "u": "ü", "ü": "u"}


def last_vowel(word: str) -> str:
    for ch in reversed(word.lower()):
        if ch in VOWELS:
            return ch
    raise ValueError(f"Ünlü yok: {word!r}")


def two_way(word: str) -> str:
    """Büyük ünlü uyumu, 2'li (A): a/e."""
    return "a" if last_vowel(word) in BACK else "e"


def four_way(word: str) -> str:
    """4'lü uyum (I): ı/i/u/ü."""
    v = last_vowel(word)
    if v in BACK:
        return "u" if v in ROUNDED else "ı"
    return "ü" if v in ROUNDED else "i"


def ends_with_vowel(word: str) -> bool:
    return word[-1].lower() in VOWELS


def d_initial(word: str) -> str:
    return "t" if word[-1].lower() in VOICELESS else "d"


# --- Ekler: (stem) -> ek metni ------------------------------------------------------------

def plural(stem: str) -> str:            # -lAr
    return "l" + two_way(stem) + "r"


def locative(stem: str) -> str:          # -DA
    return d_initial(stem) + two_way(stem)


def ablative(stem: str) -> str:          # -DAn
    return d_initial(stem) + two_way(stem) + "n"


def dative(stem: str) -> str:            # -(y)A
    return ("y" if ends_with_vowel(stem) else "") + two_way(stem)


def genitive(stem: str) -> str:          # -(n)In
    return ("n" if ends_with_vowel(stem) else "") + four_way(stem) + "n"


def poss_1pl(stem: str) -> str:          # -(I)mIz
    i = four_way(stem)
    return (i + "m" + i + "z") if not ends_with_vowel(stem) else ("m" + i + "z")


def poss_1sg(stem: str) -> str:          # -(I)m
    return "m" if ends_with_vowel(stem) else four_way(stem) + "m"


def rel_ki(stem: str) -> str:            # -ki (uyuma girmez)
    return "ki"


@dataclass(frozen=True)
class Suffix:
    name: str
    fn: callable
    harmonizes: bool = True


CHAIN = (
    Suffix("plural", plural),
    Suffix("poss1pl", poss_1pl),
    Suffix("loc", locative),
    Suffix("ki", rel_ki, harmonizes=False),
)


def build(root: str, suffixes: tuple[Suffix, ...]) -> list[str]:
    """Kök + eklerin parçalarını döndürür (birleştirince doğru form)."""
    parts = [root]
    for s in suffixes:
        parts.append(s.fn("".join(parts)))
    return parts


def violate(parts: list[str], suffixes: tuple[Suffix, ...], which: int) -> list[str]:
    """`which` indeksli ekin uyuma giren ilk ünlüsünü kalın↔ince karşıtıyla değiştirir."""
    if not suffixes[which].harmonizes:
        raise ValueError(f"{suffixes[which].name} uyuma girmez")
    seg = parts[which + 1]
    for j, ch in enumerate(seg):
        if ch in FLIP:
            new = parts.copy()
            new[which + 1] = seg[:j] + FLIP[ch] + seg[j + 1 :]
            return new
    raise ValueError(f"Ekte değiştirilecek ünlü yok: {seg!r}")


# --- Ek dizileri (kademe başına) ------------------------------------------------------------
# Her dizi çoğul ya da ünsüzle başlayan bir hal ekiyle başlar; böylece kök sonu ünsüz yumuşaması
# ve ünlü düşmesi (kitap → kitabı, şehir → şehri) hiçbir formda tetiklenmez.

SUFFIXES = {
    "plural": Suffix("plural", plural),
    "poss1pl": Suffix("poss1pl", poss_1pl),
    "loc": Suffix("loc", locative),
    "abl": Suffix("abl", ablative),
    "dat": Suffix("dat", dative),
    "gen": Suffix("gen", genitive),
    "ki": Suffix("ki", rel_ki, harmonizes=False),
}

CATEGORIES: dict[int, list[tuple[str, ...]]] = {
    1: [("plural",), ("loc",), ("abl",)],
    2: [("plural", "poss1pl"), ("plural", "loc"), ("plural", "abl"), ("plural", "dat"),
        ("plural", "gen")],
    3: [("plural", "poss1pl", "loc"), ("plural", "poss1pl", "abl"), ("plural", "poss1pl", "dat"),
        ("plural", "poss1pl", "gen")],
    4: [("plural", "poss1pl", "loc", "ki"), ("plural", "poss1pl", "gen", "ki")],
}

# Taşıyıcı cümleler, dizinin SON ekinin dilbilgisel işlevine göre seçilir.
TEMPLATES: dict[str, list[str]] = {
    "plural": ["Dün {} gördüm.", "Burada {} var.", "Masada yeni {} duruyordu.",
               "Pazarda ucuz {} satılıyordu.", "Kitapta eski {} anlatılıyordu."],
    "poss1pl": ["{} çok eski.", "Sanırım {} kayboldu.", "Annem {} hakkında sordu.",
                "{} hâlâ burada duruyor.", "Dün {} kontrol edildi."],
    "loc": ["Sorun {} değildi.", "Her şey {} saklıydı.", "Bu iz {} kalmış.",
            "Anahtarı {} bulduk.", "Bütün gün {} bekledik."],
    "abl": ["Bu ses {} geliyordu.", "Herkes {} söz ediyordu.", "Çocuk {} korkuyordu.",
            "Haber {} geldi.", "Kimse {} hoşlanmadı."],
    "dat": ["Herkes {} baktı.", "Çocuklar {} doğru koştu.", "Annem {} dokunmadı.",
            "Bunu {} ekledik.", "Kimse {} inanmadı."],
    "gen": ["{} rengi değişti.", "{} sahibi gelmedi.", "{} fiyatı arttı.",
            "Dün {} resmini çektim.", "{} sayısı azaldı."],
    "ki": ["{} herkesi şaşırttı.", "Bence {} daha güzeldi.", "{} daha yeniydi.",
           "{} bizimkine benziyordu.", "Asıl önemli olan {} değildi."],
}

PAIRS_PER_LEVEL = 1200


def read_roots(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="	"))
    return [r for r in rows if r.get("regular", "true").lower() == "true"]


def generate(roots: list[str], seed: int = 0, per_level: int = PAIRS_PER_LEVEL) -> list[dict]:
    """Kademe başına `per_level` çift üretir, kategoriler arasında eşit paylaştırarak.

    Her kategori için tüm kök × şablon birleşimleri tohuma göre karıştırılır ve ilk N tanesi
    alınır; ihlal edilecek ek de tohuma göre, uyuma giren ekler arasından seçilir.
    """
    rng = random.Random(seed)
    pairs, seen = [], set()
    for level, cats in CATEGORIES.items():
        if per_level % len(cats):
            raise ValueError(
                f"per_level={per_level}, {level}. kademenin {len(cats)} kategorisine bölünmüyor")
        quota = per_level // len(cats)
        for cat in cats:
            chain = tuple(SUFFIXES[name] for name in cat)
            harmonizing = [i for i, sfx in enumerate(chain) if sfx.harmonizes]
            templates = TEMPLATES[cat[-1]]
            combos = [(r, t) for r in roots for t in range(len(templates))]
            rng.shuffle(combos)
            taken = 0
            for root, t_id in combos:
                if taken == quota:
                    break
                parts = build(root, chain)
                which = rng.choice(harmonizing)
                good = _fill(templates[t_id], "".join(parts))
                bad = _fill(templates[t_id], "".join(violate(parts, chain, which)))
                if (good, bad) in seen:
                    continue
                seen.add((good, bad))
                taken += 1
                pairs.append({
                    "id": f"tr_{len(pairs):06d}",
                    "lang": "tr",
                    "n_suffix": level,
                    "category": "+".join(cat),
                    "violated": chain[which].name,
                    "violated_pos": which + 1,
                    "good": good,
                    "bad": bad,
                    "root": root,
                    "template_id": f"{cat[-1]}_{t_id}",
                })
            if taken < quota:
                raise RuntimeError(f"{cat}: yalnızca {taken}/{quota} benzersiz çift üretilebildi")
    return pairs


# --- Prototip (model seçimi için; DECISIONS.md D-008) ---------------------------------------
# Yalnızca `data/probes/tr_minimal_pairs_prototype.jsonl` dosyasının yeniden üretilebilmesi için
# korunur.

TEMPLATES_PROTO: dict[int, list[str]] = {
    1: ["Dün {} gördüm.", "Burada {} var.", "Masada yeni {} duruyordu."],
    2: ["{} çok eski.", "Annem {} hakkında sordu.", "Sanırım {} kayboldu."],
    3: ["Sorun {} değildi.", "Her şey {} saklıydı.", "Bu iz {} kalmış."],
    4: ["{} herkesi şaşırttı.", "Bence {} daha güzeldi.", "{} daha yeniydi."],
}


def generate_prototype(roots: list[str], seed: int = 0, levels=(1, 2, 3, 4)) -> list[dict]:
    """Her kök × kademe × şablon için bir çift (`-lAr-(I)mIz-DA-ki` zinciri)."""
    rng = random.Random(seed)
    pairs, seen = [], set()
    for level in levels:
        chain = CHAIN[:level]
        harmonizing = [i for i, s in enumerate(chain) if s.harmonizes]
        category = "+".join(s.name for s in chain)
        for root in roots:
            parts = build(root, chain)
            for t_id, template in enumerate(TEMPLATES_PROTO[level]):
                which = rng.choice(harmonizing)
                bad_parts = violate(parts, chain, which)
                good = _fill(template, "".join(parts))
                bad = _fill(template, "".join(bad_parts))
                if (good, bad) in seen:
                    continue
                seen.add((good, bad))
                pairs.append({
                    "id": f"tr_{len(pairs):06d}",
                    "lang": "tr",
                    "n_suffix": level,
                    "category": category,
                    "violated": chain[which].name,
                    "good": good,
                    "bad": bad,
                    "root": root,
                    "template_id": t_id,
                })
    return pairs


def _fill(template: str, word: str) -> str:
    if template.startswith("{}"):
        word = word[0].upper() + word[1:] if word[0] != "i" else "İ" + word[1:]
    return template.format(word)
