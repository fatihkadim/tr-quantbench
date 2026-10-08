"""FLORES-200 devtest yükleyici (PROTOCOL §6.1).

Kaynak: Meta'nın resmi arşivi (CC-BY-SA 4.0). Arşiv SHA-256 ile doğrulanır; ham ve işlenmiş
veri git'e konmaz, yalnızca bu betik ve `data/flores/VERSION.md` repoda tutulur.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

from src.utils.io import write_jsonl

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
LANGS = {"tr": "tur_Latn", "en": "eng_Latn"}
SPLIT = "devtest"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "flores200_dataset.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    digest = _sha256(archive)
    if digest != SHA256:
        raise RuntimeError(f"FLORES arşiv hash'i beklenenden farklı: {digest}")
    return archive


def read_split(archive: Path, flores_code: str, split: str = SPLIT) -> list[str]:
    member = f"./flores200_dataset/{split}/{flores_code}.{split}"
    with tarfile.open(archive, "r:gz") as tar:
        f = tar.extractfile(member)
        if f is None:
            raise FileNotFoundError(member)
        text = f.read().decode("utf-8")
    return [line.rstrip("\r") for line in text.split("\n") if line.strip()]


def to_rows(sentences: list[str], lang: str) -> list[dict]:
    return [
        {
            "item_id": f"flores_{lang}_{i:04d}",
            "lang": lang,
            "text": s,
            "n_bytes": len(s.encode("utf-8")),
            "n_words": len(s.split()),
        }
        for i, s in enumerate(sentences)
    ]


def prepare(data_dir: Path = Path("data/flores")) -> dict[str, Path]:
    """Arşivi indirir/doğrular ve `processed/devtest_{lang}.jsonl` dosyalarını yazar."""
    archive = download(data_dir / "raw")
    out: dict[str, Path] = {}
    sentences = {lang: read_split(archive, code) for lang, code in LANGS.items()}
    if len({len(v) for v in sentences.values()}) != 1:
        raise RuntimeError("TR ve EN devtest cümle sayıları eşleşmiyor")
    for lang, sents in sentences.items():
        path = data_dir / "processed" / f"{SPLIT}_{lang}.jsonl"
        write_jsonl(path, to_rows(sents, lang))
        out[lang] = path
    return out
