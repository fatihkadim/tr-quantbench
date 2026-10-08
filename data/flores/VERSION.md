# FLORES-200 sürüm notu

- Kaynak: https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz (Meta, resmi arşiv)
- Arşiv SHA-256: `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`
- Kullanılan bölüm: `devtest`, `tur_Latn` ve `eng_Latn` (1012'şer hizalı cümle)
- Lisans: CC-BY-SA 4.0 (bkz. `DATA_LICENSES.md`)
- Üretim: `uv run python -m scripts.prepare_flores` → `processed/devtest_{tr,en}.jsonl` (git dışı)
