# TR-QuantBench: Türkçe'de Nicemlemenin Morfolojik Etkisi

Düşük bitli nicemleme (INT8/INT4), dil modellerini küçültüp tüketici donanımında çalıştırmanın en yaygın yolu. Ancak nicemlemenin kalite etkisi çoğunlukla İngilizce üzerinde ölçülüyor. **TR-QuantBench**, bu kaybın Türkçe gibi eklemeli (sondan eklemeli) bir dilde İngilizce'ye kıyasla daha büyük olup olmadığını ve kaybın morfolojik karmaşıklıkla, yani bir kelimeye eklenen ek sayısıyla, ilişkili olup olmadığını ölçen yeniden üretilebilir bir değerlendirme çalışmasıdır.

## Araştırma soruları

| # | Soru |
|---|---|
| RQ1 | INT8 ve INT4 nicemleme Türkçe'de İngilizce'ye göre **orantısız** kalite kaybı yaratıyor mu? |
| RQ2 | Kayıp, **morfolojik karmaşıklıkla** (ek sayısı, kelime uzunluğu) ilişkili mi? |
| RQ3 | Kalibrasyon verisinin dili (TR/EN/karışık) GPTQ/AWQ sonucunu değiştiriyor mu? |
| RQ4 | Otomatik metrikler, **insan değerlendirmesine göre** kaybı küçümsüyor mu? |

**Hipotezler** sonuçlar görülmeden önce yazılmıştır ([ön kayıt](docs/PROTOCOL.md)) ve sonuç ne olursa olsun raporlanacaktır. "Fark yok" da geçerli bir bulgudur.

- **H1:** INT4'te Türkçe'nin göreli BPB artışı İngilizce'den büyüktür.
- **H2:** Minimal-çift doğruluk kaybı, ek sayısı arttıkça artar.
- **H3:** Türkçe kalibrasyon, Türkçe BPB kaybını İngilizce kalibrasyona göre azaltır.
- **H4:** İnsan değerlendirmesinde kayıp, otomatik metrikte görülenden büyüktür.

## Yöntem (özet)

**Modeller:** İki farklı aileden, Türkçe'yi makul düzeyde bilen küçük modeller. Seçim gerekçesi [`DECISIONS.md`](DECISIONS.md) D-010'da.

| Model | Lisans |
|---|---|
| [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | Apache-2.0 |
| [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | Llama 3.2 Community License |

**Nicemleme kolları:** BF16 referans; bitsandbytes INT8 ve NF4; GPTQ ve AWQ 4 bit. GPTQ/AWQ için kalibrasyon metni Türkçe, İngilizce ve karışık olmak üzere üç seçenekle ve üç tohumla denenir.

**Ölçümler:**
- **Bayt başına bit (BPB):** [FLORES-200](https://github.com/facebookresearch/flores) devtest'teki birbirinin çevirisi olan Türkçe ve İngilizce cümleler üzerinde. BPB token sayısına değil bayt sayısına göre normalize edildiği için tokenizer'ların iki dili farklı sayıda parçaya bölmesinden etkilenmez. Ana karşılaştırma, her dilin kendi referansına göre **göreli** BPB artışıdır.
- **Minimal çiftler:** Yalnızca tek bir ünlü uyumu ihlaliyle ayrılan cümle çiftleri. Örneğin *Dün kitaplarımızda…* ve *Dün kitapl**e**rımızda…*. Çiftler 1, 2, 3 ve 4 ekli kelimeler için ayrı ayrı üretilir. Model doğru cümleye daha yüksek olasılık veriyorsa çift doğru bilinmiş sayılır.
- **İnsan değerlendirmesi:** Referans ve 4 bitlik modelin cevapları, hangisinin hangisi olduğu gösterilmeden ikili karşılaştırılır.
- **İstatistik:** Bootstrap %95 güven aralıkları, fark-in-fark (TR − EN), ek sayısına göre regresyon.

Ayrıntılı protokol, metrik tanımları ve analiz planı: [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

## Kurulum ve yeniden üretim

Gereksinimler: Python 3.11, [uv](https://docs.astral.sh/uv/), CUDA destekli bir GPU (ölçümler 4 GB VRAM'li bir RTX 3050 Laptop GPU'da yapıldı). Llama için Hugging Face'te erişim onayı ve `hf auth login` gerekir.

```bash
uv sync                                          # ortamı kur
uv run pytest -q                                 # testler
uv run python -m scripts.prepare_flores          # FLORES-200'ü indir ve doğrula
uv run python -m scripts.run_eval --model qwen2.5-1.5b --arm fp16 --task flores_bpb
uv run python -m scripts.run_eval --model llama-3.2-1b --arm fp16 --task tr_pairs_proto
```

Her çalıştırma `results/raw/` altına cümle bazlı bir JSONL ve paket sürümleri, GPU, model revizyonu ve veri hash'lerini içeren bir `.meta.json` yazar. `make` kuruluysa aynı komutlar [`Makefile`](Makefile) hedefleriyle de çalıştırılabilir (`make test`, `make data`, `make eval-smoke`).

## Depo yapısı

```
configs/     model kimlikleri ve revizyonları, nicemleme kolları, değerlendirme ayarları
data/        kök sözlüğü, minimal çiftler, FLORES sürüm notu (FLORES metni repoda tutulmaz)
src/         veri hazırlama, minimal çift üreteci, NLL/BPB değerlendirmesi
scripts/     komut satırı giriş noktaları
tests/       birim testler (ünlü uyumu kuralları, NLL tutarlılığı)
results/     cümle bazlı sonuçlar ve çalıştırma meta verileri
docs/        araştırma protokolü
```

## Belgeler

- [`docs/PROTOCOL.md`](docs/PROTOCOL.md): Ayrıntılı araştırma protokolü ve analiz planı.
- [`DECISIONS.md`](DECISIONS.md): Protokolden sapmalar ve teknik kararlar, tarih ve gerekçeleriyle.
- [`DATA_LICENSES.md`](DATA_LICENSES.md): Veri ve model lisansları, atıflar.

## Lisans ve atıf

Kod [MIT](LICENSE) lisanslıdır. Veri ve model lisansları [`DATA_LICENSES.md`](DATA_LICENSES.md)'de. FLORES-200: CC-BY-SA 4.0, NLLB Team ve ark. (2022).
