# Kararlar

Protokolden ([`docs/PROTOCOL.md`](docs/PROTOCOL.md)) sapmalar ve protokolün açık bıraktığı teknik seçimler. Her kayıt tarihli; biçim: **Karar**, **Neden**, **Etki**.

## 2026-10-08

### D-001: Python 3.11 ve uv
- **Karar:** `requires-python = ">=3.11,<3.12"`; bağımlılıklar `uv` ile yönetilir ve `uv.lock` ile sabitlenir.
- **Neden:** Bazı nicemleme paketleri Python 3.12+ sürümlerinde henüz sorunlu.

### D-002: PyTorch CUDA 12.8 indeksi
- **Karar:** `torch`, Windows ve Linux'ta `https://download.pytorch.org/whl/cu128` indeksinden kurulur.
- **Neden:** Ölçüm donanımının sürücüsü CUDA 12.8; PyPI'deki Windows tekerleği yalnızca CPU destekli.

### D-003: Referans veri tipi BF16
- **Karar:** `fp16` kolu tüm modellerde **bfloat16** ile çalıştırılır (`configs/arms.yaml: reference_dtype`). Kol adı protokolle uyum için `fp16` olarak kalır.
- **Neden:** Protokol FP16 ile BF16 arasında tutarlı bir seçim istiyor. BF16'nın dinamik aralığı geniş olduğundan bazı modellerde FP16'da görülen taşma riskini taşımaz; ölçüm GPU'su (Ampere) BF16'yı donanımsal olarak destekler.

### D-004: Paket yerleşimi
- **Karar:** `src/` doğrudan Python paketi olarak kullanılır (`from src.eval import nll`); betikler `python -m scripts.<ad>` ile çalışır.

### D-005: Aday modeller
- **Karar:** Adaylar Hugging Face API'si üzerinden doğrulandı ve revizyonları `configs/models.yaml`'a sabitlendi: Qwen2.5-1.5B (Apache-2.0), Qwen2.5-3B (Qwen Research License, ticari değil), Gemma-2-2B (erişim onayı gerekli), Llama-3.2-1B/3B (erişim onayı gerekli).

### D-006: FLORES-200 kaynağı
- **Karar:** Veri, Meta'nın resmi arşivinden (`dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`) alınır ve SHA-256 ile doğrulanır.
- **Neden:** Hugging Face'teki `facebook/flores` ve `openlanguagedata/flores_plus` erişim onayı istiyor; `Muennighoff/flores200` yalnızca bir yükleme betiği içeriyor ve güncel `datasets` sürümü betik tabanlı veri setlerini desteklemiyor. Resmi arşiv kimlik doğrulama gerektirmez ve sürümü sabittir.
- **Etki:** FLORES+ (düzeltilmiş sürüm) yerine orijinal FLORES-200 kullanılır. Lisans aynıdır (CC-BY-SA 4.0).

### D-007: BOS'u olmayan modellerde önek token'ı
- **Karar:** Her cümlenin başına modelin BOS token'ı eklenir (Llama-3.2: `<|begin_of_text|>`). BOS'u tanımsız modellerde (Qwen2.5: `bos_token = None`) EOS token'ı (`<|endoftext|>`) önek olarak kullanılır. Önek skorlanmaz; cümlenin tüm token'ları skorlanır.
- **Neden:** PROTOCOL §7.3 başa BOS eklenmesini ve ilk token'ın da skorlanmasını istiyor. Qwen2.5 ön eğitiminde belgeler `<|endoftext|>` ile ayrıldığından bu token doğal bir belge başı bağlamıdır. Kural tüm kollarda aynıdır.

### D-008: Model seçimi için minimal çift prototipi
- **Karar:** Model seçim ölçütü (PROTOCOL §4), 32 köklü bir prototip sözlük (`data/lexicon/tr_roots_prototype.tsv`, 8 ünlünün her biri için 4 kök) ve `-lAr-(I)mIz-DA-ki` ek zinciriyle üretilen 384 çiftle ölçülür.
- **Neden:** Seçim ölçütü minimal çift doğruluğu gerektiriyor; tam sözlük ve insan doğrulaması bu aşamada henüz hazır değildi. Çoğul eki her zaman ilk sırada olduğundan ünsüz yumuşaması tetiklenmez.
- **Etki:** Prototip sonuçları yalnızca model seçimi için kullanılır; asıl analizler insan doğrulamasından geçmiş tam setle yapılır.

### D-009: Değerlendirmede batch boyutu 1
- **Karar:** `configs/eval.yaml: batch_size = 1`; dolgu kullanılmaz.
- **Neden:** Qwen2.5-1.5B (BF16) üzerinde 16 FLORES TR cümlesiyle ölçüldü: batch 4 ve 8, batch 1'e göre cümle NLL'sinde ortalama %0.2, en fazla %0.6 göreli fark veriyor. Bu, ölçülmek istenen nicemleme etkileriyle aynı büyüklükte. Batch 1'de iki çalıştırma bit düzeyinde aynı sonucu verdi. Dolgu mantığının kendisi float32 küçük bir modelde birim testle doğrulandı (`tests/test_nll.py`, fark < 1e-4).
- **Etki:** Değerlendirme daha yavaş ama belirlenimli. Tam sözlük üzerindeki float32 log-softmax satır satır alınır (batch 8'de tepe VRAM 4.26 → 3.23 GB).

## 2026-10-09

### D-010: Seçilen modeller
- **Karar:** `Qwen/Qwen2.5-1.5B` ve `meta-llama/Llama-3.2-1B`. Diğer adaylar seçilmedi.
- **Seçim ölçümleri** (BF16; FLORES-200 devtest, 1012'şer cümle; 384 prototip çift; `results/raw/`):

  | Model | TR BPB | EN BPB | TR token/kelime | EN token/kelime | Çift doğruluğu |
  |---|---|---|---|---|---|
  | Qwen2.5-1.5B | 1.578 | 0.999 | 2.55 | 1.26 | %98.4 |
  | Llama-3.2-1B | 1.402 | 0.991 | 2.16 | 1.24 | %98.7 |

- **Neden:** İki farklı aile; ikisi de seçim ölçütünü (≥ %60) geçti; ikisinin de BF16 referansı tek bir 4 GB GPU'da çalışıyor (tepe VRAM 3.08 / 2.45 GB).
- **Kapsam sapması:** Llama-3.2-1B ~1.2B parametreyle PROTOCOL §2'deki 1.5–3B aralığının biraz altında. Bulgular 1–1.5B ölçeği için geçerli olacak; raporda sınırlılık olarak belirtilecek.
- **Tavan riski:** Referans çift doğruluğu tavana yakın. Nicemleme düşüşleri yine ölçülebilir, ancak ek sayısı kademeleri arasındaki farkı görmek zorlaşabilir. Tam çift setinde daha zor çift türleri değerlendirilecek; eklenirse ayrıca kaydedilecek.
