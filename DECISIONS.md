# Kararlar ve ikameler

Her kayıt tarihli. Biçim: **Karar**, **Neden**, **Etki**.

## 2026-10-08: Faz 0

### D-001: Python 3.11 ve uv
- **Karar:** `requires-python = ">=3.11,<3.12"`, ortam `uv` ile yönetiliyor (`.python-version` = 3.11).
- **Neden:** README §3; bazı nicemleme paketleri 3.13'te sorunlu.

### D-002: PyTorch CUDA 12.8 indeksi
- **Karar:** `torch`, `https://download.pytorch.org/whl/cu128` indeksinden kuruluyor (Windows/Linux).
- **Neden:** Yerel sürücü CUDA 12.8 (RTX 3050 Laptop, 4 GB). PyPI'deki Windows tekerleği yalnızca CPU.

### D-003: Referans veri tipi BF16
- **Karar:** `fp16` kolu tüm modellerde **bfloat16** ile çalıştırılır (`configs/arms.yaml: reference_dtype`).
- **Neden:** Gemma-2 FP16'da taşma/sayısal sorunlarıyla bilinir; RTX 3050 (Ampere) ve Colab T4 dışı GPU'lar BF16 destekler. Kol adı README ile uyum için `fp16` olarak kalır.
- **Açık nokta:** Colab T4 BF16'yı donanımsal desteklemez; 3B referanslar için L4/A100 gerekebilir. Faz 3'te yeniden değerlendirilecek.

### D-004: Paket yerleşimi
- **Karar:** README §9'daki `src/` yapısı doğrudan Python paketi olarak kullanılır (`from src.eval import nll`). Betikler `python -m scripts.<ad>` ile çalışır.

### D-005: Windows'ta make
- **Karar:** Sistemde PATH'te `make` yok; `C:\msys64\usr\bin\make.exe` mevcut. (Faz 1'de geçersiz kaldı, bkz. D-011.)

### D-006: Aday model durumu (bilgi amaçlı, seçim değil)
- 2026-10-08'de HF API ile doğrulandı: Qwen2.5-1.5B (apache-2.0, açık), Qwen2.5-3B (Qwen Research lisansı, ticari değil), gemma-2-2b (erişim onayı gerekli), Llama-3.2-1B/3B (erişim onayı gerekli). Revizyonlar `configs/models.yaml`'da.
- Nihai seçim Faz 1 kapısından sonra kullanıcı onayıyla yapılacak.

## 2026-10-08: Faz 1

### D-007: FLORES-200 kaynağı
- **Karar:** Veri, Meta'nın resmi arşivinden (`dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`) alınır ve SHA-256 ile doğrulanır.
- **Neden:** HF'deki `facebook/flores` ve `openlanguagedata/flores_plus` erişim onayı ve token istiyor; `Muennighoff/flores200` yalnızca bir yükleme betiği içeriyor ve güncel `datasets` sürümü betik tabanlı veri setlerini desteklemiyor. Resmi arşiv kimlik doğrulama gerektirmez ve sürümü sabittir.
- **Etki:** FLORES+ (düzeltilmiş sürüm) yerine orijinal FLORES-200 kullanılır. Lisans aynıdır (CC-BY-SA 4.0).

### D-008: BOS'u olmayan modellerde önek token'ı
- **Karar:** Her cümlenin başına modelin BOS token'ı eklenir; BOS'u tanımsız modellerde (Qwen2.5: `bos_token = None`) EOS token'ı (`<|endoftext|>`) önek olarak kullanılır. Önek skorlanmaz; cümlenin tüm token'ları skorlanır.
- **Neden:** README §7.3 başa BOS eklenmesini ve ilk token'ın da skorlanmasını istiyor; Qwen2.5 ön eğitiminde belgeler `<|endoftext|>` ile ayrıldığı için bu token doğal bir belge başı bağlamıdır. Kural tüm kollarda aynıdır.

### D-009: Model seçim kapısı için minimal çift prototipi
- **Karar:** Faz 1 kapısı, 32 köklü prototip sözlük (`data/lexicon/tr_roots_prototype.tsv`, 8 ünlünün her biri için 4 kök) ve `-lAr-(I)mIz-DA-ki` ek zinciriyle üretilen 384 çiftle ölçülür.
- **Neden:** README §10, Faz 1'de "Faz 2'nin küçük sürümü" ile kapı ölçümü istiyor. Çoğul eki hep ilk sırada olduğundan ünsüz yumuşaması tetiklenmez. Kök listesi ve şablonlar Faz 2'de kullanıcı doğrulamasından geçecek; prototip sonuçları yalnızca kapı içindir.

### D-010: Değerlendirmede batch boyutu 1
- **Karar:** `configs/eval.yaml: batch_size = 1`; dolgu hiç kullanılmaz.
- **Neden:** Qwen2.5-1.5B (BF16, RTX 3050) üzerinde 16 FLORES TR cümlesiyle ölçüldü: batch 4 ve 8, batch 1'e göre cümle NLL'sinde ortalama %0.2, en fazla %0.6 göreli fark veriyor. Bu, ölçmek istediğimiz nicemleme etkileriyle aynı büyüklükte. Batch 1'de iki çalıştırma bit düzeyinde aynı sonucu verdi. Dolgu mantığının kendisi float32 küçük modelde birim testle doğrulandı (`tests/test_nll.py`, fark < 1e-4).
- **Etki:** Değerlendirme daha yavaş ama belirlenimli. Ayrıca float32 log-softmax satır satır alınıyor (batch 8'de tepe VRAM 4.26 → 3.23 GB).

### D-011: MSYS make kullanılmaz
- **Karar:** Makefile Windows'ta yerel bir GNU make ile çalıştırılır (öneri: `winget install ezwinports.make`). MSYS make kullanılmaz.
- **Neden:** `C:\msys64\usr\bin\make.exe` Git Bash'ten çağrıldığında alt süreçlere yalnızca 7 ortam değişkeni aktarıyor (`USERNAME`, `USERPROFILE` vb. yok). torch ve huggingface_hub ev dizinini bulamayıp çöküyor. PowerShell'den çağrıldığında da kendi kabuk araçlarını bulamıyor. Faz 0'daki `make test` geçmişti çünkü pytest bu değişkenlere ihtiyaç duymuyordu.
- **Etki:** Yerel make kurulana kadar hedeflerin karşılığı olan `uv run ...` komutları doğrudan çalıştırılır.

### D-012: Faz 1 kapı sonucu, Qwen2.5-1.5B (ölçüldü)
- FLORES-200 devtest, BF16, batch 1: TR BPB 1.578, EN BPB 0.999 (1012'şer cümle); fertility TR 2.55, EN 1.26 token/kelime.
- Prototip minimal çiftler (384 çift): doğruluk %98.4 (kademe 1–4: %100 / %96.9 / %100 / %96.9).
- Kapı ölçütü (≥ %60) sağlandı. Sonuç dosyaları: `results/raw/qwen2.5-1.5b__fp16__*.jsonl`.
- **Not (tavan riski):** FP16 doğruluğu tavana yakın. Nicemleme düşüşleri yine ölçülebilir, ancak kademeler arası fark görmek zorlaşabilir. Faz 2'de daha zor çiftler (ör. ihlalin kök yerine iç eklere dağılması, daha uzun zincirler) değerlendirilecek.

### D-013: Nihai modeller (kullanıcı onayı, 2026-10-09)
- **Karar:** `Qwen/Qwen2.5-1.5B` ve `meta-llama/Llama-3.2-1B`. Diğer adaylar `not_selected`.
- **Kapı sonuçları (ölçüldü, FLORES devtest + 384 prototip çift):**

  | Model | TR BPB | EN BPB | TR fertility | Çift doğruluğu |
  |---|---|---|---|---|
  | Qwen2.5-1.5B | 1.578 | 0.999 | 2.55 | %98.4 |
  | Llama-3.2-1B | 1.402 | 0.991 | 2.16 | %98.7 |

- **Neden:** İki farklı aile; ikisi de kapıyı geçti; ikisinin de BF16 referansı yerelde çalışıyor (tepe VRAM 3.08 / 2.45 GB), Colab gerekmez.
- **Kapsam sapması:** Llama-3.2-1B ~1.2B parametreyle README §2'deki 1.5–3B aralığının biraz altında. Kullanıcı yerel çalışabilirliği tercih etti. Raporda sınırlılık olarak belirtilecek: bulgular 1–1.5B ölçeği içindir.
- **Önek:** Llama'nın gerçek BOS'u (`<|begin_of_text|>`) kullanılır; Qwen'de D-008 geçerli.
