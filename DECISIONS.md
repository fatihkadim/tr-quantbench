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
- **Karar:** Sistemde PATH'te `make` yok; `C:\msys64\usr\bin\make.exe` mevcut ve Makefile onunla çalışır.

### D-006: Aday model durumu (bilgi amaçlı, seçim değil)
- 2026-10-08'de HF API ile doğrulandı: Qwen2.5-1.5B (apache-2.0, açık), Qwen2.5-3B (Qwen Research lisansı, ticari değil), gemma-2-2b (erişim onayı gerekli), Llama-3.2-1B/3B (erişim onayı gerekli). Revizyonlar `configs/models.yaml`'da.
- Nihai seçim Faz 1 kapısından sonra kullanıcı onayıyla yapılacak.
