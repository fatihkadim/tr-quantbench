# TR-QuantBench: Araştırma Protokolü

Bu belge çalışmanın yöntemini tanımlar: araştırma soruları, önceden belirlenmiş hipotezler, modeller, nicemleme yapılandırmaları, veri, metrikler ve analiz planı.

**Ön kayıt.** Hipotezler ve analiz planı, nicemleme sonuçları elde edilmeden önce yazılmıştır (ilk sürüm: commit `8ecacb5`, 2026-10-08). Sonuçlar, ne çıkarsa çıksın bu plana göre raporlanır; "fark yok" da geçerli bir bulgudur. Plandan her sapma tarih ve gerekçesiyle [`DECISIONS.md`](../DECISIONS.md)'ye kaydedilir. Bu sürümde hipotez, metrik ve analiz tanımları değiştirilmemiş, yalnızca belgenin biçimi düzenlenmiştir.

Bölüm numaraları kod ve karar kayıtlarındaki atıflarla (ör. "§7.3") uyumludur.

---

## 1. Araştırma soruları ve hipotezler

| # | Soru |
|---|------|
| RQ1 | INT8 ve INT4 nicemleme Türkçe'de İngilizce'ye göre **orantısız** kalite kaybı yaratıyor mu? |
| RQ2 | Kayıp, **morfolojik karmaşıklıkla** (ek sayısı, kelime uzunluğu) ilişkili mi? |
| RQ3 | Kalibrasyon verisinin dili (TR/EN/karışık) GPTQ/AWQ sonucunu değiştiriyor mu? |
| RQ4 | Otomatik metrikler, **insan değerlendirmesine göre** kaybı küçümsüyor mu? |

**Hipotezler:**

- **H1:** INT4'te Türkçe'nin göreli BPB artışı İngilizce'den büyüktür.
- **H2:** Minimal-çift doğruluk kaybı, ek sayısı arttıkça artar.
- **H3:** Türkçe kalibrasyon, Türkçe BPB kaybını İngilizce kalibrasyona göre azaltır.
- **H4:** İnsan değerlendirmesinde kayıp, otomatik metrikte görülenden büyüktür.

Analiz seçenekleri hipotezleri desteklemek amacıyla sonradan değiştirilmez.

## 2. Kapsam

**Kapsam içi:** Eğitim sonrası nicemleme (PTQ) yöntemleri (bitsandbytes INT8/NF4, GPTQ, AWQ; isteğe bağlı olarak GGUF), iki model ailesi, 1.5–3B ölçeği, Türkçe ve İngilizce. (Seçilen Llama-3.2-1B ~1.2B parametreyle bu aralığın biraz altındadır; `DECISIONS.md` D-010.)

**Kapsam dışı:** Nicemleme farkındalıklı eğitim (QAT), ince ayar, 7B ve üzeri modeller, KV-cache nicemleme, hız/enerji optimizasyonu (ölçülürse yan veri olarak raporlanır).

## 3. Hesaplama ortamı

- Python 3.11, bağımlılıklar `uv.lock` ile sabitlenmiştir.
- Yerel ölçümler: NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), CUDA 12.8.
- GPTQ/AWQ nicemleme işleri bulut GPU'da (Google Colab) yapılır; nicemlenmiş ağırlıklar repoya konmaz, cümle bazlı sonuç dosyaları konur.
- Her çalıştırmanın tam ortam bilgisi sonuç dosyasıyla birlikte kaydedilir (§9).

## 4. Modeller

**Seçim ölçütü.** İki farklı aileden, Türkçe'yi makul düzeyde bilen küçük modeller. Taban etkisini önlemek için her aday, referans hassasiyette şu ölçütü sağlamalıdır: Türkçe minimal çift doğruluğu şans düzeyinin (%50) açıkça üstünde (≥ %60) ve Türkçe BPB sonlu ve makul. Ölçütü sağlamayan model elenir.

**Değerlendirilen adaylar:** Qwen2.5-1.5B/3B, Gemma-2-2B, Llama-3.2-1B/3B.

**Seçilen modeller** (gerekçe ve ölçüm sonuçları: `DECISIONS.md` D-010):

| Model | Revizyon | Lisans |
|---|---|---|
| `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` | Apache-2.0 |
| `meta-llama/Llama-3.2-1B` | `4e20de362430cd3b72f300e6b0f18e50e7166e08` | Llama 3.2 Community License |

## 5. Nicemleme yapılandırmaları

Her model için aşağıdaki "kollar" (arms) üretilir. Yapılandırmalar [`configs/arms.yaml`](../configs/arms.yaml)'da sabittir.

| Kol | Yöntem | Bit | Kalibrasyon |
|---|---|---|---|
| `fp16` | Referans (tüm modellerde BF16, `DECISIONS.md` D-003) | 16 | — |
| `bnb_int8` | bitsandbytes LLM.int8 | 8 | — |
| `bnb_nf4` | bitsandbytes NF4, çift nicemleme açık (sabit) | 4 | — |
| `gptq_w4_{cal}` | GPTQ, grup boyutu 128 (sabit) | 4 | `en`, `tr`, `mix` |
| `awq_w4_{cal}` | AWQ, grup boyutu 128 (sabit) | 4 | `en`, `tr`, `mix` |
| `gguf_*` (isteğe bağlı, ayrı iz) | llama.cpp Q8_0, Q5_K_M, Q4_K_M | 8/5/4 | — |

**Kalibrasyon:** 128 örnek, her biri 512 token, 3 tohum (0, 1, 2). `mix` = %50 TR + %50 EN. Kalibrasyon örnekleri tüm modellerde aynı **metin** parçalarından seçilir; tokenizer farkı kaçınılmazdır.

**GGUF izi ayrı raporlanır.** llama.cpp'nin log-olasılık hesabı Hugging Face yolundakiyle birebir aynı olmayabilir; iki iz aynı tabloda karıştırılmaz.

GPTQ/AWQ için kullanılan paketler ve seçim gerekçeleri `DECISIONS.md`'de belgelenir.

## 6. Veri

### 6.1 Paralel test seti: FLORES-200 (devtest)
- Türkçe (`tur_Latn`) ve İngilizce (`eng_Latn`) olmak üzere 1012'şer hizalı cümle.
- Kullanım: cümle başına NLL ve BPB.
- Kaynak, sürüm ve doğrulama hash'i: [`data/flores/VERSION.md`](../data/flores/VERSION.md). Lisans: CC-BY-SA 4.0.

### 6.2 Kalibrasyon metinleri
- Türkçe ve İngilizce Vikipedi'den (veya lisansı uygun başka derlemlerden) rastgele paragraflar.
- Test setiyle örtüşmez: normalleştirilmiş metin eşleşmesiyle cümle düzeyinde örtüşme kontrolü yapılır, örtüşenler çıkarılır.
- Örnekleme tohuma göre yapılır; seçilen örneklerin kimlikleri `data/calibration/{lang}_seed{n}.jsonl`'a kaydedilir.

### 6.3 Morfoloji probları: minimal çiftler

Türkçe minimal çiftler bu çalışma için kural tabanlı olarak üretilir ([`src/data/tr_minimal_pairs.py`](../src/data/tr_minimal_pairs.py)).

**Kök sözlüğü** (`data/lexicon/tr_roots.tsv`): `root`, `pos`, `last_vowel_class` (a/ı/o/u/e/i/ö/ü), `final_consonant_type` (sert/yumuşak/ünlü), `regular` sütunları.
- Yalnızca düzenli kökler. Ünlü uyumu istisnası olan yabancı kökenli kelimeler (saat, kalp, rol, harf türü) dışarıda bırakılır. Ünsüz yumuşaması gösteren kökler (kitap → kitabı) ya hariç tutulur ya da ayrıca etiketlenir.
- En az 150 kök; tek ve çok heceli, kalın ve ince ünlülü, yuvarlak ve düz ünlülü kökler dengeli dağıtılır.

**Ekler:**

| Ek | Notasyon | Kural |
|---|---|---|
| Çoğul | `-lAr` | Büyük ünlü uyumu (2'li) |
| Bulunma | `-DA` | 2'li uyum + sert ünsüzden sonra d→t |
| Ayrılma | `-DAn` | 2'li uyum + d→t |
| Yönelme | `-(y)A` | 2'li uyum, ünlüyle biten köklerde y kaynaştırma |
| İlgi | `-(n)In` | 4'lü uyum, ünlüyle biten köklerde n kaynaştırma |
| İyelik 1. çoğul | `-(I)mIz` | 4'lü uyum, ünlüyle biten köklerde ilk ünlü düşer |
| İyelik 1. tekil | `-(I)m` | 4'lü uyum |

**Karmaşıklık kademesi** (RQ2'nin ana değişkeni): her kök için 1, 2, 3 ve 4 ekli formlar.
- 1 ek: `kitap-lar`
- 2 ek: `kitap-lar-ımız`
- 3 ek: `kitap-lar-ımız-da`
- 4 ek: dilbilgisel olarak geçerli ve anlamca makul bir ek dizisi (ör. `-lAr-(I)mIz-DA-ki`).

**Yanlış varyant:** Doğru formdan tek bir ünlü uyumu ihlali üretilir: bir ekin ünlüsü kalın/ince karşıtıyla değiştirilir. Çiftin iki üyesi arasında başka hiçbir fark yoktur.

**Taşıyıcı cümleler:** 5–10 sabit cümle şablonu (ör. `Dün ___ gördüm.`). Her kademe için en az 1000 çift; yinelenen çift yoktur.

**İnsan doğrulaması:** Kök listesi ve rastgele seçilen %10'luk çift örneklemi Türkçe konuşan bir kişi tarafından doğrulanır. Hatalı çift oranı %2'nin altında olmalıdır; aksi halde üreteç düzeltilip doğrulama tekrarlanır. Ünlü uyumu kuralları en az 40 elle yazılmış beklenen çıktıyla birim testlerle sınanır ([`tests/test_probes.py`](../tests/test_probes.py)).

**Çıktı şeması** (`data/probes/tr_minimal_pairs.jsonl`):
```json
{"id": "tr_000123", "lang": "tr", "n_suffix": 3, "category": "plural+poss1pl+loc",
 "good": "Dün kitaplarımızda ...", "bad": "Dün kitaplerımızda ...",
 "root": "kitap", "template_id": 4}
```

**İngilizce kontrol seti:** Tercihen BLiMP'in özne-fiil uyumu alt kümeleri (lisansı uygunsa); değilse en az 500 çiftlik şablon tabanlı bir küme (ör. `The key to the cabinets is/are on the table.`).

**Sınırlılık:** Türkçe ve İngilizce probları farklı dilbilgisel olguları ölçer. Bu yüzden iki dil doğrudan karşılaştırılmaz; her dilin kendi referansına göre göreli düşüşü karşılaştırılır.

### 6.4 İnsan değerlendirmesi istemleri
- 100 Türkçe istem (bilgi, kısa yazı, akıl yürütme, günlük dil); isteğe bağlı 50 İngilizce istem.
- İstemler insanlar tarafından yazılır veya onaylanır.

---

## 7. Metrikler

### 7.1 Birincil
1. **Göreli BPB değişimi** (her dil için):
   `ΔBPB_rel = (BPB_kol − BPB_fp16) / BPB_fp16` (FLORES devtest, cümle bazında ve toplam).
2. **Minimal çift doğruluk düşüşü:**
   `Δacc = acc_fp16 − acc_kol`. Model `good` cümlesine `bad`'den daha yüksek toplam log-olasılık verirse çift doğru sayılır.

### 7.2 İkincil
- Fark-in-fark: `DiD = ΔBPB_rel(TR) − ΔBPB_rel(EN)`.
- Ek sayısına göre `Δacc` (RQ2).
- Fertility (token/kelime), kelime başına karakter, kelime sayısı.
- İsteğe bağlı: lm-evaluation-harness ile ek çoktan seçmeli görevler (Türkçe desteği doğrulanırsa).
- Yan veri: VRAM ve token/s.

### 7.3 Hesaplama ayrıntıları
- Her cümle ayrı ayrı değerlendirilir. Başa modelin BOS token'ı eklenir; BOS'u olmayan modellerde eşdeğer belge başı token'ı kullanılır (`DECISIONS.md` D-007). Kural tüm kollar ve modeller için aynıdır.
- NLL: tüm cümle token'larının negatif log-olasılık toplamı (nat).
- `BPB = toplam_NLL_nat / (ln(2) × toplam_UTF8_bayt)`.
- Minimal çiftler: toplam log-olasılık karşılaştırması (birincil). Uzunluk-normalleştirilmiş sürüm yalnızca ikincil analizde.
- Sayısal tutarlılık: değerlendirme gradyansız ve sabit batch boyutuyla yapılır; dolgunun sonucu değiştirmediği birim testle doğrulanır (aynı cümle tek başına ve batch içinde aynı NLL'i vermelidir). Uygulamada batch boyutu 1'dir (`DECISIONS.md` D-009).

### 7.4 İstatistik
- Tüm birincil ölçümler için bootstrap %95 güven aralığı (10.000 yeniden örnekleme, cümle/çift düzeyinde).
- Kalibrasyonlu yöntemlerde 3 tohum: ortalama ± standart sapma ve bireysel değerler.
- Regresyon (RQ2): cümle bazında `ΔNLL_bayt-normalize ~ n_words + fertility + mean_word_len [+ n_suffix (TR, üretilmiş formlarda)]`. Katsayılar ve güven aralıkları raporlanır.
- Çoklu karşılaştırmada (çok sayıda kol × model) Holm düzeltmesi uygulanır veya birincil karşılaştırmalar baştan sınırlandırılır; seçilen yol raporda belirtilir.

---

## 8. İnsan değerlendirmesi (RQ4)

- **Karşılaştırma:** `fp16` ile `bnb_nf4` ve/veya en iyi 4 bit kol (seçim ve gerekçesi `DECISIONS.md`'de).
- **Üretim:** Her istem için greedy decoding (sıcaklık 0), `max_new_tokens=150`; iki kol aynı ayarlarla.
- **Kör ikili karşılaştırma:** A/B sırası rastgeledir; değerlendiriciler hangi cevabın hangi kola ait olduğunu bilmez. Eşleştirme anahtarı ayrı saklanır ve değerlendiricilerle paylaşılmaz.
- **Form sütunları:** `item_id`, `prompt`, `answer_A`, `answer_B`, `preference` (A/B/eşit), `note`.
- **Değerlendiriciler:** Türkçe'yi ana dili düzeyinde bilen en az 2, tercihen 3 kişi.
- **Rapor:** Kazanma/beraberlik/kaybetme oranları, bootstrap güven aralıkları, değerlendiriciler arası uyum (iki kişi için Cohen kappa, üç ve üzeri için Fleiss kappa).

---

## 9. Sonuç kayıtları

Her çalıştırma cümle (veya çift) bazında bir sonuç dosyası üretir: `results/raw/{model}__{arm}__{task}.jsonl`.

```json
{"model_id": "...", "model_rev": "...", "arm": "bnb_nf4", "calib": null, "seed": null,
 "task": "flores_bpb", "lang": "tr", "item_id": "flores_tr_0042",
 "nll_nat": 52.31, "n_bytes": 180, "n_tokens": 61, "n_words": 24,
 "env": {"torch": "...", "transformers": "...", "cuda": "..."}}
```

Minimal çiftler için ek alanlar: `lp_good`, `lp_bad`, `correct`, `n_suffix`, `category`.

Her sonuç dosyasının yanında bir `.meta.json` bulunur: paket sürümleri, GPU adı, CUDA sürümü, model revizyonu, yapılandırma dosyalarının ve veri dosyalarının SHA-256 hash'leri, çalışma süresi ve tepe VRAM.

---

## 10. Analiz planı

Her hipotez için sayısal kanıta dayanan bir karar kaydedilir: **destekledi / desteklemedi / belirsiz**.

**Planlanan şekiller:**
1. Bit derinliğine göre ΔBPB_rel (TR ve EN), güven aralıklarıyla.
2. Ek sayısına göre Δacc eğrisi (TR), kollara göre.
3. Kalibrasyon dili × yöntem ısı haritası (TR ve EN için ayrı).
4. Regresyon katsayıları (orman grafiği).
5. İnsan değerlendirmesi kazanma/beraberlik/kaybetme oranları.

Rapordaki her sayı bir sonuç dosyasına izlenebilir olmalı; tüm tablo ve şekiller sonuç dosyalarından betikle yeniden üretilebilir olmalıdır.

---

## 11. Geçerlilik tehditleri ve önlemler

| Tehdit | Önlem |
|---|---|
| Model Türkçe'de zaten zayıf (taban etkisi) | Model seçim ölçütü (§4) |
| Referans doğruluğu tavana yakın (tavan etkisi) | Kademe bazlı analiz; gerekirse daha zor çift türleri (`DECISIONS.md`'de belgelenir) |
| Hatalı "yanlış" cümleler | İstisna kökleri hariç tutma, insan doğrulaması, birim testler |
| Token bazlı metriklerin dilleri yanıltması | Bayt bazlı BPB; fertility regresyonda kontrol değişkeni |
| GGUF ve HF sonuçlarının karışması | Ayrı iz, ayrı tablo |
| TR ve EN problarının farklı olguları ölçmesi | Her dilin kendi referansına göre göreli düşüş |
| Küçük model ölçeği | Bulgular 1–1.5B ölçeği için geçerlidir; daha büyük modellere genelleme yapılmaz |
