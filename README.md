# TR-QuantBench: Türkçe'de Nicemlemenin Morfolojik Etkisi

Düşük bitli nicemlemenin (INT8/INT4) Türkçe gibi eklemeli bir dilde, İngilizce'ye kıyasla farklı oranda kalite kaybına yol açıp açmadığını ve bu kaybın morfolojik karmaşıklıkla ilişkisini ölçen yeniden üretilebilir bir değerlendirme çalışması.

> **Bu README bir ajan tarafından uygulanacak şekilde yazılmıştır.** Bölüm 0'daki kurallara uy, fazları sırayla ilerle, her faz sonundaki kabul ölçütleri sağlanmadan sonrakine geçme.

---

## 0. Ajan için çalışma kuralları (ZORUNLU)

1. **Sonuç uydurma.** Ölçülmemiş hiçbir sayı rapora, README'ye veya tablolara yazılmaz. Bir deney çalışmadıysa "çalıştırılamadı" yaz.
2. **Sürümleri doğrula.** Kütüphane, model ve veri seti adlarını varsaymadan önce gerçekten var olduklarını ve kurulabildiklerini kontrol et. Bu belgedeki model ve paket adları **adaydır**, kesin değildir.
3. **Sessiz ikame yok.** Bir bileşen kullanılamıyorsa (paket kurulmuyor, model yok, lisans uygun değil), alternatifini seç ve nedenini `DECISIONS.md`'ye tarihle yaz. Kullanıcıya bildir.
4. **Faz kapıları.** Her fazın "Kabul ölçütleri" bölümü sağlanmadan sonraki faza geçme. Sağlanmıyorsa nedenini `DECISIONS.md`'ye yaz ve dur.
5. **Önce duman testi.** Her yeni ölçüm hattı, ilk önce çok küçük bir girdiyle (10 cümle) uçtan uca çalıştırılır.
6. **Tekrar üretilebilirlik.** Her çalıştırma: yapılandırma dosyası (YAML), rastgele tohum, paket sürümleri, model revizyon kimliği (hash) ve veri sürümü ile birlikte kaydedilir (Bölüm 9).
7. **Donanım gerçeği.** Yerelde 4 GB VRAM var (Bölüm 3). Sığmayan işleri otomatik olarak çalıştırmaya çalışma. Colab/WSL2 gerektirenleri ayrı betik/notebook olarak hazırla ve kullanıcıya hangisinin nerede koşturulacağını söyle.
8. **Kullanıcı kararları gerektiren noktalar:** (a) model seçimi, (b) insan değerlendirmesi yapacak kişiler, (c) bütçe aşan Colab kullanımı. Bunlarda dur ve sor.
9. **Küçük, geri alınabilir değişiklikler.** Her faz için ayrı git commit'i. Büyük dosyaları (model ağırlıkları, ham veri dökümleri) git'e koyma.
10. **Lisanslar.** Kullanılan her veri setinin lisansını `DATA_LICENSES.md`'ye yaz. Lisansı belirsizse kullanma, kullanıcıya sor.

---

## 1. Araştırma soruları ve hipotezler

| # | Soru |
|---|------|
| RQ1 | INT8 ve INT4 nicemleme Türkçe'de İngilizce'ye göre **orantısız** kalite kaybı yaratıyor mu? |
| RQ2 | Kayıp, **morfolojik karmaşıklıkla** (ek sayısı, kelime uzunluğu) ilişkili mi? |
| RQ3 | Kalibrasyon verisinin dili (TR/EN/karışık) GPTQ/AWQ sonucunu değiştiriyor mu? |
| RQ4 | Otomatik metrikler, **insan değerlendirmesine göre** kaybı küçümsüyor mu? |

**Hipotezler (önceden belirlenmiştir, sonuç ne olursa olsun raporlanır):**

- **H1:** INT4'te Türkçe'nin göreli BPB artışı İngilizce'den büyüktür.
- **H2:** Minimal-çift doğruluk kaybı, ek sayısı arttıkça artar.
- **H3:** Türkçe kalibrasyon, Türkçe BPB kaybını İngilizce kalibrasyona göre azaltır.
- **H4:** İnsan değerlendirmesinde kayıp, otomatik metrikte görülenden büyüktür.

**Not:** "Fark yok" sonucu da geçerli bir bulgudur. Hipotezleri desteklemek için analiz seçeneklerini sonradan değiştirme.

## 2. Kapsam

**Kapsam içi:** PTQ yöntemleri (bitsandbytes INT8/NF4, GPTQ, AWQ; opsiyonel olarak GGUF), iki model ailesi, 1.5–3B ölçeği, Türkçe ve İngilizce.

**Kapsam dışı:** QAT, ince ayar (fine-tuning), 7B+ modeller, KV-cache nicemleme, hız/enerji optimizasyonu (ölçülürse yan veri olarak, hedef değil).

---

## 3. Ortam ve donanım

| Öğe | Değer |
|---|---|
| Yerel makine | Windows 11, RTX 3050 Laptop GPU (4 GB VRAM), CUDA 12.8 |
| Python | 3.11 (3.13'te bazı paketler sorun çıkarabilir; 3.11 öncelikli) |
| Paket yönetimi | `uv` |
| Bulut | Google Colab (FP16 3B referansları, AWQ/GPTQ nicemleme) |
| Linux gerektirenler | WSL2 veya Colab |

**Yerelde yapılacaklar:** veri hazırlama, minimal çift üretimi, 1.5B INT4/INT8 modellerin değerlendirmesi, analiz.
**Colab'da yapılacaklar:** FP16 3B referans ölçümleri, GPTQ/AWQ nicemleme işleri, kalibrasyon taramaları.
**Aktarım:** Colab çıktıları (nicemlenmiş model ağırlıkları veya sonuç JSONL'leri) Hugging Face özel deposu veya Google Drive üzerinden taşınır. Ağırlıklar repoya konmaz; sonuç JSONL dosyaları konur.

---

## 4. Modeller (ADAY, doğrulanmalı)

İki farklı aileden, 1.5–3B ölçeğinde, Türkçe'yi makul bilen modeller:

| Aday | Not |
|---|---|
| Qwen2.5-1.5B (ve/veya 3B) | Çok dilli; sürüm ve lisans doğrulanmalı. |
| Gemma-2-2B (veya güncel Gemma küçük modeli) | Lisans koşullarını (erişim onayı gerekebilir) kontrol et. |
| Llama-3.2-1B/3B | Alternatif. Lisans koşullarını kontrol et. |

**Model seçim kapısı (Faz 1):** Seçilen her model için FP16 ölçümleri şu ölçütü sağlamalı: Türkçe minimal-çift doğruluğu **şans seviyesinin (%50) açıkça üstünde** (ör. ≥ %60) ve Türkçe BPB ölçülebilir ve makul. Sağlamıyorsa model elenir ve `DECISIONS.md`'ye yazılır (taban etkisi riski). **Nihai iki modeli kullanıcı onaylar.**

Her model için kaydedilecek: Hugging Face model kimliği, revizyon (commit hash), tokenizer sürümü, lisans.

---

## 5. Nicemleme yapılandırmaları

Her model için aşağıdaki "kollar" (arms) üretilir:

| Kol ID | Yöntem | Bit | Kalibrasyon | Nerede |
|---|---|---|---|---|
| `fp16` | Referans (FP16 veya BF16; tutarlı biri seç) | 16 | yok | Colab (3B) / yerel (1.5B sığarsa) |
| `bnb_int8` | bitsandbytes LLM.int8 | 8 | yok | yerel |
| `bnb_nf4` | bitsandbytes NF4 (double quant açık/kapalı sabit tut) | 4 | yok | yerel |
| `gptq_w4_{cal}` | GPTQ, 4 bit, grup boyutu 128 (sabit) | 4 | `en`, `tr`, `mix` | Colab |
| `awq_w4_{cal}` | AWQ, 4 bit, grup boyutu 128 (sabit) | 4 | `en`, `tr`, `mix` | Colab |
| `gguf_*` (opsiyonel, ayrı iz) | llama.cpp Q8_0, Q5_K_M, Q4_K_M | 8/5/4 | yok | yerel |

**Kalibrasyon ayarları:** 128 örnek, her biri 512 token, **3 farklı tohum** (örn. 0, 1, 2). `mix` = %50 TR + %50 EN.

**Kurallar:**
- Aynı kalibrasyon örnekleri tüm modellerde aynı **metin** parçalarından (tohuma göre) seçilir; tokenizer farkı kaçınılmazdır.
- GPTQ/AWQ için güncel ve bakımı sürdürülen paketi seç (ör. GPTQModel, AutoAWQ, llm-compressor adayları). **Seçimi ve nedeni `DECISIONS.md`'ye yaz.** Paket kurulmazsa alternatif kullan.
- **GGUF izi ayrı raporlanır.** HF yolundaki log-olasılık hesabıyla GGUF'in hesabı birebir aynı olmayabilir. İki iz tablolarda karıştırılmaz.

---

## 6. Veri

### 6.1 Paralel test seti: FLORES-200 (devtest)
- TR ve EN karşılıklı cümleler (yaklaşık 1000 cümle).
- Kullanım: cümle başına NLL ve BPB.
- Lisansı doğrula ve `DATA_LICENSES.md`'ye yaz. Kullanılamazsa alternatif paralel set bul (ör. TED veya Tatoeba alt kümeleri) ve `DECISIONS.md`'ye yaz.

### 6.2 Kalibrasyon metinleri
- Türkçe ve İngilizce Vikipedi dökümlerinden (veya lisansı uygun başka derlemlerden) rastgele paragraflar.
- **Test setiyle (FLORES) örtüşmemeli.** Cümle düzeyinde örtüşme kontrolü yap (normalleştirilmiş metin eşleşmesi) ve örtüşenleri çıkar.
- Tohuma göre örnekleme; seçilen örneklerin kimlikleri `data/calibration/{lang}_seed{n}.jsonl`'a kaydedilir.

### 6.3 Morfoloji probları: minimal çiftler (kendi üretimimiz)

**Türkçe üreteç (`src/data/tr_minimal_pairs.py`):**

*Girdi:* El ile doğrulanmış kök listesi (`data/lexicon/tr_roots.tsv`). Sütunlar: `root`, `pos`, `last_vowel_class` (a/ı/o/u/e/i/ö/ü), `final_consonant_type` (sert/yumuşak/ünlü), `regular` (true/false).

*Kök seçimi kuralları:*
- Yalnızca **düzenli** kökler (`regular=true`). Ünlü uyumu istisnası olan yabancı kökenli kelimeleri (ör. saat, kalp, rol, harf türü) dışarıda bırak. Ünsüz yumuşaması (kitap → kitabı) gösteren kökleri ya hariç tut ya da ayrı etiketle.
- En az 150 kök; tek ve çok heceli; kalın ve ince ünlülü; yuvarlak ve düz ünlülü dengeli.
- **Kullanıcıdan (Türkçe konuşur biri) kök listesinin ve ilk 10%'luk çift örnekleminin doğrulanmasını iste.** Doğrulanmadan Faz 2'ye geçme.

*Ek şablonları:*

| Ek | Notasyon | Kural |
|---|---|---|
| Çoğul | `-lAr` | Büyük ünlü uyumu (2'li) |
| Bulunma | `-DA` | 2'li uyum + sert ünsüzden sonra d→t |
| Ayrılma | `-DAn` | 2'li uyum + d→t |
| Yönelme | `-(y)A` | 2'li uyum, ünlüyle biten köklerde y kaynaştırma |
| İlgi | `-(n)In` | 4'lü uyum, ünlüyle biten köklerde n kaynaştırma |
| İyelik 1. çoğul | `-(I)mIz` | 4'lü uyum, ünlüyle biten köklerde ilk ünlü düşer |
| İyelik 1. tekil | `-(I)m` | 4'lü uyum |

*Karmaşıklık kademesi (RQ2'nin ana değişkeni):* Her kök için **1, 2, 3 ve 4 ekli** formlar:
- 1 ek: `kitap-lar`
- 2 ek: `kitap-lar-ımız`
- 3 ek: `kitap-lar-ımız-da`
- 4 ek: ek yığılması için geçerli, anlamca makul bir dizi (örn. `-lAr-(I)mIz-DA-ki` biçiminde; üreteç geçerli dizileri kural tablosundan seçer). 4 ekli formların dilbilgisel geçerliliğini örneklemle doğrulat.

*Yanlış varyant üretimi:* Doğru formdan **tek bir ünlü uyumu ihlali** üret (ekin ünlüsünü karşıt sınıfla değiştir). Aynı çiftte başka bir fark olmamalı.

*Taşıyıcı cümleler:* Sabit 5–10 taşıyıcı cümle şablonu (ör. `Dün ___ gördüm.`). Hedef kelime şablona yerleştirilir. Her kademe için ≥ 1000 çift hedefle (kök × şablon karışımıyla), tekrar eden çift olmasın.

*Çıktı şeması (`data/probes/tr_minimal_pairs.jsonl`):*
```json
{"id": "tr_000123", "lang": "tr", "n_suffix": 3, "category": "case+poss+plural",
 "good": "Dün kitaplarımızda ...", "bad": "Dün kitaplerımızda ...",
 "root": "kitap", "template_id": 4}
```

**İngilizce kontrol seti (`src/data/en_controls.py`):**
- Tercihen BLiMP'ten özne-fiil uyumu alt kümeleri (lisans/erişim doğrulanırsa). Olmazsa, 500+ çiftlik kendi şablon üretecini yaz (ör. `The key to the cabinets is/are on the table.`).
- **Sınırlılık notu (README ve raporda yer alacak):** Türkçe ve İngilizce probları farklı dilbilgisel olguları ölçer; doğrudan karşılaştırma yerine her dilin kendi FP16 referansına göre göreli düşüş karşılaştırılır.

**Taban kontrolü:** `tests/test_probes.py`; ünlü uyumu fonksiyonunun birim testleri (en az 40 elle yazılmış beklenen çıktı) içermeli.

### 6.4 İnsan değerlendirmesi istemleri
- `data/human_eval/prompts_tr.jsonl`: **100 Türkçe istem**, çeşitli (bilgi, kısa yazı, akıl yürütme, günlük dil). Kullanıcı yazar veya onaylar.
- Opsiyonel: 50 İngilizce istem (karşılaştırma için).
- Ajan istemleri **uydurmaz**; kullanıcıdan alır veya taslak üretip onaya sunar.

---

## 7. Metrikler

### 7.1 Birincil
1. **Göreli BPB değişimi** (her dil için):
   `ΔBPB_rel = (BPB_kol − BPB_fp16) / BPB_fp16` (FLORES devtest, cümle bazında ve toplam).
2. **Minimal-çift doğruluk düşüşü:**
   `Δacc = acc_fp16 − acc_kol`, doğruluk = model `good` cümlesine `bad`'den daha yüksek toplam log-olasılık verirse doğru.

### 7.2 İkincil
- Fark-in-fark: `DiD = ΔBPB_rel(TR) − ΔBPB_rel(EN)`.
- Ek sayısına göre `Δacc` (RQ2).
- Fertility (token/kelime), kelime başına karakter, kelime sayısı.
- Opsiyonel: lm-evaluation-harness ile ek çoktan seçmeli görevler (varlığı ve Türkçe dil desteği doğrulanırsa).
- Yan veri (hedef değil): VRAM ve tokens/s.

### 7.3 Hesaplama ayrıntıları
- Her cümle ayrı ayrı değerlendirilir. Başa BOS token eklenir (modelin gerektirdiği şekilde); tüm kollar ve modeller için aynı kural uygulanır.
- NLL: tüm cümle token'larının negatif log-olasılık toplamı (nat).
- `BPB = toplam_NLL_nat / (ln(2) × toplam_UTF8_bayt)`.
- Minimal çiftler: **toplam** log-olasılık karşılaştırması (birincil). Uzunluk-normalleştirilmiş sürüm yalnızca ikincil analizde.
- Sayısal tutarlılık: değerlendirmede `torch.no_grad()`, sabit batch boyutu, padding'in sonuçları değiştirmediğini birim testle doğrula (aynı cümle tek tek ve batch içinde aynı NLL'i vermeli, küçük sayısal toleransla).

### 7.4 İstatistik
- Tüm birincil ölçümler için **bootstrap %95 GA** (10.000 yeniden örnekleme, cümle/çift düzeyinde).
- Kalibrasyonlu yöntemler için 3 tohum: ortalama ± standart sapma ve bireysel değerler tablo olarak.
- Regresyon (RQ2): cümle bazında `ΔNLL_bayt-normalize ~ n_words + fertility + mean_word_len [+ n_suffix (TR, üretilmiş formlarda)]`. Katsayılar ve GA'lar raporlanır.
- Çoklu karşılaştırma varsa (çok sayıda kol × model), düzeltme yöntemini (ör. Holm) raporda belirt veya birincil karşılaştırmaları baştan sınırla.

---

## 8. İnsan değerlendirmesi (RQ4)

- **Karşılaştırma:** `fp16` vs `bnb_nf4` ve/veya en iyi 4 bit kol (Faz 5 sonuçlarına göre; hangi kolun seçildiği ve nedeni `DECISIONS.md`'de).
- **Üretim:** Her istem için greedy (sıcaklık 0) decoding, `max_new_tokens=150`; her iki kol aynı ayarlarla.
- **Kör ikili karşılaştırma:** A/B sırası rastgele; hangi çıktının hangi kola ait olduğu değerlendiricilere gösterilmez.
- **Araç:** `scripts/make_human_eval_sheet.py` bir CSV (veya basit bir yerel web formu) üretir. Sütunlar: `item_id`, `prompt`, `answer_A`, `answer_B`, `preference` (A/B/eşit), `note`. Anahtar eşleştirme dosyası ayrı saklanır (`human_eval/key.json`, değerlendiricilere verilmez).
- **Değerlendiriciler:** Türkçe ana dil seviyesinde ≥ 2 kişi, tercihen 3. Değerlendirici sayısını kullanıcı belirler.
- **Rapor:** kazanma/beraberlik/kaybetme oranları, bootstrap GA, değerlendiriciler arası uyum (Cohen kappa ikili için; 3+ için Fleiss kappa).

---

## 9. Depo yapısı

```
tr-quantbench/
├── README.md
├── DECISIONS.md              # tüm karar ve ikameler (tarihli)
├── DATA_LICENSES.md
├── pyproject.toml            # uv ile yönetilir
├── Makefile
├── configs/
│   ├── models.yaml           # model kimlikleri, revizyonlar
│   ├── arms.yaml             # nicemleme kolları
│   └── eval.yaml             # batch, tohumlar, yollar
├── data/
│   ├── lexicon/tr_roots.tsv
│   ├── probes/               # tr_minimal_pairs.jsonl, en_controls.jsonl
│   ├── calibration/          # {lang}_seed{n}.jsonl (kimlik listeleri)
│   ├── flores/               # indirme betiği + sürüm notu (ham veri git'e konmaz)
│   └── human_eval/           # prompts_*.jsonl, key.json (git'e konmaz)
├── src/
│   ├── data/                 # tr_minimal_pairs.py, en_controls.py, calibration.py
│   ├── quant/                # bnb.py, gptq.py, awq.py, gguf.py
│   ├── eval/                 # nll.py, bpb.py, minimal_pairs.py, features.py
│   ├── analysis/             # bootstrap.py, did.py, regression.py, plots.py
│   └── utils/                # env_info.py, seed.py, io.py
├── scripts/                  # CLI giriş noktaları
├── notebooks/colab/          # Colab için hazır defterler
├── results/
│   ├── raw/                  # JSONL: cümle bazlı sonuçlar
│   ├── tables/               # CSV özetler
│   └── figures/              # PNG/PDF
├── tests/
└── report/                   # rapor taslağı ve son metin
```

**Sonuç JSONL şeması (`results/raw/{model}__{arm}__{task}.jsonl`):**
```json
{"model_id": "...", "model_rev": "...", "arm": "bnb_nf4", "calib": null, "seed": null,
 "task": "flores_bpb", "lang": "tr", "item_id": "flores_tr_0042",
 "nll_nat": 52.31, "n_bytes": 180, "n_tokens": 61, "n_words": 24,
 "env": {"torch": "...", "transformers": "...", "cuda": "..."}}
```
Minimal çiftler için: `lp_good`, `lp_bad`, `correct` (bool), `n_suffix`, `category`.

**Her çalıştırmada kaydedilecek ortam bilgisi:** paket sürümleri, GPU adı, CUDA sürümü, model revizyonu, yapılandırma dosyası hash'i, veri dosyası hash'leri.

**Makefile hedefleri (en az):** `make setup`, `make data`, `make probes`, `make test`, `make eval-smoke`, `make eval ARM=... MODEL=...`, `make analyze`, `make figures`, `make human-sheet`.

---

## 10. Fazlar

### Faz 0: İskelet ve ortam
**Görevler:** depo oluştur, `uv` ile ortam kur, yapılandırma dosyaları, `env_info.py`, boş betikler, `DECISIONS.md` ve `DATA_LICENSES.md` başlat.
**Kabul ölçütleri:** `make setup` ve `make test` (boş testlerle) geçer; ortam bilgisi yazdırılır; GPU görünür (`torch.cuda.is_available()`).

### Faz 1: Veri ve temel ölçüm hattı
**Görevler:**
1. FLORES-200 devtest TR/EN yükle (lisans doğrula).
2. `nll.py`: tek bir model için cümle bazlı NLL ve BPB hesapla.
3. Duman testi: 1.5B model, FP16 (sığmazsa CPU veya Colab), 10 cümle.
4. Model seçim kapısını (Bölüm 4) minimal-çift prototipiyle (Faz 2'nin küçük sürümü) çalıştır.
**Kabul ölçütleri:** BPB değerleri sonlu ve makul; batch ile tek tek NLL tutarlı (birim test); aday modeller için taban kontrolü raporlandı; **kullanıcı iki modeli onayladı**.

### Faz 2: Minimal çift üreteci ve doğrulama
**Görevler:** kök listesi, üreteç, birim testler, örnek çiftler, İngilizce kontrol seti.
**Kabul ölçütleri:** ≥ 1000 çift her ek kademesinde; birim testler geçer; **Türkçe konuşan bir kişi rastgele %10'u doğruladı** (hatalı çift oranı < %2; aksi halde üreteç düzeltilir); yinelenen çift yok.
**Durma noktası:** doğrulama olmadan Faz 3'e geçilmez.

### Faz 3: Referans (FP16) ve yerel nicemleme kolları
**Görevler:** `fp16`, `bnb_int8`, `bnb_nf4` kollarını her iki model için FLORES BPB ve minimal çiftlerde değerlendir.
**Kabul ölçütleri:** her kol için tam sonuç JSONL'leri; bootstrap GA'lı özet tablo; `make analyze` çalışır; sonuçlar tekrar çalıştırıldığında aynı (belirlenimli).
**Not:** FP16 3B yerelde sığmayacaksa Colab defteri hazırla; sonuç JSONL'lerini repoya aktar.

### Faz 4: GPTQ/AWQ ve kalibrasyon deneyi (Colab)
**Görevler:** `gptq_w4_{en,tr,mix}` ve `awq_w4_{en,tr,mix}` kolları × 3 tohum × 2 model = en fazla 36 nicemleme çalıştırması. Önce tek model ve tek tohumla duman testi yap; süre/bütçeyi tahmin edip **kullanıcıya bildir**.
**Kabul ölçütleri:** çalışan nicemlemeler değerlendirildi veya başarısız olanlar `DECISIONS.md`'de gerekçelendirildi; kalibrasyon örnek kimlikleri kaydedildi; test seti ile örtüşme kontrolü raporlandı.

### Faz 5: Analiz
**Görevler:** göreli BPB, Δacc, DiD, ek sayısı eğrileri, regresyon, kalibrasyon karşılaştırması; şekiller.
**Kabul ölçütleri:** bölüm 7.4'teki tüm istatistikler üretildi; her hipotez için "destekledi / desteklemedi / belirsiz" kaydı (sayısal kanıtla); şekiller `results/figures/`'da.

**Önerilen şekiller:**
1. Bit derinliğine göre ΔBPB_rel (TR vs EN), GA'lı.
2. Ek sayısına göre Δacc eğrisi (TR), kollara göre.
3. Kalibrasyon dili × yöntem ısı haritası (TR ve EN için ayrı).
4. Regresyon katsayıları (orman grafiği).
5. İnsan değerlendirmesi kazanma/beraberlik/kayıp.

### Faz 6: İnsan değerlendirmesi
**Görevler:** Bölüm 8. Değerlendirme formu oluştur, kullanıcıya teslim et, doldurulmuş CSV'yi analiz et.
**Kabul ölçütleri:** anahtar ayrı dosyada; analiz betiği çalışır; kappa hesaplandı.
**Not:** Bu faz kullanıcıya bağımlıdır. Form hazır olunca kullanıcıya bildir ve bekle.

### Faz 7: Rapor ve yayın hazırlığı
**Görevler:** `report/` altında rapor (Türkçe; isteğe bağlı İngilizce özet), tüm tablo ve şekiller, README sonuç bölümü, sınırlılıklar.
**Kabul ölçütleri:** rapordaki her sayı bir sonuç dosyasına izlenebilir; `make figures` ve `make analyze` ile tüm şekil/tablolar yeniden üretilebilir; lisans/atıf dosyaları tam.

---

## 11. Riskler ve karşı önlemler

| Risk | Karşı önlem |
|---|---|
| Model Türkçe'de zaten zayıf (taban etkisi) | Faz 1'de model seçim kapısı |
| Minimal çiftlerde hatalı "yanlış" cümle | İstisna kökleri hariç tut; insan doğrulaması; birim testler |
| Token bazlı metriklerin dilleri yanıltması | BPB kullan; fertility'yi regresyonda kontrol değişkeni yap |
| Paket/araç uyumsuzluğu (GPTQ/AWQ) | Faz 4'te önce duman testi; alternatif paketler; `DECISIONS.md` |
| Colab süre/bütçe aşımı | Önce küçük deney; tahmin ve kullanıcı onayı |
| GGUF ile HF sonuçlarının karışması | Ayrı iz, ayrı tablo |
| "Fark yok" bulgusu | Hipotezler önceden yazıldı; dürüstçe rapor; negatif sonuç geçerli |
| TR/EN prob farklılığı | Sınırlılık notu; göreli düşüşlerle karşılaştırma |
| Lisans sorunları | `DATA_LICENSES.md`; belirsizse kullanma |

---

## 12. Rapor iskeleti (`report/report.md`)

1. Özet
2. Giriş ve motivasyon
3. İlgili çalışmalar (Marchisio ve ark. 2024; Quantization and Machine Translation, arXiv:2508.20893; Türkçe tokenizasyon, arXiv:2502.07057; GPTQ, AWQ, QLoRA)
4. Yöntem (modeller, kollar, veri, metrikler, istatistik)
5. Bulgular (RQ1–RQ4)
6. Tartışma ve sınırlılıklar
7. Yeniden üretim talimatı
8. Ekler (prob örnekleri, hata analizi, `DECISIONS.md` özeti)

**Rapor yazım kuralları:** Yalnızca ölçülen sonuçlar; sayılar kaynağa bağlı; yorumlar sayılardan ayrı; her şekil için başlık ve eksen etiketi; güven aralıkları gösterilir.

---

## 13. Tamamlanma ölçütü (Definition of Done)

- [ ] İki model, `fp16`, `bnb_int8`, `bnb_nf4`, en az bir GPTQ/AWQ kolu değerlendirildi.
- [ ] Kalibrasyon deneyi (en/tr/mix × 3 tohum) tamamlandı veya sınırlamalarla belgelendi.
- [ ] Minimal çift seti doğrulandı (≥ %10 insan örneklemi).
- [ ] İnsan değerlendirmesi tamamlandı (≥ 2 değerlendirici).
- [ ] Tüm hipotezler (H1–H4) için kanıtlı karar kaydı var.
- [ ] `make analyze && make figures` sıfırdan çalışıyor.
- [ ] Rapor, README sonuç bölümü, lisans ve karar kayıtları tamam.
- [ ] Hiçbir sayı ölçümsüz yazılmadı.

---

## 14. Lisans ve atıf

Kod: MIT (kullanıcı onayıyla). Veri ve model lisansları `DATA_LICENSES.md` içinde; her kaynak için atıf bilgisi eksiksiz yazılmalıdır.
