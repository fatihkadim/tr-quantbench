# Özet tablolar

`scripts/summarize.py` ile `results/raw/` dosyalarından üretilir. Köşeli parantezler %95 bootstrap güven aralığıdır (10000 yeniden örnekleme, tohum 0; kol ve `fp16` aynı öğeler üzerinden eşleştirilmiştir). Kaynak dosyalar ve hash'leri: `summary.json`.

## FLORES-200 devtest: BPB ve göreli değişim

ΔBPB_rel = (BPB_kol − BPB_fp16) / BPB_fp16, yüzde olarak. DiD = ΔBPB_rel(TR) − ΔBPB_rel(EN), yüzde puan.

| Model | Kol | TR BPB | EN BPB | TR ΔBPB_rel (%) | EN ΔBPB_rel (%) | DiD (puan) |
|---|---|---|---|---|---|---|
| llama-3.2-1b | fp16 | 1.402 [1.386, 1.417] | 0.991 [0.979, 1.003] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| llama-3.2-1b | bnb_int8 | 1.411 [1.395, 1.427] | 0.993 [0.981, 1.005] | +0.64 [+0.57, +0.72] | +0.25 [+0.19, +0.31] | +0.39 [+0.29, +0.49] |
| llama-3.2-1b | bnb_nf4 | 1.477 [1.461, 1.493] | 1.009 [0.997, 1.022] | +5.39 [+5.17, +5.60] | +1.88 [+1.69, +2.07] | +3.51 [+3.24, +3.77] |
| qwen2.5-1.5b | fp16 | 1.578 [1.561, 1.595] | 0.999 [0.986, 1.012] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| qwen2.5-1.5b | bnb_int8 | 1.584 [1.567, 1.600] | 0.998 [0.985, 1.011] | +0.36 [+0.30, +0.43] | -0.08 [-0.15, -0.02] | +0.45 [+0.36, +0.54] |
| qwen2.5-1.5b | bnb_nf4 | 1.694 [1.676, 1.711] | 1.066 [1.052, 1.079] | +7.33 [+7.08, +7.58] | +6.68 [+6.44, +6.91] | +0.65 [+0.31, +1.00] |

## Minimal çiftler: doğruluk ve düşüş

Δacc = acc_fp16 − acc_kol, yüzde puan (pozitif = kayıp).

| Model | Kol | Set | Alt küme | n | Doğruluk (%) | Δacc (puan) |
|---|---|---|---|---|---|---|
| llama-3.2-1b | fp16 | tr_pairs | all | 4800 | 96.0 [95.5, 96.6] | +0.00 [+0.00, +0.00] |
| llama-3.2-1b | fp16 | tr_pairs | n_suffix=1 | 1200 | 95.6 [94.4, 96.8] | +0.00 [+0.00, +0.00] |
| llama-3.2-1b | fp16 | tr_pairs | n_suffix=2 | 1200 | 98.1 [97.2, 98.8] | +0.00 [+0.00, +0.00] |
| llama-3.2-1b | fp16 | tr_pairs | n_suffix=3 | 1200 | 98.9 [98.2, 99.4] | +0.00 [+0.00, +0.00] |
| llama-3.2-1b | fp16 | tr_pairs | n_suffix=4 | 1200 | 91.6 [90.0, 93.2] | +0.00 [+0.00, +0.00] |
| llama-3.2-1b | bnb_int8 | tr_pairs | all | 4800 | 96.2 [95.6, 96.7] | -0.15 [-0.50, +0.21] |
| llama-3.2-1b | bnb_int8 | tr_pairs | n_suffix=1 | 1200 | 96.0 [94.9, 97.1] | -0.42 [-1.25, +0.42] |
| llama-3.2-1b | bnb_int8 | tr_pairs | n_suffix=2 | 1200 | 98.0 [97.2, 98.8] | +0.08 [-0.50, +0.67] |
| llama-3.2-1b | bnb_int8 | tr_pairs | n_suffix=3 | 1200 | 98.5 [97.8, 99.2] | +0.42 [+0.08, +0.83] |
| llama-3.2-1b | bnb_int8 | tr_pairs | n_suffix=4 | 1200 | 92.2 [90.8, 93.8] | -0.67 [-1.58, +0.17] |
| llama-3.2-1b | bnb_nf4 | tr_pairs | all | 4800 | 95.5 [94.9, 96.1] | +0.54 [+0.06, +1.04] |
| llama-3.2-1b | bnb_nf4 | tr_pairs | n_suffix=1 | 1200 | 95.0 [93.8, 96.2] | +0.58 [-0.58, +1.83] |
| llama-3.2-1b | bnb_nf4 | tr_pairs | n_suffix=2 | 1200 | 97.1 [96.1, 98.0] | +1.00 [+0.17, +1.83] |
| llama-3.2-1b | bnb_nf4 | tr_pairs | n_suffix=3 | 1200 | 98.5 [97.8, 99.2] | +0.42 [-0.17, +1.00] |
| llama-3.2-1b | bnb_nf4 | tr_pairs | n_suffix=4 | 1200 | 91.4 [89.8, 93.0] | +0.17 [-0.92, +1.25] |
| llama-3.2-1b | fp16 | en_pairs | all | 5999 | 85.2 [84.3, 86.1] | +0.00 [+0.00, +0.00] |
| llama-3.2-1b | bnb_int8 | en_pairs | all | 5999 | 85.0 [84.1, 85.9] | +0.22 [-0.28, +0.72] |
| llama-3.2-1b | bnb_nf4 | en_pairs | all | 5999 | 84.4 [83.4, 85.3] | +0.85 [+0.27, +1.45] |
| qwen2.5-1.5b | fp16 | tr_pairs | all | 4800 | 96.5 [96.0, 97.0] | +0.00 [+0.00, +0.00] |
| qwen2.5-1.5b | fp16 | tr_pairs | n_suffix=1 | 1200 | 92.9 [91.5, 94.3] | +0.00 [+0.00, +0.00] |
| qwen2.5-1.5b | fp16 | tr_pairs | n_suffix=2 | 1200 | 97.3 [96.4, 98.2] | +0.00 [+0.00, +0.00] |
| qwen2.5-1.5b | fp16 | tr_pairs | n_suffix=3 | 1200 | 99.0 [98.4, 99.5] | +0.00 [+0.00, +0.00] |
| qwen2.5-1.5b | fp16 | tr_pairs | n_suffix=4 | 1200 | 96.7 [95.7, 97.7] | +0.00 [+0.00, +0.00] |
| qwen2.5-1.5b | bnb_int8 | tr_pairs | all | 4800 | 96.3 [95.8, 96.9] | +0.15 [-0.19, +0.48] |
| qwen2.5-1.5b | bnb_int8 | tr_pairs | n_suffix=1 | 1200 | 92.8 [91.2, 94.2] | +0.17 [-0.75, +1.08] |
| qwen2.5-1.5b | bnb_int8 | tr_pairs | n_suffix=2 | 1200 | 97.2 [96.2, 98.1] | +0.08 [-0.50, +0.67] |
| qwen2.5-1.5b | bnb_int8 | tr_pairs | n_suffix=3 | 1200 | 98.7 [98.0, 99.2] | +0.33 [-0.08, +0.83] |
| qwen2.5-1.5b | bnb_int8 | tr_pairs | n_suffix=4 | 1200 | 96.7 [95.7, 97.7] | +0.00 [-0.58, +0.58] |
| qwen2.5-1.5b | bnb_nf4 | tr_pairs | all | 4800 | 95.5 [94.9, 96.1] | +1.00 [+0.50, +1.50] |
| qwen2.5-1.5b | bnb_nf4 | tr_pairs | n_suffix=1 | 1200 | 91.1 [89.4, 92.7] | +1.83 [+0.42, +3.33] |
| qwen2.5-1.5b | bnb_nf4 | tr_pairs | n_suffix=2 | 1200 | 97.0 [96.0, 97.9] | +0.33 [-0.33, +1.00] |
| qwen2.5-1.5b | bnb_nf4 | tr_pairs | n_suffix=3 | 1200 | 98.7 [98.0, 99.2] | +0.33 [-0.25, +0.92] |
| qwen2.5-1.5b | bnb_nf4 | tr_pairs | n_suffix=4 | 1200 | 95.2 [93.9, 96.3] | +1.50 [+0.33, +2.67] |
| qwen2.5-1.5b | fp16 | en_pairs | all | 5999 | 87.5 [86.7, 88.4] | +0.00 [+0.00, +0.00] |
| qwen2.5-1.5b | bnb_int8 | en_pairs | all | 5999 | 87.2 [86.3, 88.1] | +0.30 [-0.15, +0.75] |
| qwen2.5-1.5b | bnb_nf4 | en_pairs | all | 5999 | 87.9 [87.1, 88.7] | -0.42 [-1.03, +0.20] |
