"""Bootstrap güven aralıkları (PROTOCOL §7.4): cümle/çift düzeyinde yeniden örnekleme.

Tüm istatistikler aynı indeks örneklemlerinden hesaplanır. Böylece eşleştirilmiş
karşılaştırmalar (kol − fp16; FLORES paralel olduğu için TR − EN) her yeniden örneklemede aynı
cümleler üzerinden yapılır.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Stat = Callable[[np.ndarray], np.ndarray]


def bootstrap(stats: dict[str, Stat], n: int, n_resamples: int, seed: int, ci: float = 0.95,
              chunk: int = 500) -> dict[str, dict[str, float]]:
    """Her istatistik için nokta tahmini ve yüzdelik güven aralığı.

    Her `stat`, (B, n) boyutlu bir indeks matrisi alır ve B değer döndürür. Nokta tahmini
    özgün örneklem (tüm indeksler bir kez) üzerinden hesaplanır.
    """
    full = np.arange(n)[None, :]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[np.ndarray]] = {k: [] for k in stats}
    for start in range(0, n_resamples, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_resamples - start), n))
        for k, f in stats.items():
            samples[k].append(f(idx))
    alpha = (1 - ci) / 2
    out = {}
    for k, f in stats.items():
        s = np.concatenate(samples[k])
        out[k] = {"value": float(f(full)[0]), "lo": float(np.quantile(s, alpha)),
                  "hi": float(np.quantile(s, 1 - alpha))}
    return out
