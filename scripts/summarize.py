"""`results/raw/` altındaki tam çalıştırmalardan bootstrap GA'lı özet tabloları üretir.

Her kol aynı modelin `fp16` koluyla öğe öğe eşleştirilir (PROTOCOL §7.1, §7.4). Duman testi ve
tekrar çalıştırma dosyaları (`__smoke10`, `__rerun` gibi ek taşıyan adlar) dahil edilmez.

Çıktılar:
    results/tables/summary.json  (tam hassasiyet + kaynak dosya hash'leri)
    results/tables/summary.md    (okunabilir tablolar)

Örnek:
    uv run python -m scripts.summarize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis.compare import LANGS, flores_comparison, pairs_comparison
from src.utils.env_info import file_sha256
from src.utils.io import read_jsonl, read_yaml

TASKS = ("flores_bpb", "tr_pairs", "en_pairs")
REF_ARM = "fp16"


def discover(raw_dir: Path) -> dict[tuple[str, str, str], Path]:
    """`{model}__{arm}__{task}.jsonl` biçimindeki tam çalıştırma dosyaları."""
    found = {}
    for p in sorted(raw_dir.glob("*.jsonl")):
        parts = p.stem.split("__")
        if len(parts) == 3 and parts[2] in TASKS:
            found[tuple(parts)] = p
    return found


def by_item(path: Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in read_jsonl(path)}


def flores_arrays(rows: dict[str, dict], ids: dict[str, list[str]]) -> dict:
    return {lang: {"nll": np.array([rows[i]["nll_nat"] for i in ids[lang]]),
                   "n_bytes": np.array([rows[i]["n_bytes"] for i in ids[lang]])}
            for lang in LANGS}


def pair_subsets(task: str, ref_rows: dict[str, dict]) -> dict[str, list[str]]:
    ids = sorted(ref_rows)
    subsets = {"all": ids}
    if task == "tr_pairs":
        for k in sorted({ref_rows[i]["n_suffix"] for i in ids}):
            subsets[f"n_suffix={k}"] = [i for i in ids if ref_rows[i]["n_suffix"] == k]
    return subsets


def fmt(s: dict, scale: float = 1.0, digits: int = 3, sign: bool = False) -> str:
    f = f"{{:{'+' if sign else ''}.{digits}f}}"
    return (f"{f.format(s['value'] * scale)} "
            f"[{f.format(s['lo'] * scale)}, {f.format(s['hi'] * scale)}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    eval_cfg = read_yaml("configs/eval.yaml")
    arm_order = list(read_yaml("configs/arms.yaml")["arms"])
    bs = {"n_resamples": eval_cfg["bootstrap"]["n_resamples"], "seed": eval_cfg["seed"],
          "ci": eval_cfg["bootstrap"]["ci"]}
    raw_dir = Path(eval_cfg["paths"]["results_raw"])
    files = discover(raw_dir)
    models = sorted({m for m, _, _ in files})

    flores_out, pairs_out, used = [], [], set()
    for model in models:
        arms = sorted({a for m, a, _ in files if m == model}, key=arm_order.index)
        for task in TASKS:
            ref_path = files.get((model, REF_ARM, task))
            if ref_path is None:
                continue
            ref_rows = by_item(ref_path)
            for arm in arms:
                path = files.get((model, arm, task))
                if path is None:
                    continue
                rows = by_item(path)
                if set(rows) != set(ref_rows):
                    raise ValueError(f"{path} ile {ref_path} öğe kümeleri farklı")
                used.update({path, ref_path})
                base = {"model": model, "arm": arm, "source": path.as_posix(),
                        "reference": ref_path.as_posix()}
                if task == "flores_bpb":
                    ids = {lang: sorted(i for i in ref_rows if ref_rows[i]["lang"] == lang)
                           for lang in LANGS}
                    res = flores_comparison(flores_arrays(ref_rows, ids),
                                            flores_arrays(rows, ids), **bs)
                    flores_out.append({**base, "n": len(ids["tr"]), **res})
                else:
                    for name, ids in pair_subsets(task, ref_rows).items():
                        res = pairs_comparison(np.array([ref_rows[i]["correct"] for i in ids]),
                                               np.array([rows[i]["correct"] for i in ids]), **bs)
                        pairs_out.append({**base, "task": task, "subset": name, "n": len(ids),
                                          **res})

    out_dir = Path(eval_cfg["paths"]["results_tables"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"bootstrap": bs, "reference_arm": REF_ARM,
               "sources": {p.as_posix(): file_sha256(p) for p in sorted(used)},
               "flores": flores_out, "pairs": pairs_out}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8", newline="\n")

    ci_pct = round(bs["ci"] * 100)
    md = ["# Özet tablolar", "",
          (f"`scripts/summarize.py` ile `results/raw/` dosyalarından üretilir. Köşeli parantezler "
           f"%{ci_pct} bootstrap güven aralığıdır ({bs['n_resamples']} yeniden örnekleme, "
           f"tohum {bs['seed']}; kol ve `{REF_ARM}` aynı öğeler üzerinden eşleştirilmiştir). "
           "Kaynak dosyalar ve hash'leri: `summary.json`."), "",
          "## FLORES-200 devtest: BPB ve göreli değişim", "",
          ("ΔBPB_rel = (BPB_kol − BPB_fp16) / BPB_fp16, yüzde olarak. "
           "DiD = ΔBPB_rel(TR) − ΔBPB_rel(EN), yüzde puan."), "",
          "| Model | Kol | TR BPB | EN BPB | TR ΔBPB_rel (%) | EN ΔBPB_rel (%) | DiD (puan) |",
          "|---|---|---|---|---|---|---|"]
    for r in flores_out:
        md.append(f"| {r['model']} | {r['arm']} | {fmt(r['bpb_tr'])} | {fmt(r['bpb_en'])} | "
                  f"{fmt(r['dbpb_rel_tr'], 100, 2, True)} | {fmt(r['dbpb_rel_en'], 100, 2, True)} "
                  f"| {fmt(r['did'], 100, 2, True)} |")
    md += ["", "## Minimal çiftler: doğruluk ve düşüş", "",
           "Δacc = acc_fp16 − acc_kol, yüzde puan (pozitif = kayıp).", "",
           "| Model | Kol | Set | Alt küme | n | Doğruluk (%) | Δacc (puan) |",
           "|---|---|---|---|---|---|---|"]
    for r in pairs_out:
        md.append(f"| {r['model']} | {r['arm']} | {r['task']} | {r['subset']} | {r['n']} | "
                  f"{fmt(r['acc'], 100, 1)} | {fmt(r['dacc'], 100, 2, True)} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(flores_out)} FLORES satırı, {len(pairs_out)} çift satırı → {out_dir.as_posix()}")


if __name__ == "__main__":
    main()
