"""Bir model × kol × görev için cümle bazlı sonuçları `results/raw/` altına yazar.

Görevler:
    flores_bpb      FLORES-200 devtest, TR ve EN (PROTOCOL §6.1)
    tr_pairs        Türkçe ünlü uyumu minimal çiftleri, 4800 çift (PROTOCOL §6.3)
    en_pairs        İngilizce kontrol çiftleri (BLiMP özne-fiil uyumu), 5999 çift
    tr_pairs_proto  Model seçiminde kullanılan 384 çiftlik prototip

Örnek:
    uv run python -m scripts.run_eval --model qwen2.5-1.5b --arm bnb_nf4 --task tr_pairs --limit 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data.tr_minimal_pairs import generate_prototype, read_roots
from src.eval.bpb import bpb, corpus_bpb
from src.eval.minimal_pairs import accuracy, score_pairs
from src.eval.models import load, quantization_summary
from src.eval.nll import score_texts
from src.utils.env_info import collect
from src.utils.io import read_jsonl, read_yaml, write_jsonl
from src.utils.seed import set_seed

CONFIGS = [Path("configs/models.yaml"), Path("configs/arms.yaml"), Path("configs/eval.yaml")]
FLORES = {lang: Path(f"data/flores/processed/devtest_{lang}.jsonl") for lang in ("tr", "en")}
PROTO_ROOTS = Path("data/lexicon/tr_roots_prototype.tsv")
PROTO_PAIRS = Path("data/probes/tr_minimal_pairs_prototype.jsonl")
PAIRS = {"tr_pairs": Path("data/probes/tr_minimal_pairs.jsonl"),
         "en_pairs": Path("data/probes/en_controls.jsonl")}
# Özet tablosunda doğruluğun kırıldığı alanlar.
PAIR_GROUPS = {"tr_pairs": ("n_suffix", "category"), "en_pairs": ("category",),
               "tr_pairs_proto": ("n_suffix",)}


def run_flores(model, tok, batch_size, limit):
    rows = []
    for lang, path in FLORES.items():
        items = list(read_jsonl(path))[:limit]
        scores = score_texts(model, tok, [x["text"] for x in items], batch_size)
        for x, s in zip(items, scores):
            rows.append({"task": "flores_bpb", "lang": lang, "item_id": x["item_id"],
                         "nll_nat": s.nll_nat, "n_bytes": x["n_bytes"], "n_tokens": s.n_tokens,
                         "n_words": x["n_words"], "bpb": bpb(s.nll_nat, x["n_bytes"])})
    summary = {}
    for lang in FLORES:
        lr = [r for r in rows if r["lang"] == lang]
        summary[lang] = {"n": len(lr), "bpb": corpus_bpb(lr),
                         "fertility": sum(r["n_tokens"] for r in lr)
                         / sum(r["n_words"] for r in lr)}
    return rows, summary


def run_pairs(model, tok, batch_size, limit, task, path):
    pairs = list(read_jsonl(path))[:limit]
    rows = [{"task": task, **r} for r in score_pairs(model, tok, pairs, batch_size)]
    summary = {"all": {"n": len(rows), "acc": accuracy(rows)}}
    for key in PAIR_GROUPS[task]:
        for v in sorted({r[key] for r in rows}):
            kr = [r for r in rows if r[key] == v]
            summary[f"{key}={v}"] = {"n": len(kr), "acc": accuracy(kr)}
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default="fp16")
    ap.add_argument("--task", required=True,
                    choices=["flores_bpb", "tr_pairs", "en_pairs", "tr_pairs_proto"])
    ap.add_argument("--limit", type=int, help="ilk N öğe (duman testi)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--out-suffix", default="",
                    help="çıktı adına ek (ör. tekrar üretilebilirlik denetimi için '__rerun')")
    args = ap.parse_args()

    models_cfg, arms_cfg, eval_cfg = (read_yaml(p) for p in CONFIGS)
    model_cfg = models_cfg["models"][args.model]
    set_seed(eval_cfg["seed"])
    batch_size = args.batch_size or eval_cfg["batch_size"]

    import torch

    t0 = time.time()
    model, tok = load(model_cfg, args.arm, arms_cfg)
    load_seconds = time.time() - t0
    torch.cuda.synchronize()
    t1 = time.time()
    if args.task == "flores_bpb":
        rows, summary = run_flores(model, tok, batch_size, args.limit)
        data_paths = list(FLORES.values())
        n_scored = sum(r["n_tokens"] for r in rows)
    else:
        if args.task == "tr_pairs_proto":
            if not PROTO_PAIRS.exists():
                roots = [r["root"] for r in read_roots(PROTO_ROOTS)]
                write_jsonl(PROTO_PAIRS, generate_prototype(roots, seed=eval_cfg["seed"]))
            data_paths = [PROTO_ROOTS, PROTO_PAIRS]
            path = PROTO_PAIRS
        else:
            path = PAIRS[args.task]
            data_paths = [path]
        rows, summary = run_pairs(model, tok, batch_size, args.limit, args.task, path)
        n_scored = sum(r["n_tokens_good"] + r["n_tokens_bad"] for r in rows)
    torch.cuda.synchronize()
    eval_seconds = time.time() - t1

    arm_cfg = arms_cfg["arms"][args.arm]
    env = collect(config_paths=CONFIGS, data_paths=data_paths)
    short_env = {"torch": env["packages"]["torch"], "transformers": env["packages"]["transformers"],
                 "bitsandbytes": env["packages"]["bitsandbytes"], "cuda": env["gpu"].get("cuda")}
    common = {"model_id": model_cfg["hf_id"], "model_rev": model_cfg["revision"], "arm": args.arm,
              "calib": None, "seed": None}
    rows = [{**common, **r, "env": short_env} for r in rows]

    suffix = (f"__smoke{args.limit}" if args.limit else "") + args.out_suffix
    stem = f"{args.model}__{args.arm}__{args.task}{suffix}"
    out = Path(eval_cfg["paths"]["results_raw"]) / f"{stem}.jsonl"
    write_jsonl(out, rows)
    quant = quantization_summary(model) if arm_cfg["method"] == "bitsandbytes" else None
    if quant:
        quant["n_linear_unquantized"] = len(quant["linear_unquantized"])
    meta = {"args": vars(args), "batch_size": batch_size, "arm_config": arm_cfg,
            "reference_dtype": arms_cfg["reference_dtype"], "quantization": quant,
            "summary": summary,
            "seconds": round(time.time() - t0, 1), "load_seconds": round(load_seconds, 1),
            "eval_seconds": round(eval_seconds, 1),
            "tokens_per_s": round(n_scored / eval_seconds, 1),
            "max_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            if torch.cuda.is_available() else None,
            "memory_footprint_gb": round(model.get_memory_footprint() / 1024**3, 2),
            "env": env}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8", newline="\n")
    print(json.dumps({"out": out.as_posix(), "summary": summary, "seconds": meta["seconds"],
                      "tokens_per_s": meta["tokens_per_s"], "max_vram_gb": meta["max_vram_gb"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
