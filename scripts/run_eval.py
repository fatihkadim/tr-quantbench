"""Bir model × kol × görev için cümle bazlı sonuçları `results/raw/` altına yazar.

Örnek:
    uv run python -m scripts.run_eval --model qwen2.5-1.5b --arm fp16 --task flores_bpb --limit 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data.tr_minimal_pairs import generate, read_roots
from src.eval.bpb import bpb, corpus_bpb
from src.eval.minimal_pairs import accuracy, score_pairs
from src.eval.models import load
from src.eval.nll import score_texts
from src.utils.env_info import collect
from src.utils.io import read_jsonl, read_yaml, write_jsonl
from src.utils.seed import set_seed

CONFIGS = [Path("configs/models.yaml"), Path("configs/arms.yaml"), Path("configs/eval.yaml")]
FLORES = {lang: Path(f"data/flores/processed/devtest_{lang}.jsonl") for lang in ("tr", "en")}
PROTO_ROOTS = Path("data/lexicon/tr_roots_prototype.tsv")
PROTO_PAIRS = Path("data/probes/tr_minimal_pairs_prototype.jsonl")


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
                         "fertility": sum(r["n_tokens"] for r in lr) / sum(r["n_words"] for r in lr)}
    return rows, summary


def run_tr_pairs_proto(model, tok, batch_size, limit, seed):
    if not PROTO_PAIRS.exists():
        roots = [r["root"] for r in read_roots(PROTO_ROOTS)]
        write_jsonl(PROTO_PAIRS, generate(roots, seed=seed))
    pairs = list(read_jsonl(PROTO_PAIRS))[:limit]
    rows = [{"task": "tr_pairs_proto", **r} for r in score_pairs(model, tok, pairs, batch_size)]
    summary = {"all": {"n": len(rows), "acc": accuracy(rows)}}
    for k in sorted({r["n_suffix"] for r in rows}):
        kr = [r for r in rows if r["n_suffix"] == k]
        summary[f"n_suffix={k}"] = {"n": len(kr), "acc": accuracy(kr)}
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default="fp16")
    ap.add_argument("--task", required=True, choices=["flores_bpb", "tr_pairs_proto"])
    ap.add_argument("--limit", type=int, help="ilk N öğe (duman testi)")
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args()

    models_cfg, arms_cfg, eval_cfg = (read_yaml(p) for p in CONFIGS)
    model_cfg = models_cfg["models"][args.model]
    set_seed(eval_cfg["seed"])
    batch_size = args.batch_size or eval_cfg["batch_size"]

    t0 = time.time()
    model, tok = load(model_cfg, args.arm, arms_cfg)
    if args.task == "flores_bpb":
        rows, summary = run_flores(model, tok, batch_size, args.limit)
        data_paths = list(FLORES.values())
    else:
        rows, summary = run_tr_pairs_proto(model, tok, batch_size, args.limit, eval_cfg["seed"])
        data_paths = [PROTO_ROOTS, PROTO_PAIRS]

    import torch

    env = collect(config_paths=CONFIGS, data_paths=data_paths)
    short_env = {"torch": env["packages"]["torch"], "transformers": env["packages"]["transformers"],
                 "cuda": env["gpu"].get("cuda")}
    common = {"model_id": model_cfg["hf_id"], "model_rev": model_cfg["revision"], "arm": args.arm,
              "calib": None, "seed": None}
    rows = [{**common, **r, "env": short_env} for r in rows]

    suffix = f"__smoke{args.limit}" if args.limit else ""
    stem = f"{args.model}__{args.arm}__{args.task}{suffix}"
    out = Path(eval_cfg["paths"]["results_raw"]) / f"{stem}.jsonl"
    write_jsonl(out, rows)
    meta = {"args": vars(args), "batch_size": batch_size, "summary": summary,
            "seconds": round(time.time() - t0, 1),
            "max_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            if torch.cuda.is_available() else None,
            "env": env}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8", newline="\n")
    print(json.dumps({"out": out.as_posix(), "summary": summary, "seconds": meta["seconds"],
                      "max_vram_gb": meta["max_vram_gb"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
