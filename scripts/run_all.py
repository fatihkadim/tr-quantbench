"""Onaylı modeller × yerel kollar × görevler için `scripts.run_eval`'i sırayla çalıştırır.

Yapılandırma ve veri hash'leri güncel olan sonuçlar atlanır; böylece yarıda kalan bir çalıştırma
kaldığı yerden sürdürülebilir. Görevler dış döngüdedir (hızlı FLORES önce biter).

Örnek:
    uv run python -m scripts.run_all
    uv run python -m scripts.run_all --tasks flores_bpb --suffix __rerun
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.utils.env_info import file_sha256
from src.utils.io import read_yaml

TASKS = ["flores_bpb", "tr_pairs", "en_pairs"]


def is_fresh(meta_path: Path) -> bool:
    """Sonuç, şimdiki yapılandırma ve veri dosyalarıyla mı üretilmiş?"""
    if not meta_path.exists():
        return False
    env = json.loads(meta_path.read_text(encoding="utf-8"))["env"]
    hashes = {**env["config_sha256"], **env["data_sha256"]}
    return all(Path(p).exists() and file_sha256(p) == h for p, h in hashes.items())


def main() -> None:
    models_cfg = read_yaml("configs/models.yaml")["models"]
    arms_cfg = read_yaml("configs/arms.yaml")["arms"]
    raw_dir = Path(read_yaml("configs/eval.yaml")["paths"]["results_raw"])
    approved = [m for m, c in models_cfg.items() if c["status"] == "approved"]
    local = [a for a, c in arms_cfg.items() if c["where"] in ("local", "local_or_colab")]

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=approved)
    ap.add_argument("--arms", nargs="+", default=local)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--suffix", default="", help="çıktı adına ek (ör. '__rerun')")
    ap.add_argument("--force", action="store_true", help="güncel sonuçları da yeniden üret")
    args = ap.parse_args()

    failed = []
    for task in args.tasks:
        for model in args.models:
            for arm in args.arms:
                name = f"{model}__{arm}__{task}{args.suffix}"
                if not args.force and is_fresh(raw_dir / f"{name}.meta.json"):
                    print(f"ATLA  {name} (güncel)", flush=True)
                    continue
                print(f"BAŞLA {time.strftime('%H:%M:%S')} {name}", flush=True)
                t0 = time.time()
                rc = subprocess.call([sys.executable, "-m", "scripts.run_eval", "--model", model,
                                      "--arm", arm, "--task", task, "--out-suffix", args.suffix])
                status = "BİTTİ" if rc == 0 else f"HATA (çıkış {rc})"
                print(f"{status} {time.strftime('%H:%M:%S')} {name} "
                      f"({(time.time() - t0) / 60:.1f} dk)", flush=True)
                if rc != 0:
                    failed.append(name)
    if failed:
        print("Başarısız çalıştırmalar:", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
