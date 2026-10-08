"""Ortam bilgisini yazdırır ve isteğe bağlı olarak dosyaya kaydeder."""

import argparse
import json
from pathlib import Path

from src.utils.env_info import collect


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="JSON çıktı yolu")
    args = ap.parse_args()
    configs = sorted(Path("configs").glob("*.yaml"))
    info = collect(config_paths=configs)
    text = json.dumps(info, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
