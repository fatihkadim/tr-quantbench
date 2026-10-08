"""Çalıştırma ortamı bilgisi (PROTOCOL §9): paket sürümleri, GPU, CUDA, dosya hash'leri."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "huggingface-hub",
    "numpy",
)


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "error": "torch kurulu değil"}
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu_name=props.name,
            gpu_mem_gb=round(props.total_memory / 1024**3, 2),
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    return info


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(config_paths: list[str | Path] = (), data_paths: list[str | Path] = ()) -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu_info(),
        "config_sha256": {Path(p).as_posix(): file_sha256(p) for p in config_paths},
        "data_sha256": {Path(p).as_posix(): file_sha256(p) for p in data_paths},
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, ensure_ascii=False))
