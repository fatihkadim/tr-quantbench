from pathlib import Path

from src.utils.env_info import collect, file_sha256
from src.utils.io import read_jsonl, read_yaml, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_configs_load():
    for name in ("models", "arms", "eval"):
        assert isinstance(read_yaml(ROOT / "configs" / f"{name}.yaml"), dict)


def test_env_info_collects_packages():
    info = collect()
    assert info["packages"]["torch"] is not None
    assert "cuda_available" in info["gpu"]


def test_jsonl_roundtrip(tmp_path):
    rows = [{"id": "tr_000001", "good": "Dün kitapları gördüm."}, {"id": "x", "n": 3}]
    p = tmp_path / "a.jsonl"
    assert write_jsonl(p, rows) == 2
    assert list(read_jsonl(p)) == rows
    assert len(file_sha256(p)) == 64
