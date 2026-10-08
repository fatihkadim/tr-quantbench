"""FLORES-200 devtest TR/EN'yi indirir, doğrular ve JSONL'e çevirir."""

from src.data.flores import prepare

if __name__ == "__main__":
    for lang, path in prepare().items():
        print(lang, path.as_posix())
