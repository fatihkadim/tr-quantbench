# Sık kullanılan komutlar. Her hedef bir `uv run` komutudur; make olmadan doğrudan da çalıştırılabilir.

UV ?= uv
RUN := $(UV) run
SMOKE_MODEL ?= qwen2.5-1.5b

.PHONY: setup env test lint data eval-smoke

setup:
	$(UV) sync
	$(RUN) python -m scripts.env_info

env:
	$(RUN) python -m scripts.env_info --out results/env_info.json

test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check src scripts tests

data:
	$(RUN) python -m scripts.prepare_flores

eval-smoke:
	$(RUN) python -m scripts.run_eval --model $(SMOKE_MODEL) --arm fp16 --task flores_bpb --limit 10
	$(RUN) python -m scripts.run_eval --model $(SMOKE_MODEL) --arm fp16 --task tr_pairs_proto --limit 10
