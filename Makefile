# TR-QuantBench görev hedefleri (README §9).
# Windows'ta yerel GNU make gerekir (ör. `winget install ezwinports.make`). MSYS make ortam
# değişkenlerini sildiği için çalışmaz (DECISIONS.md D-011).

UV ?= uv
RUN := $(UV) run
MODEL ?=
ARM ?=

.PHONY: setup env test lint data probes eval-smoke eval analyze figures human-sheet

setup:
	$(UV) sync
	$(RUN) python -m scripts.env_info

env:
	$(RUN) python -m scripts.env_info --out results/env_info.json

test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check src scripts tests

# Aşağıdaki hedefler ilgili fazlarda doldurulacak.
data:
	$(RUN) python -m scripts.prepare_flores

probes:
	@echo "Faz 2: minimal çift üreteci henüz uygulanmadı." && exit 1

SMOKE_MODEL ?= qwen2.5-1.5b

eval-smoke:
	$(RUN) python -m scripts.run_eval --model $(SMOKE_MODEL) --arm fp16 --task flores_bpb --limit 10
	$(RUN) python -m scripts.run_eval --model $(SMOKE_MODEL) --arm fp16 --task tr_pairs_proto --limit 10

eval:
	@echo "Faz 3: eval (MODEL=$(MODEL) ARM=$(ARM)) henüz uygulanmadı." && exit 1

analyze:
	@echo "Faz 3/5: analiz henüz uygulanmadı." && exit 1

figures:
	@echo "Faz 5: şekiller henüz uygulanmadı." && exit 1

human-sheet:
	@echo "Faz 6: insan değerlendirme formu henüz uygulanmadı." && exit 1
