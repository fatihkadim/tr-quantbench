# TR-QuantBench görev hedefleri (README §9).
# Windows'ta make: C:\msys64\usr\bin\make.exe (PATH'e ekleyin) veya Git Bash/WSL.

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
	@echo "Faz 1: FLORES-200 indirme henüz uygulanmadı." && exit 1

probes:
	@echo "Faz 2: minimal çift üreteci henüz uygulanmadı." && exit 1

eval-smoke:
	@echo "Faz 1: duman testi henüz uygulanmadı." && exit 1

eval:
	@echo "Faz 3: eval (MODEL=$(MODEL) ARM=$(ARM)) henüz uygulanmadı." && exit 1

analyze:
	@echo "Faz 3/5: analiz henüz uygulanmadı." && exit 1

figures:
	@echo "Faz 5: şekiller henüz uygulanmadı." && exit 1

human-sheet:
	@echo "Faz 6: insan değerlendirme formu henüz uygulanmadı." && exit 1
