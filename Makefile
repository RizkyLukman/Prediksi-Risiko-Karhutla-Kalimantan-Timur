# Jalankan dari Git Bash (Windows) atau shell POSIX lain.
# Target pipeline (data ... monitor) masih placeholder dan diisi per fase.
.DEFAULT_GOAL := help

CONFIG ?= configs/config.yaml
RUN := uv run

.PHONY: help setup lint test admin data labels features train evaluate score serve monitor all

help:
	@echo Available targets:
	@echo   help      Show this list
	@echo   setup     Create the virtual environment and install dependencies with uv sync
	@echo   lint      Run ruff check and ruff format --check
	@echo   test      Run pytest
	@echo   admin     Download and standardize admin boundaries          [Phase 1]
	@echo   data      Download raw data: admin, FIRMS, ERA5, Hansen, peat  [Phase 2]
	@echo   labels    Clean static heat sources and build the label panel  [Phase 3-4]
	@echo   features  Build weather, history, and static features          [Phase 4]
	@echo   train     Train baselines and LightGBM                         [Phase 5-6]
	@echo   evaluate  Evaluate on test years and write the report          [Phase 6]
	@echo   score     Batch-score the latest week                          [Phase 8]
	@echo   serve     Run the FastAPI server                               [Phase 8]
	@echo   monitor   Run the 2026 simulation and drift checks             [Phase 9]
	@echo   all       Run data, labels, features, train, evaluate          [Phase 7]
	@echo Config file: $(CONFIG)

setup:
	uv sync

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

test:
	$(RUN) pytest

admin:
	$(RUN) python -m karhutla.ingest.admin --config $(CONFIG)

data: admin
	@echo "[placeholder] data: implemented in Phase 2 (F-2.1 to F-2.5)"

labels: data
	@echo "[placeholder] labels: implemented in Phase 3-4 (F-3.1 to F-4.1)"

features: labels
	@echo "[placeholder] features: implemented in Phase 4 (F-4.2 to F-4.5)"

train: features
	@echo "[placeholder] train: implemented in Phase 5-6 (F-5.2, F-6.1)"

evaluate: train
	@echo "[placeholder] evaluate: implemented in Phase 6 (F-6.2)"

score:
	@echo "[placeholder] score: implemented in Phase 8 (F-8.1)"

serve:
	@echo "[placeholder] serve: implemented in Phase 8 (F-8.2)"

monitor:
	@echo "[placeholder] monitor: implemented in Phase 9 (F-9.1, F-9.3)"

all: evaluate
	@echo "all: pipeline finished"
