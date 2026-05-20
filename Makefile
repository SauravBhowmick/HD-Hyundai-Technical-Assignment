.PHONY: help install validate train evaluate drift predict api web web-install web-build test docker docker-run compose-up compose-down prepare start start-docker start-with-docker clean all

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
MID ?= 1
TS ?= 2015-06-01T06:00:00

help:
	@echo "Targets:"
	@echo "  install     install python deps into $(VENV)"
	@echo "  validate    schema + range checks on raw CSVs"
	@echo "  train       train baseline + LightGBM, MLflow runs, save model"
	@echo "  evaluate    reprint last metrics JSON"
	@echo "  drift       train-vs-test PSI report (artifacts/drift_report.md)"
	@echo "  predict     predict for MID=<id> TS=<iso>"
	@echo "  api         start FastAPI on :8000"
	@echo "  test        run pytest"
	@echo "  web-install install web deps"
	@echo "  web         start the Vite dev server (:5173)"
	@echo "  web-build   build the web app for production"
	@echo "  docker      build the all-in-one Docker image (Node + Python)"
	@echo "  docker-run  bring up API (:8000) + MLflow UI (:5050) via docker compose"
	@echo "  compose-up   same as docker-run but force-rebuilds the image first"
	@echo "  compose-down docker compose down (stop + remove the stack)"
	@echo "  prepare      one-off: build pdm image + pull MLflow image, then exit"
	@echo "  start             local API + web + MLflow UI (./start.sh)"
	@echo "  start-docker      Docker API + web + MLflow UI"
	@echo "  start-with-docker local API AND side-by-side Docker + web + MLflow UI"
	@echo "  all         install -> train -> drift -> test"
	@echo "  clean       remove build artifacts"

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

validate:
	$(PY) -m pdm.cli validate

train:
	$(PY) -m pdm.cli train

evaluate:
	$(PY) -m pdm.cli evaluate

drift:
	$(PY) -m pdm.cli drift

predict:
	$(PY) -m pdm.cli predict --machine-id $(MID) --timestamp $(TS)

api:
	$(VENV)/bin/uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

test:
	$(VENV)/bin/pytest

web-install:
	cd web && npm install

web:
	cd web && npm run dev

web-build:
	cd web && npm run build

docker:
	docker build -t pdm-digital-twin .

docker-run:
	docker compose up

compose-up:
	docker compose up --build

compose-down:
	docker compose down

prepare:
	./start.sh --prepare

start:
	./start.sh

start-docker:
	./start.sh --docker

start-with-docker:
	./start.sh --with-docker

all: install train drift test
	@echo "[all] done. Run 'make api' to start the API."

clean:
	rm -rf artifacts mlruns .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
