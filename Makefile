# ===== OpenFreqBench Makefile =====
.PHONY: default help venv install dev smoke unit test lint lint-fix typecheck run-smoke clean

# ---- Config ----
PY      ?= python3
VENV    ?= .venv-ofb
PIP     := $(VENV)/bin/pip
PYBIN   := $(VENV)/bin/python
OFB     := $(VENV)/bin/ofb
SRC     := src
TESTS   := tests

default: smoke

help:
	@echo "Targets:"
	@echo "  venv       create local $(VENV)"
	@echo "  install    pip install -e . into venv"
	@echo "  dev        pip install -e .[dev] into venv"
	@echo "  smoke      run smoke tests (fast, < 30s)"
	@echo "  unit       run unit tests"
	@echo "  test       run all tests"
	@echo "  lint       ruff check"
	@echo "  lint-fix   ruff check --fix"
	@echo "  typecheck  mypy"
	@echo "  run-smoke  ofb run examples/quick_smoke.yaml"
	@echo "  clean      remove venv & caches"

venv:
	$(PY) -m venv $(VENV)
	@echo ">> venv created at $(VENV)"

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	@echo ">> installed openfreqbench into $(VENV)"

dev: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo ">> installed openfreqbench[dev] into $(VENV)"

smoke: install
	$(PYBIN) -m pytest $(TESTS)/smoke/ -v --tb=short

unit: install
	$(PYBIN) -m pytest $(TESTS)/unit/ -v --tb=short

test: install
	$(PYBIN) -m pytest $(TESTS)/ -v --tb=short

lint:
	$(PYBIN) -m ruff check $(SRC)/ $(TESTS)/

lint-fix:
	$(PYBIN) -m ruff check --fix $(SRC)/ $(TESTS)/

typecheck:
	$(PYBIN) -m mypy $(SRC)/openfreqbench --ignore-missing-imports

run-smoke: install
	$(OFB) run examples/quick_smoke.yaml

clean:
	rm -rf $(VENV) *.egg-info build dist .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo ">> cleaned"
