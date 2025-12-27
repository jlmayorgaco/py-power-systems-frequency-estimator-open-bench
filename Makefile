# ===== OpenFreqBench minimal Makefile =====
# Goals:
#   - make venv     -> create .venv-ofb
#   - make install  -> install -e . into venv
#   - make smoke    -> run CLI sanity checks
#   - make clean    -> remove venv & caches
# Default target: smoke

# ---- Config ----
PY      ?= python3
VENV    ?= .venv-ofb
PIP     := $(VENV)/bin/pip
PYBIN   := $(VENV)/bin/python
OFB     := $(VENV)/bin/ofb

.PHONY: default help venv install smoke clean

default: smoke

help:
	@echo "Targets:"
	@echo "  venv     - create local $(VENV)"
	@echo "  install  - install -e . (uses pyproject deps)"
	@echo "  smoke    - sanity-run the CLI"
	@echo "  clean    - remove venv & caches"

venv:
	$(PY) -m venv $(VENV)
	@echo ">> venv created at $(VENV)"

install: venv
	. $(VENV)/bin/activate && pip install --upgrade pip
	. $(VENV)/bin/activate && pip install -e .
	@echo ">> installed openfreqbench into $(VENV)"

smoke: install
	@echo ">> CLI --help"
	$(OFB) --banner --help
	@echo ">> OK: ofb is runnable"

clean:
	rm -rf $(VENV) *.egg-info build dist .pytest_cache .mypy_cache **/__pycache__
	@echo ">> cleaned"
