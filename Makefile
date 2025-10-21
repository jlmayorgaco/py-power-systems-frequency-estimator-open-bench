# Makefile — OpenFreqBench command palette
# Usage examples:
#   make install ARGS='--dev --extras opendss,viz'
#   make run ARGS='--scenario s1_synthetic --case frequency_step --est ipdft --plots --json'
#   make docs-serve
#   make test-all
#   make release-test

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

# --------------------------------------------------------------------
# Tunables (override from CLI): make <target> ENV_NAME=ofb MANAGER=micromamba
# --------------------------------------------------------------------
ENV_NAME ?= openfreqbench
MANAGER  ?=
PYENV    := ENV_NAME=$(ENV_NAME) MANAGER=$(MANAGER)

S := scripts/pipelines

# Common ARGS passthrough (use: make target ARGS='...')
ARGS ?=

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
.PHONY: help
help: ## Show this help
	@awk 'BEGIN{FS":.*##"; printf "\nTargets:\n"} /^[a-zA-Z0-9_.-]+:.*##/{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make install ARGS=\"--dev --extras opendss,viz\""
	@echo "  make run ARGS=\"--scenario s1_synthetic --case frequency_step --est ipdft --plots\""
	@echo "  make docs-serve"
	@echo "  make test-all"

# --------------------------------------------------------------------
# Environment / setup
# --------------------------------------------------------------------
.PHONY: install dev
install: ## Create/update env and install (pass flags via ARGS)
	@$(PYENV) bash $(S)/install.sh $(ARGS)

dev: ## Full dev setup (recommended: --dev --extras opendss,viz,notebooks)
	@$(PYENV) bash $(S)/install.sh --dev $(ARGS)

# --------------------------------------------------------------------
# Run / inspect
# --------------------------------------------------------------------
.PHONY: run smoke list-estimators list-scenarios
run: ## Run a benchmark pipeline (pass flags via ARGS)
	@$(PYENV) bash $(S)/run.sh run $(ARGS)

smoke: ## Quick synthetic sanity check
	@$(PYENV) bash $(S)/run.sh smoke

list-estimators: ## List estimator files/classes
	@$(PYENV) bash $(S)/run.sh list-estimators

list-scenarios: ## List scenario packages & cases
	@$(PYENV) bash $(S)/run.sh list-scenarios

# --------------------------------------------------------------------
# Quality (lint / type / test)
# --------------------------------------------------------------------
.PHONY: lint fmt type type-all type-strict test test-all test-fast test-cov test-xml
lint: ## Ruff (format+lint) + mypy on smart selection
	@$(PYENV) bash $(S)/lint.sh $(ARGS)

fmt: ## Ruff format only
	@$(PYENV) bash $(S)/lint.sh --format-only $(ARGS)

type: ## Mypy on smart selection
	@$(PYENV) bash $(S)/typecheck.sh $(ARGS)

type-all: ## Mypy on entire repo
	@$(PYENV) bash $(S)/typecheck.sh --all $(ARGS)

type-strict: ## Mypy strict on entire repo
	@$(PYENV) bash $(S)/typecheck.sh --all --strict $(ARGS)

test: ## Pytest smart selection + coverage (term+XML)
	@$(PYENV) bash $(S)/test.sh $(ARGS)

test-all: ## Full test suite
	@$(PYENV) bash $(S)/test.sh --all $(ARGS)

test-fast: ## Last failed or changed since main (fast loop)
	@$(PYENV) bash -lc '$(S)/test.sh --last-failed || $(S)/test.sh --since origin/main'

test-cov: ## Full suite + HTML coverage report
	@$(PYENV) bash $(S)/test.sh --all --html $(ARGS)

test-xml: ## Full suite + JUnit XML
	@$(PYENV) bash $(S)/test.sh --all --junit $(ARGS)

# --------------------------------------------------------------------
# Docs
# --------------------------------------------------------------------
.PHONY: docs docs-serve docs-clean docs-linkcheck
docs: ## Build docs (auto-detect mkdocs/sphinx)
	@$(PYENV) bash $(S)/docs.sh $(ARGS)

docs-serve: ## Serve docs locally (live reload)
	@$(PYENV) bash $(S)/docs.sh --serve $(ARGS)

docs-clean: ## Clean docs output
	@$(PYENV) bash $(S)/docs.sh --clean $(ARGS)

docs-linkcheck: ## Sphinx linkcheck (if Sphinx stack)
	@$(PYENV) bash $(S)/docs.sh --stack sphinx --linkcheck --strict $(ARGS)

# --------------------------------------------------------------------
# Packaging / release
# --------------------------------------------------------------------
.PHONY: build package release-test release
build: ## Build sdist & wheel into dist/
	@$(PYENV) bash $(S)/package.sh --no-clean $(ARGS)

package: ## Clean build + twine check + verify install
	@$(PYENV) bash $(S)/package.sh --verify-install $(ARGS)

release-test: ## Build & upload to TestPyPI (signed)
	@$(PYENV) bash $(S)/package.sh --sign --upload testpypi $(ARGS)

release: ## Build & upload to PyPI (signed)
	@$(PYENV) bash $(S)/package.sh --sign --upload pypi $(ARGS)

# --------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------
.PHONY: clean deep-clean clean-results clean-results-all
clean: ## Remove caches and build artifacts
	@$(PYENV) bash $(S)/clean.sh $(ARGS)

deep-clean: ## Remove caches + deep artifacts (.nox/.tox/.venv, etc.)
	@$(PYENV) bash $(S)/clean.sh --deep $(ARGS)

clean-results: ## Remove data results (keep logs)
	@$(PYENV) bash $(S)/clean.sh --results $(ARGS)

clean-results-all: ## Remove data results including logs
	@$(PYENV) bash $(S)/clean.sh --results --no-logs --yes $(ARGS)
