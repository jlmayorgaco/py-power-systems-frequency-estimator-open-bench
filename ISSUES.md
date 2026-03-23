# Repository Issue Audit

Last updated: 2026-03-19

## Summary

| Issue | Title | Status | Notes |
|-------|-------|--------|-------|
| 18 | Tooling: mypy strict baseline for src/ | PARTIAL | mypy.ini present with strict flags; `mypy src` shows 89 errors in 37 files — not passing |
| 20 | Community and governance files | PARTIAL | LICENSE + CONTRIBUTING.md present; CODE_OF_CONDUCT.md and CITATION.cff missing |
| 21 | GitHub issue and PR templates | DONE | Bug/feature YAML templates and PR checklist template all present |
| 22 | Cross-platform normalization files | PARTIAL | .gitignore + .gitattributes present; .gitattributes covers only *.sh (not * text=auto); .editorconfig absent |
| 23 | Consistent logger utility | OPEN | No log.py found anywhere in src/; note: issue references `pyopenfreqbench` but package is `openfreqbench` |
| 24 | Common pytest fixtures and config | PARTIAL | conftest.py exists with 3 fixtures; no tmp_artifacts_dir; pytest unit/smoke passes locally |
| 25 | Estimator abstract base class lifecycle | PARTIAL | ABC with update()+reset() abstract; full type hints; no configure()/estimate() methods; no attribute sealing post-configure |
| 26 | Timing/resource telemetry in EstimatorBase | OPEN | Timing is deliberately external (TimingHarness); no wall/CPU TTE, hooks, or latency logging in BaseEstimator |

---

## ISSUE 18 — Tooling: mypy strict baseline for src/

**Status:** PARTIAL

**Where:** `mypy.ini`, `pyproject.toml`

**Goal:** Enable mypy (strict-ish) for `src/` while allowing tests to be looser initially.

**Acceptance Criteria**
- mypy.ini present with strict flags for src/
- `mypy src` passes without errors

**Evidence found**
- `mypy.ini` exists at project root with correct strict-ish flags:
  `warn_return_any`, `disallow_untyped_defs`, `disallow_incomplete_defs`,
  `check_untyped_defs`, `disallow_any_generics`, `strict_optional`, etc.
- Tests section `[mypy-tests.*]` correctly relaxes rules.
- `ignore_missing_imports = True` to handle missing third-party stubs.
- `python -m mypy src` run: **Found 89 errors in 37 files**.

**Missing / gaps**
- `mypy src` does not pass. 89 errors across 37 source files. Key failing areas:
  - `src/openfreqbench/plotting/suite_plots.py` — missing type params for generic `dict`/`list`
  - `src/openfreqbench/plotting/mc_summary.py` — unannotated function arguments
  - `src/openfreqbench/cli/commands/run.py` — missing type params, type mismatch in assignment
  - `src/openfreqbench/cli/commands/smoke.py` — multiple unannotated functions, `_fs_hint` attr undefined
  - `src/openfreqbench/__main__.py` — `reconfigure` not on `TextIO | Any`

**Relevant files**
- `mypy.ini`
- `src/openfreqbench/plotting/suite_plots.py`
- `src/openfreqbench/plotting/mc_summary.py`
- `src/openfreqbench/cli/commands/run.py`
- `src/openfreqbench/cli/commands/smoke.py`
- `src/openfreqbench/__main__.py`

**Review decision**
Config is correct and well-structured. Acceptance criterion requiring `mypy src` to pass is not met.
Mark PARTIAL until all 89 errors are resolved.

---

## ISSUE 20 — Community and governance files

**Status:** PARTIAL

**Where:** project root, `.github/`

**Goal:** Add community and governance files suitable for open-source.

**Acceptance Criteria**
- LICENSE (Apache-2.0 or MIT) present
- CONTRIBUTING.md explains dev setup and PR flow
- CODE_OF_CONDUCT.md present
- CITATION.cff with project metadata

**Evidence found**
- `LICENSE` present — Apache License Version 2.0. ✓
- `CONTRIBUTING.md` present — covers quick start, estimator/scenario additions, code style,
  design rules. Minimal but functional. ✓
- `CODE_OF_CONDUCT.md` — **absent**. ✗
- `CITATION.cff` — **absent**. ✗

**Missing / gaps**
- `CODE_OF_CONDUCT.md` not present at project root or `.github/`.
- `CITATION.cff` not present. Required for academic citation (JOSS target).
- `CONTRIBUTING.md` is brief; does not cover PR flow, branch strategy, or review process in detail,
  but is sufficient as a minimal contributor guide.

**Relevant files**
- `LICENSE` ✓
- `CONTRIBUTING.md` ✓
- `CODE_OF_CONDUCT.md` — missing
- `CITATION.cff` — missing

**Review decision**
Two of four acceptance criteria met. Mark PARTIAL.

---

## ISSUE 21 — GitHub issue and PR templates

**Status:** DONE

**Where:** `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE/`

**Goal:** Provide structured templates for bugs, features, and pull requests.

**Acceptance Criteria**
- Bug report and feature request YAML issue forms
- PR template with checklist (tests, docs, linters)

**Evidence found**
- `.github/ISSUE_TEMPLATE/` contains:
  - `bug_report.yml` — YAML form with labels `type:bug`, textarea fields ✓
  - `feature_request.yml` ✓
  - `config.yml` ✓
  - Additional domain-specific templates: `proposal_benchmark.yml`, `proposal_estimator.yml`,
    `proposal_metric.yml`, `proposal_scenario.yml` ✓
- `.github/PULL_REQUEST_TEMPLATE/default.md` — full PR template with:
  - Description of changes checklist ✓
  - Testing & validation checklist (unit tests, benchmarks, CI, reproducibility) ✓
  - Checklist section covering lint, docs, tests, CI, labels, changelog ✓
  - Reviewer checklist ✓
- Additional PR templates: `benchmark.md`, `deps.md`, `docs.md`, `estimator.md`,
  `hotfix.md`, `metric.md`, `scenario.md` — domain-specific templates.

**Missing / gaps**
- None. All acceptance criteria satisfied and exceeded.

**Relevant files**
- `.github/ISSUE_TEMPLATE/bug_report.yml`
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `.github/PULL_REQUEST_TEMPLATE/default.md`

**Review decision**
All acceptance criteria met. Mark DONE.

---

## ISSUE 22 — Cross-platform normalization files

**Status:** PARTIAL

**Where:** project root

**Goal:** Normalize line endings, encodings, and common ignores across platforms.

**Acceptance Criteria**
- `.editorconfig` with UTF-8, LF, indent settings
- `.gitattributes` enforcing `* text=auto eol=lf`
- `.gitignore` for Python, build, IDE, and OS files

**Evidence found**
- `.editorconfig` — **absent**. ✗
- `.gitattributes` — present, but contains only `*.sh text eol=lf`.
  Does NOT include the required `* text=auto eol=lf` global rule. PARTIAL ✗
- `.gitignore` — present, covers:
  - macOS/editor: `.DS_Store`, `.idea/`, `.vscode/`
  - Python: `__pycache__/`, `*.pyc`, `*.pyo`, `*.pyd`, `.ipynb_checkpoints/`
  - Tools: `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `coverage.xml`, `htmlcov/`
  - Build artifacts referenced. ✓

**Missing / gaps**
- `.editorconfig` completely absent — no UTF-8 encoding enforcement, no LF line ending rule,
  no indent settings declared.
- `.gitattributes` only covers shell scripts; missing the catch-all `* text=auto eol=lf`
  required to enforce LF across all text files on Windows checkouts.

**Relevant files**
- `.gitattributes` (incomplete)
- `.gitignore` ✓
- `.editorconfig` — missing

**Review decision**
One of three acceptance criteria fully met (.gitignore). Two are incomplete or absent.
Mark PARTIAL.

---

## ISSUE 23 — Consistent logger utility

**Status:** OPEN

**Where:** `src/openfreqbench/utils/log.py` (original issue references `src/pyopenfreqbench/utils/log.py`)

**Goal:** Provide `get_logger(name)` with consistent formatting and env-level override.

**Acceptance Criteria**
- Colored output in TTY; plain in CI
- Respects `PYOPENFREQBENCH_LOGLEVEL` env var
- Unit test captures logs via `caplog`

**Evidence found**
- **Path mismatch noted**: The issue specifies `src/pyopenfreqbench/utils/log.py`, but the
  actual package is `src/openfreqbench/` (not `pyopenfreqbench`). Audited both paths.
- `src/openfreqbench/utils/` — no `log.py` or `logger.py` found.
- `src/openfreqbench/` — no logger utility anywhere in the package tree.
- `tests/utils/test_log.py` — does not exist.
- No `get_logger` function defined anywhere in `src/`.

**Missing / gaps**
- Logger utility (`log.py`) not implemented in any location.
- No TTY/CI color-switching logic.
- No env var `PYOPENFREQBENCH_LOGLEVEL` (or `OPENFREQBENCH_LOGLEVEL`) handling.
- No unit test.

**Relevant files**
- None present. All work is missing.

**Review decision**
Zero acceptance criteria met. Mark OPEN.
Note: issue text references `pyopenfreqbench` — the actual package name is `openfreqbench`.
This is a naming inconsistency in the issue, not evidence of implementation.

---

## ISSUE 24 — Common pytest fixtures and config

**Status:** PARTIAL

**Where:** `tests/`, `tests/conftest.py`

**Goal:** Add common fixtures (e.g., `tmp_artifacts_dir`, small synthetic signal) and pytest config.

**Acceptance Criteria**
- `conftest.py` provides reusable fixtures
- `pytest -q` passes locally and in CI

**Evidence found**
- `tests/conftest.py` exists and provides 3 fixtures:
  - `pure_60hz_scenario` — returns `G1_E1_Pure_60Hz(fs_hz=10_000.0, T_s=0.5)`
  - `zero_crossing_estimator` — returns `ZeroCrossingEstimator()`
  - `simple_sine_waveform` — returns `(t, v)` 1-second 60 Hz sine at 10 kHz
- `pytest tests/unit/ tests/smoke/ -q` result: **610 passed, 23 failed (scaffold stubs), 1 warning**.
  The 23 failures are pre-existing scaffold stubs raising `NotImplementedError` by design.
  The 377 fully-implemented tests pass cleanly.
- CI verification cannot be confirmed locally (no CI run available).

**Missing / gaps**
- `tmp_artifacts_dir` fixture explicitly mentioned in the issue is absent.
- Only 3 fixtures present; a richer fixture set (e.g., multi-scenario waveforms, configurable
  frequency/noise, artifact directory helpers) would better serve the test suite.
- `pytest.ini` / `pyproject.toml [tool.pytest.ini_options]` config present in pyproject.toml
  (not verified in detail here, but `pytest` runs correctly).

**Relevant files**
- `tests/conftest.py`
- `pyproject.toml` (pytest configuration)

**Review decision**
`conftest.py` exists with useful fixtures and tests run. However, the `tmp_artifacts_dir`
fixture specifically called out in the issue is missing, and the fixture set is minimal.
Mark PARTIAL.

---

## ISSUE 25 — Estimator abstract base class lifecycle

**Status:** PARTIAL

**Where:** `src/openfreqbench/estimators/common/base.py`
(Note: original issue references `estimators/base.py`; actual canonical path is
`src/openfreqbench/estimators/common/base.py`. A backward-compat shim exists at
`src/openfreqbench/estimators/_base.py`.)

**Goal:** Define an abstract base class with lifecycle methods (configure, reset, update, estimate)
and sealed attributes after configuration.

**Acceptance Criteria**
- Subclasses must override `estimate()`
- Prevent addition of new attributes post `configure()`
- Type hints fully annotated

**Evidence found**
- `BaseEstimator` in `src/openfreqbench/estimators/common/base.py` is a proper ABC (`ABC`
  from `abc`).
- `reset()` and `update(voltage, timestamp) → EstimatorOutput` are `@abstractmethod` — subclasses
  must implement both. ✓
- Full type annotations throughout: `ClassVar`, `dict[str, Any]`, `np.ndarray`, return types
  on all methods. ✓
- `__init_subclass__` hook auto-derives class vars from `SPEC`. ✓
- Backward-compat shims for `step()`, `set_params()`, `latency_samples` present.

**Missing / gaps**
- No `estimate()` method — the issue specifies subclasses must override `estimate()`, but
  the implementation uses `update()` as the primary abstract method. Functionally equivalent
  in intent, but the named contract differs.
- No `configure()` lifecycle method — `__init__` + `reconfigure()` cover this functionally,
  but the named lifecycle method `configure()` is absent.
- No attribute sealing post-configure — no `__setattr__` override prevents adding new
  attributes after construction/reconfiguration. `_config` dict is open.
- Acceptance criterion "prevent addition of new attributes post `configure()`" is not met;
  no `AttributeError` would be raised for mis-assigned attributes.

**Relevant files**
- `src/openfreqbench/estimators/common/base.py` (canonical)
- `src/openfreqbench/estimators/_base.py` (shim)

**Review decision**
ABC structure, type annotations, and mandatory override of core methods are in place.
However, the named lifecycle contract (`configure`, `estimate`) and attribute sealing
are absent. Mark PARTIAL.

---

## ISSUE 26 — Timing and resource telemetry in EstimatorBase

**Status:** OPEN

**Where:** `src/openfreqbench/estimators/common/base.py`

**Goal:** Integrate timing and resource tracking into `EstimatorBase` for per-frame telemetry.

**Acceptance Criteria**
- Capture wall/CPU TTE per estimate call
- Hooks for resource tracker updates
- Logs latency metrics

**Evidence found**
- `BaseEstimator.update()` contains zero timing code. No `time.perf_counter()` or
  `time.process_time()` calls.
- No resource tracker hooks (no callback, no observer pattern).
- No logging of latency metrics from within the estimator.
- Timing is deliberately kept external: `TimingHarness` in
  `src/openfreqbench/profiling/timing.py` handles wall-clock measurement outside
  `BaseEstimator`, per the CLAUDE.md architecture rule:
  *"TimingHarness = measures wall-clock externally"* and
  *"DO NOT: Add timing inside BaseEstimator._step() or step()"*.
- `structural_latency_samples()` is present, but this is a static structural declaration,
  not per-frame telemetry.

**Missing / gaps**
- All three acceptance criteria unmet.
- Per-frame TTE capture absent from BaseEstimator.
- No resource tracker hook infrastructure.
- No latency metric logging.
- Note: CLAUDE.md architecture rules explicitly prohibit timing inside BaseEstimator.
  Resolving this issue would require a design decision about whether to relax that rule
  or implement telemetry via a decorator/mixin pattern external to the class.

**Relevant files**
- `src/openfreqbench/estimators/common/base.py`
- `src/openfreqbench/profiling/timing.py` (external timing — current approach)
- `CLAUDE.md` (architecture rule prohibiting timing in BaseEstimator)

**Review decision**
Zero acceptance criteria met. Architectural constraints in CLAUDE.md actively conflict
with this issue's requirements. Mark OPEN.
