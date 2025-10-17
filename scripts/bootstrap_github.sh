#!/usr/bin/env bash
# OpenFreqBench bootstrapper
# - Ensures labels & milestones
# - Creates tracking epics & lots of atomic issues
# - Optionally wires issues to a GitHub Project (Beta)
# - Optionally creates Git tags & releases (with semver bump)
# - Idempotent, with DRY-RUN support
set -euo pipefail

### --------------------------- ARG PARSER -------------------------------------
DRY_RUN=0
SKIP_LABELS=0
SKIP_MILESTONES=0
SKIP_EPICS=0
SKIP_ISSUES=0
SKIP_RELEASES=0
EXTRA_VERSION=""
BUMP_KIND=""
PRERELEASE=0
CLOSE_MILESTONE=1
PROJECT_NUMBER=""
REPO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift;;
    --skip-labels) SKIP_LABELS=1; shift;;
    --skip-milestones) SKIP_MILESTONES=1; shift;;
    --skip-epics) SKIP_EPICS=1; shift;;
    --skip-issues) SKIP_ISSUES=1; shift;;
    --skip-releases) SKIP_RELEASES=1; shift;;
    --version) EXTRA_VERSION="${2:-}"; shift 2;;
    --bump) BUMP_KIND="${2:-}"; shift 2;;
    --pre) PRERELEASE=1; shift;;
    --no-close) CLOSE_MILESTONE=0; shift;;
    --project) PROJECT_NUMBER="${2:-}"; shift 2;;
    --repo) REPO="${2:-}"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

### --------------------------- CONFIG ----------------------------------------
# Try to infer repo from git origin if not provided
if [[ -z "$REPO" ]]; then
  REPO="$(git -C . remote -v 2>/dev/null | awk '/origin.*(push)/{print $2}' | sed -E 's#(git@github.com:|https://github.com/)##; s/\.git$//' | head -n1 || true)"
fi
: "${REPO:?Could not detect REPO. Set --repo owner/name}"

DEFAULT_BRANCH="${DEFAULT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"

# ---- Labels (name:#RRGGBB:desc) ----
LABELS=(
  "area:infra:#0366D6:Infra / build / tooling"
  "area:orchestrator:#0E8A16:Orchestrator & co-sim loop"
  "area:estimators:#5319E7:Estimator APIs & implementations"
  "area:scenarios:#B60205:Scenarios & data adapters"
  "area:metrics:#1D76DB:Metrics, fairness & compliance"
  "area:report:#FBCA04:Reports, plots & exports"
  "kind:feature:#006B75:New feature"
  "kind:bug:#D73A4A:Bug"
  "kind:doc:#6A737D:Docs"
  "priority:P0:#E11D21:Must-do"
  "priority:P1:#D93F0B:Important"
  "priority:P2:#FBCA04:Nice-to-have"
  "size:S:#0E8A16:≤0.5d"
  "size:M:#1D76DB:0.5–1.5d"
  "size:L:#5319E7:2–4d"
  "epic:#7057FF:Tracking issue (epic)"
  "good-first-issue:#BFD4F2:Starter task"
)

# ---- Milestones (title|description|due_YYYY-MM-DD) ----
MILESTONES=(
  "M0 – Repo bootstrap|Repo skeleton, tooling, CI green on smoke.|2025-10-20"
  "M1 – Core contracts|I/O, EstimatorBase, Orchestrator, minimal metrics.|2025-10-27"
  "M2 – Profiling & Machine Card|Per-frame profiling + machine profile.|2025-11-03"
  "M3 – ComputeModel|Deadtime/jitter/throttle + backlog metrics.|2025-11-10"
  "M4 – Fairness & Compliance|FairnessGate + pass/fail thresholds.|2025-11-17"
  "M5 – Estimator Zoo (baseline)|6–8 baseline estimators.|2025-12-01"
  "M6 – Scenarios (synthetic + CSV)|Generators + CSV adapter + suites.|2025-12-08"
  "M7 – Metrics, Exports, Reports|Parquet/JSON/XLSX + PDF report.|2025-12-15"
  "M8 – Developer UX|Cookiecutter, quickstart, run_bench UX.|2025-12-22"
  "M9 – Calibration & Complexity|TTE≈a·N+b slope + emulation.|2026-01-05"
  "M10 – Docs Site|MkDocs site live.|2026-01-12"
  "M11 – v0.2 Release|Tagged release + samples + repro check.|2026-01-19"
)

# ---- Epics (title/body/milestone) ----
EPIC_ESTIMATORS_TITLE="📈 Epic: Estimators v0.1"
read -r -d '' EPIC_ESTIMATORS_BODY <<'MD'
**Goal:** deliver MVP estimator set with consistent I/O, latency accounting, and tests.

**Tasks**
- [ ] basic/fft_peak.py — parabolic interpolation
- [ ] basic/goertzel.py — Goertzel tracker
- [ ] control/pll_srf.py — SRF-PLL
- [ ] states/ekf_freq.py — EKF
- [ ] poly/taylor_fourier.py — TF-k
- [ ] spectral/idft_kay.py — IpDFT/Kay
- [ ] state/kf_phasor.py — KF
- [ ] hybrid/ensemble_blend.py — ensemble

**DoD**
- Inherits EstimatorBase, implements `estimate()`
- Sets `alg_latency_s` and validates config
- Unit tests: steady + step/ramp
- Metrics & summary columns present
MD
EPIC_ESTIMATORS_MS="M5 – Estimator Zoo (baseline)"

EPIC_SCENARIOS_TITLE="🧪 Epic: Scenarios v0.1"
read -r -d '' EPIC_SCENARIOS_BODY <<'MD'
**Goal:** core synthetic scenarios + minimal OpenDSS feeder case.

**Tasks**
- [ ] s1_synthetic/frequency_step
- [ ] s1_synthetic/frequency_ramp
- [ ] s1_synthetic/chirp_linear
- [ ] s1_synthetic/harmonics
- [ ] s2_ieee13/reg_tap_step (OpenDSS)
- [ ] s2_ieee13/fault_slg_bus671

**DoD**
- Returns (signal, truth_df), seeded
- Truth derivation documented
- CI smoke run (<5s) for at least one case
MD
EPIC_SCENARIOS_MS="M6 – Scenarios (synthetic + CSV)"

# ---- Releases (tag|title|notes) ----
RELEASES=(
  "v0.1.0|OpenFreqBench v0.1.0 — Core Skeleton|Core packaging, base estimators, CLI, smoke tests."
)

### ------------------------ UTILITIES -----------------------------------------
run() { if (( DRY_RUN )); then echo "DRY: $*"; else eval "$@"; fi; }

need_gh() {
  command -v gh >/dev/null 2>&1 || { echo "❌ gh CLI not found. Install https://cli.github.com/"; exit 1; }
  gh auth status >/dev/null || { echo "❌ gh not authenticated. Run: gh auth login"; exit 1; }
}
require_clean_git() {
  git update-index -q --refresh || true
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "❌ Working tree not clean. Commit or stash changes."; exit 1
  fi
}
ensure_label() {
  local name="$1" color="$2" desc="$3"
  if gh label list -R "$REPO" --limit 200 --search "^${name}$" | grep -q "^${name}\b"; then
    run gh label edit "$name" -R "$REPO" --color "${color#\#}" --description "$desc"
  else
    run gh label create "$name" -R "$REPO" --color "${color#\#}" --description "$desc"
  fi
}
ensure_milestone() {
  local title="$1" desc="$2" due="$3"
  if gh api -X GET "repos/$REPO/milestones" -f state=open -q '.[]|.title' | grep -qx "$title"; then
    echo "Milestone exists: $title"
  else
    run gh api -X POST "repos/$REPO/milestones" -f title="$title" -f description="$desc" -f due_on="${due}T00:00:00Z" >/dev/null
    echo "Created milestone: $title"
  fi
}
get_milestone_number() {
  local title="$1"
  gh api -X GET "repos/$REPO/milestones" -f state=open -q ".[] | select(.title==\"$title\") | .number"
}
# Create issue and return URL on stdout
create_issue() {
  local title="$1" labels_csv="$2" milestone="$3" body="$4"
  local lbl_flags=()
  IFS=',' read -ra labels <<<"$labels_csv"
  for l in "${labels[@]}"; do l="$(echo "$l" | xargs)"; [[ -n "$l" ]] && lbl_flags+=( -l "$l" ); done
  local ms_number; ms_number="$(get_milestone_number "$milestone" || true)"
  local ms_flag=(); [[ -n "$ms_number" ]] && ms_flag=( -m "$ms_number" )

  if (( DRY_RUN )); then
    echo "https://github.com/$REPO/issues/DRY-${RANDOM}" # fake URL for linking
    echo "DRY: gh issue create -R \"$REPO\" -t \"$title\" ${lbl_flags[*]} ${ms_flag[*]} -b <<BODY
$body
BODY"
    return 0
  else
    gh issue create -R "$REPO" -t "$title" "${lbl_flags[@]}" "${ms_flag[@]}" -b "$body" \
      --json url -q .url
  fi
}
project_add_item() {
  local issue_url="$1"; [[ -z "$PROJECT_NUMBER" ]] && return 0
  run gh project item-add --project "$PROJECT_NUMBER" --owner "${REPO%%/*}" --url "$issue_url"
}
append_task_to_epic() {
  local epic_number="$1" child_url="$2"
  local child_num="${child_url##*/}"
  run gh issue comment -R "$REPO" "$epic_number" -b "- [ ] #$child_num"
}
latest_tag() { git describe --tags --abbrev=0 2>/dev/null || true; }
bump_semver() {
  local last="${1#v}" part="$2"; IFS='.' read -r MA MI PA <<<"${last:-0.0.0}"
  case "$part" in major) ((MA++)); MI=0; PA=0;; minor) ((MI++)); PA=0;; patch) ((PA++));; *) echo "Unknown bump: $part"; exit 1;; esac
  echo "v${MA}.${MI}.${PA}"
}
create_tag_and_release() {
  local tag="$1" title="$2" notes="$3"
  local pre_flag=""; [[ "$PRERELEASE" = "1" ]] && pre_flag="--prerelease"

  if ! git rev-parse "$tag" >/dev/null 2>&1; then
    require_clean_git
    run git tag -a "$tag" -m "$title"
    run git push origin "$DEFAULT_BRANCH"
    run git push origin "$tag"
  else
    echo "Tag exists: $tag"
  fi
  if ! gh release view "$tag" -R "$REPO" >/dev/null 2>&1; then
    run gh release create "$tag" -R "$REPO" --generate-notes -t "$title" -n "$notes" $pre_flag
  else
    echo "Release exists: $tag"
  fi
  if (( CLOSE_MILESTONE )); then
    local ms_num
    ms_num=$(gh api -X GET "repos/$REPO/milestones?state=open" -q ".[] | select(.title | startswith(\"${tag} \") or startswith(\"${tag}\")) | .number" | head -n1 || true)
    [[ -n "$ms_num" ]] && run gh api -X PATCH "repos/$REPO/milestones/$ms_num" -f state=closed >/dev/null
  fi
}

### ------------------------- EXECUTION ----------------------------------------
need_gh
echo "Repo:   $REPO"
echo "Branch: $DEFAULT_BRANCH"
(( DRY_RUN )) && echo "Mode:   DRY-RUN (no changes will be made)"

# 1) Labels
if (( ! SKIP_LABELS )); then
  echo "==> Ensuring labels"
  for triplet in "${LABELS[@]}"; do
    name="${triplet%%:*}"; rest="${triplet#*:}"; color="${rest%%:*}"; desc="${rest#*:}"
    ensure_label "$name" "$color" "$desc"
  done
else
  echo "==> Skipping labels"
fi

# 2) Milestones
declare -A MS_NUM
if (( ! SKIP_MILESTONES )); then
  echo "==> Ensuring milestones"
  for line in "${MILESTONES[@]}"; do
    IFS='|' read -r title desc due <<<"$line"
    ensure_milestone "$title" "$desc" "$due"
    num=$(get_milestone_number "$title" || true)
    [[ -n "$num" ]] && MS_NUM["$title"]="$num"
  done
else
  echo "==> Skipping milestones"
fi

# 3) Epics (tracking issues)
EPIC_ESTIMATORS_URL=""; EPIC_SCENARIOS_URL=""
EPIC_ESTIMATORS_NUM=""; EPIC_SCENARIOS_NUM=""
if (( ! SKIP_EPICS )); then
  echo "==> Creating epics"
  EPIC_ESTIMATORS_URL="$(create_issue "$EPIC_ESTIMATORS_TITLE" "epic,area:estimators,priority:P1,size:M" "$EPIC_ESTIMATORS_MS" "$EPIC_ESTIMATORS_BODY" || true)"
  EPIC_SCENARIOS_URL="$(create_issue "$EPIC_SCENARIOS_TITLE"  "epic,area:scenarios,priority:P1,size:M"   "$EPIC_SCENARIOS_MS"   "$EPIC_SCENARIOS_BODY"  || true)"
  [[ -n "$EPIC_ESTIMATORS_URL" ]] && project_add_item "$EPIC_ESTIMATORS_URL"
  [[ -n "$EPIC_SCENARIOS_URL"  ]] && project_add_item "$EPIC_SCENARIOS_URL"
  EPIC_ESTIMATORS_NUM="${EPIC_ESTIMATORS_URL##*/}"
  EPIC_SCENARIOS_NUM="${EPIC_SCENARIOS_URL##*/}"
else
  echo "==> Skipping epics"
fi

emit() {
  local title="$1" body="$2" labels="$3" ms="$4"
  echo "• $title  [$ms]"
  local url; url="$(create_issue "$title" "$labels" "$ms" "$body" || true)"
  [[ -n "$url" && -n "$PROJECT_NUMBER" ]] && project_add_item "$url"
  # auto-append to epics by prefix
  if [[ "$title" == Estimator:* && "$EPIC_ESTIMATORS_NUM" =~ ^[0-9]+$ && -n "$url" ]]; then
    append_task_to_epic "$EPIC_ESTIMATORS_NUM" "$url"
  fi
  if [[ "$title" == Scenario:* && "$EPIC_SCENARIOS_NUM" =~ ^[0-9]+$ && -n "$url" ]]; then
    append_task_to_epic "$EPIC_SCENARIOS_NUM" "$url"
  fi
}

# 4) Issues — Core + Generators (LOTS of atomic tasks)
if (( ! SKIP_ISSUES )); then
  # ---------- Core issues (compact) ----------
  read -r -d '' CORE <<'EOF'
Repo: init pyproject + package layout|**Where:** /pyopenfreqbench/, pyproject.toml  
**Acceptance Criteria:**  
- pyproject.toml with project metadata, dependencies, ruff/black configs  
- Packages: pyopenfreqbench/{estimators,scenarios,metrics,orchestrator,utils,core}/__init__.py  
- `python -c "import pyopenfreqbench"` works  
**Test plan:** run `pip install -e .`; import smoke in CI.|area:infra,kind:feature,priority:P0,size:S|M0 – Repo bootstrap

Repo: add src layout + tests pkg|**Where:** /src/pyopenfreqbench, /tests  
**Acceptance Criteria:**  
- move package to /src layout; tests discoverable by pytest  
- pytest config picks up /src  
**Test plan:** `pytest -q` runs sample test.|area:infra,kind:feature,priority:P0,size:S|M0 – Repo bootstrap

CI: workflow skeletons|**Where:** .github/workflows/ci.yml  
**Acceptance Criteria:**  
- Jobs: lint (ruff), type (mypy), test (pytest), docs build  
- Matrix: py310–py312  
**Test plan:** open PR → all jobs green.|area:infra,kind:feature,priority:P0,size:S|M0 – Repo bootstrap

CI: cache Python deps|**Where:** .github/workflows/ci.yml  
**Acceptance Criteria:**  
- actions/setup-python cache on poetry/pip  
- test runtime reduced on second run  
**Test plan:** rerun CI → cache hit logs.|area:infra,kind:feature,size:S|M0 – Repo bootstrap

Tooling: ruff config|**Where:** pyproject.toml  
**Acceptance Criteria:**  
- ruff rules enabled: E,F,I,B,UP,SIM,PL,W  
- max line length 100  
**Test plan:** `ruff check .` passes.|area:infra,kind:feature,size:S|M0 – Repo bootstrap

Tooling: black + pre-commit|**Where:** .pre-commit-config.yaml, pyproject.toml  
**Acceptance Criteria:**  
- hooks: ruff, black, end-of-file-fixer, trailing-whitespace  
**Test plan:** `pre-commit run -a` clean.|area:infra,kind:feature,size:S|M0 – Repo bootstrap

Tooling: mypy (strict)|**Where:** mypy.ini  
**Acceptance Criteria:**  
- strict mode for src; ignore tests  
- all modules typed without errors  
**Test plan:** `mypy src` passes.|area:infra,kind:feature,size:S|M0 – Repo bootstrap

Docker: base image + MKL/OpenBLAS pins|**Where:** Dockerfile, docker/entrypoint.sh  
**Acceptance Criteria:**  
- non-root user, pinned BLAS threads (OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1)  
- `python -m pyopenfreqbench --help` works inside container  
**Test plan:** build+run minimal example.|area:infra,kind:feature,priority:P0,size:M|M0 – Repo bootstrap

Community files|**Where:** LICENSE, CONTRIBUTING.md, CODE_OF_CONDUCT.md, CITATION.cff, .github/ISSUE_TEMPLATE/  
**Acceptance Criteria:**  
- files exist and reference project name and contact  
**Test plan:** lint markdown; manual review.|area:infra,kind:doc,size:S|M0 – Repo bootstrap

Logging: unified logger|**Where:** src/pyopenfreqbench/utils/log.py  
**Acceptance Criteria:**  
- `get_logger(name)` with Rich/colored formatter + level env override  
**Test plan:** unit test captures logs via caplog.|area:infra,kind:feature,size:S|M0 – Repo bootstrap

Constants: timing|**Where:** src/pyopenfreqbench/utils/timing.py  
**Acceptance Criteria:**  
- NS_PER_SECOND, NS_PER_MILLI constants + docstrings  
**Test plan:** simple assert tests.|area:infra,kind:feature,size:S|M0 – Repo bootstrap

Scaffold tests layout|**Where:** tests/conftest.py  
**Acceptance Criteria:**  
- fixtures: tmp_artifacts_dir, small_signal()  
**Test plan:** pytest collects fixtures with no warnings.|area:infra,kind:feature,size:S|M0 – Repo bootstrap

I/O dataclasses|**Where:** src/pyopenfreqbench/estimators/io.py  
**Acceptance Criteria:**  
- PMU_Input/PMU_Output with t_delivery Optional[float]  
- frozen dataclasses  
**Test plan:** unit test for immutability, defaults.|area:estimators,kind:feature,priority:P0,size:S|M1 – Core contracts

Exceptions module|**Where:** src/pyopenfreqbench/core/exc.py  
**Acceptance Criteria:**  
- ConfigurationError, ProfilingError, EstimationError classes  
**Test plan:** raise/catch tests.|area:infra,kind:feature,size:S|M1 – Core contracts

EstimatorBase: skeleton|**Where:** src/pyopenfreqbench/estimators/base.py  
**Acceptance Criteria:**  
- final configure/reset/update, abstract estimate()  
- sealed attributes after configure()  
**Test plan:** subclass attempting stray attr → raises.|area:estimators,kind:feature,priority:P0,size:M|M1 – Core contracts

EstimatorBase: profiling integration|**Where:** src/pyopenfreqbench/estimators/base.py  
**Acceptance Criteria:**  
- records tte_wall_ns,tte_cpu_ns; calls resource.track_peak_resources()  
**Test plan:** monkeypatch resource fn; assert calls.|area:estimators,kind:feature,priority:P0,size:S|M1 – Core contracts

Resource tracker|**Where:** src/pyopenfreqbench/utils/resource.py  
**Acceptance Criteria:**  
- track_peak_resources(est, dict) updates rss_peak_bytes & obj_peak_bytes  
- comment clarifies RSS is process-wide  
**Test plan:** use small dummy object; assert monotonic peak.|area:infra,kind:feature,size:S|M2 – Profiling & Machine Card

MachineProfile|**Where:** src/pyopenfreqbench/utils/sysprobe.py  
**Acceptance Criteria:**  
- collects OS, CPU model/cores, RAM, Python, BLAS vendor, optional GPU  
- returns dict serializable to JSON  
**Test plan:** snapshot test with key presence.|area:infra,kind:feature,size:M|M2 – Profiling & Machine Card

Persist run.json|**Where:** src/pyopenfreqbench/utils/artifacts.py  
**Acceptance Criteria:**  
- write_run_json(path, manifest_dict) with git SHA + dirty flag + seeds + machine_card  
**Test plan:** create tmp dir; validate required keys.|area:infra,kind:feature,size:S|M2 – Profiling & Machine Card

ScenarioConfig (Pydantic)|**Where:** src/pyopenfreqbench/scenarios/config.py  
**Acceptance Criteria:**  
- pydantic BaseModel + sha256(config) helper  
**Test plan:** identical dict → identical sha; order-insensitive.|area:scenarios,kind:feature,priority:P0,size:M|M1 – Core contracts

SuiteConfig (Pydantic)|**Where:** src/pyopenfreqbench/scenarios/suite.py  
**Acceptance Criteria:**  
- suite model with budgets (latency, memory) and scenario list  
**Test plan:** validation errors on missing fields.|area:scenarios,kind:feature,size:M|M1 – Core contracts

Orchestrator: dual clocks|**Where:** src/pyopenfreqbench/orchestrator/runner.py  
**Acceptance Criteria:**  
- T_sim advances by Δt_sim; T_proc advances by measured TTE (wall)  
- FIFO buffer for PMU samples  
**Test plan:** deterministic toy estimator with fixed TTE; assert T_proc timeline.|area:orchestrator,kind:feature,priority:P0,size:M|M1 – Core contracts

Orchestrator: co-sim rule|**Where:** src/pyopenfreqbench/orchestrator/runner.py  
**Acceptance Criteria:**  
- only call estimator.update() when T_proc ≤ T_sim  
**Test plan:** inject TTE > Δt_sim; ensure backlog grows.|area:orchestrator,kind:feature,size:S|M1 – Core contracts

Metrics: total delay|**Where:** src/pyopenfreqbench/metrics/delay.py  
**Acceptance Criteria:**  
- function total_delay(t_delivery, t_sim_mid) vectorized  
**Test plan:** numpy array test; compare known values.|area:metrics,kind:feature,size:S|M1 – Core contracts

ComputeModel primitives|**Where:** src/pyopenfreqbench/orchestrator/compute_model.py  
**Acceptance Criteria:**  
- deadtime, jitter (norm/uniform), throttle, sleep emulation  
**Test plan:** seed RNG; assert distributions + means.|area:orchestrator,kind:feature,priority:P0,size:M|M3 – ComputeModel

ComputeModel integration|**Where:** src/pyopenfreqbench/orchestrator/runner.py  
**Acceptance Criteria:**  
- optional config flag applies compute model to TTE before T_proc increment  
**Test plan:** same input with/without model → different T_proc.|area:orchestrator,kind:feature,priority:P0,size:S|M3 – ComputeModel

Backlog metrics|**Where:** src/pyopenfreqbench/metrics/backlog.py  
**Acceptance Criteria:**  
- deadline_miss, queue_len, queuing_delay, utilization U per step  
**Test plan:** synthetic timeline unit tests.|area:orchestrator,kind:feature,size:M|M3 – ComputeModel

Warm-up frames drop|**Where:** src/pyopenfreqbench/orchestrator/runner.py  
**Acceptance Criteria:**  
- config K warm-up frames excluded from reporting  
**Test plan:** K>0 reduces rows in summary consistently.|area:orchestrator,kind:feature,size:S|M3 – ComputeModel

FairnessGate budgets|**Where:** src/pyopenfreqbench/metrics/fairness.py  
**Acceptance Criteria:**  
- window/latency/memory budgets; reason strings on fail  
**Test plan:** oversized window triggers fail with reason text.|area:metrics,kind:feature,priority:P0,size:M|M4 – Fairness & Compliance

Suite YAML loader|**Where:** src/pyopenfreqbench/scenarios/loader.py  
**Acceptance Criteria:**  
- load YAML to SuiteConfig; schema validation & hashing  
**Test plan:** invalid fields raise ValidationError.|area:scenarios,kind:feature,size:M|M4 – Fairness & Compliance

Compliance summary|**Where:** src/pyopenfreqbench/metrics/summary.py  
**Acceptance Criteria:**  
- add pass/fail column per estimator per suite  
**Test plan:** unit test with one pass, one fail.|area:metrics,kind:feature,size:S|M4 – Fairness & Compliance

Export: frames.parquet|**Where:** src/pyopenfreqbench/utils/export.py  
**Acceptance Criteria:**  
- schema documented; write_frame_table(df,path)  
**Test plan:** roundtrip read with pyarrow.|area:infra,kind:feature,priority:P0,size:M|M7 – Metrics, Exports, Reports

Export: summary.parquet|**Where:** src/pyopenfreqbench/utils/export.py  
**Acceptance Criteria:**  
- summary table writer with consistent dtypes  
**Test plan:** dtype asserts (floats/ints/strings).|area:infra,kind:feature,size:S|M7 – Metrics, Exports, Reports

Export: run.xlsx|**Where:** src/pyopenfreqbench/utils/export_xlsx.py  
**Acceptance Criteria:**  
- two tabs (summary, methods) + freeze header  
**Test plan:** openpyxl asserts on sheet names + styles.|area:infra,kind:feature,size:S|M7 – Metrics, Exports, Reports

Report PDF (headless)|**Where:** reports/build_report.py  
**Acceptance Criteria:**  
- generates Pareto, heatmaps, histograms; deterministic sizes  
**Test plan:** compare hash of PNGs within tolerance.|area:report,kind:feature,priority:P0,size:L|M7 – Metrics, Exports, Reports

CLI: run_bench.py|**Where:** cli/run_bench.py  
**Acceptance Criteria:**  
- flags: --suite, --out, --profiles, --seed  
**Test plan:** `python -m pyopenfreqbench.cli.run_bench --help` & smoke run.|area:infra,kind:feature,size:S|M8 – Developer UX

Cookiecutter: new estimator|**Where:** tools/cookiecutter/estimator/  
**Acceptance Criteria:**  
- template generates class inheriting EstimatorBase + tests  
**Test plan:** cookiecutter render + pytest.|area:infra,kind:feature,size:M|M8 – Developer UX

Cookiecutter: new scenario|**Where:** tools/cookiecutter/scenario/  
**Acceptance Criteria:**  
- template with config + truth function + tests  
**Test plan:** cookiecutter render + pytest.|area:infra,kind:feature,size:M|M8 – Developer UX

Quickstart doc|**Where:** README.md, docs/quickstart.md  
**Acceptance Criteria:**  
- ≤60s example from install to PDF+XLSX  
**Test plan:** run steps in fresh container.|area:infra,kind:doc,priority:P0,size:S|M8 – Developer UX

Calibration: fit TTE=aN+b|**Where:** src/pyopenfreqbench/metrics/complexity.py  
**Acceptance Criteria:**  
- regression with slope, intercept, R²  
**Test plan:** synthetic linear data recovers params.|area:metrics,kind:feature,size:M|M9 – Calibration & Complexity

Emulation via profile|**Where:** src/pyopenfreqbench/orchestrator/runner.py  
**Acceptance Criteria:**  
- throttle_factor scales TTE; noted in summary  
**Test plan:** factor=2 doubles avg T_proc gap.|area:orchestrator,kind:feature,size:S|M9 – Calibration & Complexity

Complexity to report|**Where:** reports/build_report.py  
**Acceptance Criteria:**  
- adds slope and R² columns/plots  
**Test plan:** assert columns present + non-null.|area:metrics,kind:feature,size:S|M9 – Calibration & Complexity

MkDocs site|**Where:** mkdocs.yml, docs/  
**Acceptance Criteria:**  
- nav with API autodoc; diagrams included  
**Test plan:** `mkdocs build` in CI.|area:infra,kind:doc,size:M|M10 – Docs Site

Notebook examples|**Where:** docs/notebooks/analysis.ipynb  
**Acceptance Criteria:**  
- demonstrates reading frames/summary; produces one chart  
**Test plan:** nbconvert executes in CI.|area:report,kind:doc,size:S|M10 – Docs Site

Dev guide: EstimatorBase|**Where:** docs/dev/estimators.md  
**Acceptance Criteria:**  
- do/don’t list; t_delivery rules; latency guidance  
**Test plan:** link checked by mkdocs.|area:infra,kind:doc,size:S|M10 – Docs Site

Release automation|**Where:** .github/workflows/release.yml  
**Acceptance Criteria:**  
- on tag: build wheels/sdist, upload artifacts  
**Test plan:** tag on sandbox repo triggers workflow.|area:infra,kind:feature,priority:P0,size:S|M11 – v0.2 Release

Repro check multi-host|**Where:** scripts/repro_check.sh  
**Acceptance Criteria:**  
- compares summaries within tolerance; prints delta table  
**Test plan:** run with two artifact dirs.|area:infra,kind:bug,size:M|M11 – v0.2 Release

Determinism: same seed same SHA|**Where:** tests/test_determinism.py  
**Acceptance Criteria:**  
- two runs with same seed → identical summary hash  
**Test plan:** CI job executes twice and compares.|area:infra,kind:bug,priority:P0,size:S|M11 – v0.2 Release

Golden summary CI|**Where:** .github/workflows/golden.yml  
**Acceptance Criteria:**  
- minimal suite; ±3% tolerance check vs golden file  
**Test plan:** change estimator → CI fails with diff.|area:infra,kind:feature,size:S|M11 – v0.2 Release
EOF

  IFS=$'\n' read -rd '' -a core_lines <<<"$CORE"
  for line in "${core_lines[@]}"; do
    [[ -z "$line" ]] && continue
    IFS='|' read -r title body labels ms <<<"$line"
    emit "$title" "$body" "$labels" "$ms"
  done

  # ---------- Programmatic generators (super-atomic) ----------
  # Estimators
  declare -a ESTIMATORS=("pll.sogi_fll" "pll.srf_pll" "pll.epll" "spectral.sdfT" "spectral.goertzel" "spectral.idft_kay" "state.kf_phasor" "state.ekf_angle")
  declare -a EST_TASKS=(
    "API conformance: estimate() returns PMU_Output with t_delivery=None|area:estimators,kind:bug,size:S|M5 – Estimator Zoo (baseline)"
    "Set alg_latency_s (derive & document method)|area:estimators,kind:feature,size:S|M5 – Estimator Zoo (baseline)"
    "Low-SNR stability (10 dB): FE/RFE tests|area:estimators,kind:bug,size:S|M5 – Estimator Zoo (baseline)"
    "Unit test: pure tone @60 Hz (FE<1e-3)|area:estimators,kind:bug,size:S|M5 – Estimator Zoo (baseline)"
    "Unit test: ramp RoCoF 1 Hz/s (RFE≤spec)|area:estimators,kind:bug,size:S|M5 – Estimator Zoo (baseline)"
    "NaN/Inf input handling & conf flags|area:estimators,kind:feature,size:S|M5 – Estimator Zoo (baseline)"
    "Vectorization: avoid Python loops (profiling delta)|area:estimators,kind:feature,size:S|M5 – Estimator Zoo (baseline)"
    "Complexity calibration: TTE vs N slope|area:metrics,kind:feature,size:S|M9 – Calibration & Complexity"
    "FairnessGate compliance (window/latency)|area:metrics,kind:bug,size:S|M4 – Fairness & Compliance"
    "Docstring + references + example config|area:infra,kind:doc,size:S|M10 – Docs Site"
  )
  for est in "${ESTIMATORS[@]}"; do
    for spec in "${EST_TASKS[@]}"; do
      title_task="${spec%%|*}"; rest="${spec#*|}"; labels="${rest%%|*}"; ms="${rest##*|}"
      emit "Estimator: ${est} — ${title_task}" "Work item for ${est}." "$labels" "$ms"
    done
  done

  # Scenarios
  declare -a SCEN_TASKS=(
    "Generator: ramp (truth f/rocof arrays + seed)|area:scenarios,kind:feature,size:S|M6 – Scenarios (synthetic + CSV)"
    "Generator: step (phase/freq) + analytic truth|area:scenarios,kind:feature,size:S|M6 – Scenarios (synthetic + CSV)"
    "Generator: harmonics/interharmonics (drift)|area:scenarios,kind:feature,size:M|M6 – Scenarios (synthetic + CSV)"
    "Generator: DC offset & colored noise|area:scenarios,kind:feature,size:M|M6 – Scenarios (synthetic + CSV)"
    "Events: sags/swells; time tags at frame mid|area:scenarios,kind:feature,size:S|M6 – Scenarios (synthetic + CSV)"
    "Seed determinism tests for generators|area:scenarios,kind:bug,size:S|M6 – Scenarios (synthetic + CSV)"
    "CSVScenario: column mapping & validation|area:scenarios,kind:feature,size:S|M6 – Scenarios (synthetic + CSV)"
    "CSVScenario: streaming backpressure test|area:scenarios,kind:bug,size:S|M6 – Scenarios (synthetic + CSV)"
    "Truth validator: compare analytic vs numeric|area:metrics,kind:feature,size:S|M6 – Scenarios (synthetic + CSV)"
  )
  for spec in "${SCEN_TASKS[@]}"; do
    title="${spec%%|*}"; rest="${spec#*|}"; labels="${rest%%|*}"; ms="${rest##*|}"
    emit "Scenario: ${title}" "Implement + tests." "$labels" "$ms"
  done

  # Metrics
  declare -a METRICS_TASKS=(
    "FE: vectorized impl + tolerance tests|area:metrics,kind:feature,size:S|M1 – Core contracts"
    "RFE: vectorized impl + tolerance tests|area:metrics,kind:feature,size:S|M1 – Core contracts"
    "TVE: phasor-based impl + unit tests|area:metrics,kind:feature,size:S|M1 – Core contracts"
    "Percentiles: FE@p50/p95, RFE@p50/p95|area:metrics,kind:feature,size:S|M7 – Metrics, Exports, Reports"
    "Deadline-aware scoring (inclusive & on-time)|area:metrics,kind:feature,size:S|M7 – Metrics, Exports, Reports"
  )
  for spec in "${METRICS_TASKS[@]}"; do
    title="${spec%%|*}"; rest="${spec#*|}"; labels="${rest%%|*}"; ms="${rest##*|}"
    emit "$title" "Implement + add to summary schema." "$labels" "$ms"
  done

  # Orchestrator & ComputeModel
  declare -a ORCH_TASKS=(
    "ComputeModel: burst jitter distribution unit test|area:orchestrator,kind:bug,size:S|M3 – ComputeModel"
    "Backlog metrics: utilization U reporting validation|area:orchestrator,kind:bug,size:S|M3 – ComputeModel"
    "Queue length distribution plot hook (report)|area:report,kind:feature,size:S|M7 – Metrics, Exports, Reports"
  )
  for spec in "${ORCH_TASKS[@]}"; do
    title="${spec%%|*}"; rest="${spec#*|}"; labels="${rest%%|*}"; ms="${rest##*|}"
    emit "$title" "Implement + tests." "$labels" "$ms"
  done

  # Reporting
  declare -a REPORT_TASKS=(
    "Pareto: FE@p95 vs avg_TTE_ms (per estimator)|area:report,kind:feature,size:S|M7 – Metrics, Exports, Reports"
    "Heatmap: deadline_miss_rate vs SNR×RoCoF|area:report,kind:feature,size:S|M7 – Metrics, Exports, Reports"
    "Histogram: queue_len per scenario|area:report,kind:feature,size:S|M7 – Metrics, Exports, Reports"
    "Methods.md autowrite (fs, window, latency, machine card)|area:report,kind:feature,size:S|M7 – Metrics, Exports, Reports"
    "Excel styling: freeze header + % formats|area:report,kind:feature,size:S|M7 – Metrics, Exports, Reports"
  )
  for spec in "${REPORT_TASKS[@]}"; do
    title="${spec%%|*}"; rest="${spec#*|}"; labels="${rest%%|*}"; ms="${rest##*|}"
    emit "$title" "Implement + save assets." "$labels" "$ms"
  done

  # Docs
  declare -a DOCS_TASKS=(
    "Doc: Scenario writing + CSV adapter how-to|area:infra,kind:doc,size:S|M10 – Docs Site"
    "Doc: FairnessGate spec + rationale|area:infra,kind:doc,size:S|M10 – Docs Site"
    "Doc: Repro checklist (seeds, Git SHA, BLAS)|area:infra,kind:doc,size:S|M10 – Docs Site"
  )
  for spec in "${DOCS_TASKS[@]}"; do
    title="${spec%%|*}"; rest="${spec#*|}"; labels="${rest%%|*}"; ms="${rest##*|}"
    emit "$title" "Write docs + examples." "$labels" "$ms"
  done
else
  echo "==> Skipping issues"
fi

# 5) Releases (static array)
if (( ! SKIP_RELEASES )); then
  echo "==> Creating tags & releases (static)"
  for rel in "${RELEASES[@]}"; do
    IFS='|' read -r tag title notes <<<"$rel"
    create_tag_and_release "$tag" "$title" "$notes"
  done
else
  echo "==> Skipping releases"
fi

# 6) Optional extra release via --version / --bump
if [[ -n "$EXTRA_VERSION" || -n "$BUMP_KIND" ]]; then
  echo "==> Creating extra release"
  ver="$EXTRA_VERSION"
  if [[ -n "$BUMP_KIND" ]]; then
    last="$(latest_tag)"; [[ -z "$last" ]] && last="v0.0.0"
    ver="$(bump_semver "$last" "$BUMP_KIND")"
  fi
  [[ "$ver" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "❌ Version must be vX.Y.Z"; exit 1; }
  create_tag_and_release "$ver" "OpenFreqBench ${ver}" "Auto-generated release."
fi

echo "✅ Done."
echo "Tips:"
echo " - Dry run:    $(basename "$0") --dry-run"
echo " - Extra bump: $(basename "$0") --bump patch"
echo " - Pre-release:$(basename "$0") --version v0.2.0 --pre"
echo " - Project:    export PROJECT_NUMBER=1   # or --project 1"
echo " - Repo:       --repo owner/name"
