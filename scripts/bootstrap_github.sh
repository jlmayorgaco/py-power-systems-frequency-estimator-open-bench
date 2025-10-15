#!/usr/bin/env bash
# Bootstrap OpenFreqBench on GitHub:
# - Ensures labels & milestones
# - Creates epics & atomic issues (linked to a Project if provided)
# - Creates tags & GitHub releases (auto notes), optionally closes matching milestone
# Reqs: gh (GitHub CLI), git remote origin->GitHub, clean tree for releases
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift;;
    --skip-labels) SKIP_LABELS=1; shift;;
    --skip-milestones) SKIP_MILESTONES=1; shift;;
    --skip-epics) SKIP_EPICS=1; shift;;
    --skip-issues) SKIP_ISSUES=1; shift;;
    --skip-releases) SKIP_RELEASES=1; shift;;
    --version) EXTRA_VERSION="$2"; shift 2;;
    --bump) BUMP_KIND="$2"; shift 2;;
    --pre) PRERELEASE=1; shift;;
    --no-close) CLOSE_MILESTONE=0; shift;;
    --project) PROJECT_NUMBER="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

### --------------------------- CONFIG ----------------------------------------
REPO="${REPO:-$(git -C . remote -v 2>/dev/null | awk '/origin.*(push)/{print $2}' | sed -E 's#(git@github.com:|https://github.com/)##; s/\.git$//' | head -n1)}"
: "${REPO:?Could not detect REPO. Set REPO=owner/name}"

PROJECT_NUMBER="${PROJECT_NUMBER:-}"   # optional Projects Beta number
DEFAULT_BRANCH="${DEFAULT_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

LABELS=(
  "type:estimator:#1D9BF0"
  "type:scenario:#0E8A16"
  "type:evaluation:#A333C8"
  "type:docs:#6A737D"
  "type:infra:#B60205"
  "epic:#FBCA04"
  "P0:#E11D21"
  "P1:#D93F0B"
  "P2:#FBCA04"
  "S:#0E8A16"
  "M:#1D76DB"
  "L:#5319E7"
  "good first issue:#7057FF"
)

MILESTONES=(
  "v0.1.0 Core Skeleton|Core packaging, base estimators, CLI & smoke tests|2025-12-31"
  "v0.2.0 Scenarios|Synthetic set & first OpenDSS case|2026-02-28"
  "v0.3.0 Evaluation|Metrics, plots, IEC envelopes|2026-04-30"
  "v0.4.0 OpenDSS|IEEE 13/39 integration & adapters|2026-06-30"
  "v1.0.0 Release|Docs, CI, DOI/Zenodo & JOSS|2026-09-30"
)

EPIC_ESTIMATORS_TITLE="📈 Epic: Estimators v0.1"
read -r -d '' EPIC_ESTIMATORS_BODY <<'MD'
**Goal:** deliver MVP estimator set with consistent I/O, latency accounting, and tests.

**Tasks**
- [ ] basic/fft_peak.py — parabolic peak interpolation
- [ ] basic/goertzel.py — Goertzel tracker
- [ ] control/pll_srf.py — SRF-PLL discrete
- [ ] states/ekf_freq.py — EKF phase/freq
- [ ] param/prony.py — Prony
- [ ] param/matrix_pencil.py — Matrix Pencil
- [ ] tf/stft_ridge.py — ridge extraction
- [ ] poly/taylor_fourier.py — TF-k
- [ ] regress/ls_phase_unwrap.py — LS slope
- [ ] sparse/spice.py — SPICE

**Definition of Done**
- Inherits `EstimatorBase`, declares `latency_samples`
- Unit tests (step & steady)
- Metrics/summary written; FE/RFE plots
- Docstring + README snippet
MD

EPIC_SCENARIOS_TITLE="🧪 Epic: Scenarios v0.1"
read -r -d '' EPIC_SCENARIOS_BODY <<'MD'
**Goal:** core synthetic scenarios + minimal OpenDSS feeder case.

**Tasks**
- [ ] s1_synthetic/frequency_step
- [ ] s1_synthetic/frequency_ramp
- [ ] s1_synthetic/chirp_linear
- [ ] s1_synthetic/harmonics
- [ ] s2_ieee13/reg_tap_step (OpenDSS minimal)
- [ ] s2_ieee13/fault_slg_bus671

**Definition of Done**
- Returns `(signal, truth_df)`; seeded
- Truth derivation documented
- CI smoke run (<5s) for at least one case
MD

# === ISSUES ARRAY (estimators, scenarios, evaluation, CI/docs) ===
ISSUES=(
  # BASIC
  "Estimator: Zero-Crossing (basic/zcd.py)|type:estimator,P2,S|v0.1.0 Core Skeleton|Baseline ZCD with debouncing; FE sanity tests on clean 60 Hz and noisy cases."
  "Estimator: FFT peak (basic/fft_peak.py)|type:estimator,P1,S|v0.1.0 Core Skeleton|Parabolic interpolation around dominant FFT bin; windowing options; steady & step tests."
  "Estimator: IpDFT (basic/ipdft.py)|type:estimator,P1,S|v0.1.0 Core Skeleton|Finalize IpDFT with leakage correction; unit tests & plots."
  "Estimator: Goertzel tracker (basic/goertzel.py)|type:estimator,P2,S|v0.1.0 Core Skeleton|Narrowband Goertzel with sliding window; compare vs FFT peak."
  "Estimator: Recursive DFT (basic/rdft.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Sliding RDFT with leakage/window handling; timing and drift tests."
  "Estimator: Hilbert Instant Freq (basic/hilbert_freq.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Analytic signal → unwrap phase → df/dt; smoothing; noise sensitivity."
  "Estimator: Phasor slope (basic/phasor_slope.py)|type:estimator,P2,S|v0.1.0 Core Skeleton|Slope of complex phasor angle; FE/ROCOF on step/ramp."
  # PARAM
  "Estimator: Prony (param/prony.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Damped sinusoids fit; frequency extraction; small-tone regime."
  "Estimator: Matrix Pencil (param/matrix_pencil.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Low-rank Hankel; robust tone separation; compare vs Prony."
  "Estimator: MUSIC (param/music.py)|type:estimator,P2,L|v0.1.0 Core Skeleton|Subspace spectrum; peak picking; resolution tests under noise."
  "Estimator: ESPRIT (param/esprit.py)|type:estimator,P2,L|v0.1.0 Core Skeleton|Rotational invariance; bias/variance evaluation."
  "Estimator: NLLS sine fit (param/nlls_sinefit.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Max-likelihood for single tone; Gauss-Newton; init vs FFT."
  "Estimator: TLS sine fit (param/tls_sinefit.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Total least squares version; robustness checks."
  # REGRESS
  "Estimator: LS phase-unwrap (regress/ls_phase_unwrap.py)|type:estimator,P2,S|v0.1.0 Core Skeleton|Phase unwrap vs time → slope; steady/ramp tests."
  "Estimator: RLS phase-unwrap (regress/rls_phase_unwrap.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Forgetting factor; track transients; latency accounting."
  "Estimator: TLS phase-unwrap (regress/tls_phase_unwrap.py)|type:estimator,P3,M|v0.1.0 Core Skeleton|TLS variant; noise robustness."
  "Estimator: WLS ROCOF (regress/wls_rocof.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Weighted LS for df/dt; window weights; compare vs SG."
  "Estimator: Savitzky-Golay freq (regress/sg_filter_freq.py)|type:estimator,P2,S|v0.1.0 Core Skeleton|SG poly regression on phase; choose order/window vs noise."
  # POLY
  "Estimator: Taylor–Fourier (poly/taylor_fourier.py)|type:estimator,P1,M|v0.1.0 Core Skeleton|TF-k with configurable order; dynamic events; leakage handling."
  "Estimator: Dynamic phasor (poly/dynamic_phasor.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Amplitude/phase derivatives; step/ramp response."
  "Estimator: Poly-phase IF (poly/ppie.py)|type:estimator,P3,L|v0.1.0 Core Skeleton|Polynomial-phase instantaneous estimator; chirp tests."
  # SPARSE
  "Estimator: SPICE (sparse/spice.py)|type:estimator,P2,L|v0.1.0 Core Skeleton|Sparse covariance-based; gridless option; harmonics stress tests."
  "Estimator: LASSO spectrum (sparse/lasso_spectrum.py)|type:estimator,P3,L|v0.1.0 Core Skeleton|L1 spectral estimation; bias/variance trade-offs."
  "Estimator: Atomic norm (sparse/atomic_norm.py)|type:estimator,P3,L|v0.1.0 Core Skeleton|Gridless line spectral; convex program; small cases."
  "Estimator: OMP multi-tone (sparse/omp_tones.py)|type:estimator,P3,M|v0.1.0 Core Skeleton|OMP pursuit; tone counting & separation."
  # TF
  "Estimator: STFT ridge (tf/stft_ridge.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Spectrogram ridge extraction; window/synch settings."
  "Estimator: Synchrosqueezed CWT (tf/sst_cwt.py)|type:estimator,P2,L|v0.1.0 Core Skeleton|ssqueezepy pipeline; chirp/step robustness."
  "Estimator: HHT/EMD (tf/hht_emd.py)|type:estimator,P3,L|v0.1.0 Core Skeleton|IMF decomposition + Hilbert IF; mode mixing notes."
  "Estimator: Wavelet IF (tf/wavelet_if.py)|type:estimator,P3,M|v0.1.0 Core Skeleton|Wavelet ridge/IF; compare vs STFT ridge."
  # CONTROL
  "Estimator: SRF-PLL (control/pll_srf.py)|type:estimator,P1,M|v0.1.0 Core Skeleton|αβ SRF-PLL (Tustin); bandwidth param; latency_samples declared."
  "Estimator: DDSRF-PLL (control/pll_ddsrf.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Decoupled double SRF; unbalance tolerance; tests."
  "Estimator: SOGI-FLL (control/fll_sogi.py)|type:estimator,P1,M|v0.1.0 Core Skeleton|SOGI quadrature + FLL loop; dynamic tests."
  "Estimator: EPLL (control/epll.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Enhanced PLL; noise vs speed study."
  "Estimator: ANF (control/anf.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Adaptive notch filter; convergence checks."
  "Estimator: PR-PLL (control/pr_pll.py)|type:estimator,P3,L|v0.1.0 Core Skeleton|Proportional-resonant PLL; discrete design."
  # STATE-SPACE
  "Estimator: KF frequency (states/kf_freq.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Linear KF on phase/freq; process/measurement noise tuning."
  "Estimator: EKF frequency (states/ekf_freq.py)|type:estimator,P1,M|v0.1.0 Core Skeleton|Nonlinear state (phase,freq); ramp/step tests."
  "Estimator: UKF frequency (states/ukf_freq.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Sigma-point filter; compare vs EKF."
  "Estimator: CKF frequency (states/ckf_freq.py)|type:estimator,P3,L|v0.1.0 Core Skeleton|Cubature KF variant; robustness study."
  "Estimator: Particle filter (states/pf_freq.py)|type:estimator,P3,L|v0.1.0 Core Skeleton|Bootstrap PF; resampling; compute cost tracking."
  "Estimator: IMM-KF (states/imm_kf.py)|type:estimator,P3,L|v0.1.0 Core Skeleton|Mode switching for step/ramp regimes."
  # HYBRID / ML
  "Estimator: PLL+KF fusion (hybrid/pll_kf_fusion.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Fuse PLL output with KF smoothing; latency vs accuracy."
  "Estimator: Distributed KF consensus (hybrid/dkf_consensus.py)|type:estimator,P3,L|v0.1.0 Core Skeleton|Multi-PMU consensus; network delays."
  "Estimator: Ensemble blend (hybrid/ensemble_blend.py)|type:estimator,P3,M|v0.1.0 Core Skeleton|Stack several base estimators; outlier rejection."
  "Estimator: CNN regressor (hybrid/ml_cnn_reg.py)|type:estimator,P3,L|v0.1.0 Core Skeleton|Supervised f(t) regressor; small demo dataset."
  "Estimator: Direct ROCOF hybrid (hybrid/rocof_direct.py)|type:estimator,P2,M|v0.1.0 Core Skeleton|Joint freq/ROCOF estimation; stability tests."
  # SCENARIOS S0/S1
  "Scenario: s0_sin_wave/clean_const60|type:scenario,P1,S|v0.2.0 Scenarios|Pure 60 Hz; baseline sanity; exact truth."
  "Scenario: s0_sin_wave/clean_const50|type:scenario,P2,S|v0.2.0 Scenarios|Pure 50 Hz; baseline sanity."
  "Scenario: s0_sin_wave/noise_snr_sweep|type:scenario,P2,M|v0.2.0 Scenarios|AWGN SNR sweep; RMSE vs SNR curves."
  "Scenario: s1_synthetic/frequency_step|type:scenario,P1,S|v0.2.0 Scenarios|Δf step at t0; param step size/time; analytic truth."
  "Scenario: s1_synthetic/frequency_ramp|type:scenario,P1,S|v0.2.0 Scenarios|Linear ramp; df/dt truth; slope param."
  "Scenario: s1_synthetic/frequency_ramp_step|type:scenario,P2,S|v0.2.0 Scenarios|Ramp followed by step; transient handling."
  "Scenario: s1_synthetic/chirp_linear|type:scenario,P2,S|v0.2.0 Scenarios|Linear chirp; bounded FE; ridge tests."
  "Scenario: s1_synthetic/harmonics|type:scenario,P2,M|v0.2.0 Scenarios|3rd/5th/7th % with random phase; leakage stress."
  "Scenario: s1_synthetic/notch_sag_swell|type:scenario,P3,M|v0.2.0 Scenarios|Amplitude events; robustness test."
  "Scenario: s1_synthetic/phase_jump|type:scenario,P3,S|v0.2.0 Scenarios|Sudden φ jump; unwrap stability."
  "Scenario: s1_synthetic/flicker_im|type:scenario,P3,M|v0.2.0 Scenarios|Inter-modulation; frequency modulation."
  "Scenario: s1_synthetic/snr_sweep|type:scenario,P2,M|v0.2.0 Scenarios|Automated SNR grid across estimators."
  # OPENDSS
  "Scenario: s2_ieee13/reg_tap_step|type:scenario,P1,M|v0.4.0 OpenDSS|LTC +1 tap at t0; export @5 kHz; truth from source profile."
  "Scenario: s2_ieee13/fault_slg_bus671|type:scenario,P1,M|v0.4.0 OpenDSS|Single-line-to-ground; short duration; decimation & anti-alias doc."
  "Scenario: s2_ieee13/pv_ramp|type:scenario,P2,M|v0.4.0 OpenDSS|DER ramp injection; ROCOF spikes; truth from schedule."
  "Scenario: s2_ieee13/motor_start|type:scenario,P3,M|v0.4.0 OpenDSS|IM start transient; voltage dips; freq estimator stress."
  "Scenario: s3_ieee8500/pv_cloud_transients|type:scenario,P3,L|v0.4.0 OpenDSS|Fast ramps; scalability focus; perf harness."
  "Scenario: s3_ieee8500/ibr_trip|type:scenario,P3,L|v0.4.0 OpenDSS|Inverter trip; ROCOF burst; timing."
  "Scenario: s4_ieee39/two_area_oscillation|type:scenario,P2,M|v0.4.0 OpenDSS|Inter-area mode; oscillation frequency tracking."
  "Scenario: s4_ieee39/gen_trip_nadir|type:scenario,P2,M|v0.4.0 OpenDSS|Gen trip; nadir & ROCOF evaluation."
  "Scenario: s4_ieee39/governor_step|type:scenario,P3,M|v0.4.0 OpenDSS|Governor step; settling metrics."
  "Scenario: s5_kundur/small_signal_mode|type:scenario,P3,M|v0.4.0 OpenDSS|Mode ID case; small signal excitation."
  "Scenario: s5_kundur/disturbance_event|type:scenario,P3,M|v0.4.0 OpenDSS|Tie-line oscillation; estimator bias/variance."
  # REAL/CSV & SWEEPS
  "Scenario: s6_real_csv/pmu_event_sample|type:scenario,P2,M|v0.2.0 Scenarios|Mock PMU CSV; include truth proxy & uncertainty."
  "Scenario: s6_real_csv/lab_pmu_waveforms|type:scenario,P3,M|v0.2.0 Scenarios|Lab dataset ingestion; metadata manifest."
  "Scenario: s7_sweeps/snr_vs_rmse_grid|type:scenario,P2,M|v0.2.0 Scenarios|Grid across SNR & estimators; produce curves."
  "Scenario: s7_sweeps/step_size_vs_overshoot|type:scenario,P3,M|v0.2.0 Scenarios|Δf vs overshoot; latency/settling trade-off."
  "Scenario: s7_sweeps/frame_len_vs_latency|type:scenario,P2,M|v0.2.0 Scenarios|Frame length sweep vs latency & RMSE."
  "Scenario: s8_harmonic_scans/kth_harm_scan|type:scenario,P2,M|v0.2.0 Scenarios|Vary 3rd/5th/7th amplitudes/phases."
  "Scenario: s8_harmonic_scans/subharm_interharm|type:scenario,P3,M|v0.2.0 Scenarios|IEC interharmonics; leakage tests."
  "Scenario: s9_ibr_low_inertia/synthetic_ibr_event|type:scenario,P3,M|v0.2.0 Scenarios|Low inertia synthetic; fast ROCOF; bursts."
  "Scenario: s9_ibr_low_inertia/mixed_events|type:scenario,P3,L|v0.2.0 Scenarios|Combined ramps + jumps + harmonics."
  # EVALUATION & CI/DOCS
  "Evaluation: FE/RFE metrics module|type:evaluation,P1,S|v0.3.0 Evaluation|Implement FE, RFE, RMSE, MAE; vectorized; tests."
  "Evaluation: Dynamic metrics (rise/settle/overshoot)|type:evaluation,P1,M|v0.3.0 Evaluation|Windowed metrics around events; definitions documented."
  "Evaluation: IEC envelopes (M/P-class steady)|type:evaluation,P1,M|v0.3.0 Evaluation|Steady-state FE/RFE bounds; pass/fail margins."
  "Evaluation: IEC envelopes (dynamic step)|type:evaluation,P1,M|v0.3.0 Evaluation|Dynamic step envelope; transient windowing."
  "Evaluation: Result schema (Parquet + manifest)|type:evaluation,P1,S|v0.3.0 Evaluation|metrics.parquet, summary.json, manifest.json; versioned."
  "Evaluation: Plotting presets (IEEE 2-col)|type:evaluation,P2,S|v0.3.0 Evaluation|Matplotlib styles; overlay envelopes; deterministic sizes."
  "Evaluation: Performance harness|type:evaluation,P1,M|v0.3.0 Evaluation|Warmup vs steady timing; CPU/frame; tracemalloc."
  "Evaluation: Reproducibility seed/provenance|type:evaluation,P2,S|v0.3.0 Evaluation|Global seed, numpy RNG state; write to manifest."
  "CI: tests.yml + smoke test|type:infra,P0,S|v0.1.0 Core Skeleton|Run pytest on ubuntu-latest py311; small synthetic smoke."
  "Docs: README header & Quickstart|type:docs,P1,S|v0.1.0 Core Skeleton|Logo, badges, architecture diagram, quick example."
)

# Default releases to create; you can add more via --version/--bump
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
  git update-index -q --refresh
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "❌ Working tree not clean. Commit or stash changes."; exit 1
  fi
}

ensure_label() {
  local name color desc; name="$1"; color="$2"; desc="${3:-}"
  if gh label list -R "$REPO" --limit 200 --search "^${name}$" | grep -q "^${name}\b"; then
    run gh label edit "$name" -R "$REPO" --color "${color#\#}" --description "$desc"
  else
    run gh label create "$name" -R "$REPO" --color "${color#\#}" --description "$desc"
  fi
}

ensure_milestone() {
  local title desc due; title="$1"; desc="$2"; due="$3"
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

create_issue() {
  local title labels_csv milestone body
  title="$1"; labels_csv="$2"; milestone="$3"; body="$4"
  local labels=(); IFS=',' read -r -a labels <<<"$labels_csv"
  local lbl_flags=(); for l in "${labels[@]}"; do lbl_flags+=( -l "$l" ); done
  local ms_number; ms_number="$(get_milestone_number "$milestone" || true)"
  local ms_flag=(); [[ -n "$ms_number" ]] && ms_flag=( -m "$ms_number" )

  if (( DRY_RUN )); then
    echo "DRY: gh issue create -R \"$REPO\" -t \"$title\" ${lbl_flags[*]} ${ms_flag[*]} -b \"$body\""
    echo "DRY: (issue number unknown in dry-run) \n \n"
  else
    gh issue create -R "$REPO" -t "$title" "${lbl_flags[@]}" "${ms_flag[@]}" -b "$body"
  fi
}

project_add_item() {
  local issue_url="$1"; [[ -z "${PROJECT_NUMBER:-}" ]] && return 0
  local issue_num="${issue_url##*/}"; issue_num="${issue_num##\#}"
  [[ "$issue_num" =~ ^[0-9]+$ ]] || return 0
  run gh project item-add --project "$PROJECT_NUMBER" --owner "${REPO%%/*}" --url "https://github.com/$REPO/issues/$issue_num"
}

append_task_to_epic() {
  local epic_num="$1" child_url="$2"
  local child_num="${child_url##*/}"; child_num="${child_num##\#}"
  [[ "$child_num" =~ ^[0-9]+$ ]] || return 0
  run gh issue comment -R "$REPO" "$epic_num" -b "- [ ] #$child_num"
}

latest_tag() { git describe --tags --abbrev=0 2>/dev/null || true; }
bump_semver() {
  local last="$1" part="$2"; last="${last#v}"
  IFS='.' read -r MA MI PA <<<"${last:-0.0.0}"
  case "$part" in major) MA=$((MA+1)); MI=0; PA=0;; minor) MI=$((MI+1)); PA=0;; patch) PA=$((PA+1));; *) echo "Unknown bump: $part"; exit 1;; esac
  echo "v${MA}.${MI}.${PA}"
}

create_tag_and_release() {
  local tag title notes pre_flag=""; [[ "$PRERELEASE" = "1" ]] && pre_flag="--prerelease"

  # create tag if missing
  if ! git rev-parse "$tag" >/dev/null 2>&1; then
    require_clean_git
    run git tag -a "$tag" -m "$title"
    run git push origin "$DEFAULT_BRANCH"
    run git push origin "$tag"
  else
    echo "Tag exists: $tag"
  fi

  # create GH release if missing
  if ! gh release view "$tag" -R "$REPO" >/dev/null 2>&1; then
    run gh release create "$tag" -R "$REPO" --generate-notes -t "$title" -n "$notes" $pre_flag
  else
    echo "Release exists: $tag"
  fi

  # close milestone that starts with tag (e.g., 'v0.1.0 ...')
  if (( CLOSE_MILESTONE )); then
    local ms_num
    ms_num=$(gh api -X GET "repos/$REPO/milestones?state=open" -q ".[] | select(.title | startswith(\"${tag} \") or startswith(\"${tag}\")) | .number" | head -n1 || true)
    if [[ -n "$ms_num" ]]; then
      echo "Closing milestone #$ms_num for ${tag}"
      run gh api -X PATCH "repos/$REPO/milestones/$ms_num" -f state=closed >/dev/null
    fi
  fi
}

### ------------------------- EXECUTION ----------------------------------------
need_gh
echo "Repo: $REPO"
echo "Branch: $DEFAULT_BRANCH"
(( DRY_RUN )) && echo "Mode: DRY-RUN (no changes will be made)"

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
if (( ! SKIP_MILESTONES )); then
  echo "==> Ensuring milestones"
  for line in "${MILESTONES[@]}"; do
    IFS='|' read -r title desc due <<<"$line"
    ensure_milestone "$title" "$desc" "$due"
  done
else
  echo "==> Skipping milestones"
fi

# 3) Epics
EPIC_ESTIMATORS_NUM=""; EPIC_SCENARIOS_NUM=""
if (( ! SKIP_EPICS )); then
  echo "==> Creating epics (tracking issues)"
  est_url=$(create_issue "$EPIC_ESTIMATORS_TITLE" "epic" "v0.1.0 Core Skeleton" "$EPIC_ESTIMATORS_BODY" || true)
  scn_url=$(create_issue "$EPIC_SCENARIOS_TITLE"  "epic" "v0.2.0 Scenarios"     "$EPIC_SCENARIOS_BODY"  || true)

  EPIC_ESTIMATORS_NUM="${est_url##*/}"; EPIC_ESTIMATORS_NUM="${EPIC_ESTIMATORS_NUM##\#}"
  EPIC_SCENARIOS_NUM="${scn_url##*/}";  EPIC_SCENARIOS_NUM="${EPIC_SCENARIOS_NUM##\#}"

  [[ -n "$est_url" ]] && project_add_item "$est_url" || true
  [[ -n "$scn_url" ]] && project_add_item "$scn_url" || true
else
  echo "==> Skipping epics"
fi

# 4) Atomic issues
if (( ! SKIP_ISSUES )); then
  echo "==> Creating atomic issues"
  for entry in "${ISSUES[@]}"; do
    IFS='|' read -r title labels milestone body <<<"$entry"
    url=$(create_issue "$title" "$labels" "$milestone" "$body" || true)
    [[ -n "$url" ]] && project_add_item "$url" || true
    if [[ "$title" == Estimator* && "$EPIC_ESTIMATORS_NUM" =~ ^[0-9]+$ && -n "$url" ]]; then
      append_task_to_epic "$EPIC_ESTIMATORS_NUM" "$url" || true
    fi
    if [[ "$title" == Scenario* && "$EPIC_SCENARIOS_NUM" =~ ^[0-9]+$ && -n "$url" ]]; then
      append_task_to_epic "$EPIC_SCENARIOS_NUM" "$url" || true
    fi
  done
else
  echo "==> Skipping issues"
fi

# 5) Releases (static list)
if (( ! SKIP_RELEASES )); then
  echo "==> Creating tags & releases (static array)"
  for rel in "${RELEASES[@]}"; do
    IFS='|' read -r tag title notes <<<"$rel"
    create_tag_and_release "$tag" "$title" "$notes"
  done
else
  echo "==> Skipping releases array"
fi

# 6) Optional extra release via --version or --bump
if [[ -n "$EXTRA_VERSION" || -n "$BUMP_KIND" ]]; then
  echo "==> Creating EXTRA release"
  ver="$EXTRA_VERSION"
  if [[ -n "$BUMP_KIND" ]]; then
    last="$(latest_tag)"; [[ -z "$last" ]] && last="v0.0.0"
    ver="$(bump_semver "$last" "$BUMP_KIND")"
  fi
  [[ "$ver" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "❌ Version must be vX.Y.Z"; exit 1; }
  create_tag_and_release "$ver" "OpenFreqBench ${ver}" "Auto-generated release via bootstrap script."
fi

echo "✅ Done."
echo "Tips:"
echo " - Dry run:    $(basename "$0") --dry-run"
echo " - Extra bump: $(basename "$0") --bump patch"
echo " - Pre-release:$(basename "$0") --version v0.2.0 --pre"
echo " - Project:    export PROJECT_NUMBER=1   # to add cards to a Project board"
echo " - Repo:       export REPO=IngJorgeLuisMayorga/py-openfreqbench"
