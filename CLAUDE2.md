═══════════════════════════════════════════════════════════════════════════════
OPENFREQBENCH v2.0 — COMPLETE ARCHITECTURE & IMPLEMENTATION PROMPT
Principal Research Software Architect + Scientific Reviewer mode
═══════════════════════════════════════════════════════════════════════════════

You are building OpenFreqBench v2.0: a world-class, open-source, publication-
ready benchmark framework for dynamic frequency estimators in IBR power grids.

Read the ENTIRE existing codebase before writing a single line.
Files to read first:
  main.py, estimators.py, ekf2.py, statistical_analysis.py,
  benchmark_plottings.py, plotting.py, scenarios.py,
  benchmark_results.json, and any existing tests or docs.

This prompt is structured in BLOCKS. Implement one BLOCK at a time.
After each BLOCK, verify it works, run the smoke test, and commit.
Do NOT attempt to implement everything at once.

═══════════════════════════════════════════════════════════════════════════════
BLOCK 0 — CHECKPOINT / RESUME SYSTEM (implement this FIRST, before anything)
═══════════════════════════════════════════════════════════════════════════════

The benchmark can take hours. It must be interruptible and resumable.
Every run unit is a (method × scenario) pair with its own Monte Carlo.

Implement a checkpoint manager BEFORE any other code:

File: src/openfreqbench/core/checkpoint.py

class CheckpointManager:
    """
    Manages incremental benchmark execution state.
    
    Each (method, scenario) pair is an atomic work unit.
    Results are written to JSON immediately after completion.
    On restart, already-completed pairs are skipped automatically.
    
    Checkpoint file: .ofb_checkpoint.json in the results directory.
    Format:
    {
      "session_id": "uuid4",
      "started_at": "ISO timestamp",
      "last_updated": "ISO timestamp",
      "config_hash": "sha256 of run config",
      "completed": {
        "EKF2::IBR_Nightmare": {
          "completed_at": "ISO timestamp",
          "n_mc": 30,
          "result_file": "results/EKF2__IBR_Nightmare.json"
        },
        ...
      },
      "in_progress": null,
      "failed": {}
    }
    
    On startup, implement this EXACT interaction:
    
    def startup_dialog(self, config) -> str:  # returns "resume" | "restart" | "partial"
        checkpoint = self.load()
        if checkpoint is None:
            print("No checkpoint found. Starting fresh.")
            return "restart"
        
        n_done = len(checkpoint["completed"])
        n_total = self.count_total_pairs(config)
        elapsed = self.format_elapsed(checkpoint)
        
        print(f"\n{'='*60}")
        print(f"  CHECKPOINT FOUND — Session: {checkpoint['session_id'][:8]}...")
        print(f"  Started: {checkpoint['started_at']}")
        print(f"  Progress: {n_done}/{n_total} pairs completed")
        print(f"  Elapsed: {elapsed}")
        print(f"  Last updated: {checkpoint['last_updated']}")
        if checkpoint["failed"]:
            print(f"  Failed pairs: {list(checkpoint['failed'].keys())}")
        print(f"{'='*60}\n")
        
        print("Options:")
        print("  [R] Resume from checkpoint (skip completed pairs)")
        print("  [S] Restart from scratch (delete checkpoint)")
        print("  [P] Partial restart (re-run only failed pairs)")
        print("  [L] List completed pairs and exit")
        
        choice = input("Choice [R/S/P/L]: ").strip().upper()
        
        if choice == "L":
            for pair in checkpoint["completed"]:
                print(f"  ✓ {pair}")
            sys.exit(0)
        elif choice == "S":
            self.delete()
            return "restart"
        elif choice == "P":
            return "partial"
        else:  # default R
            print(f"Resuming. Skipping {n_done} completed pairs.")
            return "resume"
    
    def mark_started(self, method, scenario):
        """Call before starting a (method, scenario) pair."""
        
    def mark_completed(self, method, scenario, result_path, n_mc):
        """Call immediately after a pair completes. Writes to disk atomically."""
        # Use atomic write: write to .tmp then rename
        
    def mark_failed(self, method, scenario, error_msg):
        """Call if a pair raises an exception."""
        
    def is_completed(self, method, scenario) -> bool:
        """Returns True if this pair is already done (for resume logic)."""
        
    def should_run(self, method, scenario, mode) -> bool:
        """Returns False if completed and mode is 'resume'."""

Also implement CLI integration:
  ofb run --resume          # auto-resume if checkpoint exists
  ofb run --restart         # always restart
  ofb run --status          # show checkpoint status and exit

═══════════════════════════════════════════════════════════════════════════════
BLOCK 1 — PACKAGE STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

Create this exact directory structure.
Do NOT delete existing working code — wrap it inside the new structure.

src/openfreqbench/
├── __init__.py               # version = "2.0.0"
├── config/
│   ├── __init__.py
│   ├── run_config.py         # RunConfig dataclass
│   └── defaults.py           # FS_PHYSICS, FS_DSP, TRIP_THR, etc.
├── core/
│   ├── __init__.py
│   ├── base_estimator.py     # BaseEstimator, EstimatorOutput, EstimatorSpec
│   ├── base_scenario.py      # BaseScenario, ScenarioOutput
│   ├── checkpoint.py         # CheckpointManager (BLOCK 0)
│   ├── registry.py           # EstimatorRegistry, ScenarioRegistry
│   └── runner.py             # BenchmarkRunner (orchestrates everything)
├── estimators/
│   ├── __init__.py
│   ├── common/
│   │   └── tuning.py         # TuningParam, TuningSpec, bayesian_tune()
│   ├── f0_pll/
│   │   ├── srf_pll.py        # StandardPLL → wrapped as plugin
│   │   └── sogi_fll.py       # SOGI_FLL → wrapped as plugin
│   ├── f1_kalman/
│   │   ├── ekf.py            # ClassicEKF → plugin
│   │   ├── ukf.py            # UKF_Estimator → plugin
│   │   ├── lkf.py            # LKF_Estimator → plugin
│   │   └── ra_ekf.py         # EKF2 (proposed RA-EKF) → plugin
│   ├── f2_window/
│   │   ├── ipdft.py          # TunableIpDFT → plugin
│   │   └── tft.py            # TFT_Estimator → plugin
│   ├── f3_recursive/
│   │   ├── rls.py            # RLS_Estimator → plugin
│   │   └── rls_vff.py        # RLS_VFF_Estimator → plugin
│   ├── f4_data_driven/
│   │   ├── koopman.py        # Koopman_RKDPmu → plugin
│   │   └── pi_gru.py         # PI-GRU → plugin
│   └── f5_legacy/
│       └── teager.py         # Teager_Estimator → plugin (legacy baseline)
├── scenarios/
│   ├── __init__.py
│   ├── synthetic/
│   │   ├── G0_E0_pure_60hz.py
│   │   ├── G1_E1_mag_step.py       # IEEE_Mag_Step
│   │   ├── G1_E2_freq_ramp.py      # IEEE_Freq_Ramp
│   │   ├── G1_E3_modulation.py     # IEEE_Modulation
│   │   ├── G2_E0_islanding.py      # IBR_Nightmare
│   │   ├── G2_E1_multi_event.py    # IBR_MultiEvent
│   │   ├── G2_E2_voltage_sag.py    # NEW: sag + phase jump
│   │   ├── G2_E3_subsynchronous.py # NEW: 5-20 Hz oscillations
│   │   └── G2_E4_blind_spot.py     # NEW: adversarial for EKF2
│   └── data/
│       └── chamorro_2021.py        # Real CSV loader
├── signals/
│   ├── __init__.py
│   ├── generator.py          # PhysicsSignalGenerator (1 MHz → 10 kHz)
│   └── noise.py              # AWGN, impulsive, non-Gaussian noise models
├── metrics/
│   ├── __init__.py
│   ├── accuracy.py           # RMSE, MAE, Peak, Energy
│   ├── protection.py         # Ttrip, MaxContiguous, ProtectionMargin
│   ├── dynamic.py            # Settling, RoCoF error, Latency
│   ├── resource.py           # CPU/sample, memory bytes, structural latency
│   └── compliance.py         # IEC/IEEE standard compliance checks
├── profiling/
│   ├── __init__.py
│   └── cpu_profiler.py       # run_and_time(), memory_profiler
├── tuning/
│   ├── __init__.py
│   ├── grid_search.py        # Original grid search (preserved)
│   ├── bayesian.py           # LHS + differential_evolution optimizer
│   └── sensitivity.py       # Perturbation analysis
├── experiments/
│   ├── __init__.py
│   ├── modes.py              # BenchmarkMode enum + runner per mode
│   └── pair_runner.py        # Single (method × scenario) execution unit
├── stats/
│   ├── __init__.py
│   ├── monte_carlo.py        # MC runner + raw data storage
│   ├── descriptive.py        # mean, std, IQR, CV, percentiles
│   ├── hypothesis.py         # All H0 tests with effect sizes
│   ├── corrections.py        # Bonferroni, BH-FDR
│   ├── bootstrap.py          # CI on correlations, rank stability
│   └── information.py        # PCA, redundancy, CRLB
├── plotting/
│   ├── __init__.py
│   ├── debug.py              # Quick diagnostic plots
│   ├── benchmark.py          # Standard benchmark figures
│   └── paper.py              # Publication-ready figures (Fig1, Fig2)
├── io/
│   ├── __init__.py
│   ├── result_schema.py      # PairResult, ScenarioResult, BenchmarkResult
│   ├── json_writer.py        # Atomic JSON writer
│   └── csv_exporter.py       # CSV summaries
├── reporting/
│   ├── __init__.py
│   └── report_generator.py  # paper_numbers.txt, scientific_summary.txt
├── cli/
│   ├── __init__.py
│   └── main_cli.py          # Click-based CLI
└── standards/
    ├── __init__.py
    ├── iec_60255_118_1.py    # IEC/IEEE standard thresholds + auto-tests
    └── ieee_c37_118.py       # IEEE C37.118 synchrophasor standard

tests/
├── unit/
│   ├── test_estimators.py
│   ├── test_scenarios.py
│   ├── test_metrics.py
│   └── test_checkpoint.py
├── integration/
│   └── test_smoke.py
└── conftest.py

docs/
├── CLAUDE.md
├── ARCHITECTURE.md
├── BENCHMARK_SPEC.md
├── CONTRIBUTING.md
├── RESULT_SCHEMA.md
├── TESTING.md
└── ROADMAP.md

pyproject.toml                # package metadata + dependencies
ROADMAP.md                    # top-level roadmap

═══════════════════════════════════════════════════════════════════════════════
BLOCK 2 — CORE INTERFACES
═══════════════════════════════════════════════════════════════════════════════

File: src/openfreqbench/core/base_estimator.py

from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class EstimatorOutput:
    frequency_hz: float          # primary output
    rocof_hz_s: float = 0.0      # RoCoF estimate (0 if not available)
    amplitude_pu: float = 1.0    # amplitude estimate
    phase_rad: float = 0.0       # phase estimate
    confidence: float = 1.0      # internal confidence [0,1] if available
    flags: dict = field(default_factory=dict)  # e.g. {"event_gating": True}

@dataclass
class TuningParam:
    name: str
    low: float
    high: float
    scale: str          # "log" | "linear" | "integer"
    default: float
    description: str = ""

@dataclass
class TuningSpec:
    params: list[TuningParam]
    objective: str = "RMSE"     # "RMSE" | "Ttrip" | "composite"
    
    def bounds_for_scipy(self):
        """Returns list of (low, high) for scipy optimizers, in log space if needed."""

@dataclass
class EstimatorSpec:
    name: str
    family: str          # "pll" | "kalman" | "window" | "recursive" | "data_driven"
    has_rocof_output: bool = False
    structural_latency_samples: int = 0
    requires_gpu: bool = False
    description: str = ""
    citation: str = ""

class BaseEstimator:
    """
    Plugin interface for all frequency estimators.
    
    Rules:
    - __init__ receives ONLY the tuning parameters (no signal, no scenario)
    - step() is the only method called during benchmark execution
    - reset() must restore the estimator to its initial state
    - The estimator must NOT access global state or random seeds directly
    - The estimator must NOT own timing, MC loops, or metric computation
    """
    
    @classmethod
    def spec(cls) -> EstimatorSpec:
        """Return static metadata about this estimator."""
        raise NotImplementedError
    
    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        """Return tuning parameter space. Used by all tuning modes."""
        raise NotImplementedError
    
    @classmethod
    def default_config(cls) -> dict:
        """Return default parameter values (must match tuning_spec defaults)."""
        raise NotImplementedError
    
    def reset(self):
        """Reset to initial state. Must be equivalent to __init__ with same params."""
        raise NotImplementedError
    
    def step(self, voltage: float) -> EstimatorOutput:
        """
        Process one DSP-rate sample. Returns EstimatorOutput.
        Must be callable in a tight loop with no side effects.
        """
        raise NotImplementedError


File: src/openfreqbench/core/base_scenario.py

@dataclass
class EventMarker:
    time_s: float
    event_type: str   # "phase_jump" | "freq_ramp_start" | "freq_ramp_end" | "amplitude_step"
    magnitude: float  # degrees for phase, Hz/s for ramp, pu for amplitude
    description: str = ""

@dataclass  
class ScenarioOutput:
    t_physics: np.ndarray      # time at fs_physics [s]
    v_physics: np.ndarray      # voltage at fs_physics [pu]
    f_physics: np.ndarray      # true frequency at fs_physics [Hz]
    t_dsp: np.ndarray          # time at fs_dsp [s]
    v_dsp: np.ndarray          # voltage at fs_dsp [pu]
    f_dsp: np.ndarray          # true frequency at fs_dsp [Hz]
    f_ref_available: bool = True
    events: list[EventMarker] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

class BaseScenario:
    """
    Plugin interface for all scenarios.
    
    Rules:
    - build(seed) is the only method called during benchmark execution
    - seed controls ALL stochastic elements (noise, etc.)
    - The scenario must NOT access global random state directly
    - Metadata must include: name, group, experiment, noise_model, citation
    """
    
    @classmethod
    def spec(cls) -> dict:
        """Return static metadata: name, group, experiment, description, citation."""
        raise NotImplementedError
    
    def build(self, seed: int = 42) -> ScenarioOutput:
        """
        Build the scenario with the given random seed.
        Deterministic given the same seed.
        """
        raise NotImplementedError

═══════════════════════════════════════════════════════════════════════════════
BLOCK 3 — IEEE/IEC STANDARDS MODULE
═══════════════════════════════════════════════════════════════════════════════

File: src/openfreqbench/standards/iec_60255_118_1.py

"""
IEC/IEEE 60255-118-1:2018 — Measuring relays and protection equipment
Part 118-1: Synchrophasor for power systems — Measurements

This module implements the NORMATIVE compliance tests from the standard.
All thresholds are from the published standard, not ad-hoc.

P-class (protection): fast response, lower accuracy
M-class (metering): slower response, higher accuracy
"""

# ── Frequency Error Limits ────────────────────────────────────────────────────
# Table 2 of IEC/IEEE 60255-118-1:2018
P_CLASS = {
    # Steady-state tests
    "freq_error_max_hz":   0.005,    # Hz — max |f_est - f_true| in steady state
    "rocof_error_max_hzs": 0.01,     # Hz/s — max RoCoF error in steady state
    # Dynamic tests
    "freq_step_response_ms":    70,  # ms — settling after frequency step
    "modulation_freq_error_hz": 0.01,# Hz — max error under AM/FM modulation
    # Latency
    "max_latency_ms":           20,  # ms — P1 class
}

M_CLASS = {
    "freq_error_max_hz":   0.001,
    "rocof_error_max_hzs": 0.001,
    "freq_step_response_ms":    420,
    "modulation_freq_error_hz": 0.001,
    "max_latency_ms":           40,
}

# ── IBR-specific thresholds (non-normative, motivated by IEC) ─────────────────
# Used for the compliance heatmap (Fig 2g). Must be labeled "non-normative".
IBR_PROTECTION = {
    "rmse_hz":        0.05,   # Hz
    "peak_hz":        0.50,   # Hz  
    "ttrip_s":        0.10,   # s
    "label": "IBR protection class (non-normative, benchmark-defined)"
}

class IECComplianceChecker:
    """
    Automatic compliance checker against IEC/IEEE 60255-118-1 thresholds.
    
    All methods return ComplianceResult with:
      pass_fail: bool
      margin: float        # how far from threshold (negative = failing)
      threshold: float     # the standard threshold
      actual: float        # actual measured value
      normative: bool      # True if from the standard, False if benchmark-defined
    """
    
    def check_steady_state_freq_error(self, f_hat, f_true,
                                       estimator_class="P") -> ComplianceResult:
        """
        IEC/IEEE 60255-118-1 Section 6.3: steady-state frequency error test.
        Evaluates max|f_hat - f_true| during the last 20% of signal duration.
        Threshold: P-class 0.005 Hz, M-class 0.001 Hz.
        """
    
    def check_step_response(self, f_hat, f_true, event_time_s,
                             fs_dsp, estimator_class="P") -> ComplianceResult:
        """
        IEC/IEEE 60255-118-1 Section 6.5: response time test.
        Measures time from event onset to |error| < threshold.
        Threshold: P-class 70ms, M-class 420ms.
        """
    
    def check_modulation(self, f_hat, f_true,
                          estimator_class="P") -> ComplianceResult:
        """
        IEC/IEEE 60255-118-1 Section 6.4: modulation test.
        Max frequency error under AM at 2 Hz.
        """
    
    def check_rocof_accuracy(self, rocof_hat, rocof_true,
                              estimator_class="P") -> ComplianceResult:
        """
        IEC/IEEE 60255-118-1 Section 7: RoCoF accuracy.
        Only applicable to methods with explicit RoCoF output.
        """
    
    def check_ibr_protection(self, rmse, peak, ttrip) -> ComplianceResult:
        """
        Non-normative IBR protection threshold check.
        Used for heatmap only. Must be labeled as benchmark-defined.
        """
    
    def run_full_compliance(self, method_name, results_dict,
                             scenario_name) -> dict:
        """
        Run ALL applicable tests for a (method, scenario) pair.
        Returns structured compliance report.
        
        Auto-detects which tests apply to which scenario:
          G1_E1 → check_step_response, check_steady_state_freq_error
          G1_E2 → check_rocof_accuracy (if RoCoF output available)
          G1_E3 → check_modulation
          G2_E0 → check_ibr_protection (non-normative)
          G2_E1 → check_ibr_protection (non-normative)
        """

Also add:

File: src/openfreqbench/standards/ieee_c37_118.py

"""
IEEE C37.118.2-2011 / C37.118.1-2011
Synchrophasor standard for power systems.

Relevant to: methods that output phasor + frequency (PMU-class).
Not all methods in this benchmark are PMU-class.
"""

TVE_LIMIT_P = 0.01   # 1% Total Vector Error for P-class
TVE_LIMIT_M = 0.01   # 1% Total Vector Error for M-class
FE_LIMIT_P  = 0.005  # Hz frequency error P-class
FE_LIMIT_M  = 0.001  # Hz frequency error M-class
RFE_LIMIT_P = 0.01   # Hz/s RoCoF error P-class

class C37118ComplianceChecker:
    """
    Checks against IEEE C37.118 thresholds.
    Only meaningful for methods that output amplitude + phase + frequency.
    For frequency-only methods, only frequency error tests apply.
    """
    
    def check_frequency_error(self, f_hat, f_true, pclass="P"):
        """Max |f_hat - f_true| in steady state."""
    
    def check_rocof_error(self, rocof_hat, rocof_true, pclass="P"):
        """Only for methods with explicit RoCoF output."""

═══════════════════════════════════════════════════════════════════════════════
BLOCK 4 — METRIC DEFINITIONS (complete, with known-answer tests)
═══════════════════════════════════════════════════════════════════════════════

File: src/openfreqbench/metrics/accuracy.py

def compute_rmse(f_hat, f_true) -> float:
    """Root Mean Square Error [Hz]. Excludes initialization period (first 50ms)."""

def compute_mae(f_hat, f_true) -> float:
    """Mean Absolute Error [Hz]."""

def compute_peak_error(f_hat, f_true) -> float:
    """Max |f_hat - f_true| [Hz]."""

def compute_error_energy(f_hat, f_true, fs_dsp) -> float:
    """Integral of e²(t) dt [Hz²·s]. Use np.trapz."""

File: src/openfreqbench/metrics/protection.py

def compute_trip_time(f_hat, f_true, threshold=0.5, fs_dsp=10000) -> float:
    """Cumulative time |e(t)| > threshold [s]."""

def compute_max_contiguous_trip(f_hat, f_true, threshold=0.5, fs_dsp=10000) -> float:
    """Longest unbroken episode where |e(t)| > threshold [s].
    This is the protection-engineering relevant metric:
    if max_contiguous > relay_delay, the relay WILL actuate."""

def compute_protection_margin(f_hat, f_true, threshold=0.5) -> float:
    """threshold - max(|e(t)|) during the event window.
    Positive = safe. Negative = false trip occurred."""

def compute_false_trip_probability(mc_f_hat_list, f_true, threshold=0.5) -> dict:
    """From MC runs: P(margin < 0), with Wilson CI."""

File: src/openfreqbench/metrics/dynamic.py

def compute_settling_time(f_hat, f_true, event_time_s, fs_dsp,
                           threshold=0.2) -> float:
    """Time from event onset until |e(t)| < threshold and stays below [s]."""

def compute_rocof_error_rms(rocof_hat, rocof_true) -> float:
    """RMS error of RoCoF estimate [Hz/s]. Only for methods with explicit output."""

def compute_structural_latency(method_instance, fs_dsp) -> float:
    """
    Structural latency in samples (e.g. IpDFT with Nc=6 cycles has Nc/2=3 cycles lag).
    For KF methods: 0 (sample-by-sample).
    Must be declared by each estimator via spec().structural_latency_samples.
    """

File: src/openfreqbench/metrics/resource.py

def measure_cpu_per_sample(estimator_factory, signal_dsp, n_reps=20) -> dict:
    """
    Returns:
      mean_us: float      — mean CPU time per sample [µs]
      std_us: float       — std across n_reps
      min_us: float
      max_us: float
      n_reps: int
      timer: str          — "time.process_time"
      
    Uses time.process_time() (CPU time, not wall time).
    Excludes first run (JIT warmup).
    Reports mean of runs 2..n_reps.
    """

def measure_memory_bytes(estimator_instance) -> int:
    """Memory footprint of the estimator object [bytes]. Uses sys.getsizeof."""

File: src/openfreqbench/metrics/compliance.py

def compute_all_metrics(f_hat, f_true, rocof_hat=None, rocof_true=None,
                         fs_dsp=10000, event_markers=None) -> dict:
    """
    Master metric computation function.
    Returns ALL metrics in a flat dict ready for JSON storage.
    
    Required keys in output (MUST be present for schema compliance):
      RMSE, MAE, MAX_PEAK, ENERGY, SETTLING,
      TRIP_TIME_0p5, MAX_CONTIGUOUS_0p5, PROTECTION_MARGIN,
      RFE_rms_Hz_s (NaN if no RoCoF output),
      TIME_PER_SAMPLE_US (filled by runner, not here)
    """

═══════════════════════════════════════════════════════════════════════════════
BLOCK 5 — MONTE CARLO ENGINE
═══════════════════════════════════════════════════════════════════════════════

File: src/openfreqbench/stats/monte_carlo.py

"""
Monte Carlo engine. Each (method, scenario) pair is independent.
Results are stored per-pair and aggregated later.
"""

@dataclass
class MCConfig:
    n_runs_stochastic: int = 100    # for scenarios D, E (high noise variance)
    n_runs_deterministic: int = 30  # for scenarios A, B, C (low variance)
    seeds: list = None              # if None, use range(2000, 2000+n_runs)
    
    def n_runs_for_scenario(self, scenario_name: str) -> int:
        """Scenarios G2_* get n_runs_stochastic; others get n_runs_deterministic."""
        stochastic = ["G2_E0", "G2_E1", "G2_E2", "IBR_Nightmare", "IBR_MultiEvent"]
        return self.n_runs_stochastic if any(s in scenario_name for s in stochastic) \
               else self.n_runs_deterministic

class MCRunner:
    """
    Runs Monte Carlo for a single (estimator_class, scenario, params) triple.
    
    IMPORTANT: MC results are stored as RAW ARRAYS, not just summary stats.
    This enables: survival analysis, PSD, rank stability, hazard functions.
    
    Output structure per pair:
    {
      "RMSE":   [30 or 100 values],
      "Ttrip":  [30 or 100 values],
      "Peak":   [30 or 100 values],
      "Energy": [30 or 100 values],
      "MaxContiguous": [30 or 100 values],
      "ProtectionMargin": [30 or 100 values],
      "RFE_rms": [30 or 100 values] or null,
      "seeds_used": [2000, 2001, ...],
      "n_runs": 30
    }
    
    Also store the TIMESERIES from the MEDIAN run (not all runs — too much disk):
    {
      "timeseries_seed": 42,  # seed of the run closest to median RMSE
      "f_hat": [...],
      "f_true": [...],
      "error": [...],
      "t": [...]
    }
    
    This timeseries is used for:
    - PSD analysis
    - Hazard function
    - Debug plots
    """
    
    def run(self, estimator_class, tuned_params, scenario,
            mc_config: MCConfig, checkpoint_mgr=None) -> dict:
        """
        Run MC for one (estimator, scenario) pair.
        
        Progress bar: use tqdm with format:
          "MC EKF2×IBR_Nightmare: 47/100 [████░░] ETA: 2m15s"
        
        Saves intermediate results every 10 runs (for crash recovery).
        """

File: src/openfreqbench/stats/descriptive.py

def compute_full_stats(values: list) -> dict:
    """
    Full distributional statistics for a list of scalar values.
    
    Returns:
      n, mean, std, median, IQR, CV,
      p5, p25, p75, p95, min, max,
      n_outliers, outlier_values,
      SW_stat, SW_p, is_normal_SW (Shapiro-Wilk, only if n >= 8 and not degenerate),
      is_degenerate: bool  # True if std < 1e-10 (all values identical)
    """

═══════════════════════════════════════════════════════════════════════════════
BLOCK 6 — HYPOTHESIS TEST SUITE (complete, with effect sizes)
═══════════════════════════════════════════════════════════════════════════════

File: src/openfreqbench/stats/hypothesis.py

"""
Pre-registered hypothesis test suite.
ALL tests must include: test_type, statistic, p_value, effect_size,
reject_H0, reject_H0_bonferroni, reject_H0_BH_FDR, n_per_group.

Effect size interpretation (stored as string):
  Mann-Whitney r: |r|<0.1 "negligible", 0.1-0.3 "small",
                  0.3-0.5 "medium", >0.5 "large"
  KW eta²: <0.01 "negligible", 0.01-0.06 "small",
            0.06-0.14 "medium", >0.14 "large"
"""

PRE_REGISTERED_HYPOTHESES = {
    
    "H1": {
        "description": "H0: Trip-Risk is identically distributed across ALL estimators under Composite Islanding (Scenario D). Kruskal-Wallis, two-sided, α=0.05.",
        "test": "kruskal_wallis",
        "metric": "Ttrip",
        "scenario": "IBR_Nightmare",
        "groups": "all_methods",
    },
    
    "H2": {
        "description": "H0: RA-EKF RMSE = EKF RMSE under Composite Islanding. Mann-Whitney U, two-sided.",
        "test": "mann_whitney_u",
        "metric": "RMSE",
        "scenario": "IBR_Nightmare",
        "group_a": "EKF2",
        "group_b": "EKF",
    },
    
    "H3": {
        "description": "H0: ρ=0 between RMSE on Ramp (B) and RMSE on Islanding (D) across methods. Spearman.",
        "test": "spearman",
        "metric": "RMSE",
        "scenarios": ["IEEE_Freq_Ramp", "IBR_Nightmare"],
    },
    
    "H4": {
        "description": "H0: Window-based and model-based families have identical RMSE under IBR Multi-Event. Mann-Whitney U.",
        "test": "mann_whitney_u",
        "metric": "RMSE",
        "scenario": "IBR_MultiEvent",
        "group_a": "family:window",
        "group_b": "family:kalman",
    },
    
    "H5": {
        "description": "H0: ρ=0 between CPU cost and Trip-Risk across methods. Spearman.",
        "test": "spearman",
        "metrics": ["CPU_us", "Ttrip"],
        "scenario": "IBR_MultiEvent",
    },
    
    "H6": {
        "description": "H0: RA-EKF Ramp RMSE ≥ SRF-PLL Ramp RMSE. One-sided Mann-Whitney (alternative: EKF2 < PLL).",
        "test": "mann_whitney_u_onesided",
        "metric": "RMSE",
        "scenario": "IEEE_Freq_Ramp",
        "group_a": "EKF2",
        "group_b": "PLL",
        "alternative": "less",
    },
    
    "H7": {
        "description": "H0: RMSE variance is homogeneous across estimators under Composite Islanding. Levene's test.",
        "test": "levene",
        "metric": "RMSE",
        "scenario": "IBR_Nightmare",
        "groups": "all_methods",
    },
    
    "H8": {
        "description": "H0: ρ=0 between RMSE rank on Magnitude Step (A) and RMSE rank on Islanding (D). Spearman rank correlation.",
        "test": "spearman",
        "metric": "RMSE",
        "scenarios": ["IEEE_Mag_Step", "IBR_Nightmare"],
    },
    
    "H9": {
        "description": "H0: Koopman and RA-EKF have identical Ttrip under IBR Multi-Event. Mann-Whitney U.",
        "test": "mann_whitney_u",
        "metric": "Ttrip",
        "scenario": "IBR_MultiEvent",
        "group_a": "Koopman-RKDPmu",
        "group_b": "EKF2",
    },
    
    "H10": {
        "description": "H0: RMSE under Islanding (D) is normally distributed per estimator. Shapiro-Wilk, α=0.05.",
        "test": "shapiro_wilk_per_method",
        "metric": "RMSE",
        "scenario": "IBR_Nightmare",
    },
    
    # NEW HYPOTHESES (from Block E findings):
    
    "H11": {
        "description": "H0: ρ=0 between RMSE on ALL IEC scenarios (A+B+C) and RMSE on Islanding (D). Permutation test.",
        "test": "permutation_spearman",
        "metric": "RMSE",
        "scenarios_x": ["IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation"],
        "scenario_y": "IBR_Nightmare",
        "n_permutations": 10000,
        "note": "Core IEC blindness test. p > 0.05 → IEC compliance predicts nothing about IBR.",
    },
    
    "H12": {
        "description": "H0: Methods WITH explicit RoCoF state (EKF2) have lower RFE_rms than methods WITHOUT. Mann-Whitney U.",
        "test": "mann_whitney_u",
        "metric": "RFE_rms_Hz_s",
        "scenario": "IEEE_Freq_Ramp",
        "group_a": "family:explicit_rocof",
        "group_b": "family:no_rocof",
        "note": "Tests whether ω̇ augmentation actually improves RoCoF estimation.",
    },
    
    "H13": {
        "description": "H0: ProtectionMargin ≥ 0 for RA-EKF in all MC runs (Islanding). One-sample sign test.",
        "test": "sign_test",
        "metric": "ProtectionMargin",
        "scenario": "IBR_Nightmare",
        "method": "EKF2",
        "h0_value": 0.0,
        "alternative": "greater",
        "note": "Tests whether RA-EKF is reliably safe, not just on average.",
    },
    
    "H14": {
        "description": "H0: IEC compliance and IBR pass rate are independent. Fisher's exact test.",
        "test": "fisher_exact",
        "note": "Formal IEC sufficiency test. 2×2 contingency table.",
    },
    
    "H15": {
        "description": "H0: Ttrip_EKF2 CV = 0 in Islanding (Ttrip is deterministic). F-test vs EKF.",
        "test": "variance_ratio",
        "metric": "Ttrip",
        "scenario": "IBR_Nightmare",
        "group_a": "EKF2",
        "group_b": "EKF",
        "note": "Tests whether EKF2 Ttrip is stochastic (CV=298%) vs EKF Ttrip is deterministic (CV=0%).",
    },
}

def run_hypothesis_suite(mc_raw, results_dict, n_alpha=0.05) -> dict:
    """
    Run ALL pre-registered hypotheses.
    Apply Bonferroni AND Benjamini-Hochberg corrections.
    Return complete results dict for JSON storage.
    
    For each test, compute and store:
      test_type: str            (e.g. "Mann-Whitney U")
      statistic: float          (U, H, W, rho, F, etc.)
      p_value: float
      effect_size: float        (rank-biserial r, eta², rho, etc.)
      effect_magnitude: str     ("negligible"|"small"|"medium"|"large")
      n_per_group: list[int]
      reject_H0: bool           (p < alpha)
      reject_H0_bonferroni: bool (p < alpha/n_tests)
      reject_H0_BH_FDR: bool   (after BH correction)
      skip_reason: str or null  (if degenerate data, explain WHY)
      
    Also compute and store:
      summary.n_tests_total
      summary.n_executed
      summary.n_skipped_with_reason
      summary.n_rejected_H0
      summary.n_rejected_bonferroni
      summary.n_rejected_BH_FDR
      summary.alpha_used
      summary.n_tests_for_bonferroni
    """

File: src/openfreqbench/stats/bootstrap.py

def bootstrap_spearman_ci(values_a, values_b, n_boot=2000,
                            ci_level=0.95) -> dict:
    """
    Bootstrap confidence interval on Spearman rho.
    Essential for the IEC blindness finding (n=8 methods, rho≈0).
    """

def bootstrap_rank_ci(mc_rmse_dict, n_boot=1000) -> dict:
    """
    Bootstrap rank confidence intervals per method.
    Returns rank_median, rank_p5, rank_p95, rank_std per method.
    rank_std=0 should be documented as "fully stable ranking."
    """

═══════════════════════════════════════════════════════════════════════════════
BLOCK 7 — SCIENTIFIC ANALYSIS MODULE (new findings)
═══════════════════════════════════════════════════════════════════════════════

File: src/openfreqbench/stats/information.py

def compute_crlb(scenario_output: ScenarioOutput, fs_dsp=10000) -> dict:
    """
    Cramér-Rao Lower Bound for frequency estimation.
    For A*sin(2πft+φ) in AWGN with σ:
      I(f) = (2π)² * A² * Σ(t_n²) / σ²
      CRLB_Hz = sqrt(1/I(f))
    
    Estimate A from signal RMS, σ from HF content of v_dsp.
    Returns: CRLB_Hz, SNR_dB, N_samples, A_est, sigma_est
    """

def compute_scenario_pca(results_matrix) -> dict:
    """
    PCA on (methods × scenarios) RMSE matrix.
    Returns: explained_variance_ratio, n_pcs_for_95pct,
             scenario_loadings, method_scores
    """

def compute_metric_redundancy(results_dict) -> dict:
    """
    Spearman correlation matrix across all metrics.
    Identifies redundant metrics (rho > 0.9).
    """

def compute_efficiency(method_rmse, crlb_hz) -> float:
    """Efficiency = CRLB / RMSE. Range (0, 1], 1=optimal."""

File: src/openfreqbench/stats/information.py (continued)

def iec_blindness_test(results_dict, n_permutations=10000) -> dict:
    """
    Permutation test for independence between IEC scenario RMSE and Islanding RMSE.
    
    For each IEC scenario (A, B, C):
      1. Compute Spearman rho(RMSE_IEC, RMSE_Islanding) across methods
      2. Run permutation test (shuffle Islanding rankings)
      3. p_permutation = fraction of permuted rhos >= |rho_obs|
    
    Store: rho_obs, p_permutation, CI_95_bootstrap, R_squared,
           conclusion: "IEC scenario X explains R²% of Islanding variance"
    
    Key output: "IEC compliance explains {R²×100:.1f}% of Islanding variance.
                 A method passing all IEC tests has P(IBR_pass) = {p:.1f}%,
                 not significantly different from chance ({1/n_methods:.1f}%)."
    """

def relay_coordination_table(results_dict,
                               relay_delays_ms=[12,50,100,150,200,500]) -> dict:
    """
    For each method: is it safe at each relay delay setting?
    safe = MAX_CONTIGUOUS_0p5 < relay_delay_s
    
    Returns: {method: {delay_ms: safe_bool}}
    
    Also generates the LaTeX table for the paper.
    This is the most actionable table for protection engineers.
    """

def stochastic_deterministic_classifier(mc_raw) -> dict:
    """
    Classify each (method, scenario, metric) as:
      DETERMINISTIC: CV < 0.05 (5%)
      MODERATE: 0.05 <= CV < 0.50
      STOCHASTIC: CV >= 0.50
    
    Key finding: EKF2 Ttrip in Islanding is STOCHASTIC (CV=298%)
    while EKF Ttrip is DETERMINISTIC (CV=0%).
    This means comparing single-run Ttrip values is statistically unfair.
    
    Store: {method: {scenario: {metric: classification}}}
    Also generate the "Stochastic Reproducibility Table" for the paper.
    """

═══════════════════════════════════════════════════════════════════════════════
BLOCK 8 — RESULT SCHEMA
═══════════════════════════════════════════════════════════════════════════════

File: src/openfreqbench/io/result_schema.py

"""
Every (method × scenario) pair produces a PairResult JSON file.
These are merged into a BenchmarkResult at the end.

File naming: results/{method}__{scenario}.json
"""

PAIR_RESULT_SCHEMA = {
    "schema_version": "2.0",
    "pair": {
        "method": str,
        "scenario": str,
        "completed_at": str,      # ISO timestamp
        "seed_main": int,          # seed used for main (deterministic) run
    },
    "tuning": {
        "mode": str,               # "grid" | "bayesian" | "manual"
        "best_params": dict,
        "best_rmse": float,
        "n_evaluations": int,
        "improvement_over_grid_pct": float,  # 0 if mode==grid
        "convergence_curve": list,
    },
    "main_run": {
        # Single-run results (deterministic, seed=42 or tuning seed)
        "RMSE": float, "MAE": float, "MAX_PEAK": float,
        "SETTLING": float, "ENERGY": float,
        "TRIP_TIME_0p5": float, "MAX_CONTIGUOUS_0p5": float,
        "PROTECTION_MARGIN": float,
        "RFE_rms_Hz_s": float,     # NaN if no RoCoF output
        "TIME_PER_SAMPLE_US": float,
        "MEMORY_BYTES": int,
        "STRUCTURAL_LATENCY_MS": float,
        "timeseries": {            # from seed=42 run
            "t": list, "f_hat": list, "f_true": list, "error": list
        }
    },
    "monte_carlo": {
        "n_runs": int,
        "seeds": list,
        "raw": {
            "RMSE": list, "Ttrip": list, "Peak": list,
            "Energy": list, "MaxContiguous": list, "ProtectionMargin": list
        },
        "stats": {
            # compute_full_stats() for each metric
            "RMSE": {FULL_STATS_DICT},
            "Ttrip": {FULL_STATS_DICT},
            # ...
        },
        "stochastic_classification": {
            "RMSE": str,   # "deterministic" | "moderate" | "stochastic"
            "Ttrip": str,
        },
        "timeseries_median_run": {
            # timeseries from the run with RMSE closest to median
            "seed": int, "t": list, "f_hat": list, "f_true": list
        }
    },
    "compliance": {
        "iec_60255_118_1": {COMPLIANCE_REPORT},
        "ieee_c37_118": {COMPLIANCE_REPORT},   # if applicable
        "ibr_protection": {COMPLIANCE_REPORT},  # non-normative
    },
    "tuning_sensitivity": {
        # Only if BLOCK F is run
        "sensitivity_score": float,
        "robustness_score": float,
        "worst_param": str,
    },
}

BENCHMARK_RESULT_SCHEMA = {
    "schema_version": "2.0",
    "metadata": {METADATA_DICT},
    "results": {
        "IEEE_Mag_Step": {"methods": {method: {main_run_metrics}}},
        # ...
    },
    "monte_carlo": {AGGREGATED_MC},
    "cpu_authoritative": {CPU_DICT},
    "statistical_analysis": {
        "summary": {},
        "enhanced_mc_statistics": {},
        "hypothesis_tests": {H1..H15},
        "iec_blindness_test": {},
        "crlb_analysis": {},
        "scenario_redundancy": {},
        "metric_redundancy": {},
        "relay_coordination_table": {},
        "stochastic_classification": {},
        "paper_claims_verification": {},
        "mc_citation_strings": {},
        "bootstrap_rank_confidence": {},
        "pareto_frontier": {},
        "behavioral_taxonomy": {},
    },
    "tuning_traces": {},
    "chamorro_generalization": {},   # if CSV is available
}

═══════════════════════════════════════════════════════════════════════════════
BLOCK 9 — CLI
═══════════════════════════════════════════════════════════════════════════════

File: src/openfreqbench/cli/main_cli.py

Use Click. Install as: pip install -e ".[dev]" → command: ofb

ofb version
  → "OpenFreqBench 2.0.0 | Python 3.13 | NumPy 2.x | SciPy 1.x"

ofb doctor
  → Check: scipy, numpy, sklearn, tqdm, click, pandas
  → Check: chamorro_scenario.csv exists
  → Check: results/ directory writable
  → Check: figures/ directory writable
  → Print: PASS/FAIL for each

ofb list estimators
  → Table: name | family | has_rocof | latency_ms | citation
  
ofb list scenarios
  → Table: name | group | experiment | duration_s | noise_model

ofb status
  → Show checkpoint state: n_done/n_total, elapsed, last_updated
  → List failed pairs if any

ofb run [OPTIONS]
  Options:
    --methods TEXT        comma-separated list, default=all
    --scenarios TEXT      comma-separated list, default=all
    --mode TEXT           best_case|transfer|robustness, default=best_case
    --mc-n INT            Monte Carlo runs override
    --tuning TEXT         grid|bayesian, default=bayesian
    --resume/--restart    checkpoint behavior (prompts if neither given)
    --output-dir TEXT     default=results/
    --figures/--no-figures  generate plots, default=True
    --paper-figures/--no-paper-figures  generate publication figures
    --standards/--no-standards  run IEC compliance checks, default=True
    --seed INT            global seed, default=42
  
  Behavior:
    1. Show checkpoint dialog if checkpoint exists
    2. For each (method × scenario) pair not yet completed:
       a. Run Bayesian tuning (if not cached)
       b. Run main (deterministic) benchmark
       c. Run Monte Carlo
       d. Run IEC compliance checks
       e. Compute all metrics and stats
       f. Write PairResult JSON atomically
       g. Mark as completed in checkpoint
    3. Merge all PairResults into BenchmarkResult
    4. Run statistical analysis suite
    5. Generate figures
    6. Generate reports

ofb new estimator NAME --family FAMILY
  → Generate estimators/fN_family/name.py with BaseEstimator template
  → Generate tests/unit/test_estimator_name.py with all required tests
  → Print: "Created: src/openfreqbench/estimators/f1_kalman/name.py"

ofb new scenario NAME --group G2
  → Generate scenarios/synthetic/G2_EX_name.py with BaseScenario template
  → Generate tests/unit/test_scenario_name.py
  
ofb report [--input results/] [--format pdf|md|txt]
  → Generate scientific_summary.txt and paper_numbers.txt

═══════════════════════════════════════════════════════════════════════════════
BLOCK 10 — TEST SUITE
═══════════════════════════════════════════════════════════════════════════════

File: tests/unit/test_estimators.py

REQUIRED_TESTS_PER_ESTIMATOR = [
    "test_instantiate_with_default_config",
    "test_instantiate_with_tuning_spec_defaults",
    "test_reset_restores_initial_state",
    "test_step_returns_estimator_output",
    "test_step_output_no_nan",
    "test_step_output_in_valid_range",    # 40 <= f <= 80 Hz
    "test_step_pure_60hz_converges",      # pure 60Hz signal → output ≈ 60 Hz ±0.1
    "test_spec_returns_estimator_spec",
    "test_tuning_spec_bounds_valid",      # low < high for all params
    "test_default_config_matches_spec",   # defaults within bounds
    "test_n_steps_equals_n_outputs",
    "test_mc_reproducible_with_seed",     # same seed → same output
]

File: tests/unit/test_scenarios.py

REQUIRED_TESTS_PER_SCENARIO = [
    "test_build_returns_scenario_output",
    "test_shapes_consistent",             # all arrays same length
    "test_f_dsp_in_valid_range",          # 40 <= f <= 80 Hz
    "test_v_dsp_amplitude_reasonable",    # 0.5 <= A_rms <= 1.5 pu
    "test_metadata_complete",             # all required keys present
    "test_events_list_valid",             # event times within signal duration
    "test_deterministic_with_seed",       # same seed → identical output
    "test_different_seeds_different_noise",
]

File: tests/unit/test_metrics.py

def test_rmse_known_answer():
    f_hat = np.array([60.1] * 1000)
    f_true = np.array([60.0] * 1000)
    assert abs(compute_rmse(f_hat, f_true) - 0.1) < 1e-10

def test_trip_time_zero_when_always_below_threshold():
    f_hat = np.full(10000, 60.2)   # error = 0.2 Hz < 0.5 threshold
    f_true = np.full(10000, 60.0)
    assert compute_trip_time(f_hat, f_true) == 0.0

def test_trip_time_full_when_always_above():
    f_hat = np.full(10000, 61.0)   # error = 1.0 Hz > 0.5 threshold
    f_true = np.full(10000, 60.0)
    assert abs(compute_trip_time(f_hat, f_true) - 1.0) < 1e-6  # 1s

def test_protection_margin_positive_when_safe():
    f_hat = np.full(10000, 60.3)   # error = 0.3 Hz, threshold = 0.5
    f_true = np.full(10000, 60.0)
    assert compute_protection_margin(f_hat, f_true) == pytest.approx(0.2, abs=1e-6)

def test_percentile_ordering():
    vals = np.random.randn(100)
    stats = compute_full_stats(vals.tolist())
    assert stats["p5"] <= stats["p25"] <= stats["median"] <= stats["p75"] <= stats["p95"]

File: tests/integration/test_smoke.py

def test_smoke_single_pair():
    """Run EKF2 × IBR_Nightmare for 5 MC runs. Should complete in < 30s."""
    from openfreqbench.core.runner import BenchmarkRunner
    from openfreqbench.config.run_config import RunConfig
    
    config = RunConfig(
        methods=["EKF2"],
        scenarios=["IBR_Nightmare"],
        mc_n_runs=5,
        tuning_mode="grid",
        generate_figures=False,
        run_standards=False,
    )
    runner = BenchmarkRunner(config)
    result = runner.run()
    
    assert "EKF2" in result.results["IBR_Nightmare"]["methods"]
    metrics = result.results["IBR_Nightmare"]["methods"]["EKF2"]
    assert not np.isnan(metrics["RMSE"])
    assert metrics["RMSE"] < 1.0   # sanity check

def test_checkpoint_resume():
    """Run 2 pairs, interrupt after first, resume and verify second completes."""

def test_json_output_valid_schema():
    """Run smoke, load JSON, validate against PAIR_RESULT_SCHEMA."""

═══════════════════════════════════════════════════════════════════════════════
BLOCK 11 — CHAMORRO SCENARIO (real data)
═══════════════════════════════════════════════════════════════════════════════

File: src/openfreqbench/scenarios/data/chamorro_2021.py

(Full implementation as previously specified in the CHAMORRO BLOCK H.)
Key additions:

1. Auto-detect CSV columns by name (case-insensitive).
2. Normalize voltage to ~1 pu if in Volts.
3. Resample to fs_target using scipy.signal.resample_poly.
4. Estimate THD from FFT of 1s window.
5. Store ethical_note and methodological_note in meta.
6. If f_ref not in CSV:
   - Use constant 60 Hz as placeholder
   - Set f_ref_available=False
   - Compute relative_rmse (std of residuals, not absolute error)
   - Print clear warning: "No ground truth frequency in CSV.
     RMSE will be relative (estimation noise), not absolute."

═══════════════════════════════════════════════════════════════════════════════
BLOCK 12 — DOCUMENTATION
═══════════════════════════════════════════════════════════════════════════════

Generate all these files with real content (not placeholders):

CLAUDE.md — project overview for Claude/AI assistants
  - What this project does
  - How to run it
  - Key files and their purpose
  - What NOT to change (core interfaces)
  - How to add a new estimator
  - How to add a new scenario

ARCHITECTURE.md — technical design decisions
  - Why plugin-based
  - Why separate metric computation from estimators
  - Why per-pair result files
  - Checkpoint system design
  - MC sample size justification

BENCHMARK_SPEC.md — scientific specification
  - Benchmark philosophy (best-case first, transfer second)
  - Scenario definitions with full parameter tables
  - Metric definitions with formulas
  - IEC compliance test definitions
  - Monte Carlo protocol
  - Statistical test registry (H1-H15)

CONTRIBUTING.md — how to contribute
  - How to add an estimator (with template)
  - How to add a scenario (with template)
  - Required tests
  - Code style

RESULT_SCHEMA.md — JSON schema documentation
  - Full PairResult schema with descriptions
  - Full BenchmarkResult schema
  - How to load and process results

ROADMAP.md — structured roadmap with P0/P1/P2 tickets
  (Full roadmap as specified in the original prompt section 9)

═══════════════════════════════════════════════════════════════════════════════
BLOCK 13 — TUNING (Bayesian, replaces grid search)
═══════════════════════════════════════════════════════════════════════════════

(Full implementation as specified in BLOCK F of previous prompt.)
Key additions:

1. USE_BAYESIAN_TUNING = True flag in config/defaults.py
2. Falls back to grid search if scipy differential_evolution fails
3. Stores full convergence curve per (method, scenario) in PairResult
4. Computes improvement_over_grid_pct for every pair
5. Caches tuned params to .ofb_tuning_cache.json
   → On resume, loads cached params instead of re-tuning
   → Cache key = hash(method_name + scenario_name + param_bounds)

═══════════════════════════════════════════════════════════════════════════════
VERIFICATION CHECKLIST — run after each BLOCK
═══════════════════════════════════════════════════════════════════════════════

After BLOCK 0 (Checkpoint):
  ✓ ofb run --status shows correct output on empty dir
  ✓ Checkpoint survives process kill (Ctrl+C mid-run)
  ✓ Resume skips completed pairs
  ✓ Atomic write: no partial JSON files

After BLOCK 1 (Package structure):
  ✓ import openfreqbench succeeds
  ✓ All old functionality still works (no regressions)

After BLOCK 2 (Interfaces):
  ✓ All existing estimators wrapped as plugins
  ✓ BaseEstimator.step() returns EstimatorOutput for all

After BLOCK 3 (Standards):
  ✓ IECComplianceChecker.run_full_compliance() returns dict for EKF2/G1_E1
  ✓ Non-normative tests labeled correctly

After BLOCK 4 (Metrics):
  ✓ All known-answer tests pass
  ✓ compute_all_metrics() returns required keys

After BLOCK 5 (MC Engine):
  ✓ MC produces raw arrays, not just summary stats
  ✓ Timeseries of median run saved
  ✓ Progress bar shows correct ETA

After BLOCK 6 (Hypothesis tests):
  ✓ All 15 tests run without exception
  ✓ No test has skip_reason=None when skipped
  ✓ All non-skipped tests have effect_size, test_type, statistic

After BLOCK 7 (Scientific analysis):
  ✓ IEC blindness: rho(Step, Islanding) ≈ -0.143, p_permutation > 0.05
  ✓ CRLB computed for all scenarios
  ✓ Relay coordination table has all methods

After BLOCK 8 (Schema):
  ✓ PairResult JSON validates against schema
  ✓ All required keys present

After BLOCK 9 (CLI):
  ✓ ofb doctor passes
  ✓ ofb list estimators shows all 13 methods
  ✓ ofb run --help shows all options

After BLOCK 10 (Tests):
  ✓ pytest tests/ -x passes with 0 failures
  ✓ Smoke test completes in < 60s

After BLOCK 11 (Chamorro):
  ✓ Load succeeds even without f_ref column
  ✓ All estimators run on Chamorro data
  ✓ ethical_note present in JSON

After BLOCK 12 (Docs):
  ✓ All .md files have real content
  ✓ ROADMAP.md has P0/P1/P2 tickets

After BLOCK 13 (Bayesian tuning):
  ✓ Improvement table printed after tuning
  ✓ Cache survives restart
  ✓ Convergence curves in JSON

═══════════════════════════════════════════════════════════════════════════════
IMPLEMENTATION ORDER AND TIMING
═══════════════════════════════════════════════════════════════════════════════

Implement in this exact order. Do not skip ahead.

  BLOCK 0  — Checkpoint system         (implement first, always)
  BLOCK 1  — Package structure          (scaffold, no logic yet)
  BLOCK 2  — Core interfaces            (BaseEstimator, BaseScenario)
  BLOCK 4  — Metrics                    (needed by everything)
  BLOCK 3  — Standards                  (needs metrics)
  BLOCK 8  — Result schema              (needed by runner)
  BLOCK 5  — MC engine                  (needs schema + metrics)
  BLOCK 6  — Hypothesis tests           (needs MC data)
  BLOCK 7  — Scientific analysis        (needs hypothesis + MC)
  BLOCK 13 — Bayesian tuning            (needs metrics + scenarios)
  BLOCK 11 — Chamorro scenario          (independent)
  BLOCK 9  — CLI                        (needs everything)
  BLOCK 10 — Tests                      (verify everything)
  BLOCK 12 — Documentation              (last, after code is stable)

After each block: run the smoke test (BLOCK 10 test_smoke_single_pair).
If it fails: fix before moving to next block.

First before run all, I want a fully report of all it needs to be done, in tickets, in ROADMAP2.md. also we are in windows so the commands doesn run like ofb run but python -m openfreqbench run examples/quick_smoke_step_plots.yaml instead i think. 