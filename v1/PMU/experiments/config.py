# experiments/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================
# Grid configuration (hyperparameter search spaces)
# ============================================================

@dataclass(frozen=True)
class GridConfig:
    """
    Hyperparameter grids for tuning (paper-facing).
    Keep centralized to avoid config drift.
    """

    # IpDFT
    ipdft_cycles: List[int] = field(default_factory=lambda: [2, 3, 4, 6, 8, 10, 12, 20, 50, 100])
    ipdft_decim: List[int] = field(default_factory=lambda: [1, 10, 50, 100])
    ipdft_window_type: List[str] = field(default_factory=lambda: ["hann"])

    # PLL
    pll_kp: List[float] = field(default_factory=lambda: [float(x) for x in np.linspace(1, 60, 20)])
    pll_ki: List[float] = field(default_factory=lambda: [float(x) for x in np.linspace(1, 200, 30)])

    # EKF / UKF / LKF share Q/R grids
    kf_q: List[float] = field(default_factory=lambda: [float(x) for x in np.logspace(-2, 4, 8)])
    kf_r: List[float] = field(default_factory=lambda: [float(x) for x in np.logspace(-4, 1, 8)])

    # SOGI-FLL
    sogi_k: List[float] = field(default_factory=lambda: [1.0, 1.414, 2.0])
    sogi_g: List[float] = field(default_factory=lambda: [50.0, 100.0, 200.0, 300.0])

    # RLS
    rls_lam: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995])
    rls_win: List[int] = field(default_factory=lambda: [50, 100, 200, 500])

    # Heuristic / TF
    teager_win: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50])
    tft_win: List[int] = field(default_factory=lambda: [2, 3, 4, 6])

    # VFF-RLS scenario-tuned (if you expose these in a dedicated estimator)
    vff_lam_min: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.98, 0.99])
    vff_ka: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0])
    vff_win_smooth: List[int] = field(default_factory=lambda: [10, 20, 40])
    vff_decim: List[int] = field(default_factory=lambda: [1, 10, 50, 100])

    # Koopman window
    koopman_win: List[int] = field(default_factory=lambda: [10, 40, 80, 160, 333, 800, 2000, 5000])


# ============================================================
# Monte Carlo configuration
# ============================================================

@dataclass(frozen=True)
class MonteCarloPerturbConfig:
    """Randomization knobs used for robustness claims."""
    amp_pct_jitter: float = 0.05
    snr_db_jitter: float = 2.0
    impulsive_prob: float = 0.002
    impulsive_scale: float = 6.0


@dataclass(frozen=True)
class MonteCarloConfig:
    """Monte Carlo experiment configuration."""
    n_train_seeds: int = 15
    n_test_seeds: int = 50
    base_seed: int = 42

    # Portion of the record used for tuning (rest used for evaluation)
    tune_frac: float = 0.99
    report_percentiles: List[int] = field(default_factory=lambda: [5, 50, 95])

    perturb: MonteCarloPerturbConfig = field(default_factory=MonteCarloPerturbConfig)

    # IMPORTANT: no leakage by default
    tune_on_train_segment_only: bool = True
    evaluate_on_test_segment_only: bool = True


# ============================================================
# Experiment configuration (TOP LEVEL)
# ============================================================

@dataclass
class ExperimentConfig:
    """
    Top-level experiment config used by main.py.

    Key rules (benchmark-grade):
    - downsampling_ratio must be a POSITIVE INTEGER.
    - We DO NOT silently "adjust" fs_dsp_hz; we validate instead.
      If you want a different DSP fs, change the config to a divisor of fs_physics_hz.
    """

    fs_physics_hz: float
    fs_dsp_hz: float
    seed: int

    # Human-readable identifier (paper/provenance)
    name: str = "default"

    # (Optional) legacy knob; runners can ignore it if they use mc.n_train/n_test
    n_mc: int = 10

    # Output roots
    out_dir: str = "figures_estimatores_benchmark"
    results_raw_dir: str = "results_raw"
    results_mc_dir: str = "results_mc"

    # Tuning grids (preferred)
    grids: GridConfig = field(default_factory=GridConfig)

    # Backward-compat: allow JSON to carry "tuners" separately (main.py maps it into registry cfg)
    tuners: Dict[str, Any] = field(default_factory=dict)

    # Monte Carlo
    mc: Optional[MonteCarloConfig] = field(default_factory=MonteCarloConfig)

    # Optional protocol blocks
    scenario: Dict[str, Any] = field(default_factory=dict)
    registry: Dict[str, Any] = field(default_factory=dict)

    # Methods to run (MUST match registry keys OR you provide aliases in registry)
    methods: List[str] = field(default_factory=lambda: [
        "IpDFT", "PLL", "EKF", "EKF2", "SOGI", "RLS", "Teager", "TFT",
        "RLS-VFF", "UKF", "LKF", "Koopman-RKDPmu", "PI-GRU"
    ])

    # Benchmark mode flags
    enable_landscapes: bool = True
    enable_per_scenario_plots: bool = True
    enable_global_summaries: bool = True

    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    # Derived
    downsampling_ratio: int = field(init=False)

    def __post_init__(self) -> None:
        # ---- basic checks
        if not np.isfinite(self.fs_physics_hz) or self.fs_physics_hz <= 0:
            raise ValueError(f"fs_physics_hz must be > 0 (got {self.fs_physics_hz})")
        if not np.isfinite(self.fs_dsp_hz) or self.fs_dsp_hz <= 0:
            raise ValueError(f"fs_dsp_hz must be > 0 (got {self.fs_dsp_hz})")

        # ---- strict integer ratio (reviewer-proof)
        ratio_f = float(self.fs_physics_hz) / float(self.fs_dsp_hz)
        ratio = int(round(ratio_f))

        if ratio <= 0:
            raise ValueError(f"downsampling_ratio computed invalid ({ratio}) from {ratio_f}")

        # We require fs_physics_hz / ratio == fs_dsp_hz within tight tol AND also close to integer ratio
        actual_dsp = float(self.fs_physics_hz) / float(ratio)
        if abs(actual_dsp - float(self.fs_dsp_hz)) > 1e-12:
            msg = (
                f"Invalid sampling configuration: fs_physics_hz={self.fs_physics_hz} is not an integer multiple "
                f"of fs_dsp_hz={self.fs_dsp_hz}. Closest integer ratio is {ratio}, which implies fs_dsp_hz={actual_dsp}.\n"
                f"Fix: choose fs_dsp_hz such that fs_physics_hz % fs_dsp_hz == 0 (in exact rational terms), "
                f"or set fs_dsp_hz={actual_dsp} explicitly."
            )
            raise ValueError(msg)

        self.downsampling_ratio = ratio


# ============================================================
# Backward-compat aliases
# ============================================================

GridCfg = GridConfig
MCConfig = MonteCarloConfig
MCPerturbConfig = MonteCarloPerturbConfig
