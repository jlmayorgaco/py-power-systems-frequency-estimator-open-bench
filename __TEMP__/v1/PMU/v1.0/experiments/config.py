# experiments/config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
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
    ipdft_cycles: List[int] = field(
        default_factory=lambda: [2, 3, 4, 6, 8, 10, 12, 20, 50, 100]
    )
    ipdft_decim: List[int] = field(default_factory=lambda: [1, 10, 50, 100])
    ipdft_window_type: List[str] = field(default_factory=lambda: ["hann"])

    # PLL
    pll_kp: List[float] = field(
        default_factory=lambda: [float(x) for x in np.linspace(1, 60, 20)]
    )
    pll_ki: List[float] = field(
        default_factory=lambda: [float(x) for x in np.linspace(1, 200, 30)]
    )

    # EKF / UKF / LKF share Q/R grids
    kf_q: List[float] = field(
        default_factory=lambda: [float(x) for x in np.logspace(-2, 4, 8)]
    )
    kf_r: List[float] = field(
        default_factory=lambda: [float(x) for x in np.logspace(-4, 1, 8)]
    )

    # SOGI-FLL
    sogi_k: List[float] = field(default_factory=lambda: [1.0, 1.414, 2.0])
    sogi_g: List[float] = field(default_factory=lambda: [50.0, 100.0, 200.0, 300.0])

    # RLS
    rls_lam: List[float] = field(
        default_factory=lambda: [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995]
    )
    rls_win: List[int] = field(default_factory=lambda: [50, 100, 200, 500])

    # Heuristic / TF
    teager_win: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50])
    tft_win: List[int] = field(default_factory=lambda: [2, 3, 4, 6])

    # VFF-RLS scenario-tuned
    vff_lam_min: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.98, 0.99])
    vff_ka: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0])
    vff_win_smooth: List[int] = field(default_factory=lambda: [10, 20, 40])
    vff_decim: List[int] = field(default_factory=lambda: [1, 10, 50, 100])

    # Koopman window
    koopman_win: List[int] = field(
        default_factory=lambda: [10, 40, 80, 160, 333, 800, 2000, 5000]
    )


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

    tune_frac: float = 0.99
    report_percentiles: List[int] = field(default_factory=lambda: [5, 50, 95])

    perturb: MonteCarloPerturbConfig = field(default_factory=MonteCarloPerturbConfig)

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
    """

    fs_physics_hz: float
    fs_dsp_hz: float
    seed: int

    name: str = "default"
    n_mc: int = 10

    out_dir: str = "figures_estimatores_benchmark"
    results_raw_dir: str = "results_raw"
    results_mc_dir: str = "results_mc"

    grids: GridConfig = field(default_factory=GridConfig)
    tuners: Dict[str, Any] = field(default_factory=dict)

    mc: Optional[MonteCarloConfig] = field(default_factory=MonteCarloConfig)

    scenario: Dict[str, Any] = field(default_factory=dict)
    registry: Dict[str, Any] = field(default_factory=dict)

    # IMPORTANT: default methods now match registry keys (see experiments/registry.py)
    methods: List[str] = field(
        default_factory=lambda: [
            "RA-EKF",
            "RA-EKF2",
            "CKF",
            "UKF",
            "IEKF",
            "EnKF",
            "EKF",
            "LKF",
            "IpDFT",
            "TFT",
            "SOGI-Classic",
            "SOGI-Industrial",
            "MSOGI-FLL",
            "SRF-PLL",
            "MAF-SRF-PLL",
            "DDSRF-PLL",
            "RLS",
            "RLS-VFF",
            "Teager",
        ]
    )

    enable_landscapes: bool = True
    enable_per_scenario_plots: bool = True
    enable_global_summaries: bool = True

    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    downsampling_ratio: int = field(init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.fs_physics_hz) or self.fs_physics_hz <= 0:
            raise ValueError(f"fs_physics_hz must be > 0 (got {self.fs_physics_hz})")
        if not np.isfinite(self.fs_dsp_hz) or self.fs_dsp_hz <= 0:
            raise ValueError(f"fs_dsp_hz must be > 0 (got {self.fs_dsp_hz})")

        ratio_f = float(self.fs_physics_hz) / float(self.fs_dsp_hz)
        ratio = int(round(ratio_f))
        if ratio <= 0:
            raise ValueError(
                f"downsampling_ratio computed invalid ({ratio}) from {ratio_f}"
            )

        actual_dsp = float(self.fs_physics_hz) / float(ratio)
        if abs(actual_dsp - float(self.fs_dsp_hz)) > 1e-12:
            raise ValueError(
                f"Invalid sampling configuration: fs_physics_hz={self.fs_physics_hz} is not an integer multiple "
                f"of fs_dsp_hz={self.fs_dsp_hz}. Closest integer ratio is {ratio}, which implies fs_dsp_hz={actual_dsp}."
            )

        self.downsampling_ratio = ratio

    def to_dict(self) -> Dict[str, Any]:
        """Convenience: JSON-like dict for registry/tuning runners."""
        d = asdict(self)
        # dataclasses->dict already converts nested configs; keep explicit naming
        return d


# Backward-compat aliases
GridCfg = GridConfig
MCConfig = MonteCarloConfig
MCPerturbConfig = MonteCarloPerturbConfig
