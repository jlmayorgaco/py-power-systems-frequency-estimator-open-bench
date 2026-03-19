#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/runners/run_basic.py

Clean orchestrator that reuses ALL logic from:
    pfebench/runners/validate_estimator.py

✅ No changes required to your ScenarioBase dataclass scenarios (build()).
✅ No changes required to your estimators.

We adapt ScenarioBase -> "validate_estimator expected interface" via ScenarioAdapter
(fs, f0, name, params(), generate()).

Extras:
- Logging (INFO/DEBUG)
- Progress bars (requires tqdm)
  pip install tqdm
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, is_dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

# Progress bar
try:
    from tqdm import tqdm  # type: ignore

    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore
    _HAS_TQDM = False

from pfebench.estimators.base import BaseEstimator

# --- Methods
from pfebench.estimators.e1_zc import ZeroCrossingEstimator
from pfebench.estimators.e2_izc import InterpolatedZeroCrossingEstimator
from pfebench.estimators.e3_pm import PeriodMeasurementEstimator
from pfebench.estimators.e4_if_dphi import InstantaneousFrequencyPhaseIncrementEstimator
from pfebench.estimators.e5_zc_ma import ZeroCrossingMAEstimator
from pfebench.estimators.e6_mwls import MovingWindowLeastSquaresEstimator
from pfebench.estimators.e7_rls_basic import RecursiveLeastSquaresBasicEstimator
from pfebench.estimators.e7_rls_ibr import RecursiveLeastSquaresIBREstimator
from pfebench.estimators.e7_wls import WindowedLeastSquaresEstimator
from pfebench.estimators.e8_ar import AutoregressiveFrequencyEstimator
from pfebench.estimators.e9_prony import PronyMethodEstimator
from pfebench.estimators.e10_nr import NewtonRaphsonFrequencyEstimator

# --- Your ScenarioBase dataclass scenarios
from pfebench.scenarios.G1_E1_Pure_60Hz import G1_E1_Pure_60Hz
from pfebench.scenarios.G2_E7_Freq_Step_60_to_55 import G2_E7_Freq_Step_60_to_55
from pfebench.scenarios.G2_E8_Fast_Ramp_plus5Hzs import G2_E8_Fast_Ramp_plus5Hzs

# --- Reuse the whole pipeline from validate_estimator.py
from pfebench.runners.validate_estimator import (
    NumpyEncoder,
    ensure_dir,
    safe_slug,
    method_tag,
    ts_now,
    optimize_estimator,
    run_mini_mc,
    run_statistical_analysis,
    save_artifacts,
    compare_methods,
)


# =============================================================================
# Logging
# =============================================================================


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("pfebench.run_basic")
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# =============================================================================
# 0) Adapter: ScenarioBase(build) -> validate_estimator interface (generate)
# =============================================================================


class ScenarioAdapter:
    """
    Wraps a ScenarioBase dataclass instance that implements build()
    and exposes the interface expected by validate_estimator.py:

      - .fs (float)
      - .f0 (float)
      - .name (str)
      - .params() -> dict
      - .generate(duration_s, seed) -> (t, v, f_true)
    """

    def __init__(self, scenario_obj: Any):
        self._base = scenario_obj

        # Metadata from dataclass fields (cheap, no build)
        self.name = str(
            getattr(self._base, "scenario_id", self._base.__class__.__name__)
        )
        self.fs = float(getattr(self._base, "fs_hz", 1000.0))
        self.f0 = float(getattr(self._base, "f_nom_hz", 60.0))

        # Cache for latest schema produced by build()
        self._schema: Dict[str, Any] = {}

    def params(self) -> Dict[str, Any]:
        # Prefer last built schema (rich + Q1-grade); fallback to dataclass asdict
        if self._schema:
            return dict(self._schema)
        if is_dataclass(self._base):
            return asdict(self._base)
        return {"scenario_id": self.name}

    def generate(
        self, duration_s: float, seed: Optional[int] = None
    ) -> Tuple[Any, Any, Any]:
        """
        duration_s overrides T_s if scenario has that field.
        seed overrides seed if scenario has that field.

        Returns: t, v, f_true
        """
        s = self._base

        if is_dataclass(s):
            # Override seed if present
            if seed is not None and hasattr(s, "seed"):
                s = replace(s, seed=int(seed))

            # Override duration if present
            if hasattr(s, "T_s"):
                s = replace(s, T_s=float(duration_s))

        out = s.build()
        st = out.state

        # refresh interface fields from built state (truth source)
        self.fs = float(getattr(st, "fs_hz"))
        self.f0 = float(getattr(st, "f_nom_hz"))
        self._schema = dict(getattr(st, "schema", {}))

        return st.t, st.v, st.f_true


# =============================================================================
# 1) Basic Config
# =============================================================================


def build_methods() -> List[Type[BaseEstimator]]:
    return [
        ZeroCrossingEstimator,
        InterpolatedZeroCrossingEstimator,
        PeriodMeasurementEstimator,
        InstantaneousFrequencyPhaseIncrementEstimator,
        ZeroCrossingMAEstimator,
        MovingWindowLeastSquaresEstimator,
        RecursiveLeastSquaresBasicEstimator,
    ]


def build_scenarios() -> List[ScenarioAdapter]:
    """
    Scenario instances with explicit initial conditions.
    These are ONLY configuration choices, not model changes.
    """
    scenarios_raw = [
        # ------------------------------------------------------------------
        # G1 — Baseline nominal (ideal reference)
        # ------------------------------------------------------------------
        G1_E1_Pure_60Hz(
            fs_hz=10_000.0,
            T_s=5.0,
            f_nom_hz=60.0,
            A0=1.0,
            phi0_rad=0.0,
            seed=0,
        ),
        # ------------------------------------------------------------------
        # G2-E7 — Severe frequency collapse (islanding / gen trip)
        # 60 Hz → 55 Hz at t = 2.5 s
        # ------------------------------------------------------------------
        G2_E7_Freq_Step_60_to_55(
            fs_hz=10_000.0,
            T_s=5.0,
            f_nom_hz=60.0,
            A0=1.0,
            phi0_rad=0.0,
            seed=42,
            step_time_s=2.5,
            step_size_hz=-5.0,
        ),
        # ------------------------------------------------------------------
        # G2-E8 — Fast ROCOF event (+5 Hz/s)
        # Starts at t = 1.0 s
        # ------------------------------------------------------------------
        G2_E8_Fast_Ramp_plus5Hzs(
            fs_hz=10_000.0,
            T_s=5.0,
            f_nom_hz=60.0,
            A0=1.0,
            phi0_rad=0.0,
            seed=99,
            ramp_start_time_s=1.0,
            ramp_rate_hz_s=5.0,
        ),
    ]

    return [ScenarioAdapter(s) for s in scenarios_raw]


def scenario_duration_default(adapter: ScenarioAdapter, fallback: float = 1.0) -> float:
    # Prefer dataclass field T_s if present
    base = getattr(adapter, "_base", None)
    if base is not None and hasattr(base, "T_s"):
        try:
            return float(getattr(base, "T_s"))
        except Exception:
            pass

    # Next best: schema if already built
    p = adapter.params()
    if "T_s" in p:
        try:
            return float(p["T_s"])
        except Exception:
            pass

    return float(fallback)


# =============================================================================
# Helpers
# =============================================================================


def _fmt_eta(seconds: float) -> str:
    if not (seconds >= 0.0 and seconds < 1e12):
        return "ETA=?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"ETA={h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"ETA={m:d}m{s:02d}s"
    return f"ETA={s:d}s"


# =============================================================================
# 2) Runner
# =============================================================================


def run_one_scenario(
    scenario: ScenarioAdapter,
    methods: List[Type[BaseEstimator]],
    out_root: str,
    duration_s: float,
    n_runs: int,
    alpha: float,
    targets: Dict[str, float],
    warmup: int,
    export_waveforms: bool,
    logger: logging.Logger,
    job_counter: int,
    total_jobs: int,
) -> int:
    scen_name = scenario.name
    scen_dir = os.path.join(out_root, safe_slug(scen_name))
    ensure_dir(scen_dir)

    logger.info(f"→ Building calibration waveform (seed=999) for {scen_name} ...")
    _, v_cal, f_cal = scenario.generate(duration_s=duration_s, seed=999)

    all_runs_dfs: List[pd.DataFrame] = []

    # Progress bar over methods (nice but optional)
    method_iter = methods
    method_bar = None
    if _HAS_TQDM and tqdm is not None:
        method_bar = tqdm(
            methods, desc=f"Methods @ {scen_name}", unit="method", leave=False
        )
        method_iter = method_bar  # type: ignore

    t0_scen = time.time()

    for m_idx, cls in enumerate(method_iter, start=1):
        job_counter += 1

        mtag = safe_slug(method_tag(cls))
        method_dir = os.path.join(scen_dir, mtag)
        waveforms_dir = os.path.join(method_dir, "waveforms")

        # Update progress bar text
        if method_bar is not None:
            method_bar.set_postfix_str(
                f"{m_idx}/{len(methods)} {cls.__name__} | job {job_counter}/{total_jobs}"
            )

        logger.info(
            f"[{scen_name}] "
            f"Method {m_idx}/{len(methods)} "
            f"({job_counter}/{total_jobs}) → {cls.__name__}"
        )

        t0 = time.time()

        # 1) Tune
        logger.info("   [1/4] Tuning (Grid Search Optimization)")
        tuning = optimize_estimator(
            cls,
            v=v_cal,
            f_true=f_cal,
            fs_hz=float(scenario.fs),
            warmup=int(warmup),
        )
        logger.info(
            f"       best_params={tuning['best_params']} | best_rmse={tuning['best_score']:.6f} Hz"
        )

        # 2) Validate (MC)
        logger.info(f"   [2/4] Monte Carlo validation (n_runs={n_runs})")
        results = run_mini_mc(
            cls,
            est_params=tuning["best_params"],
            scenario=scenario,
            n_runs=int(n_runs),
            duration_s=float(duration_s),
            export_waveforms=bool(export_waveforms),
            waveforms_dir=waveforms_dir,
        )

        rmse_mean = results["metrics"].get("RMSE_HZ", {}).get("mean", None)
        logger.info(f"       RMSE_HZ.mean={rmse_mean}")

        # 3) Stats
        logger.info("   [3/4] Statistical analysis + hypotheses")
        stats_report = run_statistical_analysis(
            results, tuning, targets=targets, alpha=float(alpha)
        )

        # 4) Save artifacts
        logger.info("   [4/4] Saving artifacts")
        save_artifacts(
            out_root, scen_name, cls, results, tuning, stats_report, show_plot=False
        )

        all_runs_dfs.append(pd.DataFrame(results["runs_long"]))

        elapsed = time.time() - t0
        # Scenario ETA rough based on avg method time so far
        avg_per_method = (time.time() - t0_scen) / max(1, m_idx)
        remaining_methods = max(0, len(methods) - m_idx)
        eta_scen = remaining_methods * avg_per_method

        logger.info(
            f"   DONE {cls.__name__} | time={elapsed:.1f}s | {_fmt_eta(eta_scen)}"
        )

    if method_bar is not None:
        method_bar.close()

    # 5) Compare methods (per scenario)
    logger.info(f"[{scen_name}] Building multi-method comparison artifacts ...")
    df_all = (
        pd.concat(all_runs_dfs, ignore_index=True) if all_runs_dfs else pd.DataFrame()
    )

    comp_csv = os.path.join(
        scen_dir, f"comparison_runs_long_{safe_slug(scen_name)}.csv"
    )
    df_all.to_csv(comp_csv, index=False)
    logger.info(f"[{scen_name}] Compare CSV: {os.path.basename(comp_csv)}")

    comp_report = {
        "meta": {
            "scenario_name": scen_name,
            "scenario_params": scenario.params(),
            "duration_s": float(duration_s),
            "n_runs": int(n_runs),
            "methods": [{"class": m.__name__, "tag": method_tag(m)} for m in methods],
            "timestamp": ts_now(),
            "out_root": out_root,
        },
        "comparison": compare_methods(df_all, alpha=float(alpha)),
        "note": "Welch t-test/ANOVA solo si SciPy; bootstrap siempre.",
    }

    comp_json = os.path.join(scen_dir, f"comparison_report_{safe_slug(scen_name)}.json")
    with open(comp_json, "w") as f:
        json.dump(comp_report, f, indent=2, cls=NumpyEncoder)
    logger.info(f"[{scen_name}] Compare JSON: {os.path.basename(comp_json)}")

    return job_counter


def main() -> None:
    logger = setup_logger(logging.INFO)

    out_root = os.path.join("artifacts", "mini_mc_compare")
    ensure_dir(out_root)

    # knobs
    n_runs = 5
    alpha = 0.05
    warmup = 10
    export_waveforms = True
    targets = {"RMSE_HZ_TARGET": 0.01, "IEEE_PASS_RATE_TARGET": 0.95}

    methods = build_methods()
    scenarios = build_scenarios()

    total_jobs = len(scenarios) * len(methods)
    job_counter = 0

    logger.info("=" * 72)
    logger.info("PFEBench — RUN BASIC")
    logger.info(
        f"Scenarios: {len(scenarios)} | Methods: {len(methods)} | MC runs: {n_runs}"
    )
    logger.info(f"Total method-evaluations: {total_jobs}")
    if not _HAS_TQDM:
        logger.info("Note: tqdm not installed → no progress bars (pip install tqdm).")
    logger.info("=" * 72)

    t0_all = time.time()

    for s_idx, scen in enumerate(scenarios, start=1):
        duration_s = scenario_duration_default(scen, fallback=1.0)

        logger.info(
            f"[SCENARIO {s_idx}/{len(scenarios)}] {scen.name} | duration={duration_s:.2f}s"
        )

        t0_scen = time.time()
        job_counter = run_one_scenario(
            scenario=scen,
            methods=methods,
            out_root=out_root,
            duration_s=duration_s,
            n_runs=n_runs,
            alpha=alpha,
            targets=targets,
            warmup=warmup,
            export_waveforms=export_waveforms,
            logger=logger,
            job_counter=job_counter,
            total_jobs=total_jobs,
        )
        dt = time.time() - t0_scen
        logger.info(f"[SCENARIO DONE] {scen.name} | elapsed={dt/60:.2f} min")

    dt_all = time.time() - t0_all
    logger.info("=" * 72)
    logger.info(
        f"ALL SCENARIOS COMPLETED SUCCESSFULLY | total_time={dt_all/60:.2f} min"
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
