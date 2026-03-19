#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/runners/validate_estimator.py

Q1-grade validation runner (Frequency-only).

Adds (requested):
- Grid-search progress logging with: i/total, %, best_so_far, elapsed, ETA, current params
- Throttled printing via verbose_every (prints every N candidates, plus first + new-best)
- Early-stop in grid search via rmse_stop (+ patience) to avoid useless long loops
- Optional time budget for GSO (max_seconds)
- Optional stride (tune_stride) to evaluate only every k-th combination in huge grids
- FIX: method_tag now supports E10_* (two-digit estimator numbers)
- FIX: history keys consistent (rmse_hz) everywhere

Artifacts layout:
out_root/<scenario>/<method_tag>/
  <method_tag>_<timestamp>_report.json
  <method_tag>_<timestamp>_timeseries.csv
  <method_tag>_<timestamp>_tuning_history.csv
  <method_tag>_<timestamp>_runs_long.csv
  <method_tag>_<timestamp>_hypothesis_report.csv
  <method_tag>_<timestamp>_plot.png/.pdf
  waveforms/
    meta.json
    ground_truth_seed42.csv
    waveform_seed_0000.csv
    ...

method_tag examples:
- pfebench.estimators.e4_if_dphi -> E4_IF_DPHI
- pfebench.estimators.e10_nr     -> E10_NR
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import math
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pfebench.estimators.base import BaseEstimator
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

from pfebench.metrics.metrics import (
    MetricConfig,
    aggregate_monte_carlo,
    compute_metrics,
)

# SciPy (optional) for ANOVA / Welch t-test / exact binomial CI
try:
    from scipy import stats as _stats  # type: ignore

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# RMSE (optional internal)
try:
    from pfebench.metrics.metrics_logic import rmse as calc_rmse
except Exception:

    def calc_rmse(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = min(a.size, b.size)
        if n <= 0:
            return float("inf")
        return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


# =============================================================================
# 0) UTILITIES
# =============================================================================


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ts_now() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_slug(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))


def method_tag(cls: Type[BaseEstimator]) -> str:
    """
    pfebench.estimators.e4_if_dphi -> E4_IF_DPHI
    pfebench.estimators.e10_nr -> E10_NR

    Robust parsing:
      module basename should be like: e<number>_<suffix>
      where <number> can be 1+ digits (e.g., 10).
    """
    mod = str(getattr(cls, "__module__", "")).split(".")[-1]  # e4_if_dphi / e10_nr
    if mod.startswith("e") and "_" in mod:
        head, suffix = mod.split("_", 1)  # head="e10", suffix="nr"
        num = head[1:]
        if num.isdigit():
            return f"E{num}_{suffix.upper()}"
    return str(getattr(cls, "__name__", "METHOD")).upper()


def finite(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    return a[np.isfinite(a)]


def flatten_dict(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def flatten_metrics_values(metrics_run: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k, rec in metrics_run.items():
        if isinstance(rec, dict) and "value" in rec:
            out[k] = rec.get("value", None)
    return out


def _fmt_hms(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    if h > 0:
        return f"{h:d}h{m:02d}m{ss:02d}s"
    if m > 0:
        return f"{m:d}m{ss:02d}s"
    return f"{ss:d}s"


# =============================================================================
# 1) SCENARIO
# =============================================================================


class SyntheticScenario:
    def __init__(self, fs=1000.0, f0=60.0, f1=61.0, noise_level=0.01, step_time_s=0.5):
        self.fs = float(fs)
        self.f0 = float(f0)
        self.f1 = float(f1)
        self.noise_level = float(noise_level)
        self.step_time_s = float(step_time_s)
        self.name = f"Step_Noise{int(self.noise_level * 100):02d}"

    def params(self) -> Dict[str, Any]:
        return {
            "fs": self.fs,
            "f0": self.f0,
            "f1": self.f1,
            "noise_level": self.noise_level,
            "step_time_s": self.step_time_s,
        }

    def generate(
        self, duration_s=1.0, seed=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = int(duration_s * self.fs)
        t = np.arange(n, dtype=float) / self.fs

        f_true = np.full_like(t, self.f0, dtype=float)
        f_true[t >= self.step_time_s] = self.f1

        phi = 2.0 * np.pi * np.cumsum(f_true) / self.fs
        noise = rng.normal(0.0, self.noise_level, size=n)
        v = np.sin(phi) + noise
        return t, v, f_true


# =============================================================================
# 2) GRID SEARCH OPTIMIZATION + PROGRESS LOGGING + EARLY STOP
# =============================================================================


def count_grid_sizes(tuning_defs) -> int:
    sizes = [len(p.generate_grid()) for p in tuning_defs]
    total = 1
    for s in sizes:
        total *= int(s)
    return int(total)


def _progress_log_line(
    mtag: str,
    i: int,
    total: int,
    t0: float,
    best_score: float,
    cfg: Dict[str, Any],
) -> str:
    elapsed = time.time() - t0
    rate = elapsed / max(1, i)  # sec / eval
    eta = rate * max(0, total - i)
    pct = 100.0 * (i / max(1, total))
    best_str = f"{best_score:.12e}" if np.isfinite(best_score) else "inf"

    cfg_short = ", ".join([f"{k}={cfg[k]}" for k in cfg.keys()])
    if len(cfg_short) > 180:
        cfg_short = cfg_short[:177] + "..."

    return (
        f"[GSO {mtag}] {i}/{total} ({pct:5.1f}%) | "
        f"best_rmse={best_str} | "
        f"elapsed={_fmt_hms(elapsed)} | ETA={_fmt_hms(eta)} | "
        f"try: {cfg_short}"
    )


def optimize_estimator(
    estimator_cls: Type[BaseEstimator],
    v: np.ndarray,
    f_true: np.ndarray,
    fs_hz: float,
    warmup: int = 10,
    verbose_every: int = 50,
    tune_stride: int = 1,
    max_seconds: Optional[float] = None,
    rmse_stop: Optional[float] = None,
    rmse_stop_patience: int = 1,
    min_evals_before_stop: int = 1,
    min_effective_samples: int = 32,
    stop_on_zero: bool = True,
    zero_eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Grid-search optimizer with:
      - progress logging (throttled)
      - stride sampling for huge grids
      - optional time budget
      - early-stop at acceptable RMSE threshold (+ patience)
      - optional auto-stop when RMSE ~ 0

    IMPORTANT:
      Early-stop checks are evaluated EVERY eval based on best_score,
      not only when a new-best appears.

    stop_on_zero:
      If True, break when best_score <= zero_eps (handles "best_rmse=0" cases).
    """

    mtag = method_tag(estimator_cls)
    print(
        f"\n--- Paso 1: Grid Search Optimization (GSO) [{mtag} | {estimator_cls.__name__}] ---"
    )

    tuning_defs = estimator_cls.tuning_ranges()
    if not tuning_defs:
        print(" -> El estimador no tiene parámetros ajustables.")
        return {"best_params": {}, "best_score": 0.0, "history": []}

    # sanitize
    warmup = int(max(0, warmup))
    tune_stride = int(max(1, tune_stride))
    verbose_every = int(max(1, verbose_every))
    rmse_stop_patience = int(max(1, rmse_stop_patience))
    min_evals_before_stop = int(max(1, min_evals_before_stop))
    min_effective_samples = int(max(1, min_effective_samples))
    zero_eps = float(max(0.0, zero_eps))

    param_names = [p.name for p in tuning_defs]
    param_grids = [p.generate_grid() for p in tuning_defs]

    total_full = count_grid_sizes(tuning_defs)
    total_eval_est = int(math.ceil(total_full / tune_stride))

    print(f" -> Espacio de búsqueda: {total_full} combinaciones (FULL).")
    if tune_stride > 1:
        print(f" -> tune_stride={tune_stride} => ~{total_eval_est} evaluaciones.")
    print(f" -> Logging: cada {verbose_every} evaluaciones (+ first + new-best).")
    if max_seconds is not None:
        print(f" -> max_seconds={float(max_seconds):.1f}s (time budget).")
    if rmse_stop is not None:
        print(
            f" -> rmse_stop={float(rmse_stop):.12e} | patience={rmse_stop_patience} "
            f"| min_evals_before_stop={min_evals_before_stop}"
        )
    if stop_on_zero:
        print(f" -> stop_on_zero=True (zero_eps={zero_eps:.12e})")

    best_score = float("inf")
    best_params: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []

    t0 = time.time()
    i_full = 0
    i_eval = 0

    # patience counters (consecutive evals where condition holds)
    hits_under_rmse_stop = 0
    hits_under_zero = 0

    for vals in itertools.product(*param_grids):
        i_full += 1

        # stride sampling
        if tune_stride > 1 and ((i_full - 1) % tune_stride != 0):
            continue

        i_eval += 1
        cfg = dict(zip(param_names, vals))

        # time budget (before heavy work)
        if max_seconds is not None and (time.time() - t0) >= float(max_seconds):
            print(
                f"[GSO {mtag}] ⏹ STOP: reached max_seconds={float(max_seconds):.1f}s @ eval={i_eval}"
            )
            break

        # progress log throttled
        if i_eval == 1 or (verbose_every > 0 and (i_eval % verbose_every == 0)):
            print(_progress_log_line(mtag, i_eval, total_eval_est, t0, best_score, cfg))

        est = estimator_cls({"fs": fs_hz, **cfg})
        est.reset()

        try:
            f_est = est.run(v)
            n_eff = int(min(f_est.size, f_true.size) - warmup)
            if n_eff < min_effective_samples:
                score = float("inf")
            else:
                score = float(calc_rmse(f_est[warmup:], f_true[warmup:]))

            history.append({"params": cfg.copy(), "rmse_hz": float(score)})

            # update best
            if np.isfinite(score) and score < best_score:
                best_score = float(score)
                best_params = cfg.copy()
                print(
                    f"[GSO {mtag}] ✅ NEW BEST @ eval={i_eval} (full={i_full}/{total_full}) "
                    f"| rmse={best_score:.12e} | params={best_params}"
                )

            # -------------------------
            # EARLY STOP CHECKS (EVERY EVAL)
            # -------------------------
            if i_eval >= min_evals_before_stop:

                # (A) stop_on_zero
                if stop_on_zero and np.isfinite(best_score) and best_score <= zero_eps:
                    hits_under_zero += 1
                    if hits_under_zero >= rmse_stop_patience:
                        print(
                            f"[GSO {mtag}] 🏁 STOP: best_score <= zero_eps "
                            f"({best_score:.12e} <= {zero_eps:.12e})"
                        )
                        break
                else:
                    hits_under_zero = 0

                # (B) rmse_stop threshold
                if (
                    rmse_stop is not None
                    and np.isfinite(best_score)
                    and best_score <= float(rmse_stop)
                ):
                    hits_under_rmse_stop += 1
                    if (
                        hits_under_rmse_stop == 1
                        or hits_under_rmse_stop == rmse_stop_patience
                    ):
                        print(
                            f"[GSO {mtag}] 🎯 best_score <= rmse_stop "
                            f"({hits_under_rmse_stop}/{rmse_stop_patience}) | "
                            f"{best_score:.12e} <= {float(rmse_stop):.12e}"
                        )
                    if hits_under_rmse_stop >= rmse_stop_patience:
                        print(
                            f"[GSO {mtag}] 🏁 STOP: acceptable rmse_stop reached (best={best_score:.12e})."
                        )
                        break
                else:
                    hits_under_rmse_stop = 0

        except Exception as e:
            history.append({"params": cfg.copy(), "rmse_hz": None, "error": str(e)})
            hits_under_rmse_stop = 0
            hits_under_zero = 0

    history_sorted = sorted(
        [
            h
            for h in history
            if h.get("rmse_hz") is not None and np.isfinite(h.get("rmse_hz", np.nan))
        ],
        key=lambda x: x["rmse_hz"],
    )

    print(f" -> Ganador: {best_params} (RMSE: {best_score:.12e} Hz)")
    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": history_sorted,
    }


# =============================================================================
# 3) MINI MONTE CARLO + OPTIONAL WAVEFORMS
# =============================================================================


def bootstrap_ci_mean(
    x: np.ndarray, alpha: float = 0.05, n_boot: int = 5000, seed: int = 123
) -> Tuple[float, float]:
    x = finite(x)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = x.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        boots[i] = float(np.mean(samp))
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_ci_diff_means(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 123,
) -> Dict[str, float]:
    a = finite(a)
    b = finite(b)
    if a.size == 0 or b.size == 0:
        return {
            "diff_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = rng.choice(a, size=a.size, replace=True)
        sb = rng.choice(b, size=b.size, replace=True)
        boots[i] = float(np.mean(sb) - np.mean(sa))
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"diff_mean": float(np.mean(b) - np.mean(a)), "ci_low": lo, "ci_high": hi}


def bootstrap_pvalue_one_sided_B_less_A(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 123
) -> float:
    a = finite(a)
    b = finite(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_boot):
        sa = rng.choice(a, size=a.size, replace=True)
        sb = rng.choice(b, size=b.size, replace=True)
        d = float(np.mean(sb) - np.mean(sa))
        if d >= 0:
            cnt += 1
    return float(cnt / n_boot)


def onesample_ttest_less(x: np.ndarray, mu0: float) -> Dict[str, Any]:
    x = finite(x)
    if x.size < 2:
        return {
            "n": int(x.size),
            "t": None,
            "p_value": None,
            "note": "not_enough_samples",
        }

    n = int(x.size)
    mean = float(np.mean(x))
    s = float(np.std(x, ddof=1))

    if s <= 0:
        p = 0.0 if mean < mu0 else 1.0
        return {
            "n": n,
            "t": None,
            "p_value": float(p),
            "mean": mean,
            "std": s,
            "note": "zero_variance",
        }

    t = (mean - mu0) / (s / math.sqrt(n))

    if _HAS_SCIPY:
        p = float(_stats.t.cdf(t, df=n - 1))
    else:
        p = float(0.5 * (1.0 + math.erf(t / math.sqrt(2.0))))

    return {
        "n": n,
        "t": float(t),
        "p_value": float(p),
        "mean": mean,
        "std": s,
        "mu0": float(mu0),
    }


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    k = int(k)
    n = int(n)

    if not _HAS_SCIPY:
        p = k / n
        z = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denom
        return (max(0.0, center - half), min(1.0, center + half))

    if k == 0:
        return (0.0, float(_stats.beta.ppf(1 - alpha / 2, 1, n)))
    if k == n:
        return (float(_stats.beta.ppf(alpha / 2, n, 1)), 1.0)

    lo = float(_stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = float(_stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def anova_rmse_by_method(df: pd.DataFrame) -> Dict[str, Any]:
    if "method_tag" not in df.columns or "RMSE_HZ" not in df.columns:
        return {"ok": False, "reason": "missing_columns"}
    sub = df[["method_tag", "RMSE_HZ"]].dropna()
    if sub.empty:
        return {"ok": False, "reason": "no_data"}
    groups = []
    labels = []
    for mtag, g in sub.groupby("method_tag"):
        y = finite(g["RMSE_HZ"].to_numpy())
        if y.size >= 2:
            groups.append(y)
            labels.append(str(mtag))
    if len(groups) < 2:
        return {"ok": False, "reason": "need_at_least_2_groups_with_n>=2"}
    if not _HAS_SCIPY:
        return {"ok": False, "reason": "scipy_required_for_anova"}
    F, p = _stats.f_oneway(*groups)
    return {"ok": True, "groups": labels, "F": float(F), "p_value": float(p)}


def save_waveform_triplet_csv(
    path: str, t: np.ndarray, v: np.ndarray, f_true: np.ndarray, f_hat: np.ndarray
) -> None:
    df = pd.DataFrame(
        {
            "Time_s": np.asarray(t, dtype=float),
            "Voltage_pu": np.asarray(v, dtype=float),
            "Freq_True_Hz": np.asarray(f_true, dtype=float),
            "Freq_Est_Hz": np.asarray(f_hat, dtype=float),
        }
    )
    df.to_csv(path, index=False)


def run_mini_mc(
    estimator_cls: Type[BaseEstimator],
    est_params: Dict[str, Any],
    scenario: SyntheticScenario,
    n_runs: int,
    duration_s: float,
    export_waveforms: bool,
    waveforms_dir: Optional[str],
) -> Dict[str, Any]:
    mtag = method_tag(estimator_cls)
    print(
        f"\n--- Paso 2: Validación Monte Carlo ({n_runs} runs) [{mtag} | {estimator_cls.__name__}] ---"
    )

    fs = float(scenario.fs)
    cfg = MetricConfig(fs_hz=fs, f_nom=scenario.f0)

    t_ref, v_ref_example, f_true_ref = scenario.generate(duration_s=duration_s, seed=42)

    if export_waveforms:
        if waveforms_dir is None:
            raise ValueError("export_waveforms=True requiere waveforms_dir")
        ensure_dir(waveforms_dir)

        meta = {
            "scenario_name": scenario.name,
            "scenario_params": scenario.params(),
            "method_tag": mtag,
            "method_class": estimator_cls.__name__,
            "method_params": est_params,
            "fs_hz": fs,
            "duration_s": float(duration_s),
            "n_runs": int(n_runs),
            "timestamp": ts_now(),
        }
        with open(os.path.join(waveforms_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, cls=NumpyEncoder)

        save_waveform_triplet_csv(
            os.path.join(waveforms_dir, "ground_truth_seed42.csv"),
            t_ref,
            v_ref_example,
            f_true_ref,
            np.full_like(f_true_ref, np.nan, dtype=float),
        )

    results_f_hat: List[np.ndarray] = []
    metrics_list: List[Dict[str, Any]] = []
    rows_long: List[Dict[str, Any]] = []

    for seed in range(n_runs):
        _, v, f_true = scenario.generate(duration_s=duration_s, seed=seed)

        est = estimator_cls({"fs": fs, **est_params})
        est.reset()

        f_hat = np.zeros_like(v, dtype=float)
        for k, x in enumerate(v):
            f_hat[k] = est.step(float(x))

        results_f_hat.append(f_hat)

        if export_waveforms and waveforms_dir is not None:
            wf_path = os.path.join(waveforms_dir, f"waveform_seed_{seed:04d}.csv")
            save_waveform_triplet_csv(wf_path, t_ref, v, f_true, f_hat)

        exec_time_s = (
            float(est.last_exec_us) * 1e-6 * len(v)
            if hasattr(est, "last_exec_us")
            else float("nan")
        )
        latency_samples = int(getattr(est, "latency_samples", 0))

        m_run = compute_metrics(
            f_hat=f_hat,
            f_true=f_true,
            exec_time_s=exec_time_s,
            latency_samples=latency_samples,
            cfg=cfg,
            scenario_id=f"seed_{seed}",
        )
        metrics_list.append(m_run)

        row: Dict[str, Any] = {
            "seed": int(seed),
            "scenario_name": scenario.name,
            "method_tag": mtag,
            "method_name": estimator_cls.__name__,
        }
        row.update(flatten_dict(scenario.params(), prefix="scenario_"))
        row.update(flatten_dict(est_params, prefix="method_param_"))
        row.update(flatten_metrics_values(m_run))
        row["method_latency_samples"] = latency_samples
        row["method_exec_time_s_total"] = exec_time_s
        rows_long.append(row)

    f_mat = np.asarray(results_f_hat, dtype=float)

    return {
        "t": t_ref,
        "v_example": v_ref_example,
        "f_true": f_true_ref,
        "f_mean": np.nanmean(f_mat, axis=0),
        "f_std": np.nanstd(f_mat, axis=0),
        "f_var": np.nanvar(f_mat, axis=0),
        "f_median": np.nanmedian(f_mat, axis=0),
        "f_p05": np.nanpercentile(f_mat, 5, axis=0),
        "f_p95": np.nanpercentile(f_mat, 95, axis=0),
        "runs_long": rows_long,
        "metrics": aggregate_monte_carlo(metrics_list, cfg),
        "metrics_per_run_raw": metrics_list,
        "metric_cfg": cfg,
    }


# =============================================================================
# 4) PLOTTING
# =============================================================================


def plot_validation(
    data: Dict[str, Any], tuning_info: Dict[str, Any], title: str, show: bool
) -> plt.Figure:
    t = np.asarray(data["t"], dtype=float).reshape(-1)
    f_true = np.asarray(data["f_true"], dtype=float).reshape(-1)

    f_mean = np.asarray(data["f_mean"], dtype=float).reshape(-1)
    f_p05 = np.asarray(data["f_p05"], dtype=float).reshape(-1)
    f_p95 = np.asarray(data["f_p95"], dtype=float).reshape(-1)
    f_std = np.asarray(data["f_std"], dtype=float).reshape(-1)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )

    ax1.plot(t, data["v_example"], color="gray", alpha=0.6, lw=1)
    ax1.set_ylabel("Voltage [pu]")
    ax1.grid(True, alpha=0.3)

    ax1_t = ax1.twinx()
    ax1_t.plot(t, f_true, color="tab:red", linestyle="--", lw=1.5, alpha=0.8)
    ax1_t.set_ylabel("Freq [Hz]")
    ax1_t.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title(f"{title} (Best Params: {tuning_info['best_params']})")

    ax2.fill_between(t, f_p05, f_p95, alpha=0.25, label="90% band (P05-P95)")
    if np.any(np.isfinite(f_std)):
        ax2.fill_between(
            t, f_mean - f_std, f_mean + f_std, alpha=0.20, label="Mean ± 1σ"
        )
        ax2.fill_between(
            t, f_mean - 2.0 * f_std, f_mean + 2.0 * f_std, alpha=0.12, label="Mean ± 2σ"
        )

    ax2.plot(t, f_mean, lw=1.5, label="Mean Estimate")
    ax2.plot(t, f_true, "k--", lw=1.5, label="True Ref")

    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize="small")

    m = data["metrics"]
    rmse_val = m.get("RMSE_HZ", {}).get("mean", np.nan)
    latency = m.get("LATENCY_SAMPLES", {}).get("mean", np.nan)
    stats_text = (
        f"PERFORMANCE:\n"
        f"RMSE(mean): {rmse_val:.6e} Hz\n"
        f"Latency(mean): {latency:.2f} smp\n"
        f"Tuning RMSE: {float(tuning_info['best_score']):.6e} Hz"
    )
    ax2.text(
        0.02,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# =============================================================================
# 5) STATS + HYPOTHESES (PER METHOD)
# =============================================================================


def run_statistical_analysis(
    results: Dict[str, Any],
    tuning_info: Dict[str, Any],
    targets: Dict[str, float],
    alpha: float,
) -> Dict[str, Any]:
    cfg: MetricConfig = results["metric_cfg"]
    df_runs = pd.DataFrame(results["runs_long"])

    out: Dict[str, Any] = {"alpha": float(alpha), "scipy_available": bool(_HAS_SCIPY)}

    rmse_vals = finite(df_runs.get("RMSE_HZ", pd.Series(dtype=float)).to_numpy())
    rmse_target = float(targets.get("RMSE_HZ_TARGET", 0.01))
    lo, hi = bootstrap_ci_mean(rmse_vals, alpha=alpha)
    ttest = onesample_ttest_less(rmse_vals, mu0=rmse_target)

    out["hypothesis_RMSE_mean_less_than_target"] = {
        "metric": "RMSE_HZ",
        "target": rmse_target,
        "n": int(rmse_vals.size),
        "mean": float(np.mean(rmse_vals)) if rmse_vals.size else None,
        "bootstrap_ci_mean_95": {"low": lo, "high": hi},
        "t_test_one_sided_less": ttest,
        "decision_reject_H0_at_alpha": (
            ttest.get("p_value") is not None and float(ttest["p_value"]) < alpha
        ),
        "H0": "E[RMSE_HZ] >= target",
        "H1": "E[RMSE_HZ] < target",
    }

    fe_limit = float(cfg.ieee_fe_limit_mhz)
    rfe_limit = float(cfg.ieee_rfe_limit_hzs)

    fe_max = finite(df_runs.get("FE_MAX_MHZ", pd.Series(dtype=float)).to_numpy())
    fe_pass = int(np.sum(fe_max <= fe_limit))
    fe_n = int(fe_max.size)
    fe_rate = float(fe_pass / fe_n) if fe_n > 0 else float("nan")
    fe_ci = (
        clopper_pearson_ci(fe_pass, fe_n, alpha=alpha)
        if fe_n > 0
        else (float("nan"), float("nan"))
    )

    rfe_max = finite(df_runs.get("RFE_MAX_ABS_HZS", pd.Series(dtype=float)).to_numpy())
    rfe_pass = int(np.sum(rfe_max <= rfe_limit))
    rfe_n = int(rfe_max.size)
    rfe_rate = float(rfe_pass / rfe_n) if rfe_n > 0 else float("nan")
    rfe_ci = (
        clopper_pearson_ci(rfe_pass, rfe_n, alpha=alpha)
        if rfe_n > 0
        else (float("nan"), float("nan"))
    )

    out["ieee_style_compliance_by_run"] = {
        "FE_MAX_MHZ": {
            "limit_mhz": fe_limit,
            "n_runs": fe_n,
            "pass_runs": fe_pass,
            "pass_rate": fe_rate,
            "ci_95_pass_rate": {"low": fe_ci[0], "high": fe_ci[1]},
        },
        "RFE_MAX_ABS_HZS": {
            "limit_hzs": rfe_limit,
            "n_runs": rfe_n,
            "pass_runs": rfe_pass,
            "pass_rate": rfe_rate,
            "ci_95_pass_rate": {"low": rfe_ci[0], "high": rfe_ci[1]},
        },
    }

    pass_target = float(targets.get("IEEE_PASS_RATE_TARGET", 0.95))
    out["hypothesis_ieee_pass_rate_ge_target"] = {
        "target_pass_rate": pass_target,
        "pass_FE": (fe_rate >= pass_target) if math.isfinite(fe_rate) else False,
        "pass_RFE": (rfe_rate >= pass_target) if math.isfinite(rfe_rate) else False,
    }

    out["tuning_anova_oneway"] = {
        "response": "rmse_hz",
        "results": [],
        "note": "SciPy-only. Interpreta p-values con cuidado (factor as categorical).",
    }

    return out


# =============================================================================
# 6) ARTIFACTS (per method)
# =============================================================================


def save_artifacts(
    out_root: str,
    scenario_name: str,
    estimator_cls: Type[BaseEstimator],
    results: Dict[str, Any],
    tuning_info: Dict[str, Any],
    stats_report: Dict[str, Any],
    show_plot: bool = False,
) -> Dict[str, str]:
    scen_dir = os.path.join(out_root, safe_slug(scenario_name))
    mtag = safe_slug(method_tag(estimator_cls))
    method_dir = os.path.join(scen_dir, mtag)
    ensure_dir(method_dir)

    stamp = ts_now()
    base = os.path.join(method_dir, f"{mtag}_{stamp}")

    print(f"\n[Artifacts] Root: {out_root}")
    print(f"          Path: {method_dir}")

    report = {
        "metadata": {
            "scenario": scenario_name,
            "method_tag": method_tag(estimator_cls),
            "method_class": estimator_cls.__name__,
            "timestamp": stamp,
        },
        "tuning_report": {
            "best_configuration": tuning_info["best_params"],
            "best_rmse_hz": tuning_info["best_score"],
            "exploration_history": tuning_info["history"],
        },
        "validation_metrics_aggregated": results["metrics"],
        "metric_config": asdict(results["metric_cfg"]),
        "statistical_analysis": stats_report,
    }

    # JSON
    json_path = f"{base}_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"  -> JSON: {os.path.basename(json_path)}")

    # timeseries.csv
    df_ts = pd.DataFrame(
        {
            "Time_s": results["t"],
            "Voltage_Example_pu": results["v_example"],
            "Freq_True_Hz": results["f_true"],
            "Freq_Est_Mean_Hz": results["f_mean"],
            "Freq_Est_Std_Hz": results["f_std"],
            "Freq_Est_Var_Hz2": results["f_var"],
            "Freq_Est_Median_Hz": results["f_median"],
            "Freq_Est_P05_Hz": results["f_p05"],
            "Freq_Est_P95_Hz": results["f_p95"],
        }
    )
    ts_path = f"{base}_timeseries.csv"
    df_ts.to_csv(ts_path, index=False)
    print(f"  -> CSV:  {os.path.basename(ts_path)}")

    # tuning_history.csv (FIX: uses rmse_hz key)
    hist_rows = []
    for item in tuning_info.get("history", []):
        row = {}
        row.update(flatten_dict(item.get("params", {}), prefix="method_param_"))
        row["rmse_hz"] = item.get("rmse_hz", None)
        hist_rows.append(row)
    df_hist = pd.DataFrame(hist_rows)
    hist_path = f"{base}_tuning_history.csv"
    df_hist.to_csv(hist_path, index=False)
    print(f"  -> CSV:  {os.path.basename(hist_path)}")

    # runs_long.csv
    df_long = pd.DataFrame(results["runs_long"])
    long_path = f"{base}_runs_long.csv"
    df_long.to_csv(long_path, index=False)
    print(f"  -> CSV:  {os.path.basename(long_path)}")

    # hypothesis_report.csv
    hyp_rows = []
    h1 = stats_report.get("hypothesis_RMSE_mean_less_than_target", {})
    hyp_rows.append(
        {
            "hypothesis": "RMSE_mean < target",
            "metric": h1.get("metric"),
            "target": h1.get("target"),
            "mean": h1.get("mean"),
            "ci_low": (h1.get("bootstrap_ci_mean_95") or {}).get("low"),
            "ci_high": (h1.get("bootstrap_ci_mean_95") or {}).get("high"),
            "p_value": (h1.get("t_test_one_sided_less") or {}).get("p_value"),
            "reject_H0": h1.get("decision_reject_H0_at_alpha"),
        }
    )
    comp = stats_report.get("ieee_style_compliance_by_run", {})
    for metric in ["FE_MAX_MHZ", "RFE_MAX_ABS_HZS"]:
        d = comp.get(metric, {})
        hyp_rows.append(
            {
                "hypothesis": f"{metric}_pass_rate >= target",
                "metric": metric,
                "target": stats_report.get(
                    "hypothesis_ieee_pass_rate_ge_target", {}
                ).get("target_pass_rate"),
                "mean": d.get("pass_rate"),
                "ci_low": (d.get("ci_95_pass_rate") or {}).get("low"),
                "ci_high": (d.get("ci_95_pass_rate") or {}).get("high"),
                "p_value": None,
                "reject_H0": None,
            }
        )
    hyp_path = f"{base}_hypothesis_report.csv"
    pd.DataFrame(hyp_rows).to_csv(hyp_path, index=False)
    print(f"  -> CSV:  {os.path.basename(hyp_path)}")

    # plot
    fig = plot_validation(
        results,
        tuning_info,
        title=f"{method_tag(estimator_cls)} on {scenario_name}",
        show=show_plot,
    )
    fig.savefig(f"{base}_plot.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{base}_plot.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Plot: {os.path.basename(base)}_plot.(png/pdf)")

    return {
        "method_dir": method_dir,
        "base": base,
        "report_json": json_path,
        "runs_long_csv": long_path,
    }


# =============================================================================
# 7) MULTI-METHOD COMPARISON
# =============================================================================


def compare_methods(df_all: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Any]:
    out: Dict[str, Any] = {"alpha": float(alpha), "scipy_available": bool(_HAS_SCIPY)}
    out["anova_global_rmse_by_method"] = anova_rmse_by_method(df_all)

    tags = sorted([str(x) for x in df_all["method_tag"].dropna().unique().tolist()])
    pairwise: List[Dict[str, Any]] = []

    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            A = tags[i]
            B = tags[j]
            a = finite(df_all.loc[df_all["method_tag"] == A, "RMSE_HZ"].to_numpy())
            b = finite(df_all.loc[df_all["method_tag"] == B, "RMSE_HZ"].to_numpy())

            rep = {
                "A": A,
                "B": B,
                "rmse": {
                    "A": {
                        "n": int(a.size),
                        "mean": float(np.mean(a)) if a.size else None,
                    },
                    "B": {
                        "n": int(b.size),
                        "mean": float(np.mean(b)) if b.size else None,
                    },
                    "diff_B_minus_A": bootstrap_ci_diff_means(a, b, alpha=alpha),
                    "p_bootstrap_one_sided_B_less_A": bootstrap_pvalue_one_sided_B_less_A(
                        a, b
                    ),
                },
            }

            if _HAS_SCIPY and a.size >= 2 and b.size >= 2:
                t_stat, p_two = _stats.ttest_ind(b, a, equal_var=False)
                p_one = (
                    float(p_two / 2.0)
                    if float(t_stat) < 0
                    else float(1.0 - p_two / 2.0)
                )
                rep["rmse"]["welch_ttest"] = {
                    "t": float(t_stat),
                    "p_two_sided": float(p_two),
                    "p_one_sided_B_less_A": float(p_one),
                    "reject_H0_at_alpha": bool(p_one < alpha),
                }
            else:
                rep["rmse"]["welch_ttest"] = {
                    "ok": False,
                    "reason": "scipy_required_or_not_enough_samples",
                }

            pairwise.append(rep)

    out["pairwise"] = pairwise
    return out


# =============================================================================
# 8) MAIN
# =============================================================================


def main() -> None:
    scenario = SyntheticScenario(
        fs=1000.0, f0=60.0, f1=61.0, noise_level=0.05, step_time_s=0.5
    )
    duration_s = 1.0
    n_runs = 20
    alpha = 0.05
    targets = {"RMSE_HZ_TARGET": 0.01, "IEEE_PASS_RATE_TARGET": 0.95}

    methods: List[Type[BaseEstimator]] = [
        ZeroCrossingEstimator,
        InterpolatedZeroCrossingEstimator,
        PeriodMeasurementEstimator,
        InstantaneousFrequencyPhaseIncrementEstimator,
        ZeroCrossingMAEstimator,
        MovingWindowLeastSquaresEstimator,
        RecursiveLeastSquaresBasicEstimator,
        RecursiveLeastSquaresIBREstimator,
        WindowedLeastSquaresEstimator,
        AutoregressiveFrequencyEstimator,
        PronyMethodEstimator,
        NewtonRaphsonFrequencyEstimator,
    ]

    out_root = os.path.join("artifacts", "mini_mc_compare")
    ensure_dir(out_root)
    scen_dir = os.path.join(out_root, safe_slug(scenario.name))
    ensure_dir(scen_dir)

    _, v_cal, f_cal = scenario.generate(duration_s=duration_s, seed=999)

    all_runs_dfs: List[pd.DataFrame] = []

    for cls in methods:
        mtag = safe_slug(method_tag(cls))
        method_dir = os.path.join(scen_dir, mtag)
        waveforms_dir = os.path.join(method_dir, "waveforms")

        # ---- per-method GSO budget knobs ----
        verbose_every = 50
        tune_stride = 1
        max_seconds: Optional[float] = None

        # Default acceptable threshold (you asked ~0.001)
        rmse_stop: Optional[float] = 1e-3
        rmse_stop_patience = 2
        min_evals_before_stop = 5
        min_effective_samples = 64

        tag = method_tag(cls)

        # Heavy estimators: cut cost hard
        if tag in ("E7_RLS_IBR", "E7_WLS"):
            verbose_every = 250
            tune_stride = 10
            max_seconds = 15 * 60  # 15 min
            rmse_stop = 2e-3
            rmse_stop_patience = 2

        # NR / Prony can be heavy depending on implementation
        if tag in ("E10_NR", "E9_PRONY"):
            verbose_every = 100
            tune_stride = max(tune_stride, 5)
            max_seconds = 10 * 60  # 10 min
            rmse_stop = 5e-3

        tuning = optimize_estimator(
            cls,
            v=v_cal,
            f_true=f_cal,
            fs_hz=scenario.fs,
            warmup=10,
            verbose_every=verbose_every,
            tune_stride=tune_stride,
            max_seconds=max_seconds,
            rmse_stop=rmse_stop,
            rmse_stop_patience=rmse_stop_patience,
            min_evals_before_stop=min_evals_before_stop,
            min_effective_samples=min_effective_samples,
            stop_on_zero=True,
            zero_eps=1e-12,
        )

        results = run_mini_mc(
            cls,
            est_params=tuning["best_params"],
            scenario=scenario,
            n_runs=n_runs,
            duration_s=duration_s,
            export_waveforms=True,
            waveforms_dir=waveforms_dir,
        )

        stats_report = run_statistical_analysis(
            results, tuning, targets=targets, alpha=alpha
        )
        save_artifacts(
            out_root, scenario.name, cls, results, tuning, stats_report, show_plot=False
        )
        all_runs_dfs.append(pd.DataFrame(results["runs_long"]))

    df_all = (
        pd.concat(all_runs_dfs, ignore_index=True) if all_runs_dfs else pd.DataFrame()
    )
    comp_csv = os.path.join(
        scen_dir, f"comparison_runs_long_{safe_slug(scenario.name)}.csv"
    )
    df_all.to_csv(comp_csv, index=False)
    print(f"\n[Compare] CSV combinado: {os.path.basename(comp_csv)} (en {scen_dir})")

    comp_report = {
        "meta": {
            "scenario_name": scenario.name,
            "scenario_params": scenario.params(),
            "methods": [{"class": m.__name__, "tag": method_tag(m)} for m in methods],
            "timestamp": ts_now(),
            "out_root": out_root,
        },
        "comparison": compare_methods(df_all, alpha=alpha),
        "note": "Welch t-test/ANOVA solo si SciPy; bootstrap siempre.",
    }

    comp_json = os.path.join(
        scen_dir, f"comparison_report_{safe_slug(scenario.name)}.json"
    )
    with open(comp_json, "w") as f:
        json.dump(comp_report, f, indent=2, cls=NumpyEncoder)
    print(f"[Compare] JSON: {os.path.basename(comp_json)} (en {scen_dir})")

    print("Done.")


if __name__ == "__main__":
    main()
