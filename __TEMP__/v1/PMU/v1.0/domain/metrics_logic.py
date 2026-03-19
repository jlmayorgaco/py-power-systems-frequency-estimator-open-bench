# domain/metrics_logic.py
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple

from .metrics_base import (
    MetricConfig,
    JSONValue,
    summarize,
    rmse as _rmse,
    mae as _mae,
    iae_abs as _iae_abs,
    fe_max_mhz as _fe_max_mhz,
    rocof_hz_per_s as _rocof_hz_per_s,
    settling_time as _settling_time,
    nadir_metrics as _nadir_metrics,
    overshoot_hz as _overshoot_hz,
    trip_time_threshold as _trip_time_threshold,
)

# ============================================================
# Helpers (bulletproof)
# ============================================================


def _to_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _align_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = _to_1d(a)
    b = _to_1d(b)
    n = int(min(a.size, b.size))
    if n <= 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    return a[:n], b[:n]


def _finite_or_empty(x: np.ndarray) -> np.ndarray:
    x = _to_1d(x)
    return x[np.isfinite(x)]


def _nan_if_empty(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


# ============================================================
# 1) Error primitives (logic layer)
# ============================================================


def rmse(f_hat: np.ndarray, f_true: np.ndarray) -> float:
    f_hat, f_true = _align_pair(f_hat, f_true)
    if f_hat.size == 0:
        return float("nan")
    return _rmse(f_hat - f_true)


def mae(f_hat: np.ndarray, f_true: np.ndarray) -> float:
    f_hat, f_true = _align_pair(f_hat, f_true)
    if f_hat.size == 0:
        return float("nan")
    return _mae(f_hat - f_true)


def frequency_error_mhz(f_hat: np.ndarray, f_true: np.ndarray) -> float:
    """
    Max absolute frequency error in mHz (IEEE-style FE metric).
    """
    f_hat, f_true = _align_pair(f_hat, f_true)
    if f_hat.size == 0:
        return float("nan")
    return _fe_max_mhz(f_hat - f_true)


def iae(f_hat: np.ndarray, f_true: np.ndarray, fs_hz: float) -> float:
    """
    Integral of absolute error in seconds*Hz (i.e., Hz·s).
    """
    f_hat, f_true = _align_pair(f_hat, f_true)
    if f_hat.size == 0:
        return float("nan")
    return _iae_abs(f_hat - f_true, float(fs_hz))


# ============================================================
# 2) ROCOF / dynamics (bulletproof + consistent)
# ============================================================


def rocof_error_rmse(f_hat: np.ndarray, f_true: np.ndarray, fs_hz: float) -> float:
    """
    RMSE of ROCOF error (Hz/s).

    Important:
    - Your compute_metrics currently labels this as "Hz/s^2" (typo). The quantity is Hz/s.
    - We keep the scalar output as RMSE(r_hat - r_true), units Hz/s.
    """
    f_hat, f_true = _align_pair(f_hat, f_true)
    if f_hat.size == 0:
        return float("nan")

    fs = float(fs_hz)
    # If fs is invalid, fail gracefully
    if not np.isfinite(fs) or fs <= 0:
        return float("nan")

    # Use a modest smoothing to avoid pure-diff noise; if very short signal, smoothing is auto-safe in base.
    r_true = _rocof_hz_per_s(f_true, fs, smoothing_win=10)
    r_hat = _rocof_hz_per_s(f_hat, fs, smoothing_win=10)

    r_hat, r_true = _align_pair(r_hat, r_true)
    if r_hat.size == 0:
        return float("nan")
    return _rmse(r_hat - r_true)


# ============================================================
# 3) Event-focused analysis (Chamorro)
# ============================================================


def nadir_analysis(
    f_hat: np.ndarray, f_true: np.ndarray, fs_hz: float
) -> Dict[str, float]:
    """
    Nadir magnitude and time error (Hz, ms).
    Always returns keys:
      - nadir_val_err_hz
      - nadir_time_err_ms
    """
    f_hat, f_true = _align_pair(f_hat, f_true)
    if f_hat.size == 0:
        return {"nadir_val_err_hz": float("nan"), "nadir_time_err_ms": float("nan")}
    return _nadir_metrics(f_true, f_hat, float(fs_hz))


def overshoot(f_hat: np.ndarray, f_true: np.ndarray) -> float:
    """
    Overshoot error: max(f_hat - f_true) over the aligned horizon.
    """
    f_hat, f_true = _align_pair(f_hat, f_true)
    if f_hat.size == 0:
        return float("nan")
    return _overshoot_hz(f_true, f_hat)


def settling_time(
    f_hat: np.ndarray, f_true: np.ndarray, tol_hz: float, fs_hz: float
) -> float:
    """
    Settling time of error entering and staying within ±tol_hz band.
    Returns seconds, NaN if inputs invalid.
    """
    f_hat, f_true = _align_pair(f_hat, f_true)
    if f_hat.size == 0:
        return float("nan")
    fs = float(fs_hz)
    if not np.isfinite(fs) or fs <= 0:
        return float("nan")
    err = f_hat - f_true
    return _settling_time(err, fs, float(tol_hz))


# ============================================================
# 4) Protection-style trip time (must return (val, info))
# ============================================================


def trip_time(
    f_hat: np.ndarray, f_true: np.ndarray, thr_hz: float, fs_hz: float
) -> Tuple[float, dict]:
    """
    Trip time: first instant where |f_hat - f_nom| >= thr_hz.

    Contract:
      returns (time_seconds, info_dict)
    where:
      - time_seconds can be +inf if never trips
      - info_dict includes 'reason', 'f_nom', 'threshold', 'n', 'fs_hz'

    Bulletproof behavior:
    - Aligns arrays.
    - Uses robust nominal frequency:
        * median of first 0.25s of f_true if available,
        * else first finite sample,
        * else 60.0
    """
    f_hat, f_true = _align_pair(f_hat, f_true)
    fs = float(fs_hz)

    info = {
        "threshold": float(thr_hz),
        "fs_hz": float(fs) if np.isfinite(fs) else None,
        "n": int(f_hat.size),
        "f_nom": 60.0,
        "reason": "invalid",
    }

    if f_hat.size == 0 or (not np.isfinite(fs)) or fs <= 0:
        return float("inf"), info

    # Robust nominal f from reference
    win = int(round(0.25 * fs))
    win = max(1, min(win, f_true.size))
    ref0 = f_true[:win]
    ref0_fin = ref0[np.isfinite(ref0)]
    if ref0_fin.size:
        f_nom = float(np.median(ref0_fin))
    else:
        # fallback: first finite anywhere
        ft_fin = f_true[np.isfinite(f_true)]
        f_nom = float(ft_fin[0]) if ft_fin.size else 60.0

    info["f_nom"] = f_nom

    val = _trip_time_threshold(f_hat, f_nom, fs, float(thr_hz))
    if np.isfinite(val):
        info["reason"] = "triggered"
        return float(val), info

    # base primitive returns +inf when never trips; treat that explicitly
    if np.isinf(val):
        info["reason"] = "never_tripped"
        return float("inf"), info

    info["reason"] = "nan_trip_time"
    return float("nan"), info


# ============================================================
# 5) Monte Carlo aggregation (Q1-grade)
# ============================================================


def aggregate_monte_carlo(
    per_seed_metrics: List[Dict[str, Dict[str, Any]]], config: MetricConfig
) -> Dict[str, Dict[str, JSONValue]]:
    """
    Aggregate per-seed metric dicts into summary statistics.

    Input shape (as produced by compute_metrics):
      per_seed_metrics[i][metric_name] = { "raw": ..., "value": ..., ... }

    Output is JSON-safe, stable keys, and includes ieee_compliance based on config limits.
    """
    if not per_seed_metrics:
        return {}

    metric_names: set[str] = set()
    for d in per_seed_metrics:
        metric_names |= set(d.keys())

    agg: Dict[str, Dict[str, JSONValue]] = {}
    for name in sorted(metric_names):
        vals_raw: List[float] = []
        for d in per_seed_metrics:
            m = d.get(name)
            if isinstance(m, dict):
                v = m.get("raw", None)
                if v is None:
                    v = m.get("value", None)
                vals_raw.append(float(v) if v is not None else float("nan"))
            else:
                vals_raw.append(float("nan"))

        # IEEE-like limits (only for the metrics where it makes sense)
        limit = 1e9
        if "FE_max" in name:
            limit = float(config.ieee_fe_limit_mhz)
        elif "RFE" in name:
            limit = float(config.ieee_rfe_limit_hzs)

        s = summarize(vals_raw, limit=limit)
        agg[name] = {
            "mean": s.mean,
            "std": s.std,
            "p5": s.p5,
            "p50": s.p50,
            "p95": s.p95,
            "ieee_compliance": s.ieee_compliance,
            "n_finite": s.n_finite,
        }

    return agg


__all__ = [
    "rmse",
    "mae",
    "iae",
    "frequency_error_mhz",
    "rocof_error_rmse",
    "nadir_analysis",
    "overshoot",
    "settling_time",
    "trip_time",
    "aggregate_monte_carlo",
]
