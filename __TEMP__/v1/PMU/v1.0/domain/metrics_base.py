# domain/metrics_base.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ============================================================
# 0) Types
# ============================================================

JSONValue = Union[None, bool, str, float, int, Dict[str, Any], List[Any]]


# ============================================================
# 1) Metric configuration (Q1-grade)
# ============================================================


@dataclass(frozen=True)
class MetricConfig:
    """
    Metric configuration for Q1-grade reporting + IEEE-style thresholds.

    Design goals:
    - Stable defaults (nothing "magical", everything explicit).
    - Compatible with compute_metrics() contract used in mc.py.
    - Allows *both* steady-state metrics (after warm-up) and event-window metrics.
    """

    # Sampling
    fs_hz: float
    f_nom: float = 60.0

    # Warm-up handling (steady-state metrics)
    # (your compute_metrics currently hard-codes 0.1s; we also expose it here so you can unify later)
    warm_up_s: float = 0.10

    # Optional event-window configuration (if scenario meta provides event time)
    # If event is unknown, metrics can fall back to full/clean segments.
    event_pre_s: float = 0.25
    event_post_s: float = 2.50

    # IEEE-like thresholds (keep as "limits" for pass/fail tagging)
    ieee_fe_limit_mhz: float = 5.0  # Frequency Error limit (mHz)
    ieee_rfe_limit_hzs: float = 0.1  # ROCOF Error limit (Hz/s)

    # Protection-style trip thresholds on |Δf| (Hz)
    trip_thresholds_hz: Tuple[float, ...] = (0.2, 0.5)

    # Settling tolerances on |Δf| (Hz) or |error| depending on your logic
    settling_tols_hz: Tuple[float, ...] = (0.02, 0.05)

    # RoCoF computation options (if you compute it in logic)
    rocof_smoothing_win: int = 1
    rocof_unit: str = "Hz/s"

    # Robust statistics
    percentiles: Tuple[float, ...] = (5, 50, 95, 99)
    cvar_levels: Tuple[float, ...] = (95, 99)

    # Safety: avoid reporting absurd numbers as "finite"
    # (keeps summaries sane when an estimator explodes but stays finite)
    finite_clip_abs: float = 1e6


# ============================================================
# 2) Robust helpers
# ============================================================


def _as_float_or_none(x: Any) -> Tuple[Optional[float], float, str, str]:
    """
    Returns:
      (value_or_none, raw_float, status_code, reason)

    Conventions:
    - raw_float is always a float (nan/inf allowed).
    - value is None if raw is not finite.
    """
    try:
        raw = float(x)
    except Exception:
        return None, float("nan"), "INVALID", "Value could not be cast to float."

    if math.isfinite(raw):
        return raw, raw, "SUCCESS", "Computed successfully."

    if math.isinf(raw):
        return None, raw, "DIVERGED", "Numerical instability / divergence detected."

    # NaN
    return (
        None,
        raw,
        "NOT_APPLICABLE",
        "No event, invalid data, or undefined metric for this scenario.",
    )


def _mk_metric(
    name: str,
    value_raw: Any,
    scenario_id: str = "unknown",
    *,
    units: str = "",
    threshold: Optional[float] = None,
    comparator: str = "<=",
    formula: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, JSONValue]:
    """
    Create a rich, JSON-safe metric record.

    Q1-proof features:
    - Always includes: name, value, raw, status{...}, spec{...}, compliance{...}, scenario_id
    - Computes pass/fail when threshold is provided (and value is finite).
    - Comparator supports: "<=", "<", ">=", ">", "abs<=", "abs<"
    """
    val, raw, code, reason = _as_float_or_none(value_raw)

    # compliance
    passed: Optional[bool] = None
    if threshold is not None and val is not None and math.isfinite(float(threshold)):
        thr = float(threshold)
        v = float(val)

        if comparator == "<=":
            passed = bool(v <= thr)
        elif comparator == "<":
            passed = bool(v < thr)
        elif comparator == ">=":
            passed = bool(v >= thr)
        elif comparator == ">":
            passed = bool(v > thr)
        elif comparator == "abs<=":
            passed = bool(abs(v) <= thr)
        elif comparator == "abs<":
            passed = bool(abs(v) < thr)
        else:
            # unknown comparator -> don't silently lie
            passed = None
            code = "INVALID"
            reason = f"Unknown comparator '{comparator}'."

    rec: Dict[str, JSONValue] = {
        "name": str(name),
        "scenario_id": str(scenario_id),
        "value": val,
        "raw": float(raw),
        "status": {
            "code": str(code),
            "reason": str(reason),
            # 'triggered' historically meant "this metric is meaningful / produced a value"
            "triggered": bool(val is not None),
        },
        "spec": {
            "units": str(units),
            "threshold": None if threshold is None else float(threshold),
            "comparator": str(comparator),
            "formula": str(formula),
        },
        "compliance": {
            "has_threshold": bool(threshold is not None),
            "passed": passed,
        },
    }

    if meta:
        # must be JSON-safe; keep it small (caller should sanitize)
        rec["meta"] = meta

    return rec


def _finite_1d(x: np.ndarray, *, clip_abs: Optional[float] = None) -> np.ndarray:
    """Return finite values, optionally clipping absurd magnitudes."""
    a = np.asarray(x, dtype=float).reshape(-1)
    fin = a[np.isfinite(a)]
    if clip_abs is not None and fin.size:
        c = float(abs(clip_abs))
        fin = fin[np.abs(fin) <= c]
    return fin


def _percentile(x: np.ndarray, p: float, *, clip_abs: Optional[float] = None) -> float:
    f = _finite_1d(x, clip_abs=clip_abs)
    return float(np.percentile(f, p)) if f.size else float("nan")


def _cvar(x: np.ndarray, level: float, *, clip_abs: Optional[float] = None) -> float:
    """
    Conditional Value at Risk: mean of the worst tail beyond the given percentile.
    """
    f = _finite_1d(x, clip_abs=clip_abs)
    if not f.size:
        return float("nan")
    q = float(np.percentile(f, float(level)))
    tail = f[f >= q]
    return float(np.mean(tail)) if tail.size else float(q)


# ============================================================
# 3) Core primitives (kept for compatibility / optional use)
# ============================================================


def rmse(err: np.ndarray, *, clip_abs: Optional[float] = None) -> float:
    e = _finite_1d(err, clip_abs=clip_abs)
    return float(np.sqrt(np.mean(e * e))) if e.size else float("nan")


def mae(err: np.ndarray, *, clip_abs: Optional[float] = None) -> float:
    e = _finite_1d(err, clip_abs=clip_abs)
    return float(np.mean(np.abs(e))) if e.size else float("nan")


def fe_max_mhz(err: np.ndarray, *, clip_abs: Optional[float] = None) -> float:
    e = _finite_1d(err, clip_abs=clip_abs)
    return float(np.max(np.abs(e)) * 1000.0) if e.size else float("nan")


def iae_abs(
    err: np.ndarray, fs_hz: float, *, clip_abs: Optional[float] = None
) -> float:
    e = _finite_1d(err, clip_abs=clip_abs)
    return (
        float(np.sum(np.abs(e)) / max(1e-12, float(fs_hz))) if e.size else float("nan")
    )


# ============================================================
# 4) Dynamic primitives (kept; made safer)
# ============================================================


def rocof_hz_per_s(f: np.ndarray, fs: float, smoothing_win: int = 1) -> np.ndarray:
    """
    ROCOF = df/dt computed by finite difference.
    Returns an array aligned with f, with rocof[0]=0.
    """
    ff = np.asarray(f, dtype=float).reshape(-1)
    n = ff.size
    if n == 0:
        return np.zeros((0,), dtype=float)

    # Replace NaNs/Infs to avoid diff poisoning
    ff2 = ff.copy()
    bad = ~np.isfinite(ff2)
    if np.any(bad):
        # forward-fill, then back-fill
        idx = np.where(np.isfinite(ff2))[0]
        if idx.size == 0:
            return np.zeros_like(ff2)
        first = int(idx[0])
        ff2[:first] = ff2[first]
        for i in range(first + 1, n):
            if not np.isfinite(ff2[i]):
                ff2[i] = ff2[i - 1]

    if int(smoothing_win) > 1 and n >= int(smoothing_win):
        w = int(smoothing_win)
        k = np.ones(w, dtype=float) / float(w)
        ff2 = np.convolve(ff2, k, mode="same")

    fs = float(fs)
    rocof = np.empty_like(ff2)
    rocof[0] = 0.0
    rocof[1:] = np.diff(ff2) * fs
    return rocof


def settling_time(err: np.ndarray, fs_hz: float, tol_hz: float) -> float:
    """
    Settling time: last time index where |err| > tol.
    If already within tol, returns 0.
    """
    e = np.asarray(err, dtype=float).reshape(-1)
    if e.size == 0:
        return float("nan")

    ae = np.abs(e)
    ae[~np.isfinite(ae)] = np.inf

    idx = np.where(ae > float(tol_hz))[0]
    last = int(idx[-1]) if idx.size else -1
    return float((last + 1) / max(1e-12, float(fs_hz)))


def trip_time_threshold(
    f_est: np.ndarray, f_nom: float, fs_hz: float, thr: float
) -> float:
    """
    First time where |f_est - f_nom| >= thr. Returns +inf if never trips.
    Note: This is a simple primitive; your metrics_logic.trip_time may add hysteresis, etc.
    """
    f = np.asarray(f_est, dtype=float).reshape(-1)
    if f.size == 0:
        return float("inf")

    dev = np.abs(f - float(f_nom))
    dev[~np.isfinite(dev)] = 0.0

    idx = np.where(dev >= float(thr))[0]
    return float(idx[0] / max(1e-12, float(fs_hz))) if idx.size else float("inf")


def nadir_metrics(
    f_true: np.ndarray, f_est: np.ndarray, fs_hz: float
) -> Dict[str, float]:
    """
    Nadir magnitude/time error.
    If arrays are empty or all non-finite: returns NaNs but with stable keys.
    """
    ft = np.asarray(f_true, dtype=float).reshape(-1)
    fe = np.asarray(f_est, dtype=float).reshape(-1)
    n = int(min(ft.size, fe.size))
    if n <= 0:
        return {"nadir_val_err_hz": float("nan"), "nadir_time_err_ms": float("nan")}

    ft = ft[:n]
    fe = fe[:n]

    # Replace bad values so argmin is stable
    ft2 = ft.copy()
    fe2 = fe.copy()
    ft2[~np.isfinite(ft2)] = np.inf
    fe2[~np.isfinite(fe2)] = np.inf

    if not np.isfinite(ft2).any() or not np.isfinite(fe2).any():
        return {"nadir_val_err_hz": float("nan"), "nadir_time_err_ms": float("nan")}

    idx_true = int(np.argmin(ft2))
    idx_est = int(np.argmin(fe2))

    val_err = (
        float(fe[idx_est] - ft[idx_true])
        if (np.isfinite(fe[idx_est]) and np.isfinite(ft[idx_true]))
        else float("nan")
    )
    time_err_ms = float((idx_est - idx_true) / max(1e-12, float(fs_hz)) * 1000.0)

    return {"nadir_val_err_hz": val_err, "nadir_time_err_ms": time_err_ms}


def overshoot_hz(f_true: np.ndarray, f_est: np.ndarray) -> float:
    ft = np.asarray(f_true, dtype=float).reshape(-1)
    fe = np.asarray(f_est, dtype=float).reshape(-1)
    n = int(min(ft.size, fe.size))
    if n <= 0:
        return float("nan")
    e = fe[:n] - ft[:n]
    e = _finite_1d(e)
    return float(np.max(e)) if e.size else float("nan")


# ============================================================
# 5) Summary stats (kept; fixed for NaNs)
# ============================================================


@dataclass(frozen=True)
class SummaryStats:
    mean: float
    std: float
    p5: float
    p50: float
    p95: float
    ieee_compliance: bool
    n_finite: int


def summarize(vals: Sequence[float] | np.ndarray, limit: float = 1e9) -> SummaryStats:
    v = np.asarray(vals, dtype=float).reshape(-1)
    fin = v[np.isfinite(v)]
    if not fin.size:
        return SummaryStats(
            mean=float("nan"),
            std=float("nan"),
            p5=float("nan"),
            p50=float("nan"),
            p95=float("nan"),
            ieee_compliance=False,
            n_finite=0,
        )

    avg = float(np.mean(fin))
    return SummaryStats(
        mean=avg,
        std=float(np.std(fin)),
        p5=float(np.percentile(fin, 5)),
        p50=float(np.percentile(fin, 50)),
        p95=float(np.percentile(fin, 95)),
        ieee_compliance=bool(avg <= float(limit)),
        n_finite=int(fin.size),
    )
