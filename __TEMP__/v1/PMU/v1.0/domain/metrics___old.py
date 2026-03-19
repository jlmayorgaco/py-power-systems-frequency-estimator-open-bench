# domain/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class MetricConfig:
    """
    Metric configuration (paper-facing).
    Keep these as explicit knobs to avoid "arbitrary thresholds" reviewer attacks.
    """

    fs_hz: float

    # Trip / settle thresholds (Hz)
    trip_thresholds_hz: Tuple[float, ...] = (0.5,)
    settling_tols_hz: Tuple[float, ...] = (0.1,)

    # RoCoF configuration
    rocof_enable: bool = True
    rocof_smoothing_win: int = 1  # 1 means raw diff; set >1 for smoother derivative
    rocof_unit: str = "Hz/s"

    # Tail / robustness
    percentiles: Tuple[float, ...] = (50, 95, 99)
    cvar_levels: Tuple[float, ...] = (95,)

    # Outlier rates (Hz)
    outlier_thresholds_hz: Tuple[float, ...] = (0.5,)

    # Event-weighting (to avoid "RMSE hides event")
    event_enable: bool = True
    # IMPORTANT: this threshold is for RoCoF (Hz/s), not Hz
    event_detect_threshold_hz_per_s: float = 0.05
    event_padding_s: float = 0.20
    event_weight: float = 5.0

    # Optional band metrics (off by default)
    band_metrics_enable: bool = False
    band_edges_hz: Tuple[Tuple[float, float], ...] = ((0.0, 2.0), (2.0, 10.0))


# ============================================================
# JSON-friendly metric object helpers
# ============================================================

JSONNumber = Union[float, int]
JSONValue = Union[None, bool, str, JSONNumber, Dict[str, Any], List[Any]]


def _json_float(x: Any) -> Optional[float]:
    """
    Convert to JSON-friendly float:
      - finite -> float
      - NaN/inf/non-castable -> None
    """
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _raw_float(x: Any) -> float:
    """Keep raw float for debugging (can be nan/inf)."""
    try:
        return float(x)
    except Exception:
        return float("nan")


def _mk_metric(
    name: str,
    value_raw: Any,
    scenario_id: str = "unknown",  # Nuevo parámetro
    *,
    units: str = "",
    limit: Optional[float] = None,
    threshold: Optional[float] = None,
    comment: str = "",
    formula: str = "",
    triggered: Optional[bool] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, JSONValue]:
    raw = _raw_float(value_raw)
    val = _json_float(raw)
    valid = bool(val is not None)

    # Lógica Semántica para Journal Q1
    status_code = "SUCCESS"
    reason = "Metric computed successfully."

    if not valid:
        if np.isnan(raw):
            if "EVENT" in name and scenario_id.startswith("G1"):
                status_code = "NOT_APPLICABLE"
                reason = "Baseline scenario has no dynamic events to detect."
            else:
                status_code = "DETECTION_FAILED"
                reason = "No valid data or event onset found for this calculation."
        elif np.isinf(raw):
            if "TRIP" in name:
                status_code = "NOT_TRIGGERED"
                reason = (
                    "Threshold never exceeded; signal remained within stability limits."
                )
            elif "SETTLING" in name:
                status_code = "NOT_SETTLED"
                reason = (
                    "Signal never returned to the tolerance band before simulation end."
                )
            else:
                status_code = "DIVERGED"
                reason = "Calculation resulted in infinity (numerical instability)."

    rec: Dict[str, JSONValue] = {
        "name": name,
        "value": val,  # null en JSON si era NaN/Inf
        "status": {
            "code": status_code,
            "reason": reason,
            "triggered": bool(triggered if triggered is not None else valid),
            "raw_value": str(raw),  # Guardamos el original como string por si acaso
        },
        "spec": {
            "units": units,
            "limit": _json_float(limit),
            "threshold": _json_float(threshold),
            "comment": comment,
            "formula": formula,
        },
        "debug": debug or {},
    }
    return rec


def flatten_metric_objects(
    rich: Dict[str, Dict[str, JSONValue]],
    *,
    prefer: str = "raw",
) -> Dict[str, float]:
    """
    Convert rich metric objects -> flat dict[str,float] for aggregators.

    prefer:
      - "raw":   uses 'raw' (can be NaN/inf)  <-- recommended for faithful behavior
      - "value": uses JSON-friendly 'value' (None -> NaN)
    """
    out: Dict[str, float] = {}
    use_value = prefer.strip().lower() == "value"

    for k, obj in rich.items():
        if not isinstance(obj, dict):
            continue
        v = obj.get("value") if use_value else obj.get("raw")
        if v is None:
            out[k] = float("nan")
        else:
            try:
                out[k] = float(v)
            except Exception:
                out[k] = float("nan")
    return out


# ============================================================
# Low-level helpers
# ============================================================


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _percentile(x: np.ndarray, p: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


def _cvar(x: np.ndarray, level: float) -> float:
    """
    Conditional Value-at-Risk (Expected Shortfall) at 'level' percent.
    Example: level=95 -> average of worst 5% tail.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    q = np.percentile(x, float(level))
    tail = x[x >= q]
    return float(np.mean(tail)) if tail.size else float(q)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    if win >= len(x):
        return np.full_like(x, np.mean(x))
    k = np.ones(win, dtype=float) / win
    return np.convolve(x, k, mode="same")


# ============================================================
# Primary metrics
# ============================================================


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    m = _finite_mask(x, y)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((x[m] - y[m]) ** 2)))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    m = _finite_mask(x, y)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(x[m] - y[m])))


def medae(x: np.ndarray, y: np.ndarray) -> float:
    m = _finite_mask(x, y)
    if not np.any(m):
        return float("nan")
    return float(np.median(np.abs(x[m] - y[m])))


def max_abs_error(x: np.ndarray, y: np.ndarray) -> float:
    m = _finite_mask(x, y)
    if not np.any(m):
        return float("nan")
    return float(np.max(np.abs(x[m] - y[m])))


def iae(x: np.ndarray, y: np.ndarray, dt: float) -> float:
    """Integral Absolute Error: ∫ |e(t)| dt"""
    m = _finite_mask(x, y)
    if not np.any(m):
        return float("nan")
    return float(np.sum(np.abs(x[m] - y[m])) * float(dt))


def ise(x: np.ndarray, y: np.ndarray, dt: float) -> float:
    """Integral Squared Error: ∫ e(t)^2 dt"""
    m = _finite_mask(x, y)
    if not np.any(m):
        return float("nan")
    return float(np.sum((x[m] - y[m]) ** 2) * float(dt))


def settling_time(
    f_hat: np.ndarray,
    f_true: np.ndarray,
    tol_hz: float,
    fs_hz: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Time until |error| <= tol_hz for the remainder of the signal.
    Returns (value_seconds, debug_dict).
      - NaN if cannot be evaluated (no finite overlap).
      - 0.0 if always within tolerance (on finite overlap).
      - inf if never settles.
    """
    f_hat = np.asarray(f_hat, dtype=float)
    f_true = np.asarray(f_true, dtype=float)
    if f_hat.shape != f_true.shape:
        return float("nan"), {"reason": "shape_mismatch"}

    m = _finite_mask(f_hat, f_true)
    if not np.any(m):
        return float("nan"), {"reason": "no_finite_overlap"}

    err = np.full_like(f_hat, np.nan, dtype=float)
    err[m] = np.abs(f_hat[m] - f_true[m])

    finite_idx = np.where(np.isfinite(err))[0]
    if finite_idx.size == 0:
        return float("nan"), {"reason": "no_finite_err"}

    viol = finite_idx[err[finite_idx] > float(tol_hz)]
    if viol.size == 0:
        return 0.0, {
            "reason": "already_within_tol",
            "first_finite_idx": int(finite_idx[0]),
            "last_finite_idx": int(finite_idx[-1]),
            "last_violation_idx": None,
        }

    last_violation = int(viol[-1])
    last_finite = int(finite_idx[-1])

    if last_violation >= last_finite:
        return float("inf"), {
            "reason": "never_settles",
            "last_violation_idx": last_violation,
            "last_finite_idx": last_finite,
        }

    return float(last_violation / float(fs_hz)), {
        "reason": "settled",
        "last_violation_idx": last_violation,
        "last_finite_idx": last_finite,
    }


def trip_time(
    f_hat: np.ndarray,
    f_true: np.ndarray,
    threshold_hz: float,
    fs_hz: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    First time |error| > threshold_hz.
    Returns (value_seconds, debug_dict).
      - NaN if cannot be evaluated (no finite overlap).
      - inf if never trips.
    """
    f_hat = np.asarray(f_hat, dtype=float)
    f_true = np.asarray(f_true, dtype=float)
    if f_hat.shape != f_true.shape:
        return float("nan"), {"reason": "shape_mismatch"}

    m = _finite_mask(f_hat, f_true)
    if not np.any(m):
        return float("nan"), {"reason": "no_finite_overlap"}

    err = np.full_like(f_hat, np.nan, dtype=float)
    err[m] = np.abs(f_hat[m] - f_true[m])

    finite_idx = np.where(np.isfinite(err))[0]
    if finite_idx.size == 0:
        return float("nan"), {"reason": "no_finite_err"}

    trips = finite_idx[err[finite_idx] > float(threshold_hz)]
    if trips.size == 0:
        return float("inf"), {
            "reason": "never_trips",
            "first_finite_idx": int(finite_idx[0]),
            "last_finite_idx": int(finite_idx[-1]),
            "first_trip_idx": None,
        }

    first = int(trips[0])
    return float(first / float(fs_hz)), {
        "reason": "tripped",
        "first_trip_idx": first,
    }


def outlier_rate(
    f_hat: np.ndarray,
    f_true: np.ndarray,
    threshold_hz: float,
) -> float:
    """Fraction of finite-overlap samples where |error| > threshold_hz."""
    m = _finite_mask(f_hat, f_true)
    if not np.any(m):
        return float("nan")
    err = np.abs(f_hat[m] - f_true[m])
    return float(np.mean(err > float(threshold_hz)))


# ============================================================
# RoCoF (derivative) metrics
# ============================================================


def rocof_hz_per_s(f: np.ndarray, fs_hz: float, smooth_win: int = 1) -> np.ndarray:
    """Numerical derivative of frequency (Hz/s)."""
    f = np.asarray(f, dtype=float)
    df = np.diff(f, prepend=f[0]) * float(fs_hz)
    if smooth_win and smooth_win > 1:
        df = _moving_average(df, int(smooth_win))
    return df


# ============================================================
# Event detection + event-weighted errors
# ============================================================


def detect_event_window(
    f_true: np.ndarray,
    cfg: MetricConfig,
) -> Optional[Tuple[int, int]]:
    """
    Detect event onset based on |RoCoF_true| > threshold (Hz/s).
    Returns (start_idx, end_idx) inclusive bounds with padding.
    If no event found, returns None.
    """
    if not cfg.event_enable:
        return None

    f_true = np.asarray(f_true, dtype=float)
    rocof = rocof_hz_per_s(
        f_true, float(cfg.fs_hz), smooth_win=max(1, int(cfg.rocof_smoothing_win))
    )
    rocof = np.abs(rocof)
    rocof = np.where(np.isfinite(rocof), rocof, 0.0)

    idx = np.where(rocof > float(cfg.event_detect_threshold_hz_per_s))[0]
    if idx.size == 0:
        return None

    onset = int(idx[0])
    pad = int(round(float(cfg.event_padding_s) * float(cfg.fs_hz)))
    start = max(0, onset - pad)
    end = min(len(f_true) - 1, onset + pad)
    return start, end


def weighted_rmse(
    f_hat: np.ndarray,
    f_true: np.ndarray,
    weights: np.ndarray,
) -> float:
    m = _finite_mask(f_hat, f_true, weights)
    if not np.any(m):
        return float("nan")

    e2 = (f_hat[m] - f_true[m]) ** 2
    w = weights[m]
    denom = float(np.sum(w))
    if denom <= 0.0:
        return float("nan")
    return float(np.sqrt(float(np.sum(w * e2)) / denom))


def build_event_weights(
    n: int, event_window: Optional[Tuple[int, int]], cfg: MetricConfig
) -> np.ndarray:
    w = np.ones(int(n), dtype=float)
    if event_window is None:
        return w
    a, b = int(event_window[0]), int(event_window[1])
    a = max(0, min(a, n - 1))
    b = max(0, min(b, n - 1))
    if b < a:
        a, b = b, a
    w[a : b + 1] *= float(cfg.event_weight)
    return w


def detection_delay(
    f_hat: np.ndarray,
    f_true: np.ndarray,
    cfg: MetricConfig,
) -> Tuple[float, Dict[str, Any]]:
    """
    Detection delay: onset_hat - onset_true, using RoCoF threshold crossing.
    Returns (value_seconds, debug_dict).
      - NaN if no event in f_true
      - inf if estimator never crosses
    """
    thr = float(cfg.event_detect_threshold_hz_per_s)
    fs = float(cfg.fs_hz)

    df_true = np.abs(
        rocof_hz_per_s(f_true, fs, smooth_win=max(1, int(cfg.rocof_smoothing_win)))
    )
    df_true = np.where(np.isfinite(df_true), df_true, 0.0)
    true_idx = np.where(df_true > thr)[0]
    if true_idx.size == 0:
        return float("nan"), {"reason": "no_event_in_true", "thr_hz_per_s": thr}

    onset_true = int(true_idx[0])

    df_hat = np.abs(
        rocof_hz_per_s(f_hat, fs, smooth_win=max(1, int(cfg.rocof_smoothing_win)))
    )
    df_hat = np.where(np.isfinite(df_hat), df_hat, 0.0)
    hat_idx = np.where(df_hat > thr)[0]
    if hat_idx.size == 0:
        return float("inf"), {
            "reason": "no_crossing_in_hat",
            "thr_hz_per_s": thr,
            "onset_true_idx": onset_true,
        }

    onset_hat = int(hat_idx[0])
    return float((onset_hat - onset_true) / fs), {
        "reason": "ok",
        "thr_hz_per_s": thr,
        "onset_true_idx": onset_true,
        "onset_hat_idx": onset_hat,
    }


# ============================================================
# Optional: band-limited error RMS (frequency domain)
# ============================================================


def band_error_rms(
    err: np.ndarray,
    fs_hz: float,
    band: Tuple[float, float],
) -> float:
    """
    RMS of error within a frequency band via FFT masking + inverse FFT.
    Deterministic and avoids ambiguous Parseval scaling.
    """
    err = np.asarray(err, dtype=float)
    n = int(len(err))
    if n < 4:
        return float("nan")

    x = err - np.nanmean(err)
    x = np.where(np.isfinite(x), x, 0.0)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs_hz))

    f0, f1 = float(band[0]), float(band[1])
    if f1 <= f0:
        return float("nan")

    mask = (freqs >= f0) & (freqs < f1)
    if not np.any(mask):
        return float("nan")

    Xf = np.zeros_like(X)
    Xf[mask] = X[mask]
    xf = np.fft.irfft(Xf, n=n)
    return float(np.sqrt(np.mean(xf**2)))


# ============================================================
# Main API: compute metric dictionary (RICH JSON OBJECTS)
# ============================================================


def compute_metrics_rich(
    f_hat: np.ndarray,
    f_true: np.ndarray,
    exec_time_s: float,
    latency_samples: int,
    cfg: MetricConfig,
    scenario_id: str = "unknown",  # <-- Añadir esto
) -> Dict[str, Dict[str, JSONValue]]:
    """
    Compute metrics and return JSON-debuggable metric objects.
    """
    f_hat = np.asarray(f_hat, dtype=float)
    f_true = np.asarray(f_true, dtype=float)
    if f_hat.shape != f_true.shape:
        raise ValueError("f_hat and f_true must have same shape")

    fs = float(cfg.fs_hz)
    dt = 1.0 / fs
    n = int(len(f_hat))

    err = f_hat - f_true
    abs_err = np.abs(err)

    time_per_sample_us = (float(exec_time_s) / max(n, 1)) * 1e6

    event_win = detect_event_window(f_true, cfg) if cfg.event_enable else None
    weights = build_event_weights(n, event_win, cfg)
    wrmse = weighted_rmse(f_hat, f_true, weights)

    out: Dict[str, Dict[str, JSONValue]] = {}

    out["RMSE"] = _mk_metric(
        "RMSE",
        rmse(f_hat, f_true),
        units="Hz",
        scenario_id=scenario_id,
        comment="Root Mean Square Error",
        formula="sqrt(mean((f_hat - f_true)^2))",
    )

    out["MAX_PEAK"] = _mk_metric(
        "MAX_PEAK",
        max_abs_error(f_hat, f_true),
        units="Hz",
        scenario_id=scenario_id,
        comment="Maximum absolute frequency error",
        formula="max(|f_hat - f_true|)",
    )

    out["IAE"] = _mk_metric(
        "IAE",
        iae(f_hat, f_true, dt),
        units="Hz*s",
        scenario_id=scenario_id,
        comment="Integral Absolute Error",
        formula="sum(|f_hat - f_true|) * dt",
        debug={"dt_s": dt},
    )

    out["W_RMSE_EVENT"] = _mk_metric(
        "W_RMSE_EVENT",
        wrmse,
        units="Hz",
        scenario_id=scenario_id,
        comment="Event-weighted RMSE (higher weight near detected event in f_true)",
        formula="sqrt(sum(w*e^2)/sum(w))",
        debug={
            "event_weight": float(cfg.event_weight),
            "event_window": (
                None if event_win is None else [int(event_win[0]), int(event_win[1])]
            ),
        },
    )

    out["TIME_PER_SAMPLE_US"] = _mk_metric(
        "TIME_PER_SAMPLE_US",
        float(time_per_sample_us),
        units="us/sample",
        scenario_id=scenario_id,
        comment="Execution time per sample",
        formula="exec_time_s / N * 1e6",
        debug={"exec_time_s": float(exec_time_s), "n_samples": int(n)},
    )

    out["LATENCY_SAMPLES"] = _mk_metric(
        "LATENCY_SAMPLES",
        float(latency_samples),
        units="samples",
        scenario_id=scenario_id,
        comment="Algorithmic group delay (reported by method)",
        formula="given",
    )

    for thr in cfg.trip_thresholds_hz:
        key = f"TRIP_TIME_{str(thr).replace('.', 'p')}"
        t_trip, dbg = trip_time(f_hat, f_true, float(thr), fs)
        out[key] = _mk_metric(
            key,
            t_trip,
            units="s",
            threshold=float(thr),
            comment=f"First time |error| > {thr} Hz",
            formula="min t: |f_hat(t)-f_true(t)| > threshold",
            # trip is "triggered" only if it tripped (finite time)
            triggered=bool(
                np.isfinite(t_trip) and (t_trip >= 0.0) and (not np.isinf(t_trip))
            ),
            debug={**dbg, "fs_hz": fs},
        )

    for tol in cfg.settling_tols_hz:
        key = f"SETTLING_{str(tol).replace('.', 'p')}"
        t_set, dbg = settling_time(f_hat, f_true, float(tol), fs)
        out[key] = _mk_metric(
            key,
            t_set,
            units="s",
            threshold=float(tol),
            comment=f"Settling time to band ±{tol} Hz (last-violation definition)",
            formula="t_set = (last index where |e|>tol)/fs",
            # settling is "triggered" only if it settled (finite time)
            triggered=bool(
                np.isfinite(t_set) and (t_set >= 0.0) and (not np.isinf(t_set))
            ),
            debug={**dbg, "fs_hz": fs},
        )

    out["MAE"] = _mk_metric(
        "MAE",
        mae(f_hat, f_true),
        units="Hz",
        comment="Mean Absolute Error",
        formula="mean(|f_hat - f_true|)",
    )

    out["MEDAE"] = _mk_metric(
        "MEDAE",
        medae(f_hat, f_true),
        units="Hz",
        comment="Median Absolute Error",
        formula="median(|f_hat - f_true|)",
    )

    for p in cfg.percentiles:
        k = f"P{int(p)}_ABS_ERR"
        out[k] = _mk_metric(
            k,
            _percentile(abs_err, float(p)),
            units="Hz",
            comment=f"{int(p)}th percentile of absolute error",
            formula=f"percentile(|e|, {p})",
        )

    for lvl in cfg.cvar_levels:
        k = f"CVAR{int(lvl)}_ABS_ERR"
        out[k] = _mk_metric(
            k,
            _cvar(abs_err, float(lvl)),
            units="Hz",
            comment=f"CVaR (expected shortfall) at {int(lvl)}%",
            formula=f"mean(|e| in tail >= percentile(|e|,{lvl}))",
        )

    for thr in cfg.outlier_thresholds_hz:
        k = f"OUTLIER_RATE_{str(thr).replace('.', 'p')}"
        out[k] = _mk_metric(
            k,
            outlier_rate(f_hat, f_true, float(thr)),
            units="fraction",
            threshold=float(thr),
            comment=f"Fraction of samples where |error| > {thr} Hz",
            formula="mean(|e| > threshold)",
        )

    if cfg.rocof_enable:
        rocof_true = rocof_hz_per_s(
            f_true, fs, smooth_win=max(1, int(cfg.rocof_smoothing_win))
        )
        rocof_hat = rocof_hz_per_s(
            f_hat, fs, smooth_win=max(1, int(cfg.rocof_smoothing_win))
        )

        out["ROCOF_RMSE"] = _mk_metric(
            "ROCOF_RMSE",
            rmse(rocof_hat, rocof_true),
            units=cfg.rocof_unit,
            comment="RoCoF RMSE (numerical derivative)",
            formula="rmse(d/dt f_hat, d/dt f_true)",
            debug={"smooth_win": int(cfg.rocof_smoothing_win), "fs_hz": fs},
        )

        out["ROCOF_MAX_PEAK"] = _mk_metric(
            "ROCOF_MAX_PEAK",
            max_abs_error(rocof_hat, rocof_true),
            units=cfg.rocof_unit,
            comment="RoCoF maximum absolute error",
            formula="max(|rocof_hat - rocof_true|)",
            debug={"smooth_win": int(cfg.rocof_smoothing_win), "fs_hz": fs},
        )

        out["ROCOF_MAE"] = _mk_metric(
            "ROCOF_MAE",
            mae(rocof_hat, rocof_true),
            units=cfg.rocof_unit,
            comment="RoCoF mean absolute error",
            formula="mean(|rocof_hat - rocof_true|)",
            debug={"smooth_win": int(cfg.rocof_smoothing_win), "fs_hz": fs},
        )

        if cfg.event_enable:
            dd, dbg = detection_delay(f_hat, f_true, cfg)
            trig = bool(np.isfinite(dd) and (not np.isinf(dd)))
            out["EVENT_DETECTION_DELAY_S"] = _mk_metric(
                "EVENT_DETECTION_DELAY_S",
                dd,
                units="s",
                threshold=float(cfg.event_detect_threshold_hz_per_s),
                comment="Event detection delay based on RoCoF threshold crossing",
                formula="(onset_hat - onset_true)/fs",
                triggered=trig,
                debug={**dbg, "fs_hz": fs},
            )

    if cfg.band_metrics_enable:
        for f0, f1 in cfg.band_edges_hz:
            k = f"BAND_ERR_RMS_{str(f0).replace('.', 'p')}_{str(f1).replace('.', 'p')}"
            out[k] = _mk_metric(
                k,
                band_error_rms(err, fs, (float(f0), float(f1))),
                units="Hz",
                comment=f"Band-limited RMS of error in [{f0},{f1}) Hz (FFT mask)",
                formula="rms(ifft(fft(e)*mask_band))",
                debug={"band": [float(f0), float(f1)], "fs_hz": fs},
            )

    # Event window info
    if event_win is None:
        out["EVENT_WIN_START"] = _mk_metric(
            "EVENT_WIN_START",
            float("nan"),
            units="samples",
            comment="Event window start (samples). None if no event detected in f_true.",
            formula="detect_event_window(f_true)",
            triggered=False,
            debug={"event_window": None},
        )
        out["EVENT_WIN_END"] = _mk_metric(
            "EVENT_WIN_END",
            float("nan"),
            units="samples",
            comment="Event window end (samples). None if no event detected in f_true.",
            formula="detect_event_window(f_true)",
            triggered=False,
            debug={"event_window": None},
        )
    else:
        out["EVENT_WIN_START"] = _mk_metric(
            "EVENT_WIN_START",
            float(event_win[0]),
            units="samples",
            comment="Event window start (samples)",
            formula="detect_event_window(f_true)",
            triggered=True,
            debug={"event_window": [int(event_win[0]), int(event_win[1])]},
        )
        out["EVENT_WIN_END"] = _mk_metric(
            "EVENT_WIN_END",
            float(event_win[1]),
            units="samples",
            comment="Event window end (samples)",
            formula="detect_event_window(f_true)",
            triggered=True,
            debug={"event_window": [int(event_win[0]), int(event_win[1])]},
        )

    out["_DEBUG_FS_DT"] = _mk_metric(
        "_DEBUG_FS_DT",
        0.0,
        units="",
        comment="Debug: fs and dt used by metrics (value field unused)",
        formula="",
        triggered=True,
        debug={"fs_hz": fs, "dt_s": dt, "n_samples": int(n)},
    )

    return out


# ============================================================
# Main API: compute metric dictionary (FLAT FLOATS for summary)
# ============================================================


def compute_metrics(
    f_hat: np.ndarray,
    f_true: np.ndarray,
    exec_time_s: float,
    latency_samples: int,
    cfg: MetricConfig,
    scenario_id: str = "unknown",  # <-- Añadir esto
) -> Dict[str, float]:
    rich = compute_metrics_rich(
        f_hat=f_hat,
        f_true=f_true,
        exec_time_s=exec_time_s,
        latency_samples=latency_samples,
        cfg=cfg,
        scenario_id=scenario_id,  # <-- Pasarlo aquí
    )
    # Importante: prefer="raw" para que el promedio/desviación de MC
    # siga funcionando con los floats originales.
    return flatten_metric_objects(rich, prefer="raw")
