"""
openfreqbench/metrics/frequency.py

Q1-grade metrics for monophasic frequency estimation (f_hat vs f_true).

Ported from pfebench/metrics/metrics.py (pfebench research system).

Coverage:
  A) Steady-state accuracy + IEEE-style compliance (FE, RFE) + outlier rates
  B) Event dynamics (nadir, overshoot, settling/response time, ROCOF peak)
  C) Protection (trip flag + trip time)
  D) Computational cost (time/sample) + algorithm latency

All public functions accept numpy arrays. No side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

JSONValue = Union[None, bool, str, float, int, Dict[str, Any], List[Any]]


# ---------------------------------------------------------------------------
# 1) CONFIG
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricConfig:
    """Immutable configuration for the metric suite."""

    fs_hz: float
    f_nom: float = 60.0
    warm_up_s: float = 0.10
    event_pre_s: float = 0.25
    event_post_s: float = 2.50
    ieee_fe_limit_mhz: float = 5.0
    ieee_rfe_limit_hzs: float = 0.1
    percentiles: Tuple[float, ...] = (50, 95, 99)
    cvar_levels: Tuple[float, ...] = (95,)
    trip_thresholds_hz: Tuple[float, ...] = (0.2, 0.5)
    settling_tols_hz: Tuple[float, ...] = (0.02, 0.05)
    rocof_smooth_w: int = 5
    event_persist_s: float = 0.02
    event_robust_k: float = 6.0
    response_hold_s: float = 0.05


# ---------------------------------------------------------------------------
# 2) HELPERS
# ---------------------------------------------------------------------------


def _to_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _finite_1d(x: Any) -> np.ndarray:
    a = _to_1d(x)
    return a[np.isfinite(a)]


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        xf = float(x)
        return xf if math.isfinite(xf) else None
    except (TypeError, ValueError):
        return None


def _mk_metric(
    name: str,
    value: Any,
    scenario_id: str,
    units: str = "",
    threshold: Optional[float] = None,
    compliance_mode: str = "none",
    meta: Optional[Dict[str, JSONValue]] = None,
) -> Dict[str, JSONValue]:
    val_export = _safe_float(value)
    passed: Optional[bool] = None
    if threshold is not None and compliance_mode != "none":
        thr = float(threshold)
        if val_export is None:
            passed = False
        elif compliance_mode == "abs_leq":
            passed = bool(abs(val_export) <= thr)
        elif compliance_mode == "leq":
            passed = bool(val_export <= thr)
        elif compliance_mode == "geq":
            passed = bool(val_export >= thr)
    out: Dict[str, JSONValue] = {
        "name": name,
        "scenario_id": scenario_id,
        "value": val_export,
        "units": units,
        "compliance": {
            "threshold": float(threshold) if threshold is not None else None,
            "mode": compliance_mode,
            "passed": passed,
        },
    }
    if meta:
        out["meta"] = meta
    return out


def _align_causal(
    est: np.ndarray, ref: np.ndarray, latency_samples: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    e, r = _to_1d(est), _to_1d(ref)
    n = min(e.size, r.size)
    if n <= 0:
        return np.array([]), np.array([])
    L = max(0, int(latency_samples))
    if L <= 0:
        return e[:n], r[:n]
    e_al = e[L:n]
    r_al = r[: n - L]
    m = min(e_al.size, r_al.size)
    return e_al[:m], r_al[:m]


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    x = _to_1d(x)
    w = max(1, int(w))
    if x.size == 0 or w == 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def _robust_event_index(
    f_true: np.ndarray, fs: float, persist_s: float, k: float
) -> Optional[int]:
    f = _to_1d(f_true)
    if f.size < int(0.5 * fs):
        return None
    df = np.diff(f)
    if df.size == 0:
        return None
    med = float(np.median(df))
    mad = float(np.median(np.abs(df - med))) + 1e-12
    thr = med + float(k) * 1.4826 * mad
    persist = max(3, int(max(0.0, persist_s) * fs))
    is_event = np.abs(df) > abs(thr)
    for i in range(max(0, len(is_event) - persist)):
        if np.all(is_event[i : i + persist]):
            return i
    return None


# ---------------------------------------------------------------------------
# 3) MATH CORE
# ---------------------------------------------------------------------------


def rmse(err: np.ndarray) -> float:
    e = _finite_1d(err)
    return float(np.sqrt(np.mean(e**2))) if e.size else float("nan")


def mae(err: np.ndarray) -> float:
    e = _finite_1d(err)
    return float(np.mean(np.abs(e))) if e.size else float("nan")


def bias(err: np.ndarray) -> float:
    e = _finite_1d(err)
    return float(np.mean(e)) if e.size else float("nan")


def median_abs(err: np.ndarray) -> float:
    e = _finite_1d(err)
    return float(np.median(np.abs(e))) if e.size else float("nan")


def mad_abs(err: np.ndarray) -> float:
    e = _finite_1d(err)
    if not e.size:
        return float("nan")
    ae = np.abs(e)
    return float(np.median(np.abs(ae - np.median(ae))))


def fe_max_mhz(err: np.ndarray) -> float:
    e = _finite_1d(err)
    return float(np.max(np.abs(e)) * 1000.0) if e.size else float("nan")


def percentile_abs(err: np.ndarray, p: float) -> float:
    e = _finite_1d(err)
    return float(np.percentile(np.abs(e), p)) if e.size else float("nan")


def cvar_abs(err: np.ndarray, level: float) -> float:
    e = _finite_1d(err)
    if not e.size:
        return float("nan")
    ae = np.abs(e)
    q = float(np.percentile(ae, level))
    tail = ae[ae >= q]
    return float(np.mean(tail)) if tail.size else q


def outlier_rate_abs(err: np.ndarray, thr: float) -> float:
    e = _finite_1d(err)
    return float(np.mean(np.abs(e) > float(thr))) if e.size else float("nan")


def rocof_series(f: np.ndarray, fs: float, smooth_w: int = 5) -> np.ndarray:
    f = _to_1d(f)
    if f.size < 2:
        return np.array([])
    return np.diff(_moving_average(f, smooth_w)) * float(fs)


def rocof_error_rmse(
    f_est: np.ndarray, f_ref: np.ndarray, fs: float, smooth_w: int = 5
) -> float:
    r_est = rocof_series(f_est, fs, smooth_w)
    r_ref = rocof_series(f_ref, fs, smooth_w)
    n = min(r_est.size, r_ref.size)
    return rmse(r_est[:n] - r_ref[:n]) if n > 0 else float("nan")


def rocof_error_max_abs(
    f_est: np.ndarray, f_ref: np.ndarray, fs: float, smooth_w: int = 5
) -> float:
    r_est = rocof_series(f_est, fs, smooth_w)
    r_ref = rocof_series(f_ref, fs, smooth_w)
    n = min(r_est.size, r_ref.size)
    return float(np.max(np.abs(r_est[:n] - r_ref[:n]))) if n > 0 else float("nan")


def rocof_peak_abs(f: np.ndarray, fs: float, smooth_w: int = 5) -> float:
    r = rocof_series(f, fs, smooth_w)
    return float(np.max(np.abs(r))) if r.size else float("nan")


def settling_time(err: np.ndarray, fs: float, tol: float) -> float:
    e = np.abs(_to_1d(err))
    if not e.size:
        return float("nan")
    vio = np.where(e > float(tol))[0]
    return float(vio[-1] / float(fs)) if vio.size else 0.0


def response_time(err: np.ndarray, fs: float, tol: float, hold_s: float) -> float:
    e = np.abs(_to_1d(err))
    if not e.size:
        return float("nan")
    hold = max(1, int(hold_s * float(fs)))
    ok = e <= float(tol)
    for i in range(max(0, ok.size - hold)):
        if np.all(ok[i : i + hold]):
            return float(i / float(fs))
    return float("nan")


def trip_time_and_flag(
    f_est: np.ndarray, f_nom: float, fs: float, thr: float
) -> Tuple[bool, float]:
    f = _to_1d(f_est)
    if not f.size:
        return False, float("nan")
    idx = np.where(np.abs(f - float(f_nom)) >= float(thr))[0]
    return (True, float(idx[0] / float(fs))) if idx.size else (False, float("nan"))


def nadir_stats(f_est: np.ndarray, f_ref: np.ndarray, fs: float) -> Tuple[float, float]:
    fe = _to_1d(f_est)
    fr = _to_1d(f_ref)
    n = min(fe.size, fr.size)
    if n <= 0:
        return float("nan"), float("nan")
    fe, fr = fe[:n], fr[:n]
    i_ref = int(np.argmin(fr))
    i_est = int(np.argmin(fe))
    return float(fe[i_est] - fr[i_ref]), float((i_est - i_ref) / float(fs) * 1000.0)


def overshoot_undershoot(f_est: np.ndarray, f_ref: np.ndarray) -> Tuple[float, float]:
    fe, fr = _to_1d(f_est), _to_1d(f_ref)
    n = min(fe.size, fr.size)
    if n <= 0:
        return float("nan"), float("nan")
    d = fe[:n] - fr[:n]
    return float(np.max(d)), float(np.min(d))


# ---------------------------------------------------------------------------
# 4) FULL METRIC COMPUTATION
# ---------------------------------------------------------------------------


def compute_metrics(
    f_hat: np.ndarray,
    f_true: np.ndarray,
    exec_time_s: float,
    latency_samples: int,
    cfg: MetricConfig,
    scenario_id: str = "unknown",
) -> Dict[str, Any]:
    """Compute the full Q1 metric dict for one (f_hat, f_true) pair."""

    out: Dict[str, Any] = {}
    fs = float(cfg.fs_hz)

    f_hat = _to_1d(f_hat)
    f_true = _to_1d(f_true)
    n_total = min(f_hat.size, f_true.size)
    f_h_raw = f_hat[:n_total]
    f_t_raw = f_true[:n_total]

    f_h_causal, f_t_causal = _align_causal(f_h_raw, f_t_raw, latency_samples)
    warm_up_idx = int(max(0.0, cfg.warm_up_s) * fs)
    event_idx = _robust_event_index(f_t_raw, fs, cfg.event_persist_s, cfg.event_robust_k)

    # A) Steady-state
    if f_h_causal.size > warm_up_idx + 2:
        fh_ss = f_h_causal[warm_up_idx:]
        ft_ss = f_t_causal[warm_up_idx:]
        err_ss = fh_ss - ft_ss

        out["RMSE_HZ"] = _mk_metric("RMSE_HZ", rmse(err_ss), scenario_id, "Hz")
        out["MAE_HZ"] = _mk_metric("MAE_HZ", mae(err_ss), scenario_id, "Hz")
        out["BIAS_HZ"] = _mk_metric("BIAS_HZ", bias(err_ss), scenario_id, "Hz")
        out["MED_ABS_ERR_HZ"] = _mk_metric("MED_ABS_ERR_HZ", median_abs(err_ss), scenario_id, "Hz")
        out["MAD_ABS_ERR_HZ"] = _mk_metric("MAD_ABS_ERR_HZ", mad_abs(err_ss), scenario_id, "Hz")

        fe_mhz_val = fe_max_mhz(err_ss)
        out["FE_MAX_MHZ"] = _mk_metric(
            "FE_MAX_MHZ", fe_mhz_val, scenario_id, "mHz",
            threshold=cfg.ieee_fe_limit_mhz, compliance_mode="leq",
        )
        out["FE_OUTLIER_RATE"] = _mk_metric(
            "FE_OUTLIER_RATE",
            outlier_rate_abs(err_ss * 1000.0, cfg.ieee_fe_limit_mhz),
            scenario_id, "1", meta={"thr_mhz": float(cfg.ieee_fe_limit_mhz)},
        )

        rfe_r = rocof_error_rmse(fh_ss, ft_ss, fs, smooth_w=cfg.rocof_smooth_w)
        rfe_m = rocof_error_max_abs(fh_ss, ft_ss, fs, smooth_w=cfg.rocof_smooth_w)
        out["RFE_RMSE_HZS"] = _mk_metric(
            "RFE_RMSE_HZS", rfe_r, scenario_id, "Hz/s",
            threshold=cfg.ieee_rfe_limit_hzs, compliance_mode="leq",
        )
        out["RFE_MAX_ABS_HZS"] = _mk_metric(
            "RFE_MAX_ABS_HZS", rfe_m, scenario_id, "Hz/s",
            threshold=cfg.ieee_rfe_limit_hzs, compliance_mode="leq",
        )

        r_est = rocof_series(fh_ss, fs, smooth_w=cfg.rocof_smooth_w)
        r_ref = rocof_series(ft_ss, fs, smooth_w=cfg.rocof_smooth_w)
        m = min(r_est.size, r_ref.size)
        rfe_out_val = outlier_rate_abs(r_est[:m] - r_ref[:m], cfg.ieee_rfe_limit_hzs) if m > 0 else float("nan")
        out["RFE_OUTLIER_RATE"] = _mk_metric(
            "RFE_OUTLIER_RATE", rfe_out_val, scenario_id, "1",
            meta={"thr_hzs": float(cfg.ieee_rfe_limit_hzs)},
        )

        for p in cfg.percentiles:
            out[f"P{int(p)}_ABS_ERR_HZ"] = _mk_metric(
                f"P{int(p)}_ABS_ERR_HZ", percentile_abs(err_ss, p), scenario_id, "Hz"
            )
        for c in cfg.cvar_levels:
            out[f"CVAR{int(c)}_ABS_ERR_HZ"] = _mk_metric(
                f"CVAR{int(c)}_ABS_ERR_HZ", cvar_abs(err_ss, c), scenario_id, "Hz"
            )
    else:
        _nan_ss = float("nan")
        for key in [
            "RMSE_HZ", "MAE_HZ", "BIAS_HZ", "MED_ABS_ERR_HZ", "MAD_ABS_ERR_HZ",
            "FE_MAX_MHZ", "FE_OUTLIER_RATE", "RFE_RMSE_HZS", "RFE_MAX_ABS_HZS", "RFE_OUTLIER_RATE",
        ]:
            out[key] = _mk_metric(key, _nan_ss, scenario_id)

    # B) Event dynamics
    if event_idx is not None and n_total > 0:
        w0 = max(0, int(event_idx - cfg.event_pre_s * fs))
        w1 = min(n_total, int(event_idx + cfg.event_post_s * fs))
        fh_win, ft_win = _align_causal(f_h_raw[w0:w1], f_t_raw[w0:w1], latency_samples)
        if fh_win.size > 10 and ft_win.size > 10:
            n = min(fh_win.size, ft_win.size)
            fh_win, ft_win = fh_win[:n], ft_win[:n]
            err_win = fh_win - ft_win
            for tol in cfg.settling_tols_hz:
                tag = str(tol).replace(".", "p")
                out[f"SETTLING_TIME_TOL_{tag}"] = _mk_metric(
                    f"SETTLING_TIME_TOL_{tag}", settling_time(err_win, fs, tol), scenario_id, "s"
                )
                out[f"RESPONSE_TIME_TOL_{tag}"] = _mk_metric(
                    f"RESPONSE_TIME_TOL_{tag}", response_time(err_win, fs, tol, cfg.response_hold_s), scenario_id, "s"
                )
            ov, un = overshoot_undershoot(fh_win, ft_win)
            out["OVERSHOOT_HZ"] = _mk_metric("OVERSHOOT_HZ", ov, scenario_id, "Hz")
            out["UNDERSHOOT_HZ"] = _mk_metric("UNDERSHOOT_HZ", un, scenario_id, "Hz")
            nad_mag, nad_t_ms = nadir_stats(fh_win, ft_win, fs)
            out["NADIR_ERR_HZ"] = _mk_metric("NADIR_ERR_HZ", nad_mag, scenario_id, "Hz")
            out["NADIR_TIME_ERR_MS"] = _mk_metric("NADIR_TIME_ERR_MS", nad_t_ms, scenario_id, "ms")
            out["ROCOF_PEAK_ABS_TRUE_HZS"] = _mk_metric(
                "ROCOF_PEAK_ABS_TRUE_HZS", rocof_peak_abs(ft_win, fs, cfg.rocof_smooth_w), scenario_id, "Hz/s"
            )
            out["ROCOF_PEAK_ABS_EST_HZS"] = _mk_metric(
                "ROCOF_PEAK_ABS_EST_HZS", rocof_peak_abs(fh_win, fs, cfg.rocof_smooth_w), scenario_id, "Hz/s"
            )
            out["ROCOF_ERR_MAX_ABS_HZS"] = _mk_metric(
                "ROCOF_ERR_MAX_ABS_HZS",
                rocof_error_max_abs(fh_win, ft_win, fs, cfg.rocof_smooth_w), scenario_id, "Hz/s"
            )
            est_ei = _robust_event_index(f_h_raw, fs, cfg.event_persist_s, cfg.event_robust_k)
            det_delay = float((est_ei - event_idx) / fs) if est_ei is not None else float("nan")
            out["EVENT_DETECTION_DELAY_S"] = _mk_metric("EVENT_DETECTION_DELAY_S", det_delay, scenario_id, "s")
            out["EVENT_DETECTED"] = _mk_metric("EVENT_DETECTED", 1.0, scenario_id, "flag")
        else:
            out["EVENT_DETECTED"] = _mk_metric("EVENT_DETECTED", 1.0, scenario_id, "flag")
            out["EVENT_WINDOW_TOO_SHORT"] = _mk_metric("EVENT_WINDOW_TOO_SHORT", 1.0, scenario_id, "flag")
    else:
        out["EVENT_DETECTED"] = _mk_metric("EVENT_DETECTED", 0.0, scenario_id, "flag")

    # C) Protection
    for thr in cfg.trip_thresholds_hz:
        tag = str(thr).replace(".", "p")
        tripped, tt = trip_time_and_flag(f_h_causal, cfg.f_nom, fs, thr)
        out[f"TRIPPED_{tag}"] = _mk_metric(f"TRIPPED_{tag}", 1.0 if tripped else 0.0, scenario_id, "flag")
        out[f"TRIP_TIME_{tag}"] = _mk_metric(f"TRIP_TIME_{tag}", tt, scenario_id, "s", meta={"thr_hz": float(thr)})

    # D) Computational cost
    tps = (float(exec_time_s) / float(n_total)) * 1e6 if n_total > 0 else float("nan")
    out["TIME_PER_SAMPLE_US"] = _mk_metric("TIME_PER_SAMPLE_US", tps, scenario_id, "us")
    out["LATENCY_SAMPLES"] = _mk_metric("LATENCY_SAMPLES", float(latency_samples), scenario_id, "samples")

    return out
