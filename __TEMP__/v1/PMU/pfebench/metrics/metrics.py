#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfebench/metrics/metrics.py

Suite Q1-grade de métricas SOLO para frecuencia monofásica (f_hat vs f_true).

Cubre:
  A) Steady-state accuracy + IEEE-style compliance (FE, RFE) + outlier rates
  B) Dinámica de evento (nadir, overshoot/undershoot, settling/response, ROCOF peak)
  C) Protección (trip flag + trip time)
  D) Costo computacional (time/sample) + latencia del algoritmo
  E) Estadística robusta (percentiles, CVaR, bias, MAD)

Diseño:
- Alineación causal: est[t] vs ref[t-L]
- Detección robusta de evento por cambio persistente en df (MAD-based)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# 1) CONFIG
# =============================================================================


@dataclass(frozen=True)
class MetricConfig:
    # Sampling
    fs_hz: float
    f_nom: float = 60.0

    # Ventanas
    warm_up_s: float = 0.10
    event_pre_s: float = 0.25
    event_post_s: float = 2.50

    # “IEEE-style” thresholds (steady-state)
    ieee_fe_limit_mhz: float = 5.0  # FE max (mHz) + outlier rate
    ieee_rfe_limit_hzs: float = 0.1  # RFE (Hz/s) max + outlier rate

    # Robust stats
    percentiles: Tuple[float, ...] = (50, 95, 99)
    cvar_levels: Tuple[float, ...] = (95,)

    # Dinámica
    trip_thresholds_hz: Tuple[float, ...] = (0.2, 0.5)
    settling_tols_hz: Tuple[float, ...] = (0.02, 0.05)

    # ROCOF
    rocof_smooth_w: int = 5

    # Evento (detector robusto)
    event_persist_s: float = 0.02
    event_robust_k: float = 6.0

    # Response-time: exige mantenerse dentro por hold_s
    response_hold_s: float = 0.05


# =============================================================================
# 2) TYPES + HELPERS
# =============================================================================

JSONValue = Union[None, bool, str, float, int, Dict[str, Any], List[Any]]


def _to_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _finite_1d(x: Any) -> np.ndarray:
    a = _to_1d(x)
    return a[np.isfinite(a)]


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (float, int, np.floating, np.integer)):
        xf = float(x)
        return xf if math.isfinite(xf) else None
    return None


def _mk_metric(
    name: str,
    value: Any,
    scenario_id: str,
    units: str = "",
    threshold: Optional[float] = None,
    compliance_mode: str = "abs_leq",  # abs_leq | leq | geq | none
    meta: Optional[Dict[str, JSONValue]] = None,
) -> Dict[str, JSONValue]:
    val_export = _safe_float(value)
    passed: Optional[bool] = None

    if threshold is None or compliance_mode == "none":
        passed = None
    else:
        thr = float(threshold)
        if val_export is None:
            passed = False
        else:
            if compliance_mode == "abs_leq":
                passed = bool(abs(val_export) <= thr)
            elif compliance_mode == "leq":
                passed = bool(val_export <= thr)
            elif compliance_mode == "geq":
                passed = bool(val_export >= thr)
            else:
                passed = None

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
    e = _to_1d(est)
    r = _to_1d(ref)
    n = min(e.size, r.size)
    if n <= 0:
        return np.array([]), np.array([])
    L = int(max(0, latency_samples))
    if L <= 0:
        return e[:n], r[:n]
    e_aligned = e[L:n]
    r_aligned = r[0 : n - L]
    m = min(e_aligned.size, r_aligned.size)
    return e_aligned[:m], r_aligned[:m]


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    x = _to_1d(x)
    w = int(max(1, w))
    if x.size == 0 or w == 1:
        return x
    k = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, k, mode="same")


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
    thr = med + float(k) * (1.4826 * mad)

    persist = max(3, int(max(0.0, persist_s) * fs))
    is_event = np.abs(df) > abs(thr)

    end = len(is_event) - persist
    for i in range(max(0, end)):
        if np.all(is_event[i : i + persist]):
            return i
    return None


# =============================================================================
# 3) MATH CORE (frequency-only)
# =============================================================================


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
    med = float(np.median(ae))
    return float(np.median(np.abs(ae - med)))


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
    if not e.size:
        return float("nan")
    return float(np.mean(np.abs(e) > float(thr)))


def rocof_series(f: np.ndarray, fs: float, smooth_w: int = 5) -> np.ndarray:
    f = _to_1d(f)
    if f.size < 2:
        return np.array([])
    f_s = _moving_average(f, smooth_w)
    return np.diff(f_s) * float(fs)


def rocof_error_rmse(
    f_est: np.ndarray, f_ref: np.ndarray, fs: float, smooth_w: int = 5
) -> float:
    r_est = rocof_series(f_est, fs, smooth_w=smooth_w)
    r_ref = rocof_series(f_ref, fs, smooth_w=smooth_w)
    n = min(r_est.size, r_ref.size)
    return rmse(r_est[:n] - r_ref[:n]) if n > 0 else float("nan")


def rocof_error_max_abs(
    f_est: np.ndarray, f_ref: np.ndarray, fs: float, smooth_w: int = 5
) -> float:
    r_est = rocof_series(f_est, fs, smooth_w=smooth_w)
    r_ref = rocof_series(f_ref, fs, smooth_w=smooth_w)
    n = min(r_est.size, r_ref.size)
    if n <= 0:
        return float("nan")
    return float(np.max(np.abs(r_est[:n] - r_ref[:n])))


def rocof_peak_abs(f: np.ndarray, fs: float, smooth_w: int = 5) -> float:
    r = rocof_series(f, fs, smooth_w=smooth_w)
    return float(np.max(np.abs(r))) if r.size else float("nan")


def settling_time(err: np.ndarray, fs: float, tol: float) -> float:
    e = np.abs(_to_1d(err))
    if e.size == 0:
        return float("nan")
    vio = np.where(e > float(tol))[0]
    if vio.size == 0:
        return 0.0
    return float(vio[-1] / float(fs))


def response_time(err: np.ndarray, fs: float, tol: float, hold_s: float) -> float:
    e = np.abs(_to_1d(err))
    if e.size == 0:
        return float("nan")
    fs = float(fs)
    hold = max(1, int(max(0.0, hold_s) * fs))
    ok = e <= float(tol)
    for i in range(0, max(0, ok.size - hold)):
        if np.all(ok[i : i + hold]):
            return float(i / fs)
    return float("nan")


def trip_time_and_flag(
    f_est: np.ndarray, f_nom: float, fs: float, thr: float
) -> Tuple[bool, float]:
    f_est = _to_1d(f_est)
    if f_est.size == 0:
        return False, float("nan")
    dev = np.abs(f_est - float(f_nom))
    idx = np.where(dev >= float(thr))[0]
    if idx.size == 0:
        return False, float("nan")
    return True, float(idx[0] / float(fs))


def nadir_stats(f_est: np.ndarray, f_ref: np.ndarray, fs: float) -> Tuple[float, float]:
    f_est = _to_1d(f_est)
    f_ref = _to_1d(f_ref)
    n = min(f_est.size, f_ref.size)
    if n <= 0:
        return float("nan"), float("nan")
    fe = f_est[:n]
    fr = f_ref[:n]
    i_ref = int(np.argmin(fr))
    i_est = int(np.argmin(fe))
    mag_err = float(fe[i_est] - fr[i_ref])
    time_err_ms = float((i_est - i_ref) / float(fs) * 1000.0)
    return mag_err, time_err_ms


def overshoot_undershoot(f_est: np.ndarray, f_ref: np.ndarray) -> Tuple[float, float]:
    f_est = _to_1d(f_est)
    f_ref = _to_1d(f_ref)
    n = min(f_est.size, f_ref.size)
    if n <= 0:
        return float("nan"), float("nan")
    d = f_est[:n] - f_ref[:n]
    return float(np.max(d)), float(np.min(d))


# =============================================================================
# 4) COMPUTE METRICS
# =============================================================================


def compute_metrics(
    f_hat: np.ndarray,
    f_true: np.ndarray,
    exec_time_s: float,
    latency_samples: int,
    cfg: MetricConfig,
    scenario_id: str = "unknown",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    fs = float(cfg.fs_hz)

    f_hat = _to_1d(f_hat)
    f_true = _to_1d(f_true)

    n_total = min(f_hat.size, f_true.size)
    f_h_raw = f_hat[:n_total]
    f_t_raw = f_true[:n_total]

    f_h_causal, f_t_causal = _align_causal(f_h_raw, f_t_raw, latency_samples)

    warm_up_idx = int(max(0.0, cfg.warm_up_s) * fs)
    event_idx = _robust_event_index(
        f_t_raw, fs, persist_s=cfg.event_persist_s, k=cfg.event_robust_k
    )

    # -------------------------------------------------------------------------
    # A) STEADY-STATE (causal, sin warm-up)
    # -------------------------------------------------------------------------
    if f_h_causal.size > warm_up_idx + 2:
        fh_ss = f_h_causal[warm_up_idx:]
        ft_ss = f_t_causal[warm_up_idx:]
        err_ss = fh_ss - ft_ss

        out["RMSE_HZ"] = _mk_metric("RMSE_HZ", rmse(err_ss), scenario_id, "Hz")
        out["MAE_HZ"] = _mk_metric("MAE_HZ", mae(err_ss), scenario_id, "Hz")
        out["BIAS_HZ"] = _mk_metric("BIAS_HZ", bias(err_ss), scenario_id, "Hz")
        out["MED_ABS_ERR_HZ"] = _mk_metric(
            "MED_ABS_ERR_HZ", median_abs(err_ss), scenario_id, "Hz"
        )
        out["MAD_ABS_ERR_HZ"] = _mk_metric(
            "MAD_ABS_ERR_HZ", mad_abs(err_ss), scenario_id, "Hz"
        )

        # FE max (mHz) + outlier rate (vs límite mHz)
        fe_mhz = fe_max_mhz(err_ss)
        out["FE_MAX_MHZ"] = _mk_metric(
            "FE_MAX_MHZ",
            fe_mhz,
            scenario_id,
            "mHz",
            threshold=cfg.ieee_fe_limit_mhz,
            compliance_mode="leq",
        )
        out["FE_OUTLIER_RATE"] = _mk_metric(
            "FE_OUTLIER_RATE",
            outlier_rate_abs(err_ss * 1000.0, cfg.ieee_fe_limit_mhz),
            scenario_id,
            "1",
            meta={"thr_mhz": float(cfg.ieee_fe_limit_mhz)},
        )

        # RFE (ROCOF error): RMSE + MAX abs + outlier rate (vs límite Hz/s)
        rfe_rmse = rocof_error_rmse(fh_ss, ft_ss, fs, smooth_w=cfg.rocof_smooth_w)
        rfe_max = rocof_error_max_abs(fh_ss, ft_ss, fs, smooth_w=cfg.rocof_smooth_w)
        out["RFE_RMSE_HZS"] = _mk_metric(
            "RFE_RMSE_HZS",
            rfe_rmse,
            scenario_id,
            "Hz/s",
            threshold=cfg.ieee_rfe_limit_hzs,
            compliance_mode="leq",
        )
        out["RFE_MAX_ABS_HZS"] = _mk_metric(
            "RFE_MAX_ABS_HZS",
            rfe_max,
            scenario_id,
            "Hz/s",
            threshold=cfg.ieee_rfe_limit_hzs,
            compliance_mode="leq",
        )

        r_est = rocof_series(fh_ss, fs, smooth_w=cfg.rocof_smooth_w)
        r_ref = rocof_series(ft_ss, fs, smooth_w=cfg.rocof_smooth_w)
        m = min(r_est.size, r_ref.size)
        if m > 0:
            r_err = r_est[:m] - r_ref[:m]
            out["RFE_OUTLIER_RATE"] = _mk_metric(
                "RFE_OUTLIER_RATE",
                outlier_rate_abs(r_err, cfg.ieee_rfe_limit_hzs),
                scenario_id,
                "1",
                meta={"thr_hzs": float(cfg.ieee_rfe_limit_hzs)},
            )
        else:
            out["RFE_OUTLIER_RATE"] = _mk_metric(
                "RFE_OUTLIER_RATE", float("nan"), scenario_id, "1"
            )

        # Robust tails (abs)
        for p in cfg.percentiles:
            out[f"P{p}_ABS_ERR_HZ"] = _mk_metric(
                f"P{p}_ABS_ERR_HZ",
                percentile_abs(err_ss, p),
                scenario_id,
                "Hz",
            )
        for c in cfg.cvar_levels:
            out[f"CVAR{c}_ABS_ERR_HZ"] = _mk_metric(
                f"CVAR{c}_ABS_ERR_HZ",
                cvar_abs(err_ss, c),
                scenario_id,
                "Hz",
            )
    else:
        for k in [
            "RMSE_HZ",
            "MAE_HZ",
            "BIAS_HZ",
            "MED_ABS_ERR_HZ",
            "MAD_ABS_ERR_HZ",
            "FE_MAX_MHZ",
            "FE_OUTLIER_RATE",
            "RFE_RMSE_HZS",
            "RFE_MAX_ABS_HZS",
            "RFE_OUTLIER_RATE",
        ]:
            out[k] = _mk_metric(k, float("nan"), scenario_id)

    # -------------------------------------------------------------------------
    # B) EVENT DYNAMICS (ventana evento)
    # -------------------------------------------------------------------------
    if event_idx is not None and n_total > 0:
        w0 = max(0, int(event_idx - cfg.event_pre_s * fs))
        w1 = min(n_total, int(event_idx + cfg.event_post_s * fs))

        fh_win_raw = f_h_raw[w0:w1]
        ft_win_raw = f_t_raw[w0:w1]
        fh_win, ft_win = _align_causal(fh_win_raw, ft_win_raw, latency_samples)

        if fh_win.size > 10 and ft_win.size > 10:
            n = min(fh_win.size, ft_win.size)
            fh_win = fh_win[:n]
            ft_win = ft_win[:n]
            err_win = fh_win - ft_win

            # Settling + Response (multi tol)
            for tol in cfg.settling_tols_hz:
                st = settling_time(err_win, fs, tol)
                rt = response_time(err_win, fs, tol, hold_s=cfg.response_hold_s)
                key_st = f"SETTLING_TIME_TOL_{str(tol).replace('.', 'p')}"
                key_rt = f"RESPONSE_TIME_TOL_{str(tol).replace('.', 'p')}"
                out[key_st] = _mk_metric(key_st, st, scenario_id, "s")
                out[key_rt] = _mk_metric(key_rt, rt, scenario_id, "s")

            # Overshoot/Undershoot
            ov, un = overshoot_undershoot(fh_win, ft_win)
            out["OVERSHOOT_HZ"] = _mk_metric("OVERSHOOT_HZ", ov, scenario_id, "Hz")
            out["UNDERSHOOT_HZ"] = _mk_metric("UNDERSHOOT_HZ", un, scenario_id, "Hz")

            # Nadir errors
            nad_mag, nad_t_ms = nadir_stats(fh_win, ft_win, fs)
            out["NADIR_ERR_HZ"] = _mk_metric("NADIR_ERR_HZ", nad_mag, scenario_id, "Hz")
            out["NADIR_TIME_ERR_MS"] = _mk_metric(
                "NADIR_TIME_ERR_MS", nad_t_ms, scenario_id, "ms"
            )

            # ROCOF peaks + peak error
            out["ROCOF_PEAK_ABS_TRUE_HZS"] = _mk_metric(
                "ROCOF_PEAK_ABS_TRUE_HZS",
                rocof_peak_abs(ft_win, fs, smooth_w=cfg.rocof_smooth_w),
                scenario_id,
                "Hz/s",
            )
            out["ROCOF_PEAK_ABS_EST_HZS"] = _mk_metric(
                "ROCOF_PEAK_ABS_EST_HZS",
                rocof_peak_abs(fh_win, fs, smooth_w=cfg.rocof_smooth_w),
                scenario_id,
                "Hz/s",
            )
            out["ROCOF_ERR_MAX_ABS_HZS"] = _mk_metric(
                "ROCOF_ERR_MAX_ABS_HZS",
                rocof_error_max_abs(fh_win, ft_win, fs, smooth_w=cfg.rocof_smooth_w),
                scenario_id,
                "Hz/s",
            )

            # Event detection delay: cuánto después del "event_idx" aparece el cambio en f_hat (causal)
            # Heurística: detecta evento también en f_hat raw y compara índices.
            est_event_idx = _robust_event_index(
                f_h_raw, fs, persist_s=cfg.event_persist_s, k=cfg.event_robust_k
            )
            if est_event_idx is not None:
                det_delay_s = float((est_event_idx - event_idx) / fs)
            else:
                det_delay_s = float("nan")
            out["EVENT_DETECTION_DELAY_S"] = _mk_metric(
                "EVENT_DETECTION_DELAY_S", det_delay_s, scenario_id, "s"
            )

            out["EVENT_DETECTED"] = _mk_metric(
                "EVENT_DETECTED", 1.0, scenario_id, "flag"
            )
        else:
            out["EVENT_DETECTED"] = _mk_metric(
                "EVENT_DETECTED", 1.0, scenario_id, "flag"
            )
            out["EVENT_WINDOW_TOO_SHORT"] = _mk_metric(
                "EVENT_WINDOW_TOO_SHORT", 1.0, scenario_id, "flag"
            )
    else:
        out["EVENT_DETECTED"] = _mk_metric("EVENT_DETECTED", 0.0, scenario_id, "flag")

    # -------------------------------------------------------------------------
    # C) PROTECCIÓN (trip flag + tiempo) sobre señal causal completa
    # -------------------------------------------------------------------------
    for thr in cfg.trip_thresholds_hz:
        tripped, tt = trip_time_and_flag(f_h_causal, cfg.f_nom, fs, thr)
        k_flag = f"TRIPPED_{str(thr).replace('.', 'p')}"
        k_time = f"TRIP_TIME_{str(thr).replace('.', 'p')}"
        out[k_flag] = _mk_metric(k_flag, 1.0 if tripped else 0.0, scenario_id, "flag")
        out[k_time] = _mk_metric(
            k_time, tt, scenario_id, "s", meta={"thr_hz": float(thr)}
        )

    # -------------------------------------------------------------------------
    # D) COSTO COMPUTACIONAL
    # -------------------------------------------------------------------------
    if n_total > 0:
        tps_us = (float(exec_time_s) / float(n_total)) * 1e6
        out["TIME_PER_SAMPLE_US"] = _mk_metric(
            "TIME_PER_SAMPLE_US", tps_us, scenario_id, "us"
        )
    else:
        out["TIME_PER_SAMPLE_US"] = _mk_metric(
            "TIME_PER_SAMPLE_US", float("nan"), scenario_id, "us"
        )

    out["LATENCY_SAMPLES"] = _mk_metric(
        "LATENCY_SAMPLES", float(latency_samples), scenario_id, "samples"
    )

    return out


# =============================================================================
# 5) MONTE CARLO AGG
# =============================================================================


def aggregate_monte_carlo(
    runs: List[Dict[str, Any]],
    cfg: Optional[MetricConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Agrega múltiples runs (seeds) -> mean/std/max/p95/p99/n.
    Asume formato run[k] = dict con "value".
    """
    if not runs:
        return {}

    keys = list(runs[0].keys())
    agg: Dict[str, Dict[str, float]] = {}

    for k in keys:
        vals: List[float] = []
        for r in runs:
            v = r.get(k, {}).get("value")
            if (
                v is not None
                and isinstance(v, (int, float))
                and math.isfinite(float(v))
            ):
                vals.append(float(v))

        a = np.asarray(vals, dtype=float)
        if a.size > 0:
            agg[k] = {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "max": float(np.max(a)),
                "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)) if a.size >= 5 else float(np.max(a)),
                "n": int(a.size),
            }
        else:
            agg[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}

    # metadata mínima trazable
    if cfg is not None:
        agg["_meta_fs_hz"] = {
            "mean": float(cfg.fs_hz),
            "std": 0.0,
            "max": float(cfg.fs_hz),
            "p95": float(cfg.fs_hz),
            "p99": float(cfg.fs_hz),
            "n": 1,
        }
        agg["_meta_f_nom"] = {
            "mean": float(cfg.f_nom),
            "std": 0.0,
            "max": float(cfg.f_nom),
            "p95": float(cfg.f_nom),
            "p99": float(cfg.f_nom),
            "n": 1,
        }

    return agg
