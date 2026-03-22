"""
openfreqbench/metrics/frequency.py

Q1-grade steady-state frequency metrics.
Ported from pfebench/metrics/metrics.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

JSONValue = None | bool | str | float | int | dict[str, Any] | list[Any]


@dataclass(frozen=True)
class MetricConfig:
    fs_hz: float
    f_nom: float = 60.0
    warm_up_s: float = 0.10
    event_pre_s: float = 0.25
    event_post_s: float = 2.50
    ieee_fe_limit_mhz: float = 5.0
    ieee_rfe_limit_hzs: float = 0.1
    percentiles: tuple[float, ...] = (50, 95, 99)
    cvar_levels: tuple[float, ...] = (95,)
    trip_thresholds_hz: tuple[float, ...] = (0.2, 0.5)
    settling_tols_hz: tuple[float, ...] = (0.02, 0.05)
    rocof_smooth_w: int = 5
    event_persist_s: float = 0.02
    event_robust_k: float = 6.0
    response_hold_s: float = 0.05


def _to_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _finite_1d(x: Any) -> np.ndarray:
    a = _to_1d(x)
    return a[np.isfinite(a)]  # type: ignore[return-value]


def _safe_float(x: Any) -> float | None:
    try:
        xf = float(x)
        return xf if math.isfinite(xf) else None
    except Exception:
        return None


def _mk_metric(
    name: str,
    value: Any,
    scenario_id: str,
    units: str = "",
    threshold: float | None = None,
    compliance_mode: str = "none",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    val = _safe_float(value)
    passed = None
    if threshold is not None and compliance_mode != "none":
        thr = float(threshold)
        if val is None:
            passed = False
        elif compliance_mode == "abs_leq":
            passed = bool(abs(val) <= thr)
        elif compliance_mode == "leq":
            passed = bool(val <= thr)
        elif compliance_mode == "geq":
            passed = bool(val >= thr)
    out = {
        "name": name,
        "scenario_id": scenario_id,
        "value": val,
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


def _align_causal(est: Any, ref: Any, latency_samples: int = 0) -> tuple[np.ndarray, np.ndarray]:
    e, r = _to_1d(est), _to_1d(ref)
    n = min(e.size, r.size)
    if n <= 0:
        return np.array([]), np.array([])
    L = max(0, int(latency_samples))
    if L <= 0:
        return e[:n], r[:n]
    ea, ra = e[L:n], r[: n - L]
    m = min(ea.size, ra.size)
    return ea[:m], ra[:m]


def _moving_average(x: Any, w: int) -> np.ndarray:
    x = _to_1d(x)
    w = max(1, int(w))
    return x if (x.size == 0 or w == 1) else np.convolve(x, np.ones(w) / w, mode="same")


def _robust_event_index(
    f_true: Any,
    fs: float,
    persist_s: float,
    r_true: Any | None = None,
    freq_threshold_hz: float = 0.005,
    rocof_threshold_hzs: float = 0.05,
) -> int | None:
    """
    Identifies the start of a structural parameter event (steps, ramps, modulations)
    by evaluating threshold crossover against a robust pre-fault steady-state.
    Methodologically valid for IEEE C37.118 and IEC 60255 benchmarking.
    """
    f = _to_1d(f_true)
    if f.size < int(0.5 * fs):
        return None
    
    # Baseline via median over first ~2 cycles to reject initialization discontinuities
    f_base = float(np.median(f[:max(1, int(0.05 * fs))]))
    
    if r_true is not None:
        dr = _to_1d(r_true)[:f.size]
    else:
        # Central difference mitigates phase-shifts from standard first-order numerical differentiation
        dr = np.gradient(f) * fs
        
    is_ev = (np.abs(f - f_base) > freq_threshold_hz) | (np.abs(dr) > rocof_threshold_hzs)
    persist = max(1, int(max(0.0, persist_s) * fs))
    
    for i in range(max(0, len(is_ev) - persist)):
        if np.all(is_ev[i : i + persist]):
            return i
    return None


def rmse(err: Any) -> float:
    e = _finite_1d(err)
    return float(np.sqrt(np.mean(e**2))) if e.size else float("nan")


def mae(err: Any) -> float:
    e = _finite_1d(err)
    return float(np.mean(np.abs(e))) if e.size else float("nan")


def bias(err: Any) -> float:
    e = _finite_1d(err)
    return float(np.mean(e)) if e.size else float("nan")


def median_abs(err: Any) -> float:
    e = _finite_1d(err)
    return float(np.median(np.abs(e))) if e.size else float("nan")


def mad_abs(err: Any) -> float:
    e = _finite_1d(err)
    if not e.size:
        return float("nan")
    ae = np.abs(e)
    return float(np.median(np.abs(ae - np.median(ae))))


def fe_max_mhz(err: Any) -> float:
    e = _finite_1d(err)
    return float(np.max(np.abs(e)) * 1000.0) if e.size else float("nan")


def percentile_abs(err: Any, p: float) -> float:
    e = _finite_1d(err)
    return float(np.percentile(np.abs(e), p)) if e.size else float("nan")


def cvar_abs(err: Any, level: float) -> float:
    e = _finite_1d(err)
    if not e.size:
        return float("nan")
    ae = np.abs(e)
    q = float(np.percentile(ae, level))
    tail = ae[ae >= q]
    return float(np.mean(tail)) if tail.size else q


def outlier_rate_abs(err: Any, thr: float) -> float:
    e = _finite_1d(err)
    return float(np.mean(np.abs(e) > float(thr))) if e.size else float("nan")


def rocof_series(f: Any, fs: float, smooth_w: int = 5) -> np.ndarray:
    f = _to_1d(f)
    return np.diff(_moving_average(f, smooth_w)) * float(fs) if f.size >= 2 else np.array([])


def rocof_error_rmse(f_est: Any, f_ref: Any, fs: float, smooth_w: int = 5) -> float:
    r1, r2 = rocof_series(f_est, fs, smooth_w), rocof_series(f_ref, fs, smooth_w)
    n = min(r1.size, r2.size)
    return rmse(r1[:n] - r2[:n]) if n > 0 else float("nan")


def rocof_error_max_abs(f_est: Any, f_ref: Any, fs: float, smooth_w: int = 5) -> float:
    r1, r2 = rocof_series(f_est, fs, smooth_w), rocof_series(f_ref, fs, smooth_w)
    n = min(r1.size, r2.size)
    return float(np.max(np.abs(r1[:n] - r2[:n]))) if n > 0 else float("nan")


def rocof_peak_abs(f: Any, fs: float, smooth_w: int = 5) -> float:
    r = rocof_series(f, fs, smooth_w)
    return float(np.max(np.abs(r))) if r.size else float("nan")


def settling_time(err: Any, fs: float, tol: float) -> float:
    e = np.abs(_to_1d(err))
    if not e.size:
        return float("nan")
    vio = np.where(e > float(tol))[0]
    return float(vio[-1] / float(fs)) if vio.size else 0.0


def response_time(err: Any, fs: float, tol: float, hold_s: float) -> float:
    e = np.abs(_to_1d(err))
    if not e.size:
        return float("nan")
    hold = max(1, int(hold_s * float(fs)))
    ok = e <= float(tol)
    for i in range(max(0, ok.size - hold)):
        if np.all(ok[i : i + hold]):
            return float(i / float(fs))
    return float("nan")


def trip_time_and_flag(f_est: Any, f_nom: float, fs: float, thr: float) -> tuple[bool, float]:
    f = _to_1d(f_est)
    if not f.size:
        return False, float("nan")
    idx = np.where(np.abs(f - float(f_nom)) >= float(thr))[0]
    return (True, float(idx[0] / float(fs))) if idx.size else (False, float("nan"))


def nadir_stats(f_est: Any, f_ref: Any, fs: float) -> tuple[float, float]:
    fe, fr = _to_1d(f_est), _to_1d(f_ref)
    n = min(fe.size, fr.size)
    if n <= 0:
        return float("nan"), float("nan")
    fe, fr = fe[:n], fr[:n]
    min_fr = float(np.min(fr))
    if min_fr < min(fr[0], fr[-1]) - 0.05:
        # True U-shape frequency dip event verified
        return float(fe[int(np.argmin(fe))] - fr[int(np.argmin(fr))]), float(
            (int(np.argmin(fe)) - int(np.argmin(fr))) / float(fs) * 1000.0,
        )
    return float("nan"), float("nan")


def overshoot_undershoot(f_est: Any, f_ref: Any) -> tuple[float, float]:
    fe, fr = _to_1d(f_est), _to_1d(f_ref)
    n = min(fe.size, fr.size)
    if n <= 0:
        return float("nan"), float("nan")
    d = fe[:n] - fr[:n]
    return float(np.max(d)), float(np.min(d))

def compute_metrics(
    f_hat: Any,
    f_true: Any,
    exec_time_s: float,
    latency_samples: int,
    cfg: MetricConfig,
    scenario_id: str = "unknown",
    roco_f_true: Any | None = None,
) -> dict[str, Any]:
    out = {}
    fs = float(cfg.fs_hz)
    f_hat = _to_1d(f_hat)
    f_true = _to_1d(f_true)
    n_total = min(f_hat.size, f_true.size)
    f_h_raw = f_hat[:n_total]
    f_t_raw = f_true[:n_total]
    f_h_c, f_t_c = _align_causal(f_h_raw, f_t_raw, latency_samples)
    if roco_f_true is not None:
        r_t_raw = _to_1d(roco_f_true)[:n_total]
        _, r_t_c = _align_causal(f_h_raw, r_t_raw, latency_samples)
    else:
        r_t_c = None
    wu = int(max(0.0, cfg.warm_up_s) * fs)
    r_t_eval = r_t_raw if roco_f_true is not None else None
    ev_idx = _robust_event_index(f_t_raw, fs, cfg.event_persist_s, r_true=r_t_eval)

    if f_h_c.size > wu + 2:
        # Compute full error arrays to avoid edge effects from masking
        err_full = f_h_c - f_t_c
        r1_full = rocof_series(f_h_c, fs, cfg.rocof_smooth_w)
        r2_full = rocof_series(f_t_c, fs, cfg.rocof_smooth_w) if r_t_c is None else r_t_c[:r1_full.size]
        rn = min(r1_full.size, r2_full.size)
        rerr_full = r1_full[:rn] - r2_full[:rn]

        # Construct steady-state mask
        mask = np.ones(f_h_c.size, dtype=bool)
        mask[:wu] = False  # Exclude warmup
        
        if ev_idx is not None:
            # Segregate dynamic transient window from steady-state evaluation
            w0 = max(0, int(ev_idx - cfg.event_pre_s * fs))
            w1 = min(n_total, int(ev_idx + cfg.event_post_s * fs))
            w0_c = max(0, min(w0, f_h_c.size))
            w1_c = max(0, min(w1, f_h_c.size))
            if w1_c > w0_c:
                mask[w0_c:w1_c] = False
                
        err_ss = err_full[mask]
        err_dyn = err_full[~mask] if np.any(~mask) else np.array([])
        
        # Apply matched mask to ROCOF (length is N-1 if available)
        rmask = mask[:rn] if rn > 0 else np.array([], dtype=bool)
        rerr_ss = rerr_full[rmask] if rmask.size > 0 else np.array([])
        rerr_dyn = rerr_full[~rmask] if (rmask.size > 0 and np.any(~rmask)) else np.array([])

        out["RMSE_HZ"] = _mk_metric("RMSE_HZ", rmse(err_ss), scenario_id, "Hz")
        out["MAE_HZ"] = _mk_metric("MAE_HZ", mae(err_ss), scenario_id, "Hz")
        out["BIAS_HZ"] = _mk_metric("BIAS_HZ", bias(err_ss), scenario_id, "Hz")
        out["MED_ABS_ERR_HZ"] = _mk_metric("MED_ABS_ERR_HZ", median_abs(err_ss), scenario_id, "Hz")
        out["MAD_ABS_ERR_HZ"] = _mk_metric("MAD_ABS_ERR_HZ", mad_abs(err_ss), scenario_id, "Hz")

        out["DYN_RMSE_HZ"] = _mk_metric("DYN_RMSE_HZ", rmse(err_dyn), scenario_id, "Hz")
        out["DYN_MAE_HZ"] = _mk_metric("DYN_MAE_HZ", mae(err_dyn), scenario_id, "Hz")
        out["DYN_BIAS_HZ"] = _mk_metric("DYN_BIAS_HZ", bias(err_dyn), scenario_id, "Hz")
        out["DYN_MED_ABS_ERR_HZ"] = _mk_metric("DYN_MED_ABS_ERR_HZ", median_abs(err_dyn), scenario_id, "Hz")
        out["DYN_MAD_ABS_ERR_HZ"] = _mk_metric("DYN_MAD_ABS_ERR_HZ", mad_abs(err_dyn), scenario_id, "Hz")
        
        fe_v = fe_max_mhz(err_ss)
        out["FE_MAX_MHZ"] = _mk_metric(
            "FE_MAX_MHZ",
            fe_v,
            scenario_id,
            "mHz",
            threshold=cfg.ieee_fe_limit_mhz,
            compliance_mode="leq",
        )
        fe_dyn_v = fe_max_mhz(err_dyn)
        out["DYN_FE_MAX_MHZ"] = _mk_metric(
            "DYN_FE_MAX_MHZ",
            fe_dyn_v,
            scenario_id,
            "mHz",
            threshold=cfg.ieee_fe_limit_mhz,
            compliance_mode="leq",
        )
        out["FE_OUTLIER_RATE"] = _mk_metric(
            "FE_OUTLIER_RATE",
            outlier_rate_abs(err_ss * 1000, cfg.ieee_fe_limit_mhz),
            scenario_id,
            "1",
            meta={"thr_mhz": float(cfg.ieee_fe_limit_mhz)},
        )
        out["DYN_FE_OUTLIER_RATE"] = _mk_metric(
            "DYN_FE_OUTLIER_RATE",
            outlier_rate_abs(err_dyn * 1000, cfg.ieee_fe_limit_mhz),
            scenario_id,
            "1",
            meta={"thr_mhz": float(cfg.ieee_fe_limit_mhz)},
        )
        out["RFE_RMSE_HZS"] = _mk_metric(
            "RFE_RMSE_HZS",
            rmse(rerr_ss) if rerr_ss.size > 0 else float("nan"),
            scenario_id,
            "Hz/s",
            threshold=cfg.ieee_rfe_limit_hzs,
            compliance_mode="leq",
        )
        out["DYN_RFE_RMSE_HZS"] = _mk_metric(
            "DYN_RFE_RMSE_HZS",
            rmse(rerr_dyn) if rerr_dyn.size > 0 else float("nan"),
            scenario_id,
            "Hz/s",
            threshold=cfg.ieee_rfe_limit_hzs,
            compliance_mode="leq",
        )
        rfe_max = float(np.max(np.abs(rerr_ss))) if rerr_ss.size > 0 else float("nan")
        out["RFE_MAX_ABS_HZS"] = _mk_metric(
            "RFE_MAX_ABS_HZS",
            rfe_max,
            scenario_id,
            "Hz/s",
            threshold=cfg.ieee_rfe_limit_hzs,
            compliance_mode="leq",
        )
        rfe_dyn_max = float(np.max(np.abs(rerr_dyn))) if rerr_dyn.size > 0 else float("nan")
        out["DYN_RFE_MAX_ABS_HZS"] = _mk_metric(
            "DYN_RFE_MAX_ABS_HZS",
            rfe_dyn_max,
            scenario_id,
            "Hz/s",
            threshold=cfg.ieee_rfe_limit_hzs,
            compliance_mode="leq",
        )
        out["RFE_OUTLIER_RATE"] = _mk_metric(
            "RFE_OUTLIER_RATE",
            outlier_rate_abs(rerr_ss, cfg.ieee_rfe_limit_hzs) if rerr_ss.size > 0 else float("nan"),
            scenario_id,
            "1",
            meta={"thr_hzs": float(cfg.ieee_rfe_limit_hzs)},
        )
        out["DYN_RFE_OUTLIER_RATE"] = _mk_metric(
            "DYN_RFE_OUTLIER_RATE",
            outlier_rate_abs(rerr_dyn, cfg.ieee_rfe_limit_hzs) if rerr_dyn.size > 0 else float("nan"),
            scenario_id,
            "1",
            meta={"thr_hzs": float(cfg.ieee_rfe_limit_hzs)},
        )
        for p in cfg.percentiles:
            out[f"P{int(p)}_ABS_ERR_HZ"] = _mk_metric(
                f"P{int(p)}_ABS_ERR_HZ",
                percentile_abs(err_ss, p),
                scenario_id,
                "Hz",
            )
            out[f"DYN_P{int(p)}_ABS_ERR_HZ"] = _mk_metric(
                f"DYN_P{int(p)}_ABS_ERR_HZ",
                percentile_abs(err_dyn, p),
                scenario_id,
                "Hz",
            )
        for c in cfg.cvar_levels:
            out[f"CVAR{int(c)}_ABS_ERR_HZ"] = _mk_metric(
                f"CVAR{int(c)}_ABS_ERR_HZ",
                cvar_abs(err_ss, c),
                scenario_id,
                "Hz",
            )
            out[f"DYN_CVAR{int(c)}_ABS_ERR_HZ"] = _mk_metric(
                f"DYN_CVAR{int(c)}_ABS_ERR_HZ",
                cvar_abs(err_dyn, c),
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
            out[f"DYN_{k}"] = _mk_metric(f"DYN_{k}", float("nan"), scenario_id)

    if ev_idx is not None and n_total > 0:
        w0 = max(0, int(ev_idx - cfg.event_pre_s * fs))
        w1 = min(n_total, int(ev_idx + cfg.event_post_s * fs))
        fhw, ftw = _align_causal(f_h_raw[w0:w1], f_t_raw[w0:w1], 0)
        if fhw.size > 10 and ftw.size > 10:
            n = min(fhw.size, ftw.size)
            fhw, ftw = fhw[:n], ftw[:n]
            ew = fhw - ftw
            for tol in cfg.settling_tols_hz:
                tag = str(tol).replace(".", "p")
                out[f"SETTLING_TIME_TOL_{tag}"] = _mk_metric(
                    f"SETTLING_TIME_TOL_{tag}",
                    settling_time(ew, fs, tol),
                    scenario_id,
                    "s",
                )
                out[f"RESPONSE_TIME_TOL_{tag}"] = _mk_metric(
                    f"RESPONSE_TIME_TOL_{tag}",
                    response_time(ew, fs, tol, cfg.response_hold_s),
                    scenario_id,
                    "s",
                )
            ov, un = overshoot_undershoot(fhw, ftw)
            out["OVERSHOOT_HZ"] = _mk_metric("OVERSHOOT_HZ", ov, scenario_id, "Hz")
            out["UNDERSHOOT_HZ"] = _mk_metric("UNDERSHOOT_HZ", un, scenario_id, "Hz")
            nm, nt = nadir_stats(fhw, ftw, fs)
            out["NADIR_ERR_HZ"] = _mk_metric("NADIR_ERR_HZ", nm, scenario_id, "Hz")
            out["NADIR_TIME_ERR_MS"] = _mk_metric("NADIR_TIME_ERR_MS", nt, scenario_id, "ms")
            out["ROCOF_PEAK_ABS_TRUE_HZS"] = _mk_metric(
                "ROCOF_PEAK_ABS_TRUE_HZS",
                rocof_peak_abs(ftw, fs, cfg.rocof_smooth_w),
                scenario_id,
                "Hz/s",
            )
            out["ROCOF_PEAK_ABS_EST_HZS"] = _mk_metric(
                "ROCOF_PEAK_ABS_EST_HZS",
                rocof_peak_abs(fhw, fs, cfg.rocof_smooth_w),
                scenario_id,
                "Hz/s",
            )
            out["ROCOF_ERR_MAX_ABS_HZS"] = _mk_metric(
                "ROCOF_ERR_MAX_ABS_HZS",
                rocof_error_max_abs(fhw, ftw, fs, cfg.rocof_smooth_w),
                scenario_id,
                "Hz/s",
            )
            eei = _robust_event_index(f_h_raw, fs, cfg.event_persist_s)
            out["EVENT_DETECTION_DELAY_S"] = _mk_metric(
                "EVENT_DETECTION_DELAY_S",
                float((eei - ev_idx) / fs) if eei is not None else float("nan"),
                scenario_id,
                "s",
            )
            out["EVENT_DETECTED"] = _mk_metric("EVENT_DETECTED", 1.0, scenario_id, "flag")
        else:
            out["EVENT_DETECTED"] = _mk_metric("EVENT_DETECTED", 1.0, scenario_id, "flag")
    else:
        out["EVENT_DETECTED"] = _mk_metric("EVENT_DETECTED", 0.0, scenario_id, "flag")

    for thr in cfg.trip_thresholds_hz:
        tag = str(thr).replace(".", "p")
        tripped, tt = trip_time_and_flag(f_h_raw, cfg.f_nom, fs, thr)
        out[f"TRIPPED_{tag}"] = _mk_metric(
            f"TRIPPED_{tag}",
            1.0 if tripped else 0.0,
            scenario_id,
            "flag",
        )
        out[f"TRIP_TIME_{tag}"] = _mk_metric(
            f"TRIP_TIME_{tag}",
            tt,
            scenario_id,
            "s",
            meta={"thr_hz": float(thr)},
        )

    tps = (float(exec_time_s) / float(n_total)) * 1e6 if n_total > 0 else float("nan")
    out["TIME_PER_SAMPLE_US"] = _mk_metric("TIME_PER_SAMPLE_US", tps, scenario_id, "us")
    out["LATENCY_SAMPLES"] = _mk_metric(
        "LATENCY_SAMPLES",
        float(latency_samples),
        scenario_id,
        "samples",
    )
    return out
