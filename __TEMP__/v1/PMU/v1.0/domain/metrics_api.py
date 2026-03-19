# domain/metrics_api.py  (a.k.a. your "metric.py")
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Tuple

from .metrics_base import (
    MetricConfig,
    _mk_metric,
    _percentile,
    _cvar,
)
from . import metrics_logic as logic


# ============================================================
# Utilities
# ============================================================


def _to_1d(x: Any) -> np.ndarray:
    """Safe conversion to 1D float array (prevents shape errors)."""
    return np.asarray(x, dtype=float).reshape(-1)


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if x.size else float("nan")


def _event_index_heuristic(
    f_true: np.ndarray, fs: float, scenario_id: str
) -> Optional[int]:
    """
    Best-effort event time inference WITHOUT changing the public contract.

    Priority:
      1) If scenario_id contains a hint like "...@1.50s" or "...t=1.5", parse it.
      2) Detect first significant RoCoF burst in f_true.
      3) Otherwise None (means "use full signal" for event-based metrics).
    """
    s = str(scenario_id)

    # (1) parse simple patterns: "@1.5s", "t=1.5", "t1.5s" etc.
    # Keep this extremely conservative to avoid false parses.
    import re

    m = re.search(r"@(\d+(?:\.\d+)?)s\b", s)
    if not m:
        m = re.search(r"\bt\s*=\s*(\d+(?:\.\d+)?)\b", s)
    if m:
        try:
            t_event = float(m.group(1))
            if np.isfinite(t_event) and t_event >= 0:
                return int(round(t_event * fs))
        except Exception:
            pass

    # (2) derivative-based heuristic on f_true
    f = _to_1d(f_true)
    n = f.size
    if n < int(0.5 * fs) + 5:
        return None

    # RoCoF (Hz/s) using simple diff (robust enough for heuristics)
    df = np.diff(f) * fs
    df = df[np.isfinite(df)]
    if df.size < 10:
        return None

    # "Significant burst": above 6-sigma of median absolute deviation proxy
    med = float(np.median(df))
    mad = float(np.median(np.abs(df - med)) + 1e-12)
    thr = med + 6.0 * (1.4826 * mad)

    # find first index where abs(rocof) exceeds threshold for a few samples
    # (short persistence reduces false triggers)
    rocof = np.diff(f) * fs
    rocof = np.where(np.isfinite(rocof), rocof, 0.0)

    persist = max(3, int(0.02 * fs))  # ~20ms
    for i in range(0, max(0, rocof.size - persist)):
        block = rocof[i : i + persist]
        if np.max(np.abs(block)) >= abs(thr):
            return int(i)  # rocof index corresponds to between i and i+1; close enough

    return None


def _slice_window(x: np.ndarray, i0: int, i1: int) -> np.ndarray:
    x = _to_1d(x)
    i0 = max(0, int(i0))
    i1 = min(int(i1), x.size)
    if i1 <= i0:
        return np.asarray([], dtype=float)
    return x[i0:i1]


def _align_causal(
    f_hat: np.ndarray, f_true: np.ndarray, latency_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Causal alignment: compare f_hat[t] vs f_true[t-L].
    Returns trimmed arrays with equal length.
    """
    fh = _to_1d(f_hat)
    ft = _to_1d(f_true)
    n = int(min(fh.size, ft.size))
    if n <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    L = int(max(0, latency_samples))
    if L <= 0:
        return fh[:n], ft[:n]

    # Need t-L in range => start at t=L, stop at t=n-1
    fh2 = fh[L:n]
    ft2 = ft[0 : n - L]
    m = int(min(fh2.size, ft2.size))
    return fh2[:m], ft2[:m]


def _mk_nan_pack(keys, scenario_id: str) -> Dict[str, Dict[str, Any]]:
    return {k: _mk_metric(k, float("nan"), scenario_id) for k in keys}


# ============================================================
# Public API (CONTRACT PRESERVED)
# ============================================================


def compute_metrics(
    f_hat: np.ndarray,
    f_true: np.ndarray,
    exec_time_s: float,
    latency_samples: int,
    cfg: MetricConfig,
    scenario_id: str = "unknown",
) -> Dict[str, Dict[str, Any]]:
    """
    Q1-grade metrics API.

    CONTRACTS PRESERVED:
      - same function signature
      - returns Dict[str, Dict[str, Any]] via _mk_metric(...)
      - does not require extra inputs (event time inferred best-effort)

    Key upgrades (bulletproof):
      1) RAW vs CAUSAL-aligned metrics (latency-aware).
      2) Event-windowed dynamic metrics (nadir/settling) when event detected.
      3) Warm-up applied ONLY to steady-state/tail metrics (not to event metrics).
      4) Defensive handling of NaNs/short signals.
      5) RoCoF units corrected to Hz/s (common definition).
    """
    fs = float(getattr(cfg, "fs_hz", 0.0) or 0.0)
    if not np.isfinite(fs) or fs <= 0:
        # Can't do anything meaningful without fs
        return _mk_nan_pack(
            ["RMSE_RAW", "FE_max_mHz_RAW", "TIME_PER_SAMPLE_US", "LATENCY_SAMPLES"],
            scenario_id,
        )

    f_hat_full = _to_1d(f_hat)
    f_true_full = _to_1d(f_true)
    n_total = int(min(f_hat_full.size, f_true_full.size))

    out: Dict[str, Dict[str, Any]] = {}

    if n_total <= 1:
        return _mk_nan_pack(
            ["RMSE_RAW", "FE_max_mHz_RAW", "TIME_PER_SAMPLE_US", "LATENCY_SAMPLES"],
            scenario_id,
        )

    # ------------------------------------------------------------
    # 0) Establish windows: warm-up + (optional) event window
    # ------------------------------------------------------------
    warm_up_s = (
        float(getattr(cfg, "warm_up_s", 0.1)) if hasattr(cfg, "warm_up_s") else 0.1
    )
    warm_up = int(max(0, round(warm_up_s * fs)))
    if n_total <= warm_up:
        warm_up = 0

    # Event window: best-effort inference
    ev_idx = _event_index_heuristic(f_true_full[:n_total], fs, scenario_id)
    # Default window sizes (can be overridden if cfg defines them)
    event_pre_s = (
        float(getattr(cfg, "event_pre_s", 0.0)) if hasattr(cfg, "event_pre_s") else 0.0
    )
    event_post_s = (
        float(getattr(cfg, "event_post_s", 2.5))
        if hasattr(cfg, "event_post_s")
        else 2.5
    )
    pre = int(max(0, round(event_pre_s * fs)))
    post = int(max(1, round(event_post_s * fs)))

    if ev_idx is not None:
        w0 = max(0, int(ev_idx) - pre)
        w1 = min(n_total, int(ev_idx) + post)
    else:
        w0, w1 = 0, n_total

    # ------------------------------------------------------------
    # 1) RAW and CAUSAL-aligned signals
    # ------------------------------------------------------------
    # RAW (no latency correction)
    fh_raw = f_hat_full[:n_total]
    ft_raw = f_true_full[:n_total]

    # CAUSAL alignment
    fh_c, ft_c = _align_causal(fh_raw, ft_raw, int(latency_samples))

    # ------------------------------------------------------------
    # 2) Steady-state / tail metrics (apply warm-up)
    # ------------------------------------------------------------
    # RAW steady-state
    fh_raw_clean = fh_raw[warm_up:]
    ft_raw_clean = ft_raw[warm_up:]
    m_raw = int(min(fh_raw_clean.size, ft_raw_clean.size))
    fh_raw_clean = fh_raw_clean[:m_raw]
    ft_raw_clean = ft_raw_clean[:m_raw]

    # CAUSAL steady-state (warm-up should be applied after alignment)
    fh_c_clean = fh_c[warm_up:]
    ft_c_clean = ft_c[warm_up:]
    m_c = int(min(fh_c_clean.size, ft_c_clean.size))
    fh_c_clean = fh_c_clean[:m_c]
    ft_c_clean = ft_c_clean[:m_c]

    # Fail-safe for too short clean signals
    if fh_raw_clean.size <= 1 or ft_raw_clean.size <= 1:
        # still export compute + latency
        tps_us = (float(exec_time_s) / max(1, n_total)) * 1e6
        out["TIME_PER_SAMPLE_US"] = _mk_metric(
            "TIME_PER_SAMPLE_US", tps_us, scenario_id, units="us"
        )
        out["LATENCY_SAMPLES"] = _mk_metric(
            "LATENCY_SAMPLES", float(latency_samples), scenario_id, units="samples"
        )
        # minimal
        out["RMSE_RAW"] = _mk_metric("RMSE_RAW", float("nan"), scenario_id, units="Hz")
        out["FE_max_mHz_RAW"] = _mk_metric(
            "FE_max_mHz_RAW",
            float("nan"),
            scenario_id,
            units="mHz",
            threshold=getattr(cfg, "ieee_fe_limit_mhz", None),
        )
        out["RMSE_CAUSAL"] = _mk_metric(
            "RMSE_CAUSAL", float("nan"), scenario_id, units="Hz"
        )
        out["FE_max_mHz_CAUSAL"] = _mk_metric(
            "FE_max_mHz_CAUSAL",
            float("nan"),
            scenario_id,
            units="mHz",
            threshold=getattr(cfg, "ieee_fe_limit_mhz", None),
        )
        return out

    err_raw_clean = fh_raw_clean - ft_raw_clean
    abs_err_raw_clean = np.abs(err_raw_clean)

    err_c_clean = (
        fh_c_clean - ft_c_clean
        if (fh_c_clean.size and ft_c_clean.size)
        else np.asarray([], dtype=float)
    )
    abs_err_c_clean = (
        np.abs(err_c_clean) if err_c_clean.size else np.asarray([], dtype=float)
    )

    # ------------------------------------------------------------
    # 3) Accuracy metrics: RAW + CAUSAL
    # ------------------------------------------------------------
    out["RMSE_RAW"] = _mk_metric(
        "RMSE_RAW", logic.rmse(fh_raw_clean, ft_raw_clean), scenario_id, units="Hz"
    )
    out["MAE_RAW"] = _mk_metric(
        "MAE_RAW", logic.mae(fh_raw_clean, ft_raw_clean), scenario_id, units="Hz"
    )
    out["FE_max_mHz_RAW"] = _mk_metric(
        "FE_max_mHz_RAW",
        logic.frequency_error_mhz(fh_raw_clean, ft_raw_clean),
        scenario_id,
        units="mHz",
        threshold=getattr(cfg, "ieee_fe_limit_mhz", None),
    )

    # Causal-aligned (may be empty if latency too large)
    if fh_c_clean.size > 1 and ft_c_clean.size > 1:
        out["RMSE_CAUSAL"] = _mk_metric(
            "RMSE_CAUSAL", logic.rmse(fh_c_clean, ft_c_clean), scenario_id, units="Hz"
        )
        out["MAE_CAUSAL"] = _mk_metric(
            "MAE_CAUSAL", logic.mae(fh_c_clean, ft_c_clean), scenario_id, units="Hz"
        )
        out["FE_max_mHz_CAUSAL"] = _mk_metric(
            "FE_max_mHz_CAUSAL",
            logic.frequency_error_mhz(fh_c_clean, ft_c_clean),
            scenario_id,
            units="mHz",
            threshold=getattr(cfg, "ieee_fe_limit_mhz", None),
        )
    else:
        out["RMSE_CAUSAL"] = _mk_metric(
            "RMSE_CAUSAL", float("nan"), scenario_id, units="Hz"
        )
        out["MAE_CAUSAL"] = _mk_metric(
            "MAE_CAUSAL", float("nan"), scenario_id, units="Hz"
        )
        out["FE_max_mHz_CAUSAL"] = _mk_metric(
            "FE_max_mHz_CAUSAL",
            float("nan"),
            scenario_id,
            units="mHz",
            threshold=getattr(cfg, "ieee_fe_limit_mhz", None),
        )

    # ------------------------------------------------------------
    # 4) RoCoF metrics (Hz/s) — RAW + CAUSAL
    # ------------------------------------------------------------
    # Note: keep your logic function; just fix units label here to "Hz/s" (common definition).
    out["RFE_RMSE_RAW"] = _mk_metric(
        "RFE_RMSE_RAW",
        logic.rocof_error_rmse(fh_raw_clean, ft_raw_clean, fs),
        scenario_id,
        units="Hz/s",
        threshold=getattr(cfg, "ieee_rfe_limit_hzs", None),
    )

    if fh_c_clean.size > 2 and ft_c_clean.size > 2:
        out["RFE_RMSE_CAUSAL"] = _mk_metric(
            "RFE_RMSE_CAUSAL",
            logic.rocof_error_rmse(fh_c_clean, ft_c_clean, fs),
            scenario_id,
            units="Hz/s",
            threshold=getattr(cfg, "ieee_rfe_limit_hzs", None),
        )
    else:
        out["RFE_RMSE_CAUSAL"] = _mk_metric(
            "RFE_RMSE_CAUSAL",
            float("nan"),
            scenario_id,
            units="Hz/s",
            threshold=getattr(cfg, "ieee_rfe_limit_hzs", None),
        )

    # ------------------------------------------------------------
    # 5) Dynamic/event metrics (windowed, do NOT warm-up)
    # ------------------------------------------------------------
    fh_raw_win = _slice_window(fh_raw, w0, w1)
    ft_raw_win = _slice_window(ft_raw, w0, w1)

    fh_c_win, ft_c_win = _align_causal(fh_raw_win, ft_raw_win, int(latency_samples))

    # Settling time: use windowed signal (more defensible than warm-up based)
    tol = (
        float(cfg.settling_tols_hz[0])
        if getattr(cfg, "settling_tols_hz", None)
        else 0.1
    )
    out["Settling_Time_RAW"] = _mk_metric(
        "Settling_Time_RAW",
        logic.settling_time(fh_raw_win, ft_raw_win, tol, fs),
        scenario_id,
        units="s",
    )
    if fh_c_win.size > 2 and ft_c_win.size > 2:
        out["Settling_Time_CAUSAL"] = _mk_metric(
            "Settling_Time_CAUSAL",
            logic.settling_time(fh_c_win, ft_c_win, tol, fs),
            scenario_id,
            units="s",
        )
    else:
        out["Settling_Time_CAUSAL"] = _mk_metric(
            "Settling_Time_CAUSAL", float("nan"), scenario_id, units="s"
        )

    # Nadir analysis (windowed)
    try:
        nadir_raw = logic.nadir_analysis(fh_raw_win, ft_raw_win, fs)
        out["Nadir_Mag_Err_RAW"] = _mk_metric(
            "Nadir_Mag_Err_RAW",
            float(nadir_raw.get("nadir_val_err_hz", np.nan)),
            scenario_id,
            units="Hz",
        )
        out["Nadir_Time_Err_RAW"] = _mk_metric(
            "Nadir_Time_Err_RAW",
            float(nadir_raw.get("nadir_time_err_ms", np.nan)),
            scenario_id,
            units="ms",
        )
    except Exception:
        out["Nadir_Mag_Err_RAW"] = _mk_metric(
            "Nadir_Mag_Err_RAW", float("nan"), scenario_id, units="Hz"
        )
        out["Nadir_Time_Err_RAW"] = _mk_metric(
            "Nadir_Time_Err_RAW", float("nan"), scenario_id, units="ms"
        )

    try:
        # causal window
        if fh_c_win.size > 2 and ft_c_win.size > 2:
            nadir_c = logic.nadir_analysis(fh_c_win, ft_c_win, fs)
            out["Nadir_Mag_Err_CAUSAL"] = _mk_metric(
                "Nadir_Mag_Err_CAUSAL",
                float(nadir_c.get("nadir_val_err_hz", np.nan)),
                scenario_id,
                units="Hz",
            )
            out["Nadir_Time_Err_CAUSAL"] = _mk_metric(
                "Nadir_Time_Err_CAUSAL",
                float(nadir_c.get("nadir_time_err_ms", np.nan)),
                scenario_id,
                units="ms",
            )
        else:
            out["Nadir_Mag_Err_CAUSAL"] = _mk_metric(
                "Nadir_Mag_Err_CAUSAL", float("nan"), scenario_id, units="Hz"
            )
            out["Nadir_Time_Err_CAUSAL"] = _mk_metric(
                "Nadir_Time_Err_CAUSAL", float("nan"), scenario_id, units="ms"
            )
    except Exception:
        out["Nadir_Mag_Err_CAUSAL"] = _mk_metric(
            "Nadir_Mag_Err_CAUSAL", float("nan"), scenario_id, units="Hz"
        )
        out["Nadir_Time_Err_CAUSAL"] = _mk_metric(
            "Nadir_Time_Err_CAUSAL", float("nan"), scenario_id, units="ms"
        )

    # Overshoot (windowed)
    out["Overshoot_RAW"] = _mk_metric(
        "Overshoot_RAW",
        logic.overshoot(fh_raw_win, ft_raw_win),
        scenario_id,
        units="Hz",
    )
    if fh_c_win.size > 2 and ft_c_win.size > 2:
        out["Overshoot_CAUSAL"] = _mk_metric(
            "Overshoot_CAUSAL",
            logic.overshoot(fh_c_win, ft_c_win),
            scenario_id,
            units="Hz",
        )
    else:
        out["Overshoot_CAUSAL"] = _mk_metric(
            "Overshoot_CAUSAL", float("nan"), scenario_id, units="Hz"
        )

    # ------------------------------------------------------------
    # 6) Protection / trip times (must be on FULL signal, RAW and CAUSAL)
    # ------------------------------------------------------------
    trip_thresholds = getattr(cfg, "trip_thresholds_hz", (0.5,))
    for thr in trip_thresholds:
        thr_f = float(thr)

        # RAW trip (full)
        val_raw, _info_raw = logic.trip_time(fh_raw, ft_raw, thr_f, fs)
        key_raw = f"TRIP_TIME_RAW_{str(thr_f).replace('.', 'p')}"
        out[key_raw] = _mk_metric(
            key_raw, val_raw, scenario_id, units="s", threshold=thr_f
        )

        # CAUSAL trip: align full then measure on aligned arrays
        fh_ct, ft_ct = _align_causal(fh_raw, ft_raw, int(latency_samples))
        if fh_ct.size > 2 and ft_ct.size > 2:
            val_c, _info_c = logic.trip_time(fh_ct, ft_ct, thr_f, fs)
            key_c = f"TRIP_TIME_CAUSAL_{str(thr_f).replace('.', 'p')}"
            out[key_c] = _mk_metric(
                key_c, val_c, scenario_id, units="s", threshold=thr_f
            )
        else:
            key_c = f"TRIP_TIME_CAUSAL_{str(thr_f).replace('.', 'p')}"
            out[key_c] = _mk_metric(
                key_c, float("nan"), scenario_id, units="s", threshold=thr_f
            )

    # ------------------------------------------------------------
    # 7) Tail risk metrics (percentiles / CVaR) — RAW and CAUSAL
    # ------------------------------------------------------------
    percentiles = getattr(cfg, "percentiles", (50, 95, 99))
    cvar_levels = getattr(cfg, "cvar_levels", (95,))

    for p in percentiles:
        p_i = int(p)
        out[f"P{p_i}_ABS_ERR_RAW"] = _mk_metric(
            f"P{p_i}_ABS_ERR_RAW",
            _percentile(abs_err_raw_clean, p_i),
            scenario_id,
            units="Hz",
        )
        if abs_err_c_clean.size:
            out[f"P{p_i}_ABS_ERR_CAUSAL"] = _mk_metric(
                f"P{p_i}_ABS_ERR_CAUSAL",
                _percentile(abs_err_c_clean, p_i),
                scenario_id,
                units="Hz",
            )
        else:
            out[f"P{p_i}_ABS_ERR_CAUSAL"] = _mk_metric(
                f"P{p_i}_ABS_ERR_CAUSAL", float("nan"), scenario_id, units="Hz"
            )

    for lv in cvar_levels:
        lv_i = int(lv)
        out[f"CVAR{lv_i}_ABS_ERR_RAW"] = _mk_metric(
            f"CVAR{lv_i}_ABS_ERR_RAW",
            _cvar(abs_err_raw_clean, lv_i),
            scenario_id,
            units="Hz",
        )
        if abs_err_c_clean.size:
            out[f"CVAR{lv_i}_ABS_ERR_CAUSAL"] = _mk_metric(
                f"CVAR{lv_i}_ABS_ERR_CAUSAL",
                _cvar(abs_err_c_clean, lv_i),
                scenario_id,
                units="Hz",
            )
        else:
            out[f"CVAR{lv_i}_ABS_ERR_CAUSAL"] = _mk_metric(
                f"CVAR{lv_i}_ABS_ERR_CAUSAL", float("nan"), scenario_id, units="Hz"
            )

    # Optional: small summary stats (helps debugging / analysis without extra passes)
    out["ABS_ERR_MEAN_RAW"] = _mk_metric(
        "ABS_ERR_MEAN_RAW", _safe_mean(abs_err_raw_clean), scenario_id, units="Hz"
    )
    out["ABS_ERR_STD_RAW"] = _mk_metric(
        "ABS_ERR_STD_RAW", _safe_std(abs_err_raw_clean), scenario_id, units="Hz"
    )
    if abs_err_c_clean.size:
        out["ABS_ERR_MEAN_CAUSAL"] = _mk_metric(
            "ABS_ERR_MEAN_CAUSAL", _safe_mean(abs_err_c_clean), scenario_id, units="Hz"
        )
        out["ABS_ERR_STD_CAUSAL"] = _mk_metric(
            "ABS_ERR_STD_CAUSAL", _safe_std(abs_err_c_clean), scenario_id, units="Hz"
        )
    else:
        out["ABS_ERR_MEAN_CAUSAL"] = _mk_metric(
            "ABS_ERR_MEAN_CAUSAL", float("nan"), scenario_id, units="Hz"
        )
        out["ABS_ERR_STD_CAUSAL"] = _mk_metric(
            "ABS_ERR_STD_CAUSAL", float("nan"), scenario_id, units="Hz"
        )

    # ------------------------------------------------------------
    # 8) Computational cost (keep your contract)
    # ------------------------------------------------------------
    tps_us = (float(exec_time_s) / max(1, n_total)) * 1e6
    out["TIME_PER_SAMPLE_US"] = _mk_metric(
        "TIME_PER_SAMPLE_US", tps_us, scenario_id, units="us"
    )
    out["LATENCY_SAMPLES"] = _mk_metric(
        "LATENCY_SAMPLES", float(latency_samples), scenario_id, units="samples"
    )

    # ------------------------------------------------------------
    # 9) Minimal provenance (no contract changes, just extra outputs)
    # ------------------------------------------------------------
    out["N_TOTAL"] = _mk_metric("N_TOTAL", float(n_total), scenario_id, units="samples")
    out["WARM_UP_SAMPLES"] = _mk_metric(
        "WARM_UP_SAMPLES", float(warm_up), scenario_id, units="samples"
    )
    out["EVENT_IDX"] = _mk_metric(
        "EVENT_IDX",
        float(ev_idx if ev_idx is not None else -1),
        scenario_id,
        units="samples",
    )
    out["EVENT_WIN_LEN"] = _mk_metric(
        "EVENT_WIN_LEN", float(max(0, w1 - w0)), scenario_id, units="samples"
    )

    return out
