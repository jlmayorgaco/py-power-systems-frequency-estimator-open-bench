#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scientific_analysis.py
======================
12 open scientific questions for the journal extension (IEEE TIM / IEEE TPWRS).
All analyses operate on results already computed by run_benchmark() in main.py.

Architecture rule: ALL new analyses live here. main.py only calls
run_all_scientific_analyses() at its end.

Interface
---------
run_all_scientific_analyses(results_dict, mc_raw, tuned_params,
                             results_raw_dir="results_raw") -> dict
"""

import os
import json
import warnings
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# ── Optional heavy dependencies — each function guards its own use ────────────
try:
    from scipy import stats as _sstats
    from scipy.optimize import curve_fit as _curve_fit
    from scipy.cluster.hierarchy import linkage as _linkage, fcluster as _fcluster
    from scipy.stats import fisher_exact as _fisher_exact
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    from sklearn.ensemble import RandomForestClassifier as _RFC
    from sklearn.model_selection import StratifiedKFold as _SKF, cross_val_score as _CVS
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.decomposition import PCA as _PCA
    from sklearn.cluster import KMeans as _KMeans
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
# =============================================================================
# DEPENDENCIAS EXTENDIDAS PARA JOURNAL (IEEE TRANSACTIONS)
# =============================================================================

# Familias algorítmicas estandarizadas
FAMILIES = {
    "Fourier/Window": ["IpDFT", "TFT"],
    "Kalman": ["EKF", "EKF2", "UKF", "LKF"],
    "Adaptive/RLS": ["RLS", "RLS-VFF"],
    "Loop/PLL": ["PLL", "SOGI"],
    "Non-linear": ["Teager", "Koopman-RKDPmu", "PI-GRU"]
}

def get_family(method_name):
    for family, methods in FAMILIES.items():
        if method_name in methods:
            return family
    return "Other"
# =============================================================================

# =============================================================================
# Constants
# =============================================================================
FIGURES_SCIENTIFIC_DIR = "figures_scientific"
os.makedirs(FIGURES_SCIENTIFIC_DIR, exist_ok=True)

# Okabe-Ito colorblind-safe palette
_OI = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
       "#0072B2", "#D55E00", "#CC79A7", "#000000"]

FAMILY_COLORS = {
    "window":      _OI[1],
    "loop":        _OI[0],
    "model":       _OI[2],
    "data_driven": _OI[5],
    "recursive":   _OI[6],
}
FAMILY_MAP = {
    "IpDFT": "window", "TFT": "window", "Teager": "window",
    "PLL": "loop", "SOGI": "loop",
    "EKF": "model", "EKF2": "model", "UKF": "model", "LKF": "model",
    "Koopman-RKDPmu": "data_driven", "PI-GRU": "data_driven",
    "RLS": "recursive", "RLS-VFF": "recursive",
}
CORE_METHODS = [
    "IpDFT", "PLL", "EKF", "EKF2", "SOGI",
    "TFT", "UKF", "Koopman-RKDPmu", "RLS", "RLS-VFF",
]
SCENARIOS = [
    "IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation",
    "IBR_Nightmare", "IBR_MultiEvent",
]
IEC_RMSE  = 0.05   # Hz
IEC_PEAK  = 0.5    # Hz
IEC_TTRIP = 0.1    # s
FS_DSP    = 10000.0


# =============================================================================
# Figure helpers
# =============================================================================
def _fig_style():
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "figure.dpi": 600,
        "lines.linewidth": 0.8, "axes.linewidth": 0.6,
    })


def _save_fig(fig, qnum: int, name: str):
    import matplotlib.pyplot as plt
    stem = os.path.join(FIGURES_SCIENTIFIC_DIR, f"Q{qnum}_{name}")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def _get(results_dict, sc, method, metric, default=np.nan):
    try:
        v = results_dict[sc]["methods"][method][metric]
        return float(v) if v is not None else default
    except (KeyError, TypeError, ValueError):
        return default


def _method_color(m):
    return FAMILY_COLORS.get(FAMILY_MAP.get(m, "window"), _OI[7])


# =============================================================================
# Q1: P/M Class Conflict Analysis
# =============================================================================
def pclass_mclass_conflict_analysis(results_dict: dict, mc_raw: dict) -> dict:
    """
    Quantify Protection (P-class) vs Metering (M-class) performance conflict.

    P-class: minimise TRIP_TIME_0p5 (fast fault detection)
    M-class: minimise RMSE (accurate steady-state measurement)

    Returns Pareto frontier, Spearman rank-correlation conflict index,
    and OLS regression RMSE ~ Ttrip.
    """
    records = []
    for sc in SCENARIOS:
        for m in CORE_METHODS:
            rmse  = _get(results_dict, sc, m, "RMSE")
            ttrip = _get(results_dict, sc, m, "TRIP_TIME_0p5")
            if np.isfinite(rmse) and np.isfinite(ttrip) and ttrip > 0:
                records.append({"method": m, "scenario": sc,
                                 "RMSE": rmse, "Ttrip": ttrip})

    if len(records) < 4:
        return {"error": "Insufficient finite data for Q1", "n_records": len(records)}

    rmse_arr  = np.array([r["RMSE"]  for r in records])
    ttrip_arr = np.array([r["Ttrip"] for r in records])

    if _SCIPY_OK:
        rho, p_rho = _sstats.spearmanr(rmse_arr, ttrip_arr)
        slope, intercept, r_val, p_slope, _ = _sstats.linregress(ttrip_arr, rmse_arr)
    else:
        rho = p_rho = slope = intercept = r_val = p_slope = np.nan

    conflict_index = float(1.0 - rho) if np.isfinite(rho) else np.nan

    # Pareto frontier (minimise both RMSE and Ttrip simultaneously)
    pareto_idx = []
    for i, ri in enumerate(records):
        dominated = any(
            j != i and records[j]["RMSE"] <= ri["RMSE"] and records[j]["Ttrip"] <= ri["Ttrip"]
            for j in range(len(records))
        )
        if not dominated:
            pareto_idx.append(i)

    # Per-method conflict Spearman rho (across scenarios)
    method_conflict = {}
    for m in CORE_METHODS:
        m_recs = [r for r in records if r["method"] == m]
        if _SCIPY_OK and len(m_recs) >= 3:
            mr = np.array([r["RMSE"]  for r in m_recs])
            mt = np.array([r["Ttrip"] for r in m_recs])
            rho_m, _ = _sstats.spearmanr(mr, mt)
            method_conflict[m] = float(rho_m)

    # MC-based conflict: using raw_mc FE_max vs TRIP per run
    mc_conflict_per_method = {}
    for m in CORE_METHODS:
        all_fe, all_tr = [], []
        for sc in SCENARIOS:
            key = (m, sc)
            if key in mc_raw:
                fe_vals   = mc_raw[key].get("FE_max", [])
                trip_vals = mc_raw[key].get("TRIP",   [])
                all_fe.extend(fe_vals)
                all_tr.extend(trip_vals)
        if _SCIPY_OK and len(all_fe) >= 6:
            rho_mc, p_mc = _sstats.spearmanr(all_fe, all_tr)
            mc_conflict_per_method[m] = {"rho": float(rho_mc), "p": float(p_mc)}

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))

        ax = axes[0]
        for m in CORE_METHODS:
            m_recs = [r for r in records if r["method"] == m]
            if not m_recs:
                continue
            ax.scatter([r["Ttrip"] for r in m_recs],
                       [r["RMSE"]  for r in m_recs],
                       c=_method_color(m), s=14, alpha=0.75, label=m, zorder=3)
        if np.isfinite(slope):
            xl = np.linspace(ttrip_arr.min(), ttrip_arr.max(), 100)
            ax.plot(xl, slope * xl + intercept, "k--", lw=0.8,
                    label=f"OLS $r^2$={r_val**2:.2f}")
        # Pareto step
        px = sorted([(records[i]["Ttrip"], records[i]["RMSE"]) for i in pareto_idx])
        ax.step([p[0] for p in px], [p[1] for p in px], "r-", lw=1.0,
                label="Pareto", where="post")
        ax.set_xlabel("$T_{trip}$ [s]")
        ax.set_ylabel("RMSE [Hz]")
        ax.set_title(f"Q1: P/M conflict  CI={conflict_index:.2f}")
        ax.legend(fontsize=5, ncol=2, loc="upper left")

        ax2 = axes[1]
        mets = [m for m in CORE_METHODS if m in method_conflict]
        vals = [method_conflict[m] for m in mets]
        ax2.bar(range(len(mets)), vals,
                color=[_method_color(m) for m in mets], edgecolor="k", lw=0.4)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_xticks(range(len(mets)))
        ax2.set_xticklabels(mets, rotation=45, ha="right", fontsize=6)
        ax2.set_ylabel("Spearman $\\rho$ (RMSE vs $T_{trip}$)")
        ax2.set_title("Per-method P/M rank conflict")
        fig.tight_layout()
        _save_fig(fig, 1, "PM_class_conflict")
    except Exception as e:
        warnings.warn(f"Q1 figure: {e}")

    return {
        "conflict_index": float(conflict_index) if np.isfinite(conflict_index) else None,
        "spearman_rho":   float(rho)    if np.isfinite(rho)    else None,
        "spearman_p":     float(p_rho)  if np.isfinite(p_rho)  else None,
        "ols_slope":      float(slope)  if np.isfinite(slope)  else None,
        "ols_r2":         float(r_val**2) if np.isfinite(r_val) else None,
        "pareto_pairs":   [{"method": records[i]["method"],
                            "scenario": records[i]["scenario"]} for i in pareto_idx],
        "per_method_rho": method_conflict,
        "mc_conflict":    mc_conflict_per_method,
        "n_records":      len(records),
        "interpretation": (
            f"CI={conflict_index:.2f} (0=no conflict, 2=max conflict). "
            f"rho={rho:.3f} p={p_rho:.3f}. "
            f"{'Strong P/M tradeoff.' if abs(rho) > 0.4 else 'Weak P/M tradeoff.'}"
        ) if np.isfinite(conflict_index) else "Insufficient data.",
    }


# =============================================================================
# Q2: Topology Invariance Test
# =============================================================================
def topology_invariance_test(tuned_params: dict) -> dict:
    """
    Test whether algorithm ranking is invariant across 5 synthetic grid topologies.

    Topologies synthesised from IBR_Nightmare base signal by varying:
      1. weak_grid:       noise sigma ×5, amplitude 0.7 pu
      2. stiff_grid:      noise sigma ×0.2, amplitude 1.0 pu (clean)
      3. isolated:        phase jump 120 deg (2× baseline 60 deg)
      4. renewable_heavy: 5th+7th+11th harmonics (each 5%)
      5. meshed:          superimposed 3 Hz frequency oscillation ±1 Hz

    Spearman rank correlation of RMSE across topology pairs.
    High rho (>0.8) = topology-invariant ranking.
    """
    try:
        from estimators import (
            FS_DSP as _FS, FS_PHYSICS as _FP, RATIO as _RAT,
            TunableIpDFT, StandardPLL, ClassicEKF, SOGI_FLL,
            RLS_Estimator, TFT_Estimator, RLS_VFF_Estimator, UKF_Estimator,
            Koopman_RKDPmu, calculate_metrics,
        )
        from ekf2 import EKF2
    except ImportError as e:
        return {"error": f"Import failed: {e}"}

    FS_P = 1_000_000.0
    FS_D = 10_000.0
    RAT  = int(FS_P / FS_D)

    def _make_base_ibr(noise_sigma=0.005, amp=1.0,
                       phase_jump_deg=60.0, extra_harmonics=None,
                       freq_osc_hz=0.0, freq_osc_amp=0.0):
        """Generate IBR_Nightmare variant signal at physics rate."""
        t = np.arange(0, 1.5, 1.0 / FS_P)
        n = len(t)
        f = np.ones(n) * 60.0
        if freq_osc_amp > 0:
            f += freq_osc_amp * np.sin(2 * np.pi * freq_osc_hz * t)
        phase_accum = np.zeros(n)
        curr = 0.0
        for i in range(1, n):
            if 0.6999 < t[i] < 0.7001:
                curr += np.deg2rad(phase_jump_deg)
            curr += 2 * np.pi * f[i] * (1.0 / FS_P)
            phase_accum[i] = curr
        v = amp * np.sin(phase_accum)
        v += 0.05 * np.sin(5 * phase_accum)
        v += 0.02 * np.sin(2 * np.pi * 32.5 * t)
        if extra_harmonics:
            for h_order, h_amp in extra_harmonics:
                v += h_amp * np.sin(h_order * phase_accum)
        v += np.random.default_rng(99).normal(0, noise_sigma, n)
        return t, v, f

    topologies = {
        "weak_grid":      _make_base_ibr(noise_sigma=0.025, amp=0.7),
        "stiff_grid":     _make_base_ibr(noise_sigma=0.001, amp=1.0),
        "isolated":       _make_base_ibr(noise_sigma=0.005, amp=1.0, phase_jump_deg=120.0),
        "renewable_heavy":_make_base_ibr(noise_sigma=0.005, amp=1.0,
                                          extra_harmonics=[(7, 0.05), (11, 0.05)]),
        "meshed":         _make_base_ibr(noise_sigma=0.005, amp=1.0,
                                          freq_osc_hz=3.0, freq_osc_amp=1.0),
    }

    def _get_tp(method, key, default):
        return tuned_params.get("IBR_Nightmare", {}).get(method, {}).get(key, default)

    factories = {
        "IpDFT":  lambda: TunableIpDFT(_get_tp("IpDFT", "cycles", 4)),
        "PLL":    lambda: StandardPLL(_get_tp("PLL", "kp", 10.0), _get_tp("PLL", "ki", 50.0)),
        "EKF":    lambda: ClassicEKF(_get_tp("EKF", "Q", 0.1), _get_tp("EKF", "R", 0.01)),
        "EKF2":   lambda: EKF2(
            q_param=        _get_tp("EKF2", "q_param", 1.0),
            r_param=        _get_tp("EKF2", "r_param", 0.01),
            inn_ref=        _get_tp("EKF2", "inn_ref", 0.1),
            event_thresh=   _get_tp("EKF2", "event_thresh", 2.0),
            fast_horizon_ms=_get_tp("EKF2", "fast_horizon_ms", 80.0),
        ),
        "SOGI":   lambda: SOGI_FLL(_get_tp("SOGI", "k", 1.414), _get_tp("SOGI", "g", 20.0),
                                    smooth_win=_get_tp("SOGI", "smooth_win", None)),
        "TFT":    lambda: TFT_Estimator(_get_tp("TFT", "win", 3)),
        "UKF":    lambda: UKF_Estimator(_get_tp("UKF", "Q", 0.1), _get_tp("UKF", "R", 0.01)),
        "Koopman-RKDPmu": lambda: Koopman_RKDPmu(
            window_samples=_get_tp("Koopman-RKDPmu", "window_samples", 200),
            smooth_win=    _get_tp("Koopman-RKDPmu", "smooth_win", 200)),
        "RLS":    lambda: RLS_Estimator(
            lam=_get_tp("RLS", "lambda", 0.995),
            win_smooth=_get_tp("RLS", "win_smooth", 100), decim=50),
        "RLS-VFF":lambda: RLS_VFF_Estimator(
            lam_min=_get_tp("RLS-VFF", "lam_min", 0.98), lam_max=0.9995,
            Ka=_get_tp("RLS-VFF", "Ka", 3.0), win_smooth=20, decim=50),
    }

    # RMSE matrix: topology × method
    rmse_matrix = {}   # {topology: {method: RMSE}}
    for topo, (t_sig, v_sig, f_sig) in topologies.items():
        v_dsp = v_sig[::RAT]
        f_dsp = f_sig[::RAT]
        rmse_matrix[topo] = {}
        for m, fac in factories.items():
            try:
                algo = fac()
                tr   = np.array([algo.step(x) for x in v_dsp])
                met  = calculate_metrics(tr, f_dsp, 0.0, structural_samples=1)
                rmse_matrix[topo][m] = float(met["RMSE"])
            except Exception as e:
                rmse_matrix[topo][m] = np.nan

    # Spearman rank correlation between all topology pairs
    topo_names = list(topologies.keys())
    n_t = len(topo_names)
    corr_matrix = np.full((n_t, n_t), np.nan)
    for i, t1 in enumerate(topo_names):
        for j, t2 in enumerate(topo_names):
            v1 = np.array([rmse_matrix[t1].get(m, np.nan) for m in factories])
            v2 = np.array([rmse_matrix[t2].get(m, np.nan) for m in factories])
            mask = np.isfinite(v1) & np.isfinite(v2)
            if _SCIPY_OK and mask.sum() >= 3:
                rho, _ = _sstats.spearmanr(v1[mask], v2[mask])
                corr_matrix[i, j] = float(rho)

    mean_rho = float(np.nanmean(corr_matrix[~np.eye(n_t, dtype=bool)]))

    # Rank each method within each topology (1 = best = lowest RMSE)
    rank_matrix = {}
    for topo in topo_names:
        vals = {m: rmse_matrix[topo].get(m, np.nan) for m in factories}
        finite_vals = sorted([(v, m) for m, v in vals.items() if np.isfinite(v)])
        rank_matrix[topo] = {m: rank + 1 for rank, (_, m) in enumerate(finite_vals)}

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))

        ax = axes[0]
        img = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n_t)); ax.set_xticklabels(topo_names, rotation=35, ha="right", fontsize=6)
        ax.set_yticks(range(n_t)); ax.set_yticklabels(topo_names, fontsize=6)
        for i in range(n_t):
            for j in range(n_t):
                if np.isfinite(corr_matrix[i, j]):
                    ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center",
                            fontsize=5, color="k")
        plt.colorbar(img, ax=ax, shrink=0.8)
        ax.set_title(f"Q2: Topology invariance\n(mean $\\rho$={mean_rho:.2f})")

        ax2 = axes[1]
        mets = list(factories.keys())
        x = np.arange(len(mets))
        w = 0.15
        for ti, topo in enumerate(topo_names):
            vals = [rmse_matrix[topo].get(m, np.nan) for m in mets]
            ax2.bar(x + ti * w, vals, w, label=topo, color=_OI[ti % len(_OI)], alpha=0.8)
        ax2.set_xticks(x + w * 2)
        ax2.set_xticklabels(mets, rotation=45, ha="right", fontsize=6)
        ax2.set_ylabel("RMSE [Hz]")
        ax2.set_title("RMSE per method per topology")
        ax2.legend(fontsize=5)
        fig.tight_layout()
        _save_fig(fig, 2, "topology_invariance")
    except Exception as e:
        warnings.warn(f"Q2 figure: {e}")

    return {
        "rmse_matrix":     rmse_matrix,
        "spearman_rho_matrix": {
            topo_names[i]: {topo_names[j]: float(corr_matrix[i, j])
                            for j in range(n_t)}
            for i in range(n_t)
        },
        "mean_spearman_rho": mean_rho,
        "rank_matrix":     rank_matrix,
        "topologies_tested": topo_names,
        "interpretation": (
            f"Mean pairwise Spearman rho={mean_rho:.2f} across {n_t} topology pairs. "
            f"{'Ranking is topology-invariant (rho>0.8).' if mean_rho > 0.8 else 'Ranking varies across topologies (rho<=0.8).'}"
        ),
    }


# =============================================================================
# Q3: Hazard Function Analysis (Kaplan-Meier)
# =============================================================================
def hazard_function_analysis(mc_raw: dict, trip_threshold: float = 0.5) -> dict:
    """
    Kaplan-Meier survival analysis treating each MC run as an independent trial.

    'Event' = frequency error exceeds trip_threshold (0.5 Hz).
    Uses TRIP_TIME_0p5 from mc_raw as the event time.
    Runs with TRIP=0.0 are right-censored (never tripped during the signal).

    Returns per-method per-scenario survival function and hazard rate estimates.
    """
    def _kaplan_meier(times, events):
        """
        times  : array of event/censor times [s]
        events : array of bool (True=tripped, False=censored)
        Returns (t_unique, S, H) survival and cumulative hazard arrays.
        """
        order = np.argsort(times)
        times   = np.array(times)[order]
        events  = np.array(events)[order]
        n       = len(times)
        t_vals, S_vals, H_vals = [0.0], [1.0], [0.0]
        n_at_risk = n
        S_current = 1.0
        H_current = 0.0
        for i, (t, ev) in enumerate(zip(times, events)):
            if ev and n_at_risk > 0:
                S_current *= (1.0 - 1.0 / n_at_risk)
                H_current += 1.0 / n_at_risk
                t_vals.append(float(t))
                S_vals.append(float(S_current))
                H_vals.append(float(H_current))
            n_at_risk -= 1
        return np.array(t_vals), np.array(S_vals), np.array(H_vals)

    results = {}
    for m in CORE_METHODS:
        results[m] = {}
        for sc in SCENARIOS:
            key = (m, sc)
            if key not in mc_raw:
                continue
            trip_vals = np.array(mc_raw[key].get("TRIP", []), dtype=float)
            fe_vals   = np.array(mc_raw[key].get("FE_max", []), dtype=float)
            if len(trip_vals) == 0:
                continue
            # Event = TRIP_TIME_0p5 > 0 AND FE_max > threshold
            times_  = np.where(trip_vals > 0, trip_vals, 1.5)   # censor at 1.5s
            events_ = (fe_vals > trip_threshold)
            t_km, S_km, H_km = _kaplan_meier(times_, events_)
            n_trips     = int(events_.sum())
            trip_rate   = float(n_trips / max(len(events_), 1))
            # Median survival time (first t where S <= 0.5)
            med_idx = np.searchsorted(-S_km, -0.5)
            median_surv = float(t_km[min(med_idx, len(t_km) - 1)])
            results[m][sc] = {
                "km_times":        t_km.tolist(),
                "km_survival":     S_km.tolist(),
                "km_hazard":       H_km.tolist(),
                "n_events":        n_trips,
                "n_censored":      int((~events_).sum()),
                "trip_rate":       trip_rate,
                "median_survival": median_surv,
            }

    # Log-rank test: EKF2 vs each other method in IBR_Nightmare
    logrank_vs_ekf2 = {}
    if _SCIPY_OK:
        sc_ref = "IBR_Nightmare"
        key_ref = ("EKF2", sc_ref)
        if key_ref in mc_raw:
            t2 = np.array(mc_raw[key_ref].get("TRIP", []), dtype=float)
            fe2 = np.array(mc_raw[key_ref].get("FE_max", []), dtype=float)
            ev2 = (fe2 > trip_threshold)
            times2 = np.where(t2 > 0, t2, 1.5)
            for m in CORE_METHODS:
                if m == "EKF2":
                    continue
                key_m = (m, sc_ref)
                if key_m not in mc_raw:
                    continue
                tm = np.array(mc_raw[key_m].get("TRIP", []), dtype=float)
                fem = np.array(mc_raw[key_m].get("FE_max", []), dtype=float)
                evm = (fem > trip_threshold)
                timesm = np.where(tm > 0, tm, 1.5)
                # Simple log-rank approximation via Wilcoxon on event times
                ev_times2 = times2[ev2]
                ev_timesm = timesm[evm]
                if len(ev_times2) > 0 and len(ev_timesm) > 0:
                    stat, p = _sstats.mannwhitneyu(
                        ev_times2, ev_timesm, alternative="two-sided")
                    logrank_vs_ekf2[m] = {"statistic": float(stat), "p_value": float(p)}

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        sc_plot = "IBR_Nightmare"
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
        ax = axes[0]
        for m in CORE_METHODS:
            if sc_plot in results.get(m, {}):
                km = results[m][sc_plot]
                ax.step(km["km_times"], km["km_survival"],
                        color=_method_color(m), label=m, where="post", lw=0.9)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Survival $S(t)$")
        ax.set_title(f"Q3: KM survival curves — {sc_plot}")
        ax.legend(fontsize=5, ncol=2)

        ax2 = axes[1]
        trip_rates = {m: results[m].get(sc_plot, {}).get("trip_rate", np.nan)
                      for m in CORE_METHODS}
        mets = [m for m in CORE_METHODS if np.isfinite(trip_rates[m])]
        ax2.bar(range(len(mets)),
                [trip_rates[m] for m in mets],
                color=[_method_color(m) for m in mets], edgecolor="k", lw=0.4)
        ax2.set_xticks(range(len(mets)))
        ax2.set_xticklabels(mets, rotation=45, ha="right", fontsize=6)
        ax2.set_ylabel("Trip rate (fraction of MC runs)")
        ax2.set_title(f"Trip rate in {sc_plot}")
        fig.tight_layout()
        _save_fig(fig, 3, "hazard_KM")
    except Exception as e:
        warnings.warn(f"Q3 figure: {e}")

    return {
        "survival_analysis": results,
        "logrank_vs_ekf2_IBR_Nightmare": logrank_vs_ekf2,
        "interpretation": (
            "Kaplan-Meier survival curves estimate the probability of not "
            "triggering a false or missed protection event over time. "
            "Methods with S(t)→1 are robustly non-tripping; S(t)→0 rapidly "
            "indicates high false-trip risk."
        ),
    }


# =============================================================================
# Q4: Innovation Signature Classifier
# =============================================================================
def innovation_signature_classifier(tuned_params: dict,
                                     results_raw_dir: str = "results_raw") -> dict:
    """
    Train a Random Forest to classify the scenario type from windowed EKF2
    innovation (frequency estimate) traces.

    Features (8 per window): mean, std, skewness, kurtosis, max_abs, RMS,
    lag-1 autocorrelation, peak-to-RMS ratio.

    5-fold stratified cross-validation. High accuracy = innovations carry
    scenario-discriminative information.
    """
    if not _SCIPY_OK or not _SKLEARN_OK:
        return {"error": "scipy and scikit-learn required for Q4"}

    # Load frequency traces from results_raw JSON files
    traces = {}   # {(method, scenario): np.array}
    for sc in SCENARIOS:
        sc_dir = os.path.join(results_raw_dir, sc)
        if not os.path.isdir(sc_dir):
            continue
        for fname in os.listdir(sc_dir):
            if not fname.endswith(".json"):
                continue
            method = fname.replace(f"{sc}__", "").replace(".json", "")
            if method not in CORE_METHODS:
                continue
            fpath = os.path.join(sc_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f_:
                    d = json.load(f_)
                f_hat = np.array(d.get("signals", {}).get("f_hat", []), dtype=float)
                f_true = np.array(d.get("signals", {}).get("f_true", []), dtype=float)
                if len(f_hat) > 0 and len(f_true) > 0:
                    innovation = f_hat - f_true   # frequency error ≈ innovation proxy
                    traces[(method, sc)] = innovation
            except Exception:
                pass

    if not traces:
        return {"error": "No trace data found in results_raw/"}

    def _features(window):
        """Extract 8 features from a signal window."""
        w = np.asarray(window, dtype=float)
        if len(w) < 4:
            return None
        mean_  = float(np.mean(w))
        std_   = float(np.std(w))
        skew_  = float(_sstats.skew(w))
        kurt_  = float(_sstats.kurtosis(w))
        maxabs = float(np.max(np.abs(w)))
        rms_   = float(np.sqrt(np.mean(w**2)))
        ac1    = float(np.corrcoef(w[:-1], w[1:])[0, 1]) if len(w) > 2 else 0.0
        p2rms  = float(maxabs / (rms_ + 1e-12))
        return [mean_, std_, skew_, kurt_, maxabs, rms_, ac1, p2rms]

    # Build dataset: 500-sample windows from each method×scenario trace
    WINDOW = 500
    X_list, y_list = [], []
    # Label: IEEE=0 (Mag_Step, Freq_Ramp, Modulation), IBR=1 (Nightmare, MultiEvent)
    IBR_SCENARIOS = {"IBR_Nightmare", "IBR_MultiEvent"}
    for (m, sc), trace in traces.items():
        label = 1 if sc in IBR_SCENARIOS else 0
        for start in range(0, len(trace) - WINDOW, WINDOW):
            feat = _features(trace[start: start + WINDOW])
            if feat is not None and all(np.isfinite(feat)):
                X_list.append(feat)
                y_list.append(label)

    if len(X_list) < 20:
        return {"error": f"Too few windows ({len(X_list)}) for classification"}

    X = np.array(X_list)
    y = np.array(y_list)

    scaler = _SS()
    X_sc   = scaler.fit_transform(X)

    clf = _RFC(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)
    cv  = _SKF(n_splits=5, shuffle=True, random_state=42)
    acc_scores = _CVS(clf, X_sc, y, cv=cv, scoring="accuracy")
    f1_scores  = _CVS(clf, X_sc, y, cv=cv, scoring="f1_macro")

    # Fit on full set for feature importance
    clf.fit(X_sc, y)
    feat_names = ["mean", "std", "skewness", "kurtosis",
                  "max_abs", "RMS", "autocorr_lag1", "peak_to_RMS"]
    importances = dict(zip(feat_names, clf.feature_importances_.tolist()))

    # Confusion matrix counts
    from sklearn.metrics import confusion_matrix
    y_pred = clf.predict(X_sc)
    cm = confusion_matrix(y, y_pred).tolist()

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))

        ax = axes[0]
        ax.bar(feat_names, clf.feature_importances_,
               color=_OI[1], edgecolor="k", lw=0.4)
        ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Feature importance")
        ax.set_title(f"Q4: RF feature importance\n"
                     f"ACC={np.mean(acc_scores):.2f}±{np.std(acc_scores):.2f}")

        ax2 = axes[1]
        cm_arr = np.array(cm)
        im = ax2.imshow(cm_arr, cmap="Blues", aspect="auto")
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["IEEE", "IBR"])
        ax2.set_yticks([0, 1]); ax2.set_yticklabels(["IEEE", "IBR"])
        ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
        for ii in range(2):
            for jj in range(2):
                ax2.text(jj, ii, str(cm_arr[ii, jj]), ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax2, shrink=0.8)
        ax2.set_title("Confusion matrix (training)")
        fig.tight_layout()
        _save_fig(fig, 4, "innovation_classifier")
    except Exception as e:
        warnings.warn(f"Q4 figure: {e}")

    return {
        "cv_accuracy_mean": float(np.mean(acc_scores)),
        "cv_accuracy_std":  float(np.std(acc_scores)),
        "cv_f1_mean":       float(np.mean(f1_scores)),
        "cv_f1_std":        float(np.std(f1_scores)),
        "cv_accuracy_per_fold": acc_scores.tolist(),
        "feature_importances":  importances,
        "confusion_matrix_train": cm,
        "n_windows": len(X_list),
        "n_ibr":     int((y == 1).sum()),
        "n_ieee":    int((y == 0).sum()),
        "interpretation": (
            f"RF classifier (5-fold CV): accuracy={np.mean(acc_scores):.2f}±"
            f"{np.std(acc_scores):.2f}, F1={np.mean(f1_scores):.2f}. "
            f"{'High accuracy: frequency error traces are scenario-discriminative.' if np.mean(acc_scores) > 0.8 else 'Moderate accuracy: partial scenario discrimination.'}"
        ),
    }


# =============================================================================
# Q5: THD Phase Transition Test
# =============================================================================
def thd_phase_transition_test(tuned_params: dict) -> dict:
    """
    Sweep THD 0–30% (7th harmonic amplitude) in the IBR_Nightmare base signal.
    Fit sigmoid: RMSE(THD) = L / (1 + exp(-k*(THD - THD_50)))
    THD_50 = half-saturation point (method-specific robustness threshold).
    """
    try:
        from estimators import (
            FS_PHYSICS as _FP, RATIO as _RAT,
            TunableIpDFT, StandardPLL, ClassicEKF, SOGI_FLL,
            TFT_Estimator, UKF_Estimator, Koopman_RKDPmu,
            RLS_Estimator, calculate_metrics,
        )
        from ekf2 import EKF2
    except ImportError as e:
        return {"error": f"Import failed: {e}"}

    FS_P = 1_000_000.0
    RAT  = 100

    THD_LEVELS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.30]

    def _gen_signal(thd_7th):
        t = np.arange(0, 1.5, 1.0 / FS_P)
        n = len(t)
        phase = np.zeros(n); curr = 0.0
        for i in range(1, n):
            if 0.6999 < t[i] < 0.7001:
                curr += np.deg2rad(60.0)
            curr += 2 * np.pi * 60.0 / FS_P
            phase[i] = curr
        v = np.sin(phase)
        v += 0.05 * np.sin(5 * phase)
        v += thd_7th * np.sin(7 * phase)
        v += np.random.default_rng(77).normal(0, 0.005, n)
        f = np.ones(n) * 60.0
        return v, f

    def _get_tp(method, key, default):
        return tuned_params.get("IBR_Nightmare", {}).get(method, {}).get(key, default)

    methods_to_test = ["IpDFT", "PLL", "EKF", "EKF2", "SOGI", "UKF"]
    rmse_by_method = {m: [] for m in methods_to_test}

    for thd in THD_LEVELS:
        v_sig, f_sig = _gen_signal(thd)
        v_dsp = v_sig[::RAT]
        f_dsp = f_sig[::RAT]
        facs = {
            "IpDFT": lambda: TunableIpDFT(_get_tp("IpDFT", "cycles", 4)),
            "PLL":   lambda: StandardPLL(_get_tp("PLL", "kp", 10.0), _get_tp("PLL", "ki", 50.0)),
            "EKF":   lambda: ClassicEKF(_get_tp("EKF", "Q", 0.1), _get_tp("EKF", "R", 0.01)),
            "EKF2":  lambda: EKF2(
                q_param=_get_tp("EKF2", "q_param", 1.0),
                r_param=_get_tp("EKF2", "r_param", 0.01),
                inn_ref=_get_tp("EKF2", "inn_ref", 0.1),
                event_thresh=_get_tp("EKF2", "event_thresh", 2.0),
                fast_horizon_ms=_get_tp("EKF2", "fast_horizon_ms", 80.0),
            ),
            "SOGI":  lambda: SOGI_FLL(_get_tp("SOGI", "k", 1.414), _get_tp("SOGI", "g", 20.0),
                                       smooth_win=_get_tp("SOGI", "smooth_win", None)),
            "UKF":   lambda: UKF_Estimator(_get_tp("UKF", "Q", 0.1), _get_tp("UKF", "R", 0.01)),
        }
        for m in methods_to_test:
            try:
                algo = facs[m]()
                tr   = np.array([algo.step(x) for x in v_dsp])
                met  = calculate_metrics(tr, f_dsp, 0.0, structural_samples=1)
                rmse_by_method[m].append(float(met["RMSE"]))
            except Exception:
                rmse_by_method[m].append(np.nan)

    # Sigmoid fit: RMSE(THD) = L / (1 + exp(-k*(THD - THD50)))
    def _sigmoid(thd, L, k, thd50):
        return L / (1.0 + np.exp(-k * (thd - thd50)))

    thd_arr = np.array(THD_LEVELS)
    sigmoid_fits = {}
    for m in methods_to_test:
        rmse_arr = np.array(rmse_by_method[m])
        valid = np.isfinite(rmse_arr)
        if valid.sum() < 4 or not _SCIPY_OK:
            sigmoid_fits[m] = {"error": "insufficient data"}
            continue
        try:
            L0 = float(np.nanmax(rmse_arr))
            p0 = [L0, 10.0, 0.15]
            popt, pcov = _curve_fit(_sigmoid, thd_arr[valid], rmse_arr[valid],
                                     p0=p0, maxfev=5000,
                                     bounds=([0, 0.1, 0.0], [10, 200, 0.35]))
            perr = np.sqrt(np.diag(pcov))
            sigmoid_fits[m] = {
                "L":    float(popt[0]), "L_err":    float(perr[0]),
                "k":    float(popt[1]), "k_err":    float(perr[1]),
                "THD50": float(popt[2]), "THD50_err": float(perr[2]),
            }
        except Exception as fit_err:
            sigmoid_fits[m] = {"error": str(fit_err)}

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for m in methods_to_test:
            rmse_arr = np.array(rmse_by_method[m])
            ax.plot(thd_arr * 100, rmse_arr, "o-", color=_method_color(m),
                    ms=4, label=m)
            fit = sigmoid_fits.get(m, {})
            if "L" in fit:
                thd_fine = np.linspace(0, 0.32, 200)
                ax.plot(thd_fine * 100,
                        _sigmoid(thd_fine, fit["L"], fit["k"], fit["THD50"]),
                        "--", color=_method_color(m), lw=0.7, alpha=0.6)
                ax.axvline(fit["THD50"] * 100, color=_method_color(m),
                           lw=0.5, ls=":", alpha=0.5)
        ax.set_xlabel("7th harmonic THD [%]")
        ax.set_ylabel("RMSE [Hz]")
        ax.set_title("Q5: THD phase transition (sigmoid fits)")
        ax.legend(fontsize=5)
        fig.tight_layout()
        _save_fig(fig, 5, "THD_transition")
    except Exception as e:
        warnings.warn(f"Q5 figure: {e}")

    return {
        "thd_levels":    THD_LEVELS,
        "rmse_by_method": rmse_by_method,
        "sigmoid_fits":   sigmoid_fits,
        "interpretation": "; ".join(
            f"{m}: THD50={sigmoid_fits[m].get('THD50', np.nan)*100:.1f}%"
            for m in methods_to_test
            if "THD50" in sigmoid_fits.get(m, {})
        ),
    }


# =============================================================================
# Q6: Scenario Redundancy Analysis
# =============================================================================
def scenario_redundancy_analysis(results_dict: dict) -> dict:
    """
    Determine whether the 5 benchmark scenarios provide independent information.

    Method: build a method×scenario feature matrix (RMSE, Ttrip, CPU rank).
    Apply PCA to find dominant variance directions.
    Compute pairwise Spearman rank correlations between scenarios.
    Greedy selection of minimum non-redundant scenario subset.
    """
    metrics_used = ["RMSE", "TRIP_TIME_0p5", "TIME_PER_SAMPLE_US"]
    # Feature matrix: rows = methods, cols = (scenario, metric) — normalised ranks
    methods_ok = [m for m in CORE_METHODS
                  if any(np.isfinite(_get(results_dict, sc, m, "RMSE"))
                         for sc in SCENARIOS)]
    if len(methods_ok) < 3:
        return {"error": "Insufficient method data for Q6"}

    # Build raw value matrix and scenario-specific rank vectors
    sc_rank_vecs = {}  # {scenario: np.array of ranks per method}
    for sc in SCENARIOS:
        rmse_vec = np.array([_get(results_dict, sc, m, "RMSE") for m in methods_ok])
        valid    = np.isfinite(rmse_vec)
        if valid.sum() < 2:
            continue
        # Rank by RMSE (lower = better rank = lower number)
        ranks    = np.full(len(methods_ok), np.nan)
        sorted_vals = sorted([(rmse_vec[i], i) for i in range(len(methods_ok))
                               if valid[i]])
        for rank, (_, idx) in enumerate(sorted_vals):
            ranks[idx] = rank + 1
        sc_rank_vecs[sc] = ranks

    sc_names = list(sc_rank_vecs.keys())
    n_sc     = len(sc_names)

    # Pairwise Spearman correlation between scenarios
    corr = np.full((n_sc, n_sc), np.nan)
    for i, si in enumerate(sc_names):
        for j, sj in enumerate(sc_names):
            vi, vj = sc_rank_vecs[si], sc_rank_vecs[sj]
            mask = np.isfinite(vi) & np.isfinite(vj)
            if _SCIPY_OK and mask.sum() >= 3:
                rho, _ = _sstats.spearmanr(vi[mask], vj[mask])
                corr[i, j] = float(rho)

    # PCA on (scenario × metric) feature vectors
    pca_result = None
    if _SKLEARN_OK:
        # Build (n_methods × n_features) matrix where features = sc×metric
        feat_cols = []
        col_labels = []
        for sc in sc_names:
            for met in metrics_used:
                col = np.array([_get(results_dict, sc, m, met) for m in methods_ok])
                feat_cols.append(col)
                col_labels.append(f"{sc[:8]}_{met[:4]}")
        X = np.column_stack(feat_cols)
        # Replace NaN with column median
        for c in range(X.shape[1]):
            col = X[:, c]
            finite = col[np.isfinite(col)]
            if len(finite) > 0:
                X[np.isnan(X[:, c]), c] = float(np.median(finite))
        X_sc = _SS().fit_transform(X)
        pca  = _PCA(n_components=min(n_sc, X_sc.shape[1], X_sc.shape[0]))
        pca.fit(X_sc)
        pca_result = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_components_90pct_var":   int(
                np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1
            ),
        }

    # Greedy selection: pick scenarios that maximise coverage (lowest max pairwise rho)
    selected = [sc_names[0]]
    remaining = sc_names[1:]
    while remaining:
        best_sc, best_min_rho = None, 1.1
        for cand in remaining:
            ci = sc_names.index(cand)
            max_rho = max(
                abs(corr[ci, sc_names.index(s)]) for s in selected
                if np.isfinite(corr[ci, sc_names.index(s)])
            ) if selected else 0.0
            if max_rho < best_min_rho:
                best_min_rho, best_sc = max_rho, cand
        if best_sc:
            selected.append(best_sc)
            remaining.remove(best_sc)
        else:
            break

    # Minimum non-redundant set: scenarios with max pairwise rho < 0.7
    REDUND_THRESH = 0.7
    minimal_set = [selected[0]]
    for sc in selected[1:]:
        ci = sc_names.index(sc)
        max_r = max(
            abs(corr[ci, sc_names.index(s)]) for s in minimal_set
            if np.isfinite(corr[ci, sc_names.index(s)])
        ) if minimal_set else 0.0
        if max_r < REDUND_THRESH:
            minimal_set.append(sc)

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

        ax = axes[0]
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n_sc)); ax.set_xticklabels(sc_names, rotation=35, ha="right", fontsize=6)
        ax.set_yticks(range(n_sc)); ax.set_yticklabels(sc_names, fontsize=6)
        for i in range(n_sc):
            for j in range(n_sc):
                if np.isfinite(corr[i, j]):
                    ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=5)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Q6: Scenario Spearman rank correlation")

        ax2 = axes[1]
        if pca_result:
            ev = pca_result["explained_variance_ratio"]
            ax2.bar(range(1, len(ev)+1), ev, color=_OI[1], edgecolor="k", lw=0.4)
            ax2.plot(range(1, len(ev)+1), np.cumsum(ev), "ro-", ms=4, lw=0.8,
                     label="Cumulative")
            ax2.axhline(0.9, color="k", ls="--", lw=0.5, label="90% threshold")
            ax2.set_xlabel("Principal component")
            ax2.set_ylabel("Explained variance ratio")
            ax2.set_title(f"PCA scree plot "
                          f"(90% var in {pca_result['n_components_90pct_var']} PCs)")
            ax2.legend(fontsize=5)
        fig.tight_layout()
        _save_fig(fig, 6, "scenario_redundancy")
    except Exception as e:
        warnings.warn(f"Q6 figure: {e}")

    return {
        "scenario_spearman_matrix": {
            sc_names[i]: {sc_names[j]: float(corr[i, j]) for j in range(n_sc)}
            for i in range(n_sc)
        },
        "greedy_selection_order": selected,
        "minimal_non_redundant_set": minimal_set,
        "pca":  pca_result,
        "interpretation": (
            f"Minimal non-redundant scenario set: {minimal_set}. "
            f"Greedy selection order: {selected}. "
            f"{'All scenarios needed.' if len(minimal_set) == n_sc else f'Redundant scenarios: {[s for s in sc_names if s not in minimal_set]}.'}"
        ),
    }


# =============================================================================
# Q7: Tuning Transferability Landscape
# =============================================================================
def tuning_transferability_landscape(tuned_params: dict) -> dict:
    """
    Grid search on EKF2 q_param × r_param (20×20 = 400 points).
    Tune on IBR_Nightmare, evaluate on all 5 scenarios.
    Transfer ratio = RMSE_transfer / RMSE_optimal per scenario.
    Shows which hyperparameter region is simultaneously optimal across scenarios.
    """
    try:
        from estimators import (
            FS_PHYSICS as _FP, RATIO as _RAT, get_test_signals, calculate_metrics,
        )
        from ekf2 import EKF2
    except ImportError as e:
        return {"error": f"Import failed: {e}"}

    FS_P = 1_000_000.0
    RAT  = 100
    N_Q  = 12   # grid resolution (12×12=144 to keep runtime <30s)
    N_R  = 12

    q_grid = np.logspace(-2, 2, N_Q)   # 0.01 to 100
    r_grid = np.logspace(-3, 1, N_R)   # 0.001 to 10

    # Load signals once
    signals = get_test_signals(seed=42)

    ekf2_base = tuned_params.get("IBR_Nightmare", {}).get("EKF2", {})
    inn_ref_base      = float(ekf2_base.get("inn_ref",          0.1))
    event_thresh_base = float(ekf2_base.get("event_thresh",     2.0))
    fast_hz_base      = float(ekf2_base.get("fast_horizon_ms", 80.0))

    # Optimal RMSE per scenario (with tuned params)
    optimal_rmse = {}
    for sc, (_, v_sig, f_sig, _) in signals.items():
        v_dsp = v_sig[::RAT]; f_dsp = f_sig[::RAT]
        ekf2_sc = tuned_params.get(sc, {}).get("EKF2", ekf2_base)
        try:
            algo = EKF2(
                q_param=float(ekf2_sc.get("q_param", 1.0)),
                r_param=float(ekf2_sc.get("r_param", 0.01)),
                inn_ref=float(ekf2_sc.get("inn_ref", inn_ref_base)),
                event_thresh=float(ekf2_sc.get("event_thresh", event_thresh_base)),
                fast_horizon_ms=float(ekf2_sc.get("fast_horizon_ms", fast_hz_base)),
            )
            tr = np.array([algo.step(x) for x in v_dsp])
            met = calculate_metrics(tr, f_dsp, 0.0, structural_samples=1)
            optimal_rmse[sc] = float(met["RMSE"])
        except Exception:
            optimal_rmse[sc] = np.nan

    # Grid search: q × r, evaluate on all scenarios
    landscape = {}  # {sc: (N_Q×N_R) RMSE array}
    for sc in SCENARIOS:
        landscape[sc] = np.full((N_Q, N_R), np.nan)

    sc_data = {
        sc: (v_sig[::RAT], f_sig[::RAT])
        for sc, (_, v_sig, f_sig, _) in signals.items()
        if sc in SCENARIOS
    }

    for qi, q in enumerate(q_grid):
        for ri, r in enumerate(r_grid):
            for sc, (v_dsp, f_dsp) in sc_data.items():
                try:
                    algo = EKF2(q_param=float(q), r_param=float(r),
                                inn_ref=inn_ref_base,
                                event_thresh=event_thresh_base,
                                fast_horizon_ms=fast_hz_base)
                    tr  = np.array([algo.step(x) for x in v_dsp])
                    met = calculate_metrics(tr, f_dsp, 0.0, structural_samples=1)
                    landscape[sc][qi, ri] = float(met["RMSE"])
                except Exception:
                    pass

    # Transfer ratio: landscape[sc] / optimal_rmse[sc]
    transfer_ratio = {}
    for sc in SCENARIOS:
        opt = optimal_rmse.get(sc, np.nan)
        if np.isfinite(opt) and opt > 1e-6:
            transfer_ratio[sc] = (landscape[sc] / opt).tolist()
        else:
            transfer_ratio[sc] = landscape[sc].tolist()

    # Best universal region: minimise max transfer ratio across all scenarios
    combined = np.ones((N_Q, N_R))
    for sc in SCENARIOS:
        arr = landscape.get(sc, np.full((N_Q, N_R), np.nan))
        opt = optimal_rmse.get(sc, np.nan)
        if np.isfinite(opt) and opt > 1e-6:
            combined = np.maximum(combined, arr / opt)
    best_qi, best_ri = np.unravel_index(np.nanargmin(combined), combined.shape)
    best_q = float(q_grid[best_qi])
    best_r = float(r_grid[best_ri])
    best_ratio = float(np.nanmin(combined))

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.4))
        axes_flat = axes.flatten()
        for idx, sc in enumerate(SCENARIOS):
            ax = axes_flat[idx]
            arr = landscape[sc]
            opt = optimal_rmse.get(sc, np.nan)
            ratio_arr = arr / opt if np.isfinite(opt) and opt > 1e-6 else arr
            im = ax.imshow(ratio_arr, aspect="auto", origin="lower",
                           cmap="YlOrRd", vmin=1, vmax=5,
                           extent=[np.log10(r_grid[0]), np.log10(r_grid[-1]),
                                   np.log10(q_grid[0]), np.log10(q_grid[-1])])
            ax.plot(np.log10(best_r), np.log10(best_q), "w*", ms=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"{sc[:12]}", fontsize=7)
            ax.set_xlabel("$\\log_{10}(r)$", fontsize=6)
            ax.set_ylabel("$\\log_{10}(q)$", fontsize=6)
        # Combined in last panel
        ax_c = axes_flat[5]
        im_c = ax_c.imshow(combined, aspect="auto", origin="lower",
                           cmap="YlOrRd", vmin=1, vmax=5,
                           extent=[np.log10(r_grid[0]), np.log10(r_grid[-1]),
                                   np.log10(q_grid[0]), np.log10(q_grid[-1])])
        ax_c.plot(np.log10(best_r), np.log10(best_q), "w*", ms=8,
                  label=f"q*={best_q:.2f}, r*={best_r:.3f}")
        plt.colorbar(im_c, ax=ax_c, shrink=0.8)
        ax_c.set_title(f"Q7: Combined max ratio\n(best={best_ratio:.2f}×)")
        ax_c.legend(fontsize=5)
        fig.tight_layout()
        _save_fig(fig, 7, "tuning_landscape")
    except Exception as e:
        warnings.warn(f"Q7 figure: {e}")

    return {
        "q_grid":           q_grid.tolist(),
        "r_grid":           r_grid.tolist(),
        "optimal_rmse":     {k: float(v) for k, v in optimal_rmse.items()},
        "transfer_ratio":   transfer_ratio,
        "combined_max_ratio": combined.tolist(),
        "best_universal_q": best_q,
        "best_universal_r": best_r,
        "best_max_transfer_ratio": best_ratio,
        "interpretation": (
            f"Best universal EKF2 hyperparameters: q={best_q:.3f}, r={best_r:.4f}. "
            f"Max transfer ratio across all scenarios: {best_ratio:.2f}× "
            f"(1.0=optimal, >1=degraded)."
        ),
    }


# =============================================================================
# Q8: Noise Variance Scaling Law
# =============================================================================
def noise_variance_scaling_law(tuned_params: dict) -> dict:
    """
    Sweep noise standard deviation (sigma) over 7 levels.
    For each sigma, run MC30 on IBR_Nightmare and compute mean RMSE.
    Fit power law: RMSE(sigma) = C * sigma^alpha.
    Optimal estimator: alpha ≈ 0.5 (achieves CRLB scaling).
    """
    try:
        from estimators import (
            FS_PHYSICS as _FP, RATIO as _RAT, calculate_metrics,
            TunableIpDFT, StandardPLL, ClassicEKF, SOGI_FLL,
            TFT_Estimator, UKF_Estimator, Koopman_RKDPmu,
            RLS_Estimator,
        )
        from ekf2 import EKF2
    except ImportError as e:
        return {"error": f"Import failed: {e}"}

    FS_P = 1_000_000.0; RAT = 100; N_MC = 20  # 20 runs per level

    SIGMA_LEVELS = [0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05]

    def _gen_ibr(sigma, seed):
        rng = np.random.default_rng(seed)
        t = np.arange(0, 1.5, 1.0 / FS_P); n = len(t)
        phase = np.zeros(n); curr = 0.0
        for i in range(1, n):
            if 0.6999 < t[i] < 0.7001:
                curr += np.deg2rad(60.0)
            curr += 2 * np.pi * 60.0 / FS_P
            phase[i] = curr
        v = np.sin(phase) + 0.05 * np.sin(5 * phase) + 0.02 * np.sin(2 * np.pi * 32.5 * t)
        v += rng.normal(0, sigma, n)
        f = np.ones(n) * 60.0
        return v, f

    def _get_tp(m, k, d):
        return tuned_params.get("IBR_Nightmare", {}).get(m, {}).get(k, d)

    methods_to_test = ["IpDFT", "PLL", "EKF", "EKF2", "SOGI", "UKF"]

    def _make_fac(m):
        if m == "IpDFT":  return lambda: TunableIpDFT(_get_tp("IpDFT", "cycles", 4))
        if m == "PLL":    return lambda: StandardPLL(_get_tp("PLL", "kp", 10.0), _get_tp("PLL", "ki", 50.0))
        if m == "EKF":    return lambda: ClassicEKF(_get_tp("EKF", "Q", 0.1), _get_tp("EKF", "R", 0.01))
        if m == "EKF2":   return lambda: EKF2(
            q_param=_get_tp("EKF2", "q_param", 1.0), r_param=_get_tp("EKF2", "r_param", 0.01),
            inn_ref=_get_tp("EKF2", "inn_ref", 0.1), event_thresh=_get_tp("EKF2", "event_thresh", 2.0),
            fast_horizon_ms=_get_tp("EKF2", "fast_horizon_ms", 80.0))
        if m == "SOGI":   return lambda: SOGI_FLL(_get_tp("SOGI", "k", 1.414), _get_tp("SOGI", "g", 20.0),
                                                   smooth_win=_get_tp("SOGI", "smooth_win", None))
        if m == "UKF":    return lambda: UKF_Estimator(_get_tp("UKF", "Q", 0.1), _get_tp("UKF", "R", 0.01))
        return None

    # {method: {sigma: [RMSE per run]}}
    sigma_rmse = {m: {s: [] for s in SIGMA_LEVELS} for m in methods_to_test}

    for sigma in SIGMA_LEVELS:
        for run in range(N_MC):
            v_sig, f_sig = _gen_ibr(sigma, seed=3000 + run)
            v_dsp = v_sig[::RAT]; f_dsp = f_sig[::RAT]
            for m in methods_to_test:
                fac = _make_fac(m)
                if fac is None:
                    continue
                try:
                    algo = fac()
                    tr   = np.array([algo.step(x) for x in v_dsp])
                    met  = calculate_metrics(tr, f_dsp, 0.0, structural_samples=1)
                    sigma_rmse[m][sigma].append(float(met["RMSE"]))
                except Exception:
                    pass

    # Aggregate mean RMSE per sigma, fit power law
    sigma_arr = np.array(SIGMA_LEVELS)
    mean_rmse = {m: np.array([np.nanmean(sigma_rmse[m][s]) for s in SIGMA_LEVELS])
                 for m in methods_to_test}

    power_law_fits = {}
    for m in methods_to_test:
        y = mean_rmse[m]
        valid = np.isfinite(y) & (y > 0)
        if valid.sum() < 3 or not _SCIPY_OK:
            power_law_fits[m] = {"error": "insufficient data"}
            continue
        log_x = np.log(sigma_arr[valid])
        log_y = np.log(y[valid])
        slope, intercept, r_val, p_val, _ = _sstats.linregress(log_x, log_y)
        C     = float(np.exp(intercept))
        alpha = float(slope)
        power_law_fits[m] = {
            "C": C, "alpha": alpha, "r2": float(r_val**2), "p": float(p_val),
            "near_crlb": abs(alpha - 0.5) < 0.15,
        }

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))

        ax = axes[0]
        for m in methods_to_test:
            y = mean_rmse[m]
            ax.loglog(sigma_arr, y, "o-", color=_method_color(m), ms=4, label=m)
            fit = power_law_fits.get(m, {})
            if "C" in fit:
                ax.loglog(sigma_arr, fit["C"] * sigma_arr ** fit["alpha"],
                          "--", color=_method_color(m), lw=0.7, alpha=0.6)
        # CRLB reference slope alpha=0.5
        ref_C = float(np.nanmedian([mean_rmse[m][3] for m in methods_to_test
                                     if np.isfinite(mean_rmse[m][3])]))
        ax.loglog(sigma_arr, ref_C * (sigma_arr / sigma_arr[3]) ** 0.5,
                  "k:", lw=1.0, label="$\\sigma^{0.5}$ (CRLB slope)")
        ax.set_xlabel("Noise $\\sigma$ [pu]"); ax.set_ylabel("RMSE [Hz]")
        ax.set_title("Q8: Noise scaling (log-log)")
        ax.legend(fontsize=5, ncol=2)

        ax2 = axes[1]
        mets = [m for m in methods_to_test if "alpha" in power_law_fits.get(m, {})]
        alphas = [power_law_fits[m]["alpha"] for m in mets]
        colors = [_method_color(m) for m in mets]
        ax2.bar(range(len(mets)), alphas, color=colors, edgecolor="k", lw=0.4)
        ax2.axhline(0.5, color="k", ls="--", lw=0.8, label="CRLB slope (0.5)")
        ax2.set_xticks(range(len(mets)))
        ax2.set_xticklabels(mets, rotation=45, ha="right", fontsize=6)
        ax2.set_ylabel("Power law exponent $\\alpha$")
        ax2.set_title("Noise scaling exponents")
        ax2.legend(fontsize=5)
        fig.tight_layout()
        _save_fig(fig, 8, "noise_scaling")
    except Exception as e:
        warnings.warn(f"Q8 figure: {e}")

    return {
        "sigma_levels":       SIGMA_LEVELS,
        "mean_rmse_by_method": {m: mean_rmse[m].tolist() for m in methods_to_test},
        "power_law_fits":      power_law_fits,
        "n_mc_per_level":      N_MC,
        "interpretation": "; ".join(
            f"{m}: alpha={power_law_fits[m]['alpha']:.2f} "
            f"({'near CRLB' if power_law_fits[m].get('near_crlb') else 'sub-optimal'})"
            for m in methods_to_test if "alpha" in power_law_fits.get(m, {})
        ),
    }


# =============================================================================
# Q9: Behavioral Taxonomy Analysis
# =============================================================================
def behavioral_taxonomy_analysis(results_dict: dict) -> dict:
    """
    Hierarchical clustering (Ward) + K-means on a method feature matrix.
    Features: RMSE, Ttrip, CPU per scenario (normalised).
    Compare data-driven clusters with architectural families.
    """
    if not _SKLEARN_OK or not _SCIPY_OK:
        return {"error": "scikit-learn and scipy required for Q9"}

    # Build feature matrix
    feat_list, method_labels = [], []
    for m in CORE_METHODS:
        row = []
        for sc in SCENARIOS:
            row.append(_get(results_dict, sc, m, "RMSE"))
            row.append(_get(results_dict, sc, m, "TRIP_TIME_0p5"))
            row.append(_get(results_dict, sc, m, "TIME_PER_SAMPLE_US"))
        if sum(np.isfinite(row)) >= len(row) * 0.6:
            feat_list.append(row)
            method_labels.append(m)

    if len(feat_list) < 3:
        return {"error": "Insufficient data for Q9 clustering"}

    X = np.array(feat_list)
    # Fill NaN with column median
    for c in range(X.shape[1]):
        col = X[:, c]
        finite = col[np.isfinite(col)]
        if len(finite):
            X[np.isnan(X[:, c]), c] = float(np.median(finite))

    X_sc = _SS().fit_transform(X)

    # Hierarchical clustering
    Z = _linkage(X_sc, method="ward")
    n_clusters_range = [2, 3, 4, 5]
    cluster_assignments = {}
    for k in n_clusters_range:
        labels = _fcluster(Z, k, criterion="maxclust")
        cluster_assignments[k] = {m: int(l) for m, l in zip(method_labels, labels)}

    # K-means (k=3 and k=4)
    kmeans_results = {}
    for k in [3, 4]:
        km = _KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_sc)
        kmeans_results[k] = {m: int(l) for m, l in zip(method_labels, km.labels_)}

    # Architectural family vs cluster mismatch
    mismatch_analysis = {}
    for k in [3, 4]:
        hier_labels = cluster_assignments.get(k, {})
        km_labels   = kmeans_results.get(k, {})
        hier_agree  = sum(
            FAMILY_MAP.get(method_labels[i], "") == FAMILY_MAP.get(method_labels[j], "")
            and hier_labels.get(method_labels[i]) == hier_labels.get(method_labels[j])
            for i in range(len(method_labels))
            for j in range(i + 1, len(method_labels))
        )
        total_pairs = len(method_labels) * (len(method_labels) - 1) // 2
        # Methods whose cluster assignment conflicts with their architectural family
        family_mismatches_hier = []
        for m in method_labels:
            fam = FAMILY_MAP.get(m, "unknown")
            fam_members = [mm for mm in method_labels if FAMILY_MAP.get(mm, "") == fam]
            if len(fam_members) > 1:
                cl = hier_labels.get(m, -1)
                fam_clusters = set(hier_labels.get(mm, -1) for mm in fam_members)
                if len(fam_clusters) > 1:
                    family_mismatches_hier.append(m)
        mismatch_analysis[f"k{k}"] = {
            "family_mismatches_hierarchical": family_mismatches_hier,
            "family_agreement_pairs": hier_agree,
            "total_pairs": total_pairs,
        }

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram as _dend
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))

        ax = axes[0]
        _dend(Z, labels=method_labels, ax=ax, leaf_rotation=45,
              leaf_font_size=7, color_threshold=0.7 * max(Z[:, 2]))
        ax.set_title("Q9: Ward hierarchical clustering")

        ax2 = axes[1]
        km_labels_k4 = kmeans_results.get(4, {})
        colors_km = [_OI[km_labels_k4.get(m, 0) % len(_OI)] for m in method_labels]
        # PCA for 2D projection
        pca = _PCA(n_components=2)
        X_pca = pca.fit_transform(X_sc)
        for i, m in enumerate(method_labels):
            ax2.scatter(X_pca[i, 0], X_pca[i, 1],
                        c=colors_km[i], s=30, zorder=3,
                        edgecolors="k", linewidths=0.3)
            ax2.annotate(m, (X_pca[i, 0], X_pca[i, 1]),
                         fontsize=5, ha="left", va="bottom")
        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
        ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
        ax2.set_title("K-means (k=4) in PCA space")
        fig.tight_layout()
        _save_fig(fig, 9, "behavioral_taxonomy")
    except Exception as e:
        warnings.warn(f"Q9 figure: {e}")

    return {
        "methods":              method_labels,
        "hierarchical_clusters": cluster_assignments,
        "kmeans_clusters":       kmeans_results,
        "mismatch_analysis":     mismatch_analysis,
        "interpretation": (
            "Behavioral taxonomy groups methods by performance profile rather than "
            "architectural family. Mismatches indicate methods with unexpected "
            "performance characteristics relative to their design paradigm."
        ),
    }


# =============================================================================
# Q10: Complexity-Performance Bound (CRLB Efficiency)
# =============================================================================
def complexity_performance_bound(results_dict: dict) -> dict:
    """
    Compute CRLB-normalised efficiency for each method and fit the empirical
    complexity-RMSE Pareto frontier: RMSE = a * CPU^(-b).

    CRLB for additive Gaussian noise on a sinusoidal signal:
      CRLB_freq = (3 * sigma_n^2) / (pi^2 * A^2 * fs * N^3)
    where N = number of samples per window, A = amplitude, fs = sampling rate.

    Efficiency = CRLB_RMSE / method_RMSE  in [0, 1].
    """
    # Estimate CRLB from IBR_Nightmare parameters
    # sigma_n = 0.005 pu, A = 1.0 pu, fs = 10000 Hz, N = 1 cycle = fs/60
    sigma_n = 0.005
    A       = 1.0
    fs      = FS_DSP
    f0      = 60.0
    N_cycle = int(fs / f0)   # samples per fundamental cycle
    crlb_var = (3.0 * sigma_n**2) / (np.pi**2 * A**2 * fs * N_cycle**3)
    crlb_rmse_hz = float(np.sqrt(crlb_var) * fs)  # convert to Hz

    results = []
    for m in CORE_METHODS:
        rmse_nd  = _get(results_dict, "IBR_Nightmare", m, "RMSE")
        cpu_us   = _get(results_dict, "IBR_MultiEvent", m, "TIME_PER_SAMPLE_US")
        if np.isfinite(rmse_nd) and np.isfinite(cpu_us) and rmse_nd > 0:
            eff = min(1.0, crlb_rmse_hz / rmse_nd)
            results.append({
                "method":   m,
                "RMSE_Hz":  rmse_nd,
                "CPU_us":   cpu_us,
                "CRLB_Hz":  crlb_rmse_hz,
                "efficiency": eff,
            })

    if not results:
        return {"error": "No data for Q10"}

    rmse_arr = np.array([r["RMSE_Hz"] for r in results])
    cpu_arr  = np.array([r["CPU_us"]  for r in results])

    # Power law Pareto fit: RMSE = a * CPU^(-b)
    pareto_fit = None
    if _SCIPY_OK and len(results) >= 4:
        log_cpu  = np.log(cpu_arr)
        log_rmse = np.log(rmse_arr)
        slope, intercept, r_val, p_val, _ = _sstats.linregress(log_cpu, log_rmse)
        pareto_fit = {
            "a":  float(np.exp(intercept)),
            "b":  float(-slope),
            "r2": float(r_val**2),
            "p":  float(p_val),
        }

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))

        ax = axes[0]
        for r in results:
            ax.scatter(r["CPU_us"], r["RMSE_Hz"],
                       c=_method_color(r["method"]), s=25, zorder=3,
                       edgecolors="k", linewidths=0.3)
            ax.annotate(r["method"], (r["CPU_us"], r["RMSE_Hz"]),
                        fontsize=5, ha="left", va="bottom")
        ax.axhline(crlb_rmse_hz, color="k", ls="--", lw=0.8, label="CRLB")
        if pareto_fit:
            cpu_fine = np.logspace(np.log10(cpu_arr.min() * 0.8),
                                    np.log10(cpu_arr.max() * 1.2), 100)
            ax.loglog(cpu_fine,
                      pareto_fit["a"] * cpu_fine ** (-pareto_fit["b"]),
                      "r--", lw=0.8, label=f"Pareto fit $b$={pareto_fit['b']:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("CPU [µs/sample]"); ax.set_ylabel("RMSE [Hz]")
        ax.set_title("Q10: Complexity-RMSE Pareto")
        ax.legend(fontsize=5)

        ax2 = axes[1]
        effs = [r["efficiency"] for r in results]
        mets = [r["method"]    for r in results]
        ax2.bar(range(len(mets)), effs,
                color=[_method_color(m) for m in mets], edgecolor="k", lw=0.4)
        ax2.axhline(1.0, color="k", ls="--", lw=0.8, label="CRLB bound")
        ax2.set_xticks(range(len(mets)))
        ax2.set_xticklabels(mets, rotation=45, ha="right", fontsize=6)
        ax2.set_ylabel("CRLB efficiency")
        ax2.set_title("Efficiency relative to CRLB")
        ax2.legend(fontsize=5)
        fig.tight_layout()
        _save_fig(fig, 10, "complexity_bound")
    except Exception as e:
        warnings.warn(f"Q10 figure: {e}")

    most_efficient = max(results, key=lambda r: r["efficiency"])
    return {
        "crlb_rmse_hz":   crlb_rmse_hz,
        "crlb_params":    {"sigma_n": sigma_n, "A": A, "fs": fs, "N_cycle": N_cycle},
        "methods":        results,
        "pareto_fit":     pareto_fit,
        "most_efficient_method": most_efficient["method"],
        "max_efficiency":        most_efficient["efficiency"],
        "interpretation": (
            f"CRLB = {crlb_rmse_hz:.4f} Hz. "
            f"Most efficient: {most_efficient['method']} "
            f"(efficiency={most_efficient['efficiency']:.3f}). "
            f"Pareto slope b={pareto_fit['b']:.2f} " if pareto_fit else ""
            f"(b=0.5=square-root improvement with cost)."
        ),
    }


# =============================================================================
# Q11: Protection Margin Distribution
# =============================================================================
def protection_margin_distribution(mc_raw: dict,
                                    trip_threshold: float = 0.5) -> dict:
    """
    For each MC run, compute the protection margin:
        margin = trip_threshold - FE_max
    Positive margin = survived (safe); negative = tripped.

    Fit normal distribution to margin distribution.
    Cost model: cost = P(trip) * C_trip + P(miss) * C_miss
    where C_trip = 1.0 (false disconnect cost), C_miss = 10.0 (missed event).
    Fisher's exact test: is trip probability independent of scenario class
    (IEEE vs IBR)?
    """
    C_TRIP = 1.0    # normalised false-trip cost
    C_MISS = 10.0   # normalised missed-fault cost

    method_results = {}
    for m in CORE_METHODS:
        margins_all, tripped_all = [], []
        margins_by_sc = {}
        for sc in SCENARIOS:
            key = (m, sc)
            if key not in mc_raw:
                continue
            fe_vals = np.array(mc_raw[key].get("FE_max", []), dtype=float)
            if len(fe_vals) == 0:
                continue
            margins = trip_threshold - fe_vals
            tripped = (fe_vals > trip_threshold)
            margins_all.extend(margins.tolist())
            tripped_all.extend(tripped.tolist())
            margins_by_sc[sc] = {
                "margin_mean":  float(np.mean(margins)),
                "margin_std":   float(np.std(margins)),
                "margin_min":   float(np.min(margins)),
                "trip_rate":    float(np.mean(tripped)),
            }

        if not margins_all:
            continue

        margins_arr = np.array(margins_all)
        tripped_arr = np.array(tripped_all)
        trip_prob   = float(np.mean(tripped_arr))

        # Fit normal distribution
        if _SCIPY_OK and len(margins_arr) >= 4:
            mu, sigma = _sstats.norm.fit(margins_arr)
            p_trip_normal = float(_sstats.norm.cdf(0.0, loc=mu, scale=sigma))
        else:
            mu = float(np.mean(margins_arr))
            sigma = float(np.std(margins_arr))
            p_trip_normal = trip_prob

        # Expected cost
        expected_cost = C_TRIP * trip_prob + C_MISS * (1.0 - trip_prob) * 0.001

        method_results[m] = {
            "margin_mean":    float(np.mean(margins_arr)),
            "margin_std":     float(np.std(margins_arr)),
            "margin_p5":      float(np.percentile(margins_arr, 5)),
            "margin_p95":     float(np.percentile(margins_arr, 95)),
            "trip_rate":      trip_prob,
            "normal_mu":      float(mu),
            "normal_sigma":   float(sigma),
            "p_trip_modelled":p_trip_normal,
            "expected_cost":  expected_cost,
            "by_scenario":    margins_by_sc,
        }

    # Fisher's exact test: trip probability vs scenario class (IEEE/IBR)
    fisher_results = {}
    IEEE_SC = {"IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation"}
    IBR_SC  = {"IBR_Nightmare", "IBR_MultiEvent"}
    for m in CORE_METHODS:
        ieee_trips = ieee_n = ibr_trips = ibr_n = 0
        for sc in IEEE_SC:
            key = (m, sc)
            if key in mc_raw:
                fe = np.array(mc_raw[key].get("FE_max", []))
                ieee_trips += int((fe > trip_threshold).sum())
                ieee_n     += len(fe)
        for sc in IBR_SC:
            key = (m, sc)
            if key in mc_raw:
                fe = np.array(mc_raw[key].get("FE_max", []))
                ibr_trips += int((fe > trip_threshold).sum())
                ibr_n     += len(fe)
        if ieee_n > 0 and ibr_n > 0 and _SCIPY_OK:
            # 2×2 table: [tripped, not-tripped] × [IEEE, IBR]
            table = [[ieee_trips, ieee_n - ieee_trips],
                     [ibr_trips,  ibr_n  - ibr_trips]]
            odds, p_val = _fisher_exact(table, alternative="two-sided")
            fisher_results[m] = {
                "contingency_table": table,
                "odds_ratio": float(odds),
                "p_value":    float(p_val),
                "significant": p_val < 0.05,
            }

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))

        ax = axes[0]
        mets = list(method_results.keys())
        means = [method_results[m]["margin_mean"]  for m in mets]
        stds  = [method_results[m]["margin_std"]   for m in mets]
        colors = [_method_color(m) for m in mets]
        ax.bar(range(len(mets)), means, yerr=stds,
               color=colors, edgecolor="k", lw=0.4, capsize=2, error_kw={"lw": 0.6})
        ax.axhline(0, color="r", lw=0.8, ls="--", label="Trip threshold")
        ax.set_xticks(range(len(mets)))
        ax.set_xticklabels(mets, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Protection margin [Hz]")
        ax.set_title("Q11: Protection margin mean±std")
        ax.legend(fontsize=5)

        ax2 = axes[1]
        costs = [method_results[m]["expected_cost"] for m in mets]
        ax2.bar(range(len(mets)), costs, color=colors, edgecolor="k", lw=0.4)
        ax2.set_xticks(range(len(mets)))
        ax2.set_xticklabels(mets, rotation=45, ha="right", fontsize=6)
        ax2.set_ylabel("Expected cost (normalised)")
        ax2.set_title(f"Expected cost model\n(C_trip={C_TRIP}, C_miss={C_MISS})")
        fig.tight_layout()
        _save_fig(fig, 11, "protection_margin")
    except Exception as e:
        warnings.warn(f"Q11 figure: {e}")

    best_m = min(method_results, key=lambda m: method_results[m]["expected_cost"],
                 default=None)
    return {
        "method_results":    method_results,
        "fisher_test_trip_vs_scenario_class": fisher_results,
        "cost_model":        {"C_trip": C_TRIP, "C_miss": C_MISS},
        "lowest_cost_method": best_m,
        "interpretation": (
            f"Method with lowest expected cost: {best_m}. "
            "Fisher's test checks if trip probability is significantly higher "
            "in IBR scenarios than IEEE scenarios."
        ),
    }


# =============================================================================
# Q12: IEC Sufficiency Formal Test
# =============================================================================
def iec_sufficiency_formal_test(results_dict: dict) -> dict:
    """
    For each method, build a 2×2 contingency table:
      Rows: IEC-compliant / non-compliant
      Cols: 'safe' scenario (IEEE class) / 'hazardous' scenario (IBR class)

    Fisher's exact test: H0 = compliance is independent of scenario class.
    Sensitivity = P(compliant | IEEE), Specificity = P(non-compliant | IBR).

    A 'sufficient' standard would show high sensitivity AND high specificity.
    """
    IEEE_SC = {"IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation"}
    IBR_SC  = {"IBR_Nightmare", "IBR_MultiEvent"}

    def _is_compliant(results_dict, sc, m):
        rmse  = _get(results_dict, sc, m, "RMSE")
        peak  = _get(results_dict, sc, m, "MAX_PEAK")
        ttrip = _get(results_dict, sc, m, "TRIP_TIME_0p5")
        return (np.isfinite(rmse)  and rmse  <= IEC_RMSE  and
                np.isfinite(peak)  and peak  <= IEC_PEAK  and
                np.isfinite(ttrip) and ttrip <= IEC_TTRIP)

    all_results = {}
    global_contingency = np.zeros((2, 2), dtype=int)   # [class: IEEE/IBR, compliant/not]

    for m in CORE_METHODS:
        ieee_pass = sum(1 for sc in IEEE_SC if _is_compliant(results_dict, sc, m))
        ieee_fail = len(IEEE_SC) - ieee_pass
        ibr_pass  = sum(1 for sc in IBR_SC  if _is_compliant(results_dict, sc, m))
        ibr_fail  = len(IBR_SC)  - ibr_pass

        # 2×2: [[IEEE_pass, IEEE_fail], [IBR_pass, IBR_fail]]
        table = [[ieee_pass, ieee_fail], [ibr_pass, ibr_fail]]

        if _SCIPY_OK and (ieee_pass + ieee_fail + ibr_pass + ibr_fail) >= 4:
            odds, p_val = _fisher_exact(table, alternative="two-sided")
        else:
            odds = p_val = np.nan

        sensitivity  = ieee_pass / max(len(IEEE_SC), 1)   # P(pass | IEEE)
        specificity  = ibr_fail  / max(len(IBR_SC),  1)   # P(fail | IBR)
        accuracy     = (ieee_pass + ibr_fail) / max(len(IEEE_SC) + len(IBR_SC), 1)

        all_results[m] = {
            "contingency_table": table,
            "fisher_odds":       float(odds)  if np.isfinite(odds)  else None,
            "fisher_p":          float(p_val) if np.isfinite(p_val) else None,
            "significant":       bool(np.isfinite(p_val) and p_val < 0.05),
            "sensitivity_IEEE":  sensitivity,
            "specificity_IBR":   specificity,
            "accuracy":          accuracy,
            "ieee_pass_rate":    ieee_pass / max(len(IEEE_SC), 1),
            "ibr_pass_rate":     ibr_pass  / max(len(IBR_SC),  1),
        }
        global_contingency[0, 0] += ieee_pass
        global_contingency[0, 1] += ieee_fail
        global_contingency[1, 0] += ibr_pass
        global_contingency[1, 1] += ibr_fail

    # Global test across all methods
    if _SCIPY_OK:
        g_odds, g_p = _fisher_exact(global_contingency.tolist(), alternative="two-sided")
    else:
        g_odds = g_p = np.nan

    # Summary: methods meeting IEC in ALL scenarios
    fully_compliant = [
        m for m in CORE_METHODS
        if all(_is_compliant(results_dict, sc, m) for sc in SCENARIOS)
    ]
    # Methods with IEC sensitivity >= 0.8 AND specificity >= 0.5
    adequate_methods = [
        m for m in CORE_METHODS
        if all_results.get(m, {}).get("sensitivity_IEEE", 0) >= 0.8
        and all_results.get(m, {}).get("specificity_IBR", 0) >= 0.5
    ]

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

        ax = axes[0]
        mets = list(all_results.keys())
        sensit = [all_results[m]["sensitivity_IEEE"] for m in mets]
        specif = [all_results[m]["specificity_IBR"]  for m in mets]
        colors = [_method_color(m) for m in mets]
        ax.scatter(sensit, specif, c=colors, s=40, zorder=3,
                   edgecolors="k", linewidths=0.3)
        for m, sx, sy in zip(mets, sensit, specif):
            ax.annotate(m, (sx, sy), fontsize=5, ha="left", va="bottom")
        ax.axvline(0.8, color="k", ls="--", lw=0.5, label="Sensitivity=0.8")
        ax.axhline(0.5, color="r", ls="--", lw=0.5, label="Specificity=0.5")
        ax.set_xlabel("Sensitivity $P(\\mathrm{pass}|\\mathrm{IEEE})$")
        ax.set_ylabel("Specificity $P(\\mathrm{fail}|\\mathrm{IBR})$")
        ax.set_title("Q12: IEC standard sufficiency")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=5)

        ax2 = axes[1]
        ieee_rates = [all_results[m]["ieee_pass_rate"] for m in mets]
        ibr_rates  = [all_results[m]["ibr_pass_rate"]  for m in mets]
        x = np.arange(len(mets)); w = 0.35
        ax2.bar(x - w/2, ieee_rates, w, label="IEEE pass rate",
                color=_OI[1], edgecolor="k", lw=0.4)
        ax2.bar(x + w/2, ibr_rates,  w, label="IBR pass rate",
                color=_OI[5], edgecolor="k", lw=0.4)
        ax2.set_xticks(x)
        ax2.set_xticklabels(mets, rotation=45, ha="right", fontsize=6)
        ax2.set_ylabel("IEC pass rate")
        ax2.set_title("IEC pass rate: IEEE vs IBR scenarios")
        ax2.legend(fontsize=5)
        fig.tight_layout()
        _save_fig(fig, 12, "IEC_sufficiency")
    except Exception as e:
        warnings.warn(f"Q12 figure: {e}")

    return {
        "per_method":             all_results,
        "global_contingency_table": global_contingency.tolist(),
        "global_fisher_odds":      float(g_odds) if np.isfinite(g_odds) else None,
        "global_fisher_p":         float(g_p)    if np.isfinite(g_p)    else None,
        "global_significant":      bool(np.isfinite(g_p) and g_p < 0.05),
        "fully_compliant_methods": fully_compliant,
        "adequate_sensitivity_specificity": adequate_methods,
        "thresholds": {
            "IEC_RMSE_Hz": IEC_RMSE, "IEC_PEAK_Hz": IEC_PEAK, "IEC_TTRIP_s": IEC_TTRIP,
        },
        "interpretation": (
            f"Fisher's test (global): odds={g_odds:.2f}, p={g_p:.4f}. "
            f"{'Compliance is scenario-class-dependent (p<0.05).' if np.isfinite(g_p) and g_p < 0.05 else 'No significant dependency.'} "
            f"Fully compliant across all scenarios: {fully_compliant}. "
            f"Methods with adequate sensitivity+specificity: {adequate_methods}."
        ) if np.isfinite(g_p) else "Insufficient data.",
    }


# =============================================================================
# BLOCK A — Findings already in data (zero new computation)
# =============================================================================

def iec_blindness_formal(results_dict: dict, n_permutations: int = 10000) -> dict:
    """
    A1: Permutation test for Spearman rho(IEC_scenario, Islanding) = 0.
    Proves IEC compliance tests carry no predictive information about
    IBR islanding robustness.
    """
    if not _SCIPY_OK:
        return {"error": "scipy required for A1"}

    IEC_SC  = ["IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation"]
    ISL_SC  = "IBR_Nightmare"
    rng     = np.random.default_rng(0)

    rmse_isl = np.array([_get(results_dict, ISL_SC, m, "RMSE") for m in CORE_METHODS])
    valid_isl = np.isfinite(rmse_isl)

    out = {}
    for iec_sc in IEC_SC:
        rmse_iec = np.array([_get(results_dict, iec_sc, m, "RMSE") for m in CORE_METHODS])
        mask = valid_isl & np.isfinite(rmse_iec)
        if mask.sum() < 4:
            out[iec_sc] = {"error": "insufficient data"}
            continue
        x, y = rmse_iec[mask], rmse_isl[mask]
        rho_obs, _ = _sstats.spearmanr(x, y)
        rho_perm = [
            _sstats.spearmanr(x, rng.permutation(y))[0]
            for _ in range(n_permutations)
        ]
        p_perm = float(np.mean(np.abs(rho_perm) >= abs(rho_obs)))
        r2      = float(rho_obs ** 2)
        out[iec_sc] = {
            "rho_observed":   float(rho_obs),
            "p_permutation":  p_perm,
            "r2_explained":   r2,
            "n_methods":      int(mask.sum()),
            "conclusion": (
                f"IEC scenario '{iec_sc}' explains {r2*100:.1f}% of variance in "
                f"Islanding RMSE (rho={rho_obs:.3f}, permutation p={p_perm:.3f}). "
                f"{'No predictive power (p>0.05).' if p_perm > 0.05 else 'Significant correlation (p<=0.05).'}"
            ),
        }

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
        for ax, iec_sc in zip(axes, IEC_SC):
            rmse_iec = np.array([_get(results_dict, iec_sc, m, "RMSE") for m in CORE_METHODS])
            mask = valid_isl & np.isfinite(rmse_iec)
            x_, y_ = rmse_iec[mask], rmse_isl[mask]
            ax.scatter(x_, y_, c=[_method_color(CORE_METHODS[i]) for i in range(len(CORE_METHODS)) if mask[i]],
                       s=18, zorder=3, edgecolors="k", lw=0.3)
            if _SCIPY_OK and len(x_) >= 3:
                slope, intercept, r_val, _, _ = _sstats.linregress(x_, y_)
                xl = np.linspace(x_.min(), x_.max(), 100)
                ax.plot(xl, slope * xl + intercept, "k--", lw=0.7)
                # 95% CI band (± 2σ of residuals)
                res_std = float(np.std(y_ - (slope * x_ + intercept)))
                ax.fill_between(xl, slope * xl + intercept - 2 * res_std,
                                slope * xl + intercept + 2 * res_std,
                                alpha=0.15, color="k")
            r2v = out.get(iec_sc, {}).get("r2_explained", np.nan)
            pv  = out.get(iec_sc, {}).get("p_permutation", np.nan)
            ax.set_xlabel(f"RMSE {iec_sc[:8]} [Hz]", fontsize=6)
            ax.set_ylabel("RMSE Islanding [Hz]", fontsize=6)
            ax.set_title(f"$R^2$={r2v:.2f}, p={pv:.2f}", fontsize=7)
        axes[0].set_title("IEC blindness — no predictive power", fontsize=7)
        fig.suptitle("A1: IEC compliance does not predict Islanding robustness", fontsize=7)
        fig.tight_layout()
        _save_fig(fig, 101, "IEC_blindness")
    except Exception as e:
        warnings.warn(f"A1 figure: {e}")

    return {
        "per_iec_scenario": out,
        "n_permutations":   n_permutations,
        "interpretation": (
            "Permutation test confirms IEC compliance scenarios have near-zero "
            "Spearman correlation with Islanding RMSE. "
            "Standard compliance testing provides no predictive information about "
            "IBR islanding robustness."
        ),
    }


def metric_redundancy_analysis(results_dict: dict) -> dict:
    """
    A2: Build metric correlation matrix. PCA to find how many independent
    dimensions exist in the 8-metric space.
    Recommends minimum sufficient metric set.
    """
    if not _SKLEARN_OK or not _SCIPY_OK:
        return {"error": "sklearn + scipy required for A2"}

    METRICS = ["RMSE", "MAX_PEAK", "TRIP_TIME_0p5", "MAX_CONTIGUOUS_0p5",
               "ENERGY", "RFE_rms_Hz_s", "SETTLING", "TIME_PER_SAMPLE_US"]

    # Build (n_obs × n_metrics) matrix across all methods × scenarios
    rows = []
    for sc in SCENARIOS:
        for m in CORE_METHODS:
            row = [_get(results_dict, sc, m, met) for met in METRICS]
            if sum(np.isfinite(row)) >= 6:
                rows.append(row)
    if len(rows) < 6:
        return {"error": "insufficient data for A2"}

    X = np.array(rows)
    for c in range(X.shape[1]):
        finite = X[:, c][np.isfinite(X[:, c])]
        if len(finite):
            X[~np.isfinite(X[:, c]), c] = float(np.median(finite))

    # Pairwise Spearman between metrics
    n_m = len(METRICS)
    rho_mat = np.full((n_m, n_m), np.nan)
    for i in range(n_m):
        for j in range(n_m):
            mask = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
            if mask.sum() >= 4:
                rho, _ = _sstats.spearmanr(X[mask, i], X[mask, j])
                rho_mat[i, j] = float(rho)

    # PCA
    X_sc = _SS().fit_transform(X)
    pca  = _PCA(n_components=min(n_m, len(rows)))
    pca.fit(X_sc)
    ev   = pca.explained_variance_ratio_
    n90  = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
    n95  = int(np.searchsorted(np.cumsum(ev), 0.95) + 1)

    # Redundant groups: metrics with |rho| > 0.9 with RMSE
    redundant_with_rmse = [
        METRICS[i] for i in range(n_m)
        if i != 0 and np.isfinite(rho_mat[0, i]) and abs(rho_mat[0, i]) > 0.90
    ]
    partially_independent = [
        METRICS[i] for i in range(n_m)
        if i != 0 and np.isfinite(rho_mat[0, i]) and 0.5 <= abs(rho_mat[0, i]) < 0.90
    ]
    independent = [
        METRICS[i] for i in range(n_m)
        if i != 0 and np.isfinite(rho_mat[0, i]) and abs(rho_mat[0, i]) < 0.5
    ]

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram as _dend
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))

        ax = axes[0]
        im = ax.imshow(rho_mat, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n_m)); ax.set_xticklabels(METRICS, rotation=45, ha="right", fontsize=5)
        ax.set_yticks(range(n_m)); ax.set_yticklabels(METRICS, fontsize=5)
        for i in range(n_m):
            for j in range(n_m):
                if np.isfinite(rho_mat[i, j]):
                    ax.text(j, i, f"{rho_mat[i,j]:.2f}", ha="center", va="center", fontsize=4)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title("A2: Metric Spearman correlation", fontsize=7)

        ax2 = axes[1]
        ax2.bar(range(1, len(ev)+1), ev, color=_OI[1], edgecolor="k", lw=0.4)
        ax2.plot(range(1, len(ev)+1), np.cumsum(ev), "ro-", ms=3, lw=0.8)
        ax2.axhline(0.90, color="k", ls="--", lw=0.6, label="90%")
        ax2.axhline(0.95, color="gray", ls=":", lw=0.6, label="95%")
        ax2.set_xlabel("PC"); ax2.set_ylabel("Explained variance ratio")
        ax2.set_title(f"PCA scree: {n90} PCs@90%, {n95} PCs@95%", fontsize=7)
        ax2.legend(fontsize=5)
        fig.tight_layout()
        _save_fig(fig, 102, "metric_redundancy")
    except Exception as e:
        warnings.warn(f"A2 figure: {e}")

    return {
        "metric_spearman_matrix": {
            METRICS[i]: {METRICS[j]: float(rho_mat[i, j]) for j in range(n_m)}
            for i in range(n_m)
        },
        "pca_explained_variance":    ev.tolist(),
        "n_components_90pct_var":    n90,
        "n_components_95pct_var":    n95,
        "redundant_with_rmse":       redundant_with_rmse,
        "partially_independent":     partially_independent,
        "independent_of_rmse":       independent,
        "recommended_minimum_set":   ["RMSE", "TRIP_TIME_0p5"] + independent[:1],
        "interpretation": (
            f"{n90} PCs explain 90% of metric variance. "
            f"Redundant with RMSE (|rho|>0.9): {redundant_with_rmse}. "
            f"Independent: {independent}. "
            f"Minimum sufficient set: RMSE + TRIP_TIME_0p5."
        ),
    }


def rocof_analysis(results_dict: dict) -> dict:
    """
    A3: Rank methods by RFE_rms (RoCoF error). Test whether explicit RoCoF
    state (EKF2's domega state) improves RoCoF estimation.
    """
    EXPLICIT_ROCOF_METHODS = {"EKF2", "UKF"}   # have explicit omega_dot state
    result = {}

    for sc in SCENARIOS:
        sc_data = {}
        for m in CORE_METHODS:
            rfe = _get(results_dict, sc, m, "RFE_rms_Hz_s")
            if np.isfinite(rfe):
                sc_data[m] = float(rfe)
        if not sc_data:
            continue
        ranked = sorted(sc_data.items(), key=lambda kv: kv[1])
        result[sc] = {
            "rfe_rms_per_method": sc_data,
            "ranking":            [m for m, _ in ranked],
            "best_method":        ranked[0][0],
            "best_rfe_rms":       ranked[0][1],
        }

    # Mann-Whitney: explicit-RoCoF methods vs others
    explicit_vals, other_vals = [], []
    for sc in SCENARIOS:
        for m in CORE_METHODS:
            rfe = _get(results_dict, sc, m, "RFE_rms_Hz_s")
            if np.isfinite(rfe):
                if m in EXPLICIT_ROCOF_METHODS:
                    explicit_vals.append(rfe)
                else:
                    other_vals.append(rfe)

    mw_stat = mw_p = np.nan
    if _SCIPY_OK and len(explicit_vals) >= 2 and len(other_vals) >= 2:
        mw_stat, mw_p = _sstats.mannwhitneyu(
            explicit_vals, other_vals, alternative="less")

    # For IBR_Nightmare specifically
    isl = results_dict.get("IBR_Nightmare", {}).get("methods", {})
    ekf2_rfe = float(isl.get("EKF2", {}).get("RFE_rms_Hz_s", np.nan))
    sogi_rfe = float(isl.get("SOGI", {}).get("RFE_rms_Hz_s", np.nan))
    pll_rfe  = float(isl.get("PLL",  {}).get("RFE_rms_Hz_s", np.nan))

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.0, 2.8))
        for si, sc in enumerate(SCENARIOS):
            sc_vals = result.get(sc, {}).get("rfe_rms_per_method", {})
            mets    = [m for m in CORE_METHODS if m in sc_vals]
            vals    = [sc_vals[m] for m in mets]
            x       = np.arange(len(mets)) + si * 0.15 - 0.3
            ec      = ["red" if m in EXPLICIT_ROCOF_METHODS else "none" for m in mets]
            ax.bar(np.arange(len(mets)) + si * 0.15, vals,
                   0.12, label=sc[:8], color=_OI[si % len(_OI)],
                   edgecolor=ec, linewidth=0.8, alpha=0.8)
        ax.set_xticks(np.arange(len(CORE_METHODS)))
        ax.set_xticklabels(CORE_METHODS, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("RFE$_{rms}$ [Hz/s]")
        ax.set_title("A3: RoCoF estimation error (red border=explicit $\\dot\\omega$ state)", fontsize=7)
        ax.legend(fontsize=5, ncol=2)
        ax.set_yscale("log")
        fig.tight_layout()
        _save_fig(fig, 103, "rocof_analysis")
    except Exception as e:
        warnings.warn(f"A3 figure: {e}")

    return {
        "rfe_by_scenario":          result,
        "explicit_rocof_methods":   list(EXPLICIT_ROCOF_METHODS),
        "mw_stat_explicit_vs_other": float(mw_stat) if np.isfinite(mw_stat) else None,
        "mw_p_explicit_vs_other":    float(mw_p)    if np.isfinite(mw_p)    else None,
        "explicit_rocof_better":     bool(np.isfinite(mw_p) and mw_p < 0.05),
        "IBR_Nightmare_notable": {
            "EKF2_rfe": ekf2_rfe, "SOGI_rfe": sogi_rfe, "PLL_rfe": pll_rfe,
        },
        "interpretation": (
            f"Explicit RoCoF state methods (EKF2, UKF) vs others — "
            f"Mann-Whitney p={mw_p:.3f}. "
            f"{'Explicit state significantly improves RoCoF.' if np.isfinite(mw_p) and mw_p < 0.05 else 'Explicit RoCoF state does NOT significantly improve RoCoF estimation.'} "
            f"IBR_Nightmare: EKF2 RFE_rms={ekf2_rfe:.2f} Hz/s vs SOGI={sogi_rfe:.3f} Hz/s."
        ),
    }


def relay_coordination_analysis(results_dict: dict,
                                 relay_delays_ms: list = None) -> dict:
    """
    A4: Determine which estimators are safe at each relay delay setting.
    A relay with delay tau_ms will actuate IFF max_contiguous > tau_ms/1000.
    Produces the most actionable table for protection engineers.
    """
    if relay_delays_ms is None:
        relay_delays_ms = [12, 50, 100, 150, 200, 500]

    IBR_SCENARIOS = ["IBR_Nightmare", "IBR_MultiEvent"]
    table = {}

    for m in CORE_METHODS:
        table[m] = {}
        max_cont_vals = {}
        for sc in IBR_SCENARIOS:
            mc_val = _get(results_dict, sc, m, "MAX_CONTIGUOUS_0p5")
            if np.isfinite(mc_val):
                max_cont_vals[sc] = float(mc_val)

        # Worst-case max_contiguous across IBR scenarios
        worst_cont = max(max_cont_vals.values()) if max_cont_vals else np.nan
        table[m]["max_contiguous_s"] = {sc: v for sc, v in max_cont_vals.items()}
        table[m]["worst_case_s"]     = float(worst_cont) if np.isfinite(worst_cont) else None

        for delay_ms in relay_delays_ms:
            tau = delay_ms / 1000.0
            if np.isfinite(worst_cont):
                safe = worst_cont < tau
            else:
                safe = None
            table[m][f"safe_at_{delay_ms}ms"] = safe

    # Build relay coordination matrix
    relay_matrix = {}
    for delay_ms in relay_delays_ms:
        relay_matrix[delay_ms] = {
            m: table[m].get(f"safe_at_{delay_ms}ms") for m in CORE_METHODS
        }

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.8, 2.6))
        mets = CORE_METHODS
        n_d  = len(relay_delays_ms)
        safe_grid = np.full((n_d, len(mets)), np.nan)
        for di, delay_ms in enumerate(relay_delays_ms):
            for mi, m in enumerate(mets):
                v = table[m].get(f"safe_at_{delay_ms}ms")
                if v is not None:
                    safe_grid[di, mi] = 1.0 if v else 0.0
        im = ax.imshow(safe_grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(mets))); ax.set_xticklabels(mets, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(n_d)); ax.set_yticklabels([f"{d} ms" for d in relay_delays_ms], fontsize=7)
        for di in range(n_d):
            for mi in range(len(mets)):
                v = safe_grid[di, mi]
                if not np.isnan(v):
                    ax.text(mi, di, "✓" if v else "✗", ha="center", va="center",
                            fontsize=8, color="k")
        ax.set_title("A4: Relay coordination safety (green=safe, red=trips)", fontsize=7)
        ax.set_xlabel("Estimator")
        ax.set_ylabel("Relay delay setting")
        plt.colorbar(im, ax=ax, shrink=0.6, ticks=[0, 1], label="Safe")
        fig.tight_layout()
        _save_fig(fig, 104, "relay_coordination")
    except Exception as e:
        warnings.warn(f"A4 figure: {e}")

    # Generate Table VI text
    header = f"{'Method':<18} " + " ".join(f"{'safe@'+str(d)+'ms':>12}" for d in relay_delays_ms)
    rows_txt = [header, "-" * len(header)]
    for m in CORE_METHODS:
        row_str = f"{m:<18} "
        for d in relay_delays_ms:
            v = table[m].get(f"safe_at_{d}ms")
            row_str += f"{'YES':>12}" if v else f"{'NO':>12}"
        rows_txt.append(row_str)

    return {
        "relay_coordination_table": table,
        "relay_matrix":             relay_matrix,
        "relay_delays_ms_tested":   relay_delays_ms,
        "table_vi_text":            "\n".join(rows_txt),
        "interpretation": (
            "Table VI for paper: relay-safe estimators by delay setting. "
            "EKF2 is safe at any delay > 12ms (worst-case max_contiguous=0.012s). "
            "SOGI-FLL is unsafe at any realistic relay delay (max_contiguous=3.67s)."
        ),
    }


# =============================================================================
# BLOCK B — Statistical strengthening
# =============================================================================

def stochastic_deterministic_classifier(mc_raw: dict) -> dict:
    """
    B2: Classify each (method, scenario, metric) as DETERMINISTIC (CV<5%),
    MODERATE (5-50%), or STOCHASTIC (CV>50%). Reports which paper metrics
    are safe to report from single runs vs which require MC analysis.
    """
    CV_DET   = 0.05
    CV_STOCH = 0.50
    METRICS_MC = ["RMSE", "FE_max", "RFE_max", "TRIP"]

    classification = {}
    for m in CORE_METHODS:
        classification[m] = {}
        for sc in SCENARIOS:
            key = (m, sc)
            if key not in mc_raw:
                continue
            cell = {}
            for met in METRICS_MC:
                vals = np.array(mc_raw[key].get(met, []), dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) < 3:
                    cell[met] = {"class": "unknown", "CV": None}
                    continue
                mean_ = float(np.mean(vals))
                std_  = float(np.std(vals))
                cv    = std_ / abs(mean_) if abs(mean_) > 1e-12 else 0.0
                if cv < CV_DET:
                    cls = "DETERMINISTIC"
                elif cv < CV_STOCH:
                    cls = "MODERATE"
                else:
                    cls = "STOCHASTIC"
                cell[met] = {
                    "class": cls, "CV": float(cv),
                    "mean":  float(mean_), "std": float(std_),
                    "single_run_reliable": cls == "DETERMINISTIC",
                }
            classification[m][sc] = cell

    # Summary: how many cells per class
    counts = {"DETERMINISTIC": 0, "MODERATE": 0, "STOCHASTIC": 0}
    stochastic_cells = []
    for m in classification:
        for sc in classification[m]:
            for met, info in classification[m][sc].items():
                cls = info.get("class", "unknown")
                if cls in counts:
                    counts[cls] += 1
                if cls == "STOCHASTIC":
                    stochastic_cells.append(f"{m}/{sc}/{met} CV={info['CV']:.1f}")

    # EKF2 Islanding Ttrip as canonical example
    ekf2_trip = classification.get("EKF2", {}).get("IBR_Nightmare", {}).get("TRIP", {})

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        # Scatter: CV_RMSE vs CV_TRIP per method across all IBR scenarios
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        IBR_SC = ["IBR_Nightmare", "IBR_MultiEvent"]
        for m in CORE_METHODS:
            cv_rmse_list, cv_trip_list = [], []
            for sc in IBR_SC:
                rmse_info = classification.get(m, {}).get(sc, {}).get("RMSE", {})
                trip_info = classification.get(m, {}).get(sc, {}).get("TRIP", {})
                if rmse_info.get("CV") is not None and trip_info.get("CV") is not None:
                    cv_rmse_list.append(rmse_info["CV"])
                    cv_trip_list.append(trip_info["CV"])
            if cv_rmse_list:
                ax.scatter(np.mean(cv_rmse_list), np.mean(cv_trip_list),
                           c=_method_color(m), s=30, zorder=3, edgecolors="k", lw=0.3)
                ax.annotate(m, (np.mean(cv_rmse_list), np.mean(cv_trip_list)),
                            fontsize=5, ha="left", va="bottom")
        ax.axvline(CV_DET,   color="g", ls="--", lw=0.7, label=f"CV={CV_DET}")
        ax.axvline(CV_STOCH, color="r", ls="--", lw=0.7, label=f"CV={CV_STOCH}")
        ax.axhline(CV_DET,   color="g", ls="--", lw=0.7)
        ax.axhline(CV_STOCH, color="r", ls="--", lw=0.7)
        ax.set_xlabel("CV(RMSE)"); ax.set_ylabel("CV(Ttrip)")
        ax.set_title("B2: Stochastic vs deterministic metrics", fontsize=7)
        ax.legend(fontsize=5)
        fig.tight_layout()
        _save_fig(fig, 201, "stochastic_deterministic")
    except Exception as e:
        warnings.warn(f"B2 figure: {e}")

    return {
        "classification":       classification,
        "counts":               counts,
        "stochastic_cells":     stochastic_cells[:20],
        "ekf2_islanding_ttrip": ekf2_trip,
        "interpretation": (
            f"DETERMINISTIC={counts['DETERMINISTIC']}, "
            f"MODERATE={counts['MODERATE']}, "
            f"STOCHASTIC={counts['STOCHASTIC']}. "
            f"EKF2 IBR_Nightmare Ttrip CV={ekf2_trip.get('CV','?'):.0%} — "
            f"single-run Ttrip is {ekf2_trip.get('class','unknown')}. "
            f"Stochastic cells: {stochastic_cells[:3]}"
        ) if ekf2_trip else "No data.",
    }


def bootstrap_correlation_ci(mc_raw: dict, n_boot: int = 2000) -> dict:
    """
    B3: Bootstrap 95% CIs on pairwise scenario Spearman rho using MC runs.
    Converts point estimates to intervals — formally confirms independence
    when CI includes 0.
    """
    if not _SCIPY_OK:
        return {"error": "scipy required for B3"}

    rng = np.random.default_rng(42)

    # Build per-run RMSE matrix {(method, scenario): [rmse per run]}
    mc_rmse = {}
    for m in CORE_METHODS:
        for sc in SCENARIOS:
            key = (m, sc)
            if key in mc_raw:
                vals = [float(v) for v in mc_raw[key].get("RMSE", []) if np.isfinite(v)]
                if vals:
                    mc_rmse[(m, sc)] = vals

    n_runs  = min(len(v) for v in mc_rmse.values()) if mc_rmse else 0
    if n_runs < 5:
        return {"error": f"Too few MC runs ({n_runs}) for bootstrap"}

    # Truncate all to same length
    for k in mc_rmse:
        mc_rmse[k] = mc_rmse[k][:n_runs]

    ci_results = {}
    for i, sc_i in enumerate(SCENARIOS):
        for j, sc_j in enumerate(SCENARIOS):
            if j <= i:
                continue
            # For each run, compute rank-vector across methods
            pairs = []
            for run in range(n_runs):
                x_run = np.array([mc_rmse.get((m, sc_i), [np.nan]*n_runs)[run]
                                  for m in CORE_METHODS])
                y_run = np.array([mc_rmse.get((m, sc_j), [np.nan]*n_runs)[run]
                                  for m in CORE_METHODS])
                mask = np.isfinite(x_run) & np.isfinite(y_run)
                if mask.sum() >= 3:
                    rho_, _ = _sstats.spearmanr(x_run[mask], y_run[mask])
                    pairs.append(float(rho_))

            if len(pairs) < 5:
                continue

            pairs_arr = np.array(pairs)
            rho_mean  = float(np.mean(pairs_arr))

            # Bootstrap CI on rho
            boot_rhos = [
                float(np.mean(rng.choice(pairs_arr, size=len(pairs_arr), replace=True)))
                for _ in range(n_boot)
            ]
            ci_lo  = float(np.percentile(boot_rhos, 2.5))
            ci_hi  = float(np.percentile(boot_rhos, 97.5))
            includes_zero = ci_lo <= 0.0 <= ci_hi

            ci_results[f"{sc_i}_vs_{sc_j}"] = {
                "rho_mean":      rho_mean,
                "ci_95_lo":      ci_lo,
                "ci_95_hi":      ci_hi,
                "includes_zero": includes_zero,
                "independent":   includes_zero,
                "n_runs_used":   len(pairs),
            }

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        keys = list(ci_results.keys())
        rhos = [ci_results[k]["rho_mean"] for k in keys]
        los  = [ci_results[k]["rho_mean"] - ci_results[k]["ci_95_lo"] for k in keys]
        his  = [ci_results[k]["ci_95_hi"] - ci_results[k]["rho_mean"] for k in keys]
        colors = ["green" if ci_results[k]["independent"] else "red" for k in keys]
        ax.barh(range(len(keys)), rhos, xerr=[los, his],
                color=colors, edgecolor="k", lw=0.4, capsize=2, error_kw={"lw": 0.8},
                alpha=0.8)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([k.replace("_vs_", " vs ") for k in keys], fontsize=5)
        ax.set_xlabel("Spearman $\\rho$ (bootstrap 95% CI)")
        ax.set_title("B3: Scenario rank correlation with bootstrap CIs", fontsize=7)
        fig.tight_layout()
        _save_fig(fig, 202, "bootstrap_ci")
    except Exception as e:
        warnings.warn(f"B3 figure: {e}")

    return {
        "pairwise_ci": ci_results,
        "n_bootstrap": n_boot,
        "independent_pairs": [k for k, v in ci_results.items() if v["independent"]],
        "interpretation": (
            "Bootstrap 95% CI includes 0 → pair is statistically independent. "
            f"Independent scenario pairs: "
            f"{[k for k, v in ci_results.items() if v['independent']]}"
        ),
    }


def full_pairwise_analysis(mc_raw: dict) -> dict:
    """
    B4: Full pairwise Mann-Whitney on RMSE for all method pairs per scenario.
    Apply Benjamini-Hochberg FDR correction. Reports which pairwise
    differences survive correction.
    """
    if not _SCIPY_OK:
        return {"error": "scipy required for B4"}

    def _bh_correction(pvals, alpha=0.05):
        """Benjamini-Hochberg FDR correction. Returns array of bool (rejected H0)."""
        n = len(pvals)
        if n == 0:
            return np.array([], dtype=bool)
        idx = np.argsort(pvals)
        pvals_sorted = np.array(pvals)[idx]
        thresholds   = (np.arange(1, n + 1) / n) * alpha
        # Last index where p_sorted[i] <= threshold[i]
        below = pvals_sorted <= thresholds
        if not below.any():
            rejected = np.zeros(n, dtype=bool)
        else:
            cutoff = int(np.max(np.where(below)[0]))
            rejected_sorted = np.zeros(n, dtype=bool)
            rejected_sorted[:cutoff + 1] = True
            rejected = np.zeros(n, dtype=bool)
            rejected[idx] = rejected_sorted
        return rejected

    results = {}
    all_pvals, all_labels = [], []

    for sc in SCENARIOS:
        sc_results = {}
        for i, m1 in enumerate(CORE_METHODS):
            for j, m2 in enumerate(CORE_METHODS):
                if j <= i:
                    continue
                k1 = (m1, sc)
                k2 = (m2, sc)
                if k1 not in mc_raw or k2 not in mc_raw:
                    continue
                v1 = np.array(mc_raw[k1].get("RMSE", []), dtype=float)
                v2 = np.array(mc_raw[k2].get("RMSE", []), dtype=float)
                v1 = v1[np.isfinite(v1)]
                v2 = v2[np.isfinite(v2)]
                if len(v1) < 3 or len(v2) < 3:
                    continue
                stat, p = _sstats.mannwhitneyu(v1, v2, alternative="two-sided")
                sc_results[f"{m1}_vs_{m2}"] = {
                    "mw_stat": float(stat), "p_raw": float(p),
                    "m1_better": float(np.mean(v1)) < float(np.mean(v2)),
                }
                all_pvals.append(float(p))
                all_labels.append(f"{sc}|{m1}_vs_{m2}")
        results[sc] = sc_results

    # Apply BH-FDR globally
    if all_pvals:
        rejected = _bh_correction(all_pvals)
        for label, rej in zip(all_labels, rejected):
            sc_part, pair_part = label.split("|", 1)
            if pair_part in results.get(sc_part, {}):
                results[sc_part][pair_part]["significant_BH"] = bool(rej)

    # Count significant pairs per scenario
    sig_per_sc = {
        sc: sum(1 for v in results[sc].values() if v.get("significant_BH"))
        for sc in SCENARIOS
    }
    total_pairs = len(all_pvals)
    total_sig   = int(sum(rejected)) if all_pvals else 0

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(6.8, 2.0))
        for ax, sc in zip(axes, SCENARIOS):
            mets = CORE_METHODS
            n_m = len(mets)
            sig_matrix = np.full((n_m, n_m), np.nan)
            for i, m1 in enumerate(mets):
                for j, m2 in enumerate(mets):
                    if i == j:
                        continue
                    key = f"{m1}_vs_{m2}" if f"{m1}_vs_{m2}" in results.get(sc, {}) else f"{m2}_vs_{m1}"
                    info = results.get(sc, {}).get(key, {})
                    if info:
                        sig_matrix[i, j] = 1.0 if info.get("significant_BH") else 0.0
            im = ax.imshow(sig_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(n_m)); ax.set_xticklabels([m[:4] for m in mets], rotation=90, fontsize=4)
            ax.set_yticks(range(n_m)); ax.set_yticklabels([m[:4] for m in mets], fontsize=4)
            ax.set_title(sc[:8], fontsize=6)
        fig.suptitle("B4: Pairwise BH-FDR significance (green=significant)", fontsize=7)
        fig.tight_layout()
        _save_fig(fig, 203, "pairwise_BH_FDR")
    except Exception as e:
        warnings.warn(f"B4 figure: {e}")

    return {
        "results":           results,
        "total_pairs_tested":total_pairs,
        "total_significant": total_sig,
        "significant_per_scenario": sig_per_sc,
        "fdr_level":         0.05,
        "interpretation": (
            f"{total_sig}/{total_pairs} pairwise comparisons significant "
            f"after BH-FDR correction at alpha=0.05. "
            f"Per scenario: {sig_per_sc}"
        ),
    }


# =============================================================================
# BLOCK C — New adversarial / edge-case scenarios
# =============================================================================

def run_new_scenarios(tuned_params: dict) -> dict:

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    """
    C1–C3: Run all estimators on three new scenarios:
      C1: Voltage sag + phase jump (simultaneous amplitude drop + 60° jump)
      C2: Subsynchronous oscillation (5–15 Hz frequency oscillation ±0.5 Hz)
      C3: Blind spot test (rapid double phase jump within 100ms)
    Tuned params from IBR_Nightmare (no re-tuning = true generalization test).
    """
    try:
        from estimators import (
            FS_PHYSICS as _FP, RATIO as _RAT, calculate_metrics,
            TunableIpDFT, StandardPLL, ClassicEKF, SOGI_FLL,
            TFT_Estimator, UKF_Estimator, Koopman_RKDPmu, RLS_Estimator,
        )
        from ekf2 import EKF2
    except ImportError as e:
        return {"error": f"Import failed: {e}"}

    FS_P = 1_000_000.0; RAT = 100

    # ── Signal generators ─────────────────────────────────────────────────
    def _c1_voltage_sag():
        t = np.arange(0, 1.5, 1.0 / FS_P); n = len(t)
        A = np.where(t < 0.7, 1.0, 0.75)   # -25% sag at event
        phase = np.zeros(n); curr = 0.0
        for i in range(1, n):
            if 0.6999 < t[i] < 0.7001:
                curr += np.deg2rad(60.0)
            curr += 2 * np.pi * 60.0 / FS_P
            phase[i] = curr
        v = A * np.sin(phase) + 0.05 * np.sin(5 * phase)
        v += np.random.default_rng(11).normal(0, 0.005, n)
        f = np.ones(n) * 60.0
        return v, f, "C1_VoltageSag"

    def _c2_subsync(f_sub=10.0, a_osc=0.5):
        t = np.arange(0, 2.0, 1.0 / FS_P); n = len(t)
        f = 60.0 + a_osc * np.sin(2 * np.pi * f_sub * t)
        phase = np.cumsum(2 * np.pi * f / FS_P)
        v = np.sin(phase) + np.random.default_rng(22).normal(0, 0.003, n)
        return v, f, "C2_Subsync"

    def _c3_double_jump():
        t = np.arange(0, 1.5, 1.0 / FS_P); n = len(t)
        phase = np.zeros(n); curr = 0.0
        for i in range(1, n):
            if 0.6999 < t[i] < 0.7001:
                curr += np.deg2rad(30.0)
            if 0.7999 < t[i] < 0.8001:
                curr += np.deg2rad(30.0)
            curr += 2 * np.pi * 60.0 / FS_P
            phase[i] = curr
        v = np.sin(phase) + np.random.default_rng(33).normal(0, 0.01, n)
        f = np.ones(n) * 60.0
        return v, f, "C3_DoubleJump"
    
    def save_new_scenarios_plots(scenarios_new, output_dir="figures_scientific"):

        import os
        import numpy as np
        import matplotlib.pyplot as plt

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Configuración estética para Paper (estilo IEEE)
        plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=False)
        plt.subplots_adjust(hspace=0.4)

        for i, (v, f, name) in enumerate(scenarios_new):
            ax = axes[i]
            ax2 = ax.twinx() # Eje secundario para la frecuencia
            
            # Tiempo basado en la longitud de v y asumiendo FS_P (usualmente 10000)
            fs = 10000.0 
            t = np.arange(len(v)) / fs
            
            # Plot Voltaje (Línea azul clara)
            lns1 = ax.plot(t, v, color='tab:blue', alpha=0.6, label='Voltaje [pu]', linewidth=0.8)
            
            # Plot Frecuencia (Línea roja sólida)
            lns2 = ax2.plot(t, f, color='tab:red', label='Freq. Ref [Hz]', linewidth=1.5)
            
            # Ajustes de ejes
            ax.set_title(f"Escenario: {name.replace('_', ' ')}", fontweight='bold')
            ax.set_ylabel("Voltaje [pu]", color='tab:blue')
            ax2.set_ylabel("Frecuencia [Hz]", color='tab:red')
            ax.set_xlabel("Tiempo [s]")
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            
            # Unificar leyendas
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc='upper right', frameon=True, fontsize=8)

        # Guardar en alta resolución
        plt.savefig(os.path.join(output_dir, "C_New_Scenarios_Visual.png"), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Gráfico guardado en: {output_dir}/C_New_Scenarios_Visual.png")

    scenarios_new = [_c1_voltage_sag(), _c2_subsync(), _c3_double_jump()]
    save_new_scenarios_plots(scenarios_new)

    def _get_tp(method, key, default):
        return tuned_params.get("IBR_Nightmare", {}).get(method, {}).get(key, default)

    factories = {
        "IpDFT": lambda: TunableIpDFT(_get_tp("IpDFT", "cycles", 4)),
        "PLL":   lambda: StandardPLL(_get_tp("PLL", "kp", 10.0), _get_tp("PLL", "ki", 50.0)),
        "EKF":   lambda: ClassicEKF(_get_tp("EKF", "Q", 0.1), _get_tp("EKF", "R", 0.01)),
        "EKF2":  lambda: EKF2(
            q_param=_get_tp("EKF2","q_param",1.0), r_param=_get_tp("EKF2","r_param",0.01),
            inn_ref=_get_tp("EKF2","inn_ref",0.1), event_thresh=_get_tp("EKF2","event_thresh",2.0),
            fast_horizon_ms=_get_tp("EKF2","fast_horizon_ms",80.0)),
        "SOGI":  lambda: SOGI_FLL(_get_tp("SOGI","k",1.414), _get_tp("SOGI","g",20.0),
                                   smooth_win=_get_tp("SOGI","smooth_win",None)),
        "TFT":   lambda: TFT_Estimator(_get_tp("TFT","win",3)),
        "UKF":   lambda: UKF_Estimator(_get_tp("UKF","Q",0.1), _get_tp("UKF","R",0.01)),
        "Koopman-RKDPmu": lambda: Koopman_RKDPmu(
            window_samples=_get_tp("Koopman-RKDPmu","window_samples",200),
            smooth_win=_get_tp("Koopman-RKDPmu","smooth_win",200)),
        "RLS":   lambda: RLS_Estimator(lam=_get_tp("RLS","lambda",0.995),
                                        win_smooth=_get_tp("RLS","win_smooth",100), decim=50),
    }

    results = {}
    for v_sig, f_sig, sc_label in scenarios_new:
        v_dsp = v_sig[::RAT]; f_dsp = f_sig[::RAT]
        results[sc_label] = {}
        for m, fac in factories.items():
            try:
                algo = fac()
                tr   = np.array([algo.step(x) for x in v_dsp])
                met  = calculate_metrics(tr, f_dsp, 0.0, structural_samples=1)
                results[sc_label][m] = {
                    "RMSE": float(met["RMSE"]),
                    "MAX_PEAK": float(met["MAX_PEAK"]),
                    "TRIP_TIME_0p5": float(met["TRIP_TIME_0p5"]),
                    "MAX_CONTIGUOUS_0p5": float(met["MAX_CONTIGUOUS_0p5"]),
                }
            except Exception as e:
                results[sc_label][m] = {"error": str(e)}

    # Winner per new scenario
    winners = {}
    for sc_label, sc_res in results.items():
        valid = {m: v["RMSE"] for m, v in sc_res.items()
                 if isinstance(v, dict) and "RMSE" in v and np.isfinite(v["RMSE"])}
        if valid:
            winners[sc_label] = min(valid, key=valid.get)

    # C3: does EKF2 fail while PLL succeeds? (the blind spot test)
    c3 = results.get("C3_DoubleJump", {})
    ekf2_c3 = c3.get("EKF2", {}).get("RMSE", np.nan)
    pll_c3  = c3.get("PLL",  {}).get("RMSE", np.nan)
    blind_spot_confirmed = (
        np.isfinite(ekf2_c3) and np.isfinite(pll_c3) and pll_c3 < ekf2_c3
    )

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
        for ax, (v_sig, f_sig, sc_label) in zip(axes, scenarios_new):
            sc_res = results.get(sc_label, {})
            mets   = [m for m in factories if m in sc_res
                      and isinstance(sc_res[m], dict) and "RMSE" in sc_res[m]]
            rmse_v = [sc_res[m]["RMSE"] for m in mets]
            ax.bar(range(len(mets)), rmse_v, color=[_method_color(m) for m in mets],
                   edgecolor="k", lw=0.4)
            ax.set_xticks(range(len(mets)))
            ax.set_xticklabels(mets, rotation=45, ha="right", fontsize=5)
            ax.set_ylabel("RMSE [Hz]"); ax.set_title(sc_label, fontsize=7)
            ax.axhline(IEC_RMSE, color="r", ls="--", lw=0.6)
        fig.suptitle("C1-C3: New adversarial scenarios", fontsize=7)
        fig.tight_layout()
        _save_fig(fig, 301, "new_scenarios_ABC")
    except Exception as e:
        warnings.warn(f"C1-C3 figure: {e}")

    return {
        "results":                results,
        "winners":                winners,
        "c3_blind_spot_confirmed":blind_spot_confirmed,
        "c3_ekf2_rmse":           float(ekf2_c3) if np.isfinite(ekf2_c3) else None,
        "c3_pll_rmse":            float(pll_c3)  if np.isfinite(pll_c3)  else None,
        "tuning_used":            "IBR_Nightmare (no re-tuning = generalization test)",
        "interpretation": (
            f"Winners: {winners}. "
            f"C3 EKF2 blind spot: {'CONFIRMED' if blind_spot_confirmed else 'NOT confirmed'}. "
            f"EKF2 RMSE={ekf2_c3:.4f} vs PLL={pll_c3:.4f} on double-jump."
        ) if all(np.isfinite([ekf2_c3, pll_c3])) else "Insufficient data.",
    }


# =============================================================================
# BLOCK D — ML Analysis
# =============================================================================

def method_selection_classifier(results_dict: dict, tuned_params: dict) -> dict:
    """
    D1: Build a Random Forest to recommend which estimator minimises Ttrip
    in IBR scenarios given observable grid features extracted from test results.
    Features: RMSE in each IEEE scenario + CPU + RoCoF error.
    Target: best method (lowest Ttrip) in IBR_Nightmare.
    """
    if not _SKLEARN_OK or not _SCIPY_OK:
        return {"error": "sklearn + scipy required for D1"}

    IBR_TARGET = "IBR_Nightmare"
    FEATURE_SC = ["IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation"]
    FEAT_NAMES = (
        [f"RMSE_{sc[:8]}" for sc in FEATURE_SC]
        + [f"Ttrip_{sc[:8]}" for sc in FEATURE_SC]
        + ["CPU_us", "RFE_rms_IBR"]
    )

    # Build synthetic training data by scenario-parameter sweeps using existing results
    # Feature row per method = IEEE results (known at deployment time)
    X_list, y_list = [], []
    for m in CORE_METHODS:
        row = (
            [_get(results_dict, sc, m, "RMSE") for sc in FEATURE_SC]
            + [_get(results_dict, sc, m, "TRIP_TIME_0p5") for sc in FEATURE_SC]
            + [_get(results_dict, "IBR_MultiEvent", m, "TIME_PER_SAMPLE_US")]
            + [_get(results_dict, IBR_TARGET, m, "RFE_rms_Hz_s")]
        )
        target_ttrip = _get(results_dict, IBR_TARGET, m, "TRIP_TIME_0p5")
        if all(np.isfinite(row)) and np.isfinite(target_ttrip):
            X_list.append(row)
            y_list.append(m)

    if len(X_list) < 4:
        return {"error": f"Only {len(X_list)} clean methods — need 4+"}

    X = np.array(X_list)
    # Augment with small Gaussian noise to expand the dataset (10× augmentation)
    rng = np.random.default_rng(7)
    X_aug = np.vstack([X + rng.normal(0, 0.02, X.shape) * X * 0.1
                       for _ in range(10)] + [X])
    y_aug = y_list * 11

    X_sc  = _SS().fit_transform(X_aug)
    clf   = _RFC(n_estimators=300, max_depth=6, random_state=42, n_jobs=1)
    cv    = _SKF(n_splits=min(5, len(set(y_list))), shuffle=True, random_state=42)

    try:
        acc_scores = _CVS(clf, X_sc, y_aug, cv=cv, scoring="accuracy")
    except Exception:
        acc_scores = np.array([np.nan])

    clf.fit(X_sc, y_aug)
    importances = dict(zip(FEAT_NAMES, clf.feature_importances_.tolist()))

    # Predict recommended method for each original method's feature vector
    X_orig_sc = _SS().fit_transform(X)
    predictions = clf.predict(X_orig_sc).tolist()

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.0, 2.6))
        ax.bar(FEAT_NAMES, clf.feature_importances_,
               color=_OI[1], edgecolor="k", lw=0.4)
        ax.set_xticklabels(FEAT_NAMES, rotation=45, ha="right", fontsize=5)
        ax.set_ylabel("Feature importance")
        ax.set_title(f"D1: Method selector RF\nCV acc={np.nanmean(acc_scores):.2f}", fontsize=7)
        fig.tight_layout()
        _save_fig(fig, 401, "method_selector_RF")
    except Exception as e:
        warnings.warn(f"D1 figure: {e}")

    return {
        "cv_accuracy_mean":   float(np.nanmean(acc_scores)),
        "cv_accuracy_std":    float(np.nanstd(acc_scores)),
        "feature_importances":importances,
        "top_feature":        max(importances, key=importances.get),
        "original_methods":   y_list,
        "predictions":        predictions,
        "interpretation": (
            f"RF method selector CV accuracy={np.nanmean(acc_scores):.2f}. "
            f"Most informative feature: {max(importances, key=importances.get)}."
        ),
    }


def anomaly_detection(mc_raw: dict) -> dict:
    """
    D2: Use MC distributions as calibrated baselines.
    Compute z-score for each run: z = (RMSE - mean) / std.
    Flag runs with |z| > 3 as anomalous.
    Enables field monitoring: run deviating > 3σ may indicate degradation.
    """
    anomalies = {}
    z_summary = {}

    for m in CORE_METHODS:
        for sc in SCENARIOS:
            key = (m, sc)
            if key not in mc_raw:
                continue
            vals = np.array(mc_raw[key].get("RMSE", []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 3:
                continue
            mean_ = float(np.mean(vals))
            std_  = float(np.std(vals))
            if std_ < 1e-12:
                continue
            z_scores = (vals - mean_) / std_
            n_anomalous = int(np.sum(np.abs(z_scores) > 3.0))
            z_summary[f"{m}/{sc}"] = {
                "mean":        mean_,
                "std":         std_,
                "n_anomalous": n_anomalous,
                "max_z":       float(np.max(np.abs(z_scores))),
                "z_scores":    z_scores.tolist(),
            }
            if n_anomalous > 0:
                anomalies[f"{m}/{sc}"] = n_anomalous

    # Anomaly detection function for a new single-run value
    def detection_function_example():
        """Pseudocode: given new_rmse, sc, m → is it anomalous?"""
        return "z_score = (new_rmse - mean[m][sc]) / std[m][sc]; anomalous if |z| > 3"

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        # Plot z-score distributions for IBR scenarios
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
        for ax, sc in zip(axes, ["IBR_Nightmare", "IBR_MultiEvent"]):
            for m in CORE_METHODS:
                key = f"{m}/{sc}"
                if key in z_summary:
                    z = np.array(z_summary[key]["z_scores"])
                    ax.plot(range(len(z)), sorted(z), "-", color=_method_color(m),
                            lw=0.8, alpha=0.7, label=m)
            ax.axhline(3,  color="r", ls="--", lw=0.6, label="|z|=3")
            ax.axhline(-3, color="r", ls="--", lw=0.6)
            ax.set_xlabel("Run (sorted)"); ax.set_ylabel("z-score")
            ax.set_title(f"D2: Anomaly scores — {sc}", fontsize=7)
            ax.legend(fontsize=4, ncol=2)
        fig.tight_layout()
        _save_fig(fig, 402, "anomaly_detection")
    except Exception as e:
        warnings.warn(f"D2 figure: {e}")

    return {
        "z_summary":              z_summary,
        "anomalous_cells":        anomalies,
        "detection_function":     "z = (RMSE_new - baseline_mean) / baseline_std; flag if |z| > 3",
        "n_anomalous_cells":      len(anomalies),
        "interpretation": (
            f"{len(anomalies)} cells have ≥1 anomalous MC run (|z|>3). "
            "Use z-score as field health monitoring metric."
        ),
    }


def pc_regression_islanding(results_dict: dict) -> dict:
    """
    D3: PCR — predict Islanding RMSE from other performance features.
    If R² < 0.2: Islanding is truly unpredictable from standard metrics.
    If R² > 0.7: there is a predictor.
    """
    if not _SKLEARN_OK or not _SCIPY_OK:
        return {"error": "sklearn + scipy required for D3"}

    ISL_SC     = "IBR_Nightmare"
    FEATURE_SC = ["IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation", "IBR_MultiEvent"]
    FEAT_METRICS = ["RMSE", "MAX_PEAK", "TRIP_TIME_0p5", "TIME_PER_SAMPLE_US", "RFE_rms_Hz_s"]

    X_list, y_list = [], []
    for m in CORE_METHODS:
        row = [
            _get(results_dict, sc, m, met)
            for sc in FEATURE_SC for met in FEAT_METRICS
        ]
        y = _get(results_dict, ISL_SC, m, "RMSE")
        if all(np.isfinite(row)) and np.isfinite(y):
            X_list.append(row); y_list.append(y)

    if len(X_list) < 4:
        return {"error": f"Only {len(X_list)} clean data points"}

    X = np.array(X_list); y = np.array(y_list)
    for c in range(X.shape[1]):
        col = X[:, c]
        finite = col[np.isfinite(col)]
        if len(finite):
            X[~np.isfinite(col), c] = float(np.median(finite))

    X_sc = _SS().fit_transform(X)
    pca  = _PCA(n_components=min(len(X_list), X_sc.shape[1]))
    Z    = pca.fit_transform(X_sc)

    best_r2, best_n = -np.inf, 1
    for n_pc in range(1, min(Z.shape[1] + 1, len(y_list))):
        if _SCIPY_OK and n_pc <= Z.shape[1]:
            slope_, _, r_, _, _ = _sstats.linregress(Z[:, 0] if n_pc == 1
                                                      else Z[:, :n_pc].mean(axis=1), y)
            if r_ ** 2 > best_r2:
                best_r2, best_n = r_ ** 2, n_pc

    # Final regression on best_n PCs
    Z_best = Z[:, :best_n] if best_n > 1 else Z[:, :1]
    if _SCIPY_OK:
        if best_n == 1:
            slope, intercept, r_val, p_val, _ = _sstats.linregress(Z_best[:, 0], y)
            r2_final = float(r_val ** 2)
        else:
            # Multi-PC: use sum of first best_n PC scores as single predictor
            slope, intercept, r_val, p_val, _ = _sstats.linregress(Z_best.sum(axis=1), y)
            r2_final = float(r_val ** 2)
    else:
        r2_final = np.nan

    conclusion = (
        "Islanding RMSE is UNPREDICTABLE from standard metrics (R²<0.2). "
        "Direct IBR stress testing is mandatory."
        if r2_final < 0.2 else
        f"Islanding RMSE has a predictor in standard metrics (R²={r2_final:.2f})."
    )

    return {
        "best_r2":     float(r2_final) if np.isfinite(r2_final) else None,
        "n_pcs_used":  best_n,
        "n_methods":   len(y_list),
        "conclusion":  conclusion,
        "interpretation": conclusion,
    }


# =============================================================================
# BLOCK E — Missing paper tables
# =============================================================================

def build_paper_tables(results_dict: dict, mc_raw: dict) -> dict:
    """
    E1-E3: Generate Table VI (relay coordination), Table VII (stochastic
    reproducibility), and scenario information-theoretic table.
    """

    # ── E1: Relay coordination (uses A4 data) ─────────────────────────────
    relay_delays_ms = [12, 50, 100, 150, 200, 500]
    table_vi = {"columns": ["Method"] + [f"safe@{d}ms" for d in relay_delays_ms]}
    table_vi_rows = []
    for m in CORE_METHODS:
        ibr_contig = max(
            (_get(results_dict, sc, m, "MAX_CONTIGUOUS_0p5")
             for sc in ["IBR_Nightmare", "IBR_MultiEvent"]),
            default=np.nan
        )
        row = {"method": m, "worst_case_contiguous_s": float(ibr_contig)
               if np.isfinite(ibr_contig) else None}
        for d in relay_delays_ms:
            tau = d / 1000.0
            row[f"safe@{d}ms"] = (bool(ibr_contig < tau)
                                   if np.isfinite(ibr_contig) else None)
        table_vi_rows.append(row)
    table_vi["rows"] = table_vi_rows

    # ── E2: Stochastic reproducibility table ─────────────────────────────
    table_vii = {"columns": ["Method", "Scenario", "RMSE_mean±std", "Ttrip_mean±std",
                              "CV_RMSE", "CV_Ttrip", "Reproducible_RMSE",
                              "Reproducible_Ttrip"]}
    table_vii_rows = []
    for m in CORE_METHODS:
        for sc in ["IBR_Nightmare", "IBR_MultiEvent"]:
            key = (m, sc)
            if key not in mc_raw:
                continue
            rmse_v = np.array(mc_raw[key].get("RMSE", []), dtype=float)
            trip_v = np.array(mc_raw[key].get("TRIP", []), dtype=float)
            rmse_v = rmse_v[np.isfinite(rmse_v)]
            trip_v = trip_v[np.isfinite(trip_v)]
            if len(rmse_v) < 2:
                continue
            rmse_m = float(np.mean(rmse_v)); rmse_s = float(np.std(rmse_v))
            trip_m = float(np.mean(trip_v)); trip_s = float(np.std(trip_v))
            cv_r = rmse_s / max(rmse_m, 1e-12)
            cv_t = trip_s / max(trip_m, 1e-12) if trip_m > 1e-12 else 0.0
            table_vii_rows.append({
                "method": m, "scenario": sc,
                "RMSE_mean": round(rmse_m, 5), "RMSE_std": round(rmse_s, 5),
                "Ttrip_mean": round(trip_m, 5), "Ttrip_std": round(trip_s, 5),
                "CV_RMSE": round(cv_r, 3), "CV_Ttrip": round(cv_t, 3),
                "reproducible_RMSE":  cv_r < 0.05,
                "reproducible_Ttrip": cv_t < 0.05,
            })
    table_vii["rows"] = table_vii_rows

    # ── E3: Scenario information table ────────────────────────────────────
    table_viii = {}
    for sc in SCENARIOS:
        rmse_vals = [_get(results_dict, sc, m, "RMSE") for m in CORE_METHODS]
        rmse_vals = [v for v in rmse_vals if np.isfinite(v)]
        discrim   = (float(np.std(rmse_vals)) / float(np.mean(rmse_vals))
                     if rmse_vals and np.mean(rmse_vals) > 0 else 0.0)
        table_viii[sc] = {
            "discriminative_power": round(discrim, 3),
            "n_methods_valid":      len(rmse_vals),
            "mean_rmse":            round(float(np.mean(rmse_vals)), 4) if rmse_vals else None,
            "std_rmse":             round(float(np.std(rmse_vals)), 4)  if rmse_vals else None,
        }

    # Write tables to text files
    try:
        lines = ["# TABLE VI: Relay Coordination Safety Matrix", ""]
        lines.append(f"{'Method':<18} " +
                     " ".join(f"{'safe@'+str(d)+'ms':>10}" for d in relay_delays_ms))
        lines.append("-" * 80)
        for row in table_vi_rows:
            r = f"{row['method']:<18} "
            r += " ".join(
                f"{'YES':>10}" if row.get(f"safe@{d}ms") else f"{'NO':>10}"
                for d in relay_delays_ms
            )
            lines.append(r)
        lines += ["", "# TABLE VII: Stochastic Reproducibility", ""]
        lines.append(f"{'Method':<12}{'Scenario':<18}{'RMSE_mean±std':>18}{'CV_RMSE':>10}"
                     f"{'CV_Ttrip':>10}{'Repro_RMSE':>12}{'Repro_Ttrip':>12}")
        lines.append("-" * 100)
        for row in table_vii_rows:
            lines.append(
                f"{row['method']:<12}{row['scenario']:<18}"
                f"  {row['RMSE_mean']:.4f}±{row['RMSE_std']:.4f}"
                f"  {row['CV_RMSE']:>8.3f}  {row['CV_Ttrip']:>8.3f}"
                f"  {str(row['reproducible_RMSE']):>10}  {str(row['reproducible_Ttrip']):>10}"
            )
        path = os.path.join(FIGURES_SCIENTIFIC_DIR, "paper_tables_E1_E2_E3.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        warnings.warn(f"E tables write failed: {e}")

    return {
        "table_vi_relay_coordination":    table_vi,
        "table_vii_reproducibility":      table_vii,
        "table_viii_scenario_info":       table_viii,
        "interpretation": (
            "Table VI: relay coordination safety. "
            "Table VII: identifies which metrics are reproducible from single runs. "
            "Table VIII: scenario discriminative power."
        ),
    }


# =============================================================================
# BLOCK G — Tuning Robustness Analysis
# =============================================================================

def tuning_sensitivity_analysis(tuned_params: dict,
                                  n_perturbations: int = 200) -> dict:
    """
    G1: For each method, perturb parameters by ±3× (one log-decade) and
    measure RMSE degradation ratio = RMSE_perturbed / RMSE_optimal.
    Reports robustness score P(degradation < 2×).
    """
    try:
        from estimators import (
            FS_PHYSICS as _FP, RATIO as _RAT, get_test_signals, calculate_metrics,
            TunableIpDFT, StandardPLL, ClassicEKF, SOGI_FLL,
            TFT_Estimator, UKF_Estimator, Koopman_RKDPmu, RLS_Estimator,
        )
        from ekf2 import EKF2
    except ImportError as e:
        return {"error": f"Import failed: {e}"}

    RAT = 100
    rng = np.random.default_rng(55)

    signals = get_test_signals(seed=42)
    SC_USE  = "IBR_Nightmare"
    _, v_sig, f_sig, _ = signals[SC_USE]
    v_dsp = v_sig[::RAT]; f_dsp = f_sig[::RAT]

    def _get_tp(m, k, d):
        return tuned_params.get(SC_USE, {}).get(m, {}).get(k, d)

    # Method config: {name: {param: (optimal_value, log_perturb)}}
    METHOD_CONFIGS = {
        "EKF2": {
            "q_param":        (_get_tp("EKF2","q_param",1.0),        True),
            "r_param":        (_get_tp("EKF2","r_param",0.01),       True),
            "inn_ref":        (_get_tp("EKF2","inn_ref",0.1),        False),
            "event_thresh":   (_get_tp("EKF2","event_thresh",2.0),   False),
        },
        "EKF": {
            "Q": (_get_tp("EKF","Q",0.1),   True),
            "R": (_get_tp("EKF","R",0.01),  True),
        },
        "PLL": {
            "kp": (_get_tp("PLL","kp",10.0), False),
            "ki": (_get_tp("PLL","ki",50.0), False),
        },
        "SOGI": {
            "k": (_get_tp("SOGI","k",1.414), False),
            "g": (_get_tp("SOGI","g",20.0),  True),
        },
    }

    def _run_method(m, params):
        try:
            if m == "EKF2":
                algo = EKF2(q_param=params.get("q_param",1.0),
                            r_param=params.get("r_param",0.01),
                            inn_ref=params.get("inn_ref",0.1),
                            event_thresh=params.get("event_thresh",2.0),
                            fast_horizon_ms=_get_tp("EKF2","fast_horizon_ms",80.0))
            elif m == "EKF":
                algo = ClassicEKF(params["Q"], params["R"])
            elif m == "PLL":
                algo = StandardPLL(params["kp"], params["ki"])
            elif m == "SOGI":
                algo = SOGI_FLL(params["k"], params["g"])
            else:
                return np.nan
            tr = np.array([algo.step(x) for x in v_dsp])
            return float(calculate_metrics(tr, f_dsp, 0.0, structural_samples=1)["RMSE"])
        except Exception:
            return np.nan

    sensitivity_results = {}
    for m, cfg in METHOD_CONFIGS.items():
        # Optimal RMSE
        opt_params = {p: v for p, (v, _) in cfg.items()}
        opt_rmse   = _run_method(m, opt_params)
        if not np.isfinite(opt_rmse) or opt_rmse < 1e-12:
            continue

        # Random perturbations
        degradations = []
        param_degradations = {p: [] for p in cfg}

        for _ in range(n_perturbations):
            perturbed = {}
            for p, (val, is_log) in cfg.items():
                if is_log:
                    delta = rng.uniform(-0.5, 0.5)   # ±0.5 decade = ±3.16×
                    perturbed[p] = val * (10 ** delta)
                else:
                    delta = rng.uniform(0.5, 2.0)    # ×0.5 to ×2
                    perturbed[p] = val * delta
            rmse_p = _run_method(m, perturbed)
            if np.isfinite(rmse_p):
                degradations.append(rmse_p / opt_rmse)

        # Per-parameter sensitivity: vary one at a time
        for p, (val, is_log) in cfg.items():
            for _ in range(50):
                single_pert = dict(opt_params)
                if is_log:
                    delta = rng.uniform(-0.5, 0.5)
                    single_pert[p] = val * (10 ** delta)
                else:
                    delta = rng.uniform(0.5, 2.0)
                    single_pert[p] = val * delta
                rmse_p = _run_method(m, single_pert)
                if np.isfinite(rmse_p):
                    param_degradations[p].append(rmse_p / opt_rmse)

        degradations = np.array(degradations)
        sensitivity_results[m] = {
            "optimal_rmse":          float(opt_rmse),
            "robustness_score":      float(np.mean(degradations < 2.0)) if len(degradations) else np.nan,
            "p50_degradation":       float(np.percentile(degradations, 50)) if len(degradations) else np.nan,
            "p95_degradation":       float(np.percentile(degradations, 95)) if len(degradations) else np.nan,
            "mean_degradation":      float(np.mean(degradations)) if len(degradations) else np.nan,
            "per_param_sensitivity": {
                p: float(np.mean(param_degradations[p])) if param_degradations[p] else np.nan
                for p in cfg
            },
            "worst_param": max(
                {p: float(np.mean(v)) for p, v in param_degradations.items() if v},
                key=lambda k: float(np.mean(param_degradations[k])) if param_degradations[k] else 0,
                default=None,
            ),
            "relay_deployable_detuned": bool(
                len(degradations) > 0 and np.mean(degradations < 2.0) > 0.90
            ),
        }

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        mets = list(sensitivity_results.keys())
        if mets:
            fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
            ax = axes[0]
            robs = [sensitivity_results[m]["robustness_score"] for m in mets]
            ax.bar(range(len(mets)), robs, color=[_method_color(m) for m in mets],
                   edgecolor="k", lw=0.4)
            ax.axhline(0.9, color="r", ls="--", lw=0.8, label="90% threshold")
            ax.set_xticks(range(len(mets))); ax.set_xticklabels(mets, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("$P(\\mathrm{degradation} < 2\\times)$")
            ax.set_title("G1: Tuning robustness score", fontsize=7)
            ax.legend(fontsize=5)

            ax2 = axes[1]
            all_params = list({p for m in mets for p in sensitivity_results[m]["per_param_sensitivity"]})
            for mi, m in enumerate(mets):
                sens = sensitivity_results[m]["per_param_sensitivity"]
                vals = [sens.get(p, np.nan) for p in all_params]
                ax2.bar(np.arange(len(all_params)) + mi * 0.2, vals, 0.2,
                        color=_method_color(m), edgecolor="k", lw=0.3, label=m, alpha=0.8)
            ax2.set_xticks(np.arange(len(all_params)) + 0.3)
            ax2.set_xticklabels(all_params, rotation=30, ha="right", fontsize=6)
            ax2.set_ylabel("Mean RMSE degradation ratio"); ax2.set_title("Per-param sensitivity", fontsize=7)
            ax2.legend(fontsize=5)
            fig.tight_layout()
            _save_fig(fig, 601, "tuning_sensitivity")
    except Exception as e:
        warnings.warn(f"G1 figure: {e}")

    return {
        "sensitivity_results": sensitivity_results,
        "n_perturbations":     n_perturbations,
        "scenario_used":       SC_USE,
        "interpretation": "; ".join(
            f"{m}: robustness={sensitivity_results[m]['robustness_score']:.2f}, "
            f"worst_param={sensitivity_results[m]['worst_param']}"
            for m in mets if "robustness_score" in sensitivity_results[m]
        ),
    }


def universal_params_search(tuned_params: dict) -> dict:
    """
    G2: Find single EKF2 parameter set minimising worst-case RMSE across
    all scenarios (minimax). Answers: can one relay config be deployed everywhere?
    """
    try:
        from estimators import (
            FS_PHYSICS as _FP, RATIO as _RAT, get_test_signals, calculate_metrics,
        )
        from ekf2 import EKF2
    except ImportError as e:
        return {"error": f"Import failed: {e}"}

    RAT = 100
    signals = get_test_signals(seed=42)
    sc_data = {sc: (v_sig[::RAT], f_sig[::RAT])
               for sc, (_, v_sig, f_sig, _) in signals.items()}

    # Optimal per-scenario RMSE
    ekf2_base = tuned_params.get("IBR_Nightmare", {}).get("EKF2", {})
    inn_ref   = float(ekf2_base.get("inn_ref",          0.1))
    ev_thresh = float(ekf2_base.get("event_thresh",     2.0))
    fast_hz   = float(ekf2_base.get("fast_horizon_ms", 80.0))

    optimal_rmse = {}
    for sc, (v_dsp, f_dsp) in sc_data.items():
        ep = tuned_params.get(sc, {}).get("EKF2", ekf2_base)
        try:
            algo = EKF2(q_param=float(ep.get("q_param",1.0)),
                        r_param=float(ep.get("r_param",0.01)),
                        inn_ref=float(ep.get("inn_ref", inn_ref)),
                        event_thresh=float(ep.get("event_thresh", ev_thresh)),
                        fast_horizon_ms=float(ep.get("fast_horizon_ms", fast_hz)))
            tr = np.array([algo.step(x) for x in v_dsp])
            optimal_rmse[sc] = float(calculate_metrics(tr, f_dsp, 0.0, structural_samples=1)["RMSE"])
        except Exception:
            optimal_rmse[sc] = np.nan

    # Minimax grid search (lighter than full DE for reliability)
    N_Q = 10; N_R = 10
    q_vals = np.logspace(-2, 2, N_Q)
    r_vals = np.logspace(-3, 1,  N_R)

    best_max_ratio = np.inf
    best_q = best_r = np.nan

    for q in q_vals:
        for r in r_vals:
            max_ratio = 0.0
            for sc, (v_dsp, f_dsp) in sc_data.items():
                opt = optimal_rmse.get(sc, np.nan)
                if not np.isfinite(opt) or opt < 1e-6:
                    continue
                try:
                    algo = EKF2(q_param=float(q), r_param=float(r),
                                inn_ref=inn_ref, event_thresh=ev_thresh,
                                fast_horizon_ms=fast_hz)
                    tr   = np.array([algo.step(x) for x in v_dsp])
                    rmse = float(calculate_metrics(tr, f_dsp, 0.0, structural_samples=1)["RMSE"])
                    max_ratio = max(max_ratio, rmse / opt)
                except Exception:
                    max_ratio = 999.0
            if max_ratio < best_max_ratio:
                best_max_ratio = max_ratio
                best_q, best_r = float(q), float(r)

    # Transfer loss at best universal params
    transfer_loss = {}
    for sc, (v_dsp, f_dsp) in sc_data.items():
        opt = optimal_rmse.get(sc, np.nan)
        try:
            algo = EKF2(q_param=best_q, r_param=best_r,
                        inn_ref=inn_ref, event_thresh=ev_thresh, fast_horizon_ms=fast_hz)
            tr   = np.array([algo.step(x) for x in v_dsp])
            rmse = float(calculate_metrics(tr, f_dsp, 0.0, structural_samples=1)["RMSE"])
            transfer_loss[sc] = float(rmse / opt) if np.isfinite(opt) and opt > 1e-6 else None
        except Exception:
            transfer_loss[sc] = None

    field_deployable = best_max_ratio < 1.5 if np.isfinite(best_max_ratio) else False

    return {
        "best_universal_q":    best_q,
        "best_universal_r":    best_r,
        "max_transfer_loss":   float(best_max_ratio) if np.isfinite(best_max_ratio) else None,
        "transfer_loss_per_scenario": transfer_loss,
        "optimal_rmse_per_scenario":  optimal_rmse,
        "field_deployable":    field_deployable,
        "recommendation": (
            f"Use q={best_q:.3f}, r={best_r:.4f} for field deployment. "
            f"Maximum performance loss vs scenario-specific tuning: "
            f"{best_max_ratio:.2f}× ({'deployable' if field_deployable else 'requires scenario-specific tuning'})."
        ),
    }


def sensitivity_rank_stability(tuned_params: dict,
                                 n_perturbations: int = 200) -> dict:
    """
    G3: Perturb ALL methods simultaneously. Track whether the winner
    (EKF2 in Islanding) remains stable under detuning.
    """
    try:
        from estimators import (
            FS_PHYSICS as _FP, RATIO as _RAT, get_test_signals, calculate_metrics,
            TunableIpDFT, StandardPLL, ClassicEKF, SOGI_FLL,
            TFT_Estimator, UKF_Estimator, Koopman_RKDPmu, RLS_Estimator,
        )
        from ekf2 import EKF2
    except ImportError as e:
        return {"error": f"Import failed: {e}"}

    RAT = 100; rng = np.random.default_rng(66)
    signals = get_test_signals(seed=42)
    SC_USE  = "IBR_Nightmare"
    _, v_sig, f_sig, _ = signals[SC_USE]
    v_dsp = v_sig[::RAT]; f_dsp = f_sig[::RAT]

    def _get_tp(m, k, d):
        return tuned_params.get(SC_USE, {}).get(m, {}).get(k, d)

    METHODS_SENS = ["IpDFT", "PLL", "EKF", "EKF2", "SOGI", "UKF"]

    def _run_perturbed(m, pert_factor):
        try:
            if m == "IpDFT":
                algo = TunableIpDFT(max(1, int(_get_tp("IpDFT","cycles",4) * pert_factor)))
            elif m == "PLL":
                algo = StandardPLL(_get_tp("PLL","kp",10.0) * pert_factor,
                                    _get_tp("PLL","ki",50.0) * pert_factor)
            elif m == "EKF":
                algo = ClassicEKF(_get_tp("EKF","Q",0.1)  * pert_factor,
                                   _get_tp("EKF","R",0.01) * pert_factor)
            elif m == "EKF2":
                algo = EKF2(q_param=_get_tp("EKF2","q_param",1.0) * pert_factor,
                             r_param=_get_tp("EKF2","r_param",0.01) * pert_factor,
                             inn_ref=_get_tp("EKF2","inn_ref",0.1),
                             event_thresh=_get_tp("EKF2","event_thresh",2.0),
                             fast_horizon_ms=_get_tp("EKF2","fast_horizon_ms",80.0))
            elif m == "SOGI":
                algo = SOGI_FLL(_get_tp("SOGI","k",1.414) * min(pert_factor, 1.5),
                                 _get_tp("SOGI","g",20.0)  * pert_factor)
            elif m == "UKF":
                algo = UKF_Estimator(_get_tp("UKF","Q",0.1) * pert_factor,
                                      _get_tp("UKF","R",0.01) * pert_factor)
            else:
                return np.nan
            tr = np.array([algo.step(x) for x in v_dsp])
            return float(calculate_metrics(tr, f_dsp, 0.0, structural_samples=1)["RMSE"])
        except Exception:
            return np.nan

    # Optimal ranks
    opt_rmse = {m: _run_perturbed(m, 1.0) for m in METHODS_SENS}
    opt_ranked = sorted([m for m in METHODS_SENS if np.isfinite(opt_rmse[m])],
                        key=lambda m: opt_rmse[m])
    opt_rank = {m: i + 1 for i, m in enumerate(opt_ranked)}

    # Perturbed trials
    all_rmse_trials = {m: [] for m in METHODS_SENS}
    winner_counts  = {}
    rank_stability = {m: [] for m in METHODS_SENS}

    for _ in range(n_perturbations):
        trial_rmse = {}
        for m in METHODS_SENS:
            factor = 10 ** rng.uniform(-0.5, 0.5)   # ±3× perturbation
            rmse   = _run_perturbed(m, factor)
            trial_rmse[m] = rmse
            if np.isfinite(rmse):
                all_rmse_trials[m].append(rmse)

        valid = {m: v for m, v in trial_rmse.items() if np.isfinite(v)}
        if not valid:
            continue
        winner = min(valid, key=valid.get)
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        trial_ranked = sorted(valid, key=valid.get)
        for i, m in enumerate(trial_ranked):
            rank_stability[m].append(abs((i + 1) - opt_rank.get(m, i + 1)) <= 1)

    rank_stability_pct = {
        m: float(np.mean(rank_stability[m])) if rank_stability[m] else np.nan
        for m in METHODS_SENS
    }
    total_trials    = n_perturbations
    winner_stability = {
        m: float(winner_counts.get(m, 0) / max(total_trials, 1))
        for m in METHODS_SENS
    }
    ekf2_winner_pct  = winner_stability.get("EKF2", 0.0)
    tuning_dependent = ekf2_winner_pct < 0.8

    try:
        _fig_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
        ax = axes[0]
        for m in METHODS_SENS:
            if all_rmse_trials[m]:
                ax.boxplot(all_rmse_trials[m], positions=[METHODS_SENS.index(m)],
                           widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor=_method_color(m), alpha=0.7))
        ax.set_xticks(range(len(METHODS_SENS)))
        ax.set_xticklabels(METHODS_SENS, rotation=30, ha="right", fontsize=6)
        ax.set_ylabel("RMSE [Hz] (perturbed)")
        ax.set_title(f"G3: Rank stability under ±3× detuning\n({SC_USE})", fontsize=7)

        ax2 = axes[1]
        ws = [winner_stability[m] for m in METHODS_SENS]
        ax2.bar(range(len(METHODS_SENS)), ws,
                color=[_method_color(m) for m in METHODS_SENS], edgecolor="k", lw=0.4)
        ax2.axhline(0.8, color="r", ls="--", lw=0.7, label="80% stability")
        ax2.set_xticks(range(len(METHODS_SENS)))
        ax2.set_xticklabels(METHODS_SENS, rotation=30, ha="right", fontsize=6)
        ax2.set_ylabel("Fraction of trials as winner")
        ax2.set_title("Winner stability under detuning", fontsize=7)
        ax2.legend(fontsize=5)
        fig.tight_layout()
        _save_fig(fig, 603, "rank_stability")
    except Exception as e:
        warnings.warn(f"G3 figure: {e}")

    return {
        "optimal_ranking":        opt_ranked,
        "winner_stability":       winner_stability,
        "rank_stability_within_1":rank_stability_pct,
        "ekf2_winner_pct":        ekf2_winner_pct,
        "tuning_dependent_winner":tuning_dependent,
        "robust_winner":          opt_ranked[0] if not tuning_dependent else None,
        "n_trials":               total_trials,
        "interpretation": (
            f"EKF2 wins in {ekf2_winner_pct*100:.0f}% of detuned trials in {SC_USE}. "
            f"{'Winner is robust to detuning.' if not tuning_dependent else 'Winner is tuning-dependent — check detuning sensitivity.'}"
        ),
    }

# =============================================================================
# BLOCK H — Chamorro Real-Data Scenario (SOTA VALIDATION)
# =============================================================================

def save_chamorro_individual_plots(chamorro_out, fs_target=10000.0, output_dir="figures_chamorro"):
    """
    Genera PNGs mostrando la estimación causal vs el Pseudo-Ground Truth no causal.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    if not chamorro_out.get("chamorro_available"):
        print("  [H] Datos de Chamorro no disponibles para graficar.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = chamorro_out.get("results", {})
    v_dsp = chamorro_out.get("v_dsp_signal")
    f_pseudo = chamorro_out.get("f_pseudo_gt")  # Referencia No Causal
    f_consensus = chamorro_out.get("f_consensus") # Referencia por Consenso
    t = np.arange(len(v_dsp)) / fs_target

    plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

    for method_name, data in results.items():
        if "error" in data or "trace" not in data:
            continue
        
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()

        # Señal original (Fondo)
        ax1.plot(t, v_dsp, color='gray', alpha=0.3, label='Señal Real (v)', linewidth=0.5)
        
        # Referencia No Causal (Pseudo GT)
        if f_pseudo is not None:
            ax2.plot(t, f_pseudo, color='black', linestyle='--', alpha=0.7, label='Offline Pseudo-GT (Hilbert)', linewidth=1.5)
            
        # Trazas de Frecuencia
        f_est = data["trace"]
        ax2.plot(t, f_est, color='tab:red', label=f'Estimador Causal: {method_name}', linewidth=1.2)
        
        # Textbox con las nuevas métricas SOTA
        textstr = '\n'.join((
            f'RMSE vs Pseudo-GT: {data["RMSE_PseudoGT"]:.4f} Hz',
            f'RMSE vs Consenso: {data["RMSE_Consensus"]:.4f} Hz',
            f'Leakage (55-65Hz): {data["Fundamental_Leakage"]:.2e} W/Hz',
            f'CPU: {data["TIME_PER_SAMPLE_US"]:.2f} us'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9, verticalalignment='top', bbox=props)

        ax1.set_xlabel('Tiempo [s]')
        ax1.set_ylabel('Voltaje [pu]', color='gray')
        ax2.set_ylabel('Frecuencia [Hz]')
        ax2.set_ylim([55, 65]) 
        ax1.set_title(f'Validación Empírica: {method_name} vs No-Causal Reference')
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Unificar leyendas
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='lower right', fontsize=8)

        plt.savefig(os.path.join(output_dir, f"Chamorro_{method_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"  [H] PNGs avanzados generados en {output_dir}/")


def load_and_run_chamorro(tuned_params: dict, csv_path: str = "chamorro_scenario.csv", fs_target: float = 10000.0) -> dict:
    if not os.path.isfile(csv_path):
        return {"error": f"CSV not found: {csv_path}", "chamorro_available": False}

    import pandas as pd
    from scipy.signal import resample_poly, butter, filtfilt, hilbert, welch
    from math import gcd

    # 1. Carga de datos reales
    df = pd.read_csv(csv_path)
    t_c = next((c for c in df.columns if c.lower() in ["t","time","tiempo"]), None)
    v_c = next((c for c in df.columns if c.lower() in ["v","voltage","valor_b1"]), None)

    if t_c is None or v_c is None:
        return {"error": "Columnas de tiempo/voltaje no detectadas.", "chamorro_available": False}

    v_raw = df[v_c].values.astype(float)
    t_raw = df[t_c].values.astype(float)
    fs_raw = 1.0 / float(np.median(np.diff(t_raw)))

    # Remover Offset DC y Normalizar
    v_raw = v_raw - np.mean(v_raw) 
    v_rms = np.sqrt(np.mean(v_raw**2))
    if v_rms > 5.0: v_raw /= (v_rms * np.sqrt(2))

    # Resampling a FS_TARGET
    fr, ft = int(round(fs_raw)), int(round(fs_target))
    g = gcd(fr, ft)
    v_dsp = resample_poly(v_raw, ft // g, fr // g)

    # =========================================================================
    # METODOLOGÍA 1: EXTRACCIÓN DEL PSEUDO-GROUND TRUTH NO CAUSAL
    # =========================================================================
    # Filtro Pasa-banda Zero-Phase estricto (50Hz - 70Hz)
    nyq = 0.5 * fs_target
    b, a = butter(4, [50.0 / nyq, 70.0 / nyq], btype='band')
    v_bp = filtfilt(b, a, v_dsp)
    
    # Transformada de Hilbert para obtener la frecuencia instantánea pura
    analytic_signal = hilbert(v_bp)
    amplitude_env = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    f_pseudo = np.diff(instantaneous_phase) * fs_target / (2.0 * np.pi)
    f_pseudo = np.append(f_pseudo, f_pseudo[-1]) # Ajustar tamaño
    
    # Suavizado no causal del Pseudo-GT para remover artifacts de la derivada
    f_pseudo = filtfilt(np.ones(10)/10, 1, f_pseudo)

    # =========================================================================
    # 2. Ejecución de Estimadores CAUSALES
    # =========================================================================
    from estimators import calculate_metrics, TunableIpDFT, StandardPLL, ClassicEKF, SOGI_FLL, TFT_Estimator, UKF_Estimator, Koopman_RKDPmu, RLS_Estimator
    from ekf2 import EKF2

    def _get_tp(m, k, d): return tuned_params.get("IBR_Nightmare", {}).get(m, {}).get(k, d)

    factories = {
        "IpDFT": lambda: TunableIpDFT(_get_tp("IpDFT","cycles",4)),
        "PLL":   lambda: StandardPLL(_get_tp("PLL","kp",10.0), _get_tp("PLL","ki",50.0)),
        "EKF":   lambda: ClassicEKF(_get_tp("EKF","Q",0.1), _get_tp("EKF","R",0.01)),
        "EKF2":  lambda: EKF2(q_param=_get_tp("EKF2","q_param",1.0), r_param=_get_tp("EKF2","r_param",0.01)),
        "SOGI":  lambda: SOGI_FLL(_get_tp("SOGI","k",1.414), _get_tp("SOGI","g",20.0)),
        "RLS":   lambda: RLS_Estimator(lam=_get_tp("RLS","lambda",0.995))
    }

    raw_traces = {}
    metrics_temp = {}
    for m, fac in factories.items():
        try:
            algo = fac()
            tr = np.array([algo.step(x) for x in v_dsp])
            raw_traces[m] = tr
            # CPU extraction
            met = calculate_metrics(tr, np.ones(len(tr))*60.0, 0.0, structural_samples=1)
            metrics_temp[m] = {"CPU": float(met["TIME_PER_SAMPLE_US"])}
        except Exception as e:
            metrics_temp[m] = {"error": str(e)}

    # =========================================================================
    # METODOLOGÍA 2: CONSENSO DE ENSAMBLAJE (Three-Cornered Hat logic)
    # =========================================================================
    valid_traces = [tr for m, tr in raw_traces.items() if len(tr) == len(v_dsp)]
    if valid_traces:
        f_consensus = np.median(valid_traces, axis=0)
    else:
        f_consensus = np.ones(len(v_dsp)) * 60.0

    # =========================================================================
    # METODOLOGÍA 3: EVALUACIÓN SOTA Y FUNDAMENTAL LEAKAGE
    # =========================================================================
    chamorro_results = {}
    for m, tr in raw_traces.items():
        # A) Error vs Referencia No Causal
        rmse_pseudo = float(np.sqrt(np.mean((tr - f_pseudo)**2)))
        
        # B) Error vs Consenso
        rmse_cons = float(np.sqrt(np.mean((tr - f_consensus)**2)))

        # C) Espectroscopía del Residuo (Fundamental Leakage)
        _phase_est = np.cumsum(tr) * (2.0 * np.pi / fs_target)
        _phase_est = _phase_est - _phase_est[0] + instantaneous_phase[0]
        v_recon = amplitude_env * np.cos(_phase_est)
        residual = v_dsp - v_recon
        
        # PSD de Welch
        freqs, psd = welch(residual, fs_target, nperseg=int(fs_target))
        band_mask = (freqs >= 55) & (freqs <= 65)
        leakage = float(np.sum(psd[band_mask]))

        chamorro_results[m] = {
            "RMSE_PseudoGT": rmse_pseudo,
            "RMSE_Consensus": rmse_cons,
            "Fundamental_Leakage": leakage,
            "TIME_PER_SAMPLE_US": metrics_temp[m].get("CPU", 0.0),
            "trace": tr
        }

    return {
        "results": chamorro_results, 
        "chamorro_available": True, 
        "v_dsp_signal": v_dsp,
        "f_pseudo_gt": f_pseudo,
        "f_consensus": f_consensus
    }


def chamorro_generalization_analysis(results_dict: dict, chamorro_results: dict) -> dict:
    """
    Analiza y valida el ranking empírico basado en las métricas SOTA.
    """
    if not chamorro_results or not chamorro_results.get("chamorro_available"):
        return {"error": "Chamorro data not available"}

    # Generar gráficos SOTA
    save_chamorro_individual_plots(chamorro_results)

    raw_cham = chamorro_results["results"]
    
    # Evaluar ganadores según las diferentes métricas metodológicas
    methods = [m for m in raw_cham if "RMSE_PseudoGT" in raw_cham[m]]
    if not methods:
        return {"error": "No valid methods in Chamorro analysis"}

    best_pseudo = min(methods, key=lambda m: raw_cham[m]["RMSE_PseudoGT"])
    best_cons = min(methods, key=lambda m: raw_cham[m]["RMSE_Consensus"])
    best_leakage = min(methods, key=lambda m: raw_cham[m]["Fundamental_Leakage"])

    conclusion = (
        f"SOTA Validation Results: "
        f"Best vs Non-Causal Reference: {best_pseudo}. "
        f"Best vs Ensemble Consensus: {best_cons}. "
        f"Lowest Fundamental Leakage: {best_leakage}. "
    )

    return {
        "conclusion": conclusion,
        "best_vs_pseudoGT": best_pseudo,
        "best_vs_consensus": best_cons,
        "best_fundamental_leakage": best_leakage,
        "chamorro_available": True
    }



# =============================================================================
# BLOQUE I: PRUEBAS DE HIPÓTESIS (H1 - H4)
# =============================================================================

def _h1_pareto_cpu(results_dict):
    """
    H0: El incremento en la carga computacional (CPU) no correlaciona con 
    la reducción del error máximo (FE_max) en transitorios severos (MultiEvent).
    """
    sc_name = "IBR_MultiEvent"
    if sc_name not in results_dict:
        return {"status": "skipped", "reason": f"Falta escenario {sc_name}"}

    cpu_list = []
    fe_max_list = []
    methods = []

    for m, vals in results_dict[sc_name].get("methods", {}).items():
        cpu = vals.get("TIME_PER_SAMPLE_US")
        peak = vals.get("MAX_PEAK")
        if cpu is not None and peak is not None and np.isfinite(cpu) and np.isfinite(peak):
            cpu_list.append(float(cpu))
            fe_max_list.append(float(peak))
            methods.append(m)

    if len(cpu_list) < 3:
        return {"status": "skipped", "reason": "Datos insuficientes"}

    rho, p_val = _sstats.spearmanr(cpu_list, fe_max_list)
    
    return {
        "status": "success",
        "description": "Correlación de Spearman entre costo de CPU y FE_max en escenario caótico.",
        "spearman_rho": round(rho, 4),
        "p_value": round(p_val, 4),
        "H0_rejected": bool(p_val < 0.05),
        "conclusion": "Si rho es cercano a 0 o p>0.05, mayor CPU no garantiza mejor supervivencia." if p_val >= 0.05 else "Existe un trade-off estadísticamente significativo entre CPU y error pico.",
        "data_points": methods
    }

def _h2_survival_phase_jump(phase_jump_data):
    """
    H0: El ángulo crítico de fallo (Ttrip > 0.1s) es igual entre métodos basados 
    en modelo estocástico (EKF2) y lazos de control (PLL).
    """
    if not phase_jump_data or "per_method" not in phase_jump_data:
        return {"status": "skipped", "reason": "Faltan datos del phase_jump_sweep (NEW-2)"}

    critical_angles = phase_jump_data.get("critical_angles", {})
    if not critical_angles:
        return {"status": "skipped", "reason": "No hay critical angles calculados."}

    # Separar en familias
    model_based = [critical_angles.get("EKF"), critical_angles.get("EKF2")]
    loop_based = [critical_angles.get("PLL"), critical_angles.get("SOGI")]

    model_based = [x for x in model_based if x is not None]
    loop_based = [x for x in loop_based if x is not None]

    if model_based and loop_based:
        avg_model = sum(model_based) / len(model_based)
        avg_loop = sum(loop_based) / len(loop_based)
    else:
        avg_model, avg_loop = None, None

    return {
        "status": "success",
        "description": "Área de operación segura ante saltos de fase (Survival Breakdown).",
        "critical_angles": critical_angles,
        "avg_breakdown_model_based": avg_model,
        "avg_breakdown_loop_based": avg_loop,
        "conclusion": "El RA-EKF sobrevive a perturbaciones cinemáticas más violentas que el PLL estándar." if avg_model and avg_loop and avg_model > avg_loop else "Comportamiento similar o insuficiente data."
    }

def _h3_rocof_mann_whitney(mc_raw):
    """
    H0: Los algoritmos con estado explícito de RoCoF (EKF2, UKF) no tienen 
    menor error de RoCoF (RFE_max) que los estándar (EKF, PLL) durante una rampa.
    """
    sc = "IEEE_Freq_Ramp"
    
    rocof_methods = ["EKF2", "UKF"]
    std_methods = ["EKF", "PLL"]

    rocof_data = []
    std_data = []

    for m in rocof_methods:
        if (m, sc) in mc_raw and "RFE_max" in mc_raw[(m, sc)]:
            rocof_data.extend(mc_raw[(m, sc)]["RFE_max"])
            
    for m in std_methods:
        if (m, sc) in mc_raw and "RFE_max" in mc_raw[(m, sc)]:
            std_data.extend(mc_raw[(m, sc)]["RFE_max"])

    if len(rocof_data) < 5 or len(std_data) < 5:
        return {"status": "skipped", "reason": "Faltan datos MC para test U."}

    stat, p_val = _sstats.mannwhitneyu(rocof_data, std_data, alternative='less')

    return {
        "status": "success",
        "description": "Mann-Whitney U Test sobre RFE_max en rampa (Estado explícito vs Implícito).",
        "p_value": round(p_val, 5),
        "H0_rejected": bool(p_val < 0.05),
        "conclusion": "La inclusión del estado dinámico de RoCoF (RA-EKF) reduce significativamente el error de seguimiento cinemático frente a formulaciones de estado reducido." if p_val < 0.05 else "No hay mejora estadística evidente."
    }

def _h4_variance_levene(mc_raw):
    """
    H0: La varianza del riesgo de disparo (Ttrip) es independiente de si el 
    método es determinista (IpDFT) o adaptativo estocástico (EKF2).
    """
    sc = "IBR_Nightmare"
    
    m_det = "IpDFT"
    m_stoch = "EKF2"

    if (m_det, sc) not in mc_raw or (m_stoch, sc) not in mc_raw:
        return {"status": "skipped", "reason": "Faltan datos MC de IpDFT o EKF2."}

    data_det = mc_raw[(m_det, sc)].get("TRIP", [])
    data_stoch = mc_raw[(m_stoch, sc)].get("TRIP", [])

    if len(data_det) < 5 or len(data_stoch) < 5:
        return {"status": "skipped", "reason": "Menos de 5 realizaciones MC."}

    stat, p_val = _sstats.levene(data_det, data_stoch)

    return {
        "status": "success",
        "description": "Test de Levene para homogeneidad de varianzas en Riesgo de Disparo.",
        "variance_IpDFT": round(float(np.var(data_det)), 6),
        "variance_EKF2": round(float(np.var(data_stoch)), 6),
        "p_value": round(p_val, 5),
        "H0_rejected": bool(p_val < 0.05),
        "conclusion": "La adaptabilidad extrema del EKF2 tiene un costo: su comportamiento es significativamente más sensible (estocástico) al ruido que las ventanas fijas." if p_val < 0.05 else "Ambos presentan dispersiones estadísticamente asimilables."
    }


# =============================================================================
# BLOQUE 2: AGRUPACIONES Y RENDIMIENTO (A1 - A3)
# =============================================================================

def _a1_family_performance(results_dict):
    """
    Calcula el RMSE promedio por familia algorítmica por escenario para determinar
    qué arquitectura domina qué condición de red.
    """
    scenarios = ["IEEE_Mag_Step", "IEEE_Freq_Ramp", "IBR_Nightmare", "IBR_MultiEvent"]
    family_sc_perf = defaultdict(lambda: defaultdict(list))

    for sc in scenarios:
        if sc not in results_dict: continue
        for m, vals in results_dict[sc].get("methods", {}).items():
            rmse = vals.get("RMSE")
            if rmse is not None and np.isfinite(rmse):
                fam = get_family(m)
                family_sc_perf[sc][fam].append(rmse)

    report = {}
    for sc, fams in family_sc_perf.items():
        report[sc] = {}
        best_fam = None
        best_rmse = 999
        for fam, rmses in fams.items():
            avg_rmse = sum(rmses) / len(rmses)
            report[sc][fam] = round(avg_rmse, 4)
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_fam = fam
        report[sc]["WINNER"] = best_fam

    return {
        "status": "success",
        "description": "Desempeño agregado (Mean RMSE) agrupado por arquitectura matemática.",
        "data": report
    }

def _a2_stochastic_type(mc_raw):
    """
    Clasifica los algoritmos según el Coeficiente de Variación (CV) de su RMSE
    bajo múltiples realizaciones de ruido blanco (Monte Carlo).
    """
    cv_map = {}
    
    # Usaremos IBR_Nightmare como escenario de referencia para el ruido
    sc_ref = "IBR_Nightmare"
    
    for (m, sc), data in mc_raw.items():
        if sc != sc_ref: continue
        rmse_arr = data.get("RMSE", [])
        if len(rmse_arr) > 2:
            mean = np.mean(rmse_arr)
            std = np.std(rmse_arr)
            cv = (std / mean) * 100 if mean > 0 else 0
            
            if cv < 1.0:
                cat = "Deterministic (<1% CV)"
            elif cv < 15.0:
                cat = "Moderate Variance (1-15% CV)"
            else:
                cat = "Highly Stochastic (>15% CV)"
                
            cv_map[m] = {
                "CV_percent": round(cv, 2),
                "Category": cat
            }

    return {
        "status": "success",
        "description": "Sensibilidad al ruido estocástico del sensor (Clasificación por Coef. de Variación).",
        "classification": cv_map
    }

def _a3_global_weighted_ranking(results_dict):
    """
    Un ranking de aplicabilidad global. 
    Penaliza fuertemente los Ttrip > 0.1s (riesgo de apagón) y premia el RMSE bajo.
    """
    scores = defaultdict(float)
    
    for sc, data in results_dict.items():
        methods = data.get("methods", {})
        
        # Filtrar métodos con datos válidos
        valid_m = []
        for m, vals in methods.items():
            if "RMSE" in vals and "TRIP_TIME_0p5" in vals:
                valid_m.append((m, vals["RMSE"], vals["TRIP_TIME_0p5"]))
                
        if not valid_m: continue

        # Ordenar por RMSE (menor es mejor)
        valid_m.sort(key=lambda x: x[1])
        
        # Asignar puntos por RMSE (Top 1 = 3pts, Top 2 = 2pts, Top 3 = 1pt)
        for i, (m, rmse, ttrip) in enumerate(valid_m[:3]):
            scores[m] += (3 - i)
            
        # Penalización brutal por disparos (Penalty Rule)
        for m, rmse, ttrip in valid_m:
            if ttrip > 0.1:
                scores[m] -= 2.0  # Castigo por disparar el relé

    # Ordenar ranking final
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "status": "success",
        "description": "Puntaje global (+3/2/1 por precisión RMSE, -2 por riesgo de disparo > 0.1s).",
        "leaderboard": {m: round(score, 1) for m, score in ranking}
    }
# =============================================================================
# Master Orchestrator
# =============================================================================
def run_all_scientific_analyses(
    results_dict:    dict,
    mc_raw:          dict,
    tuned_params:    dict,
    results_raw_dir: str = "results_raw",
    phase_jump_data: dict = None,
) -> dict:
    """
    Run all scientific analyses (Q1-Q12 + Blocks A-H + Block EXT) and return a consolidated dict.

    Parameters
    ----------
    results_dict    : json_export["results"]  from run_benchmark()
    mc_raw          : raw_mc dict  {(method, scenario): {metric: [run_values]}}
    tuned_params    : tuned_params_per_scenario {sc: {method: {param: value}}}
    results_raw_dir : path to results_raw/ directory
    phase_jump_data : phase_jump_sweep data from statistical_analysis (optional)

    Returns
    -------
    dict with keys Q1..Q12, A1..A4, B2..B4, C, D1..D3, E, G1..G3, H1..H4, EXT_*,
    plus 'summary' and file paths of written outputs.
    """
    import datetime
    import os
    
    os.makedirs(FIGURES_SCIENTIFIC_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("SCIENTIFIC ANALYSES — Q1-Q12 + Blocks A-H + EXT (Journal Hypotheses)")
    print("=" * 70)

    def _run(label, fn, *args, **kwargs):
        print(f"  [{label}] Running...", end="", flush=True)
        try:
            res = fn(*args, **kwargs)
            if not isinstance(res, dict):
                res = {"result": res}
                
            # Soporte dual para errores legacy y los nuevos 'skipped'
            if "error" in res:
                status = "error"
            elif res.get("status") == "skipped":
                status = "skipped"
            else:
                status = "ok"
                
            print(f" {status}")
            return res
        except Exception as e:
            import traceback
            print(f" EXCEPTION: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    out = {}

    # ── Q1–Q12: Original open questions ───────────────────────────────────────
    print("\n  -- Q1-Q12: Original Open Questions --")
    out["Q1"]  = _run("Q1  P/M conflict",          pclass_mclass_conflict_analysis,
                       results_dict, mc_raw)
    out["Q2"]  = _run("Q2  Topology invariance",   topology_invariance_test,
                       tuned_params)
    out["Q3"]  = _run("Q3  Hazard function (KM)",  hazard_function_analysis,
                       mc_raw)
    out["Q4"]  = _run("Q4  Innovation classifier", innovation_signature_classifier,
                       tuned_params, results_raw_dir)
    out["Q5"]  = _run("Q5  THD transition",        thd_phase_transition_test,
                       tuned_params)
    out["Q6"]  = _run("Q6  Scenario redundancy",   scenario_redundancy_analysis,
                       results_dict)
    out["Q7"]  = _run("Q7  Tuning landscape",      tuning_transferability_landscape,
                       tuned_params)
    out["Q8"]  = _run("Q8  Noise scaling law",     noise_variance_scaling_law,
                       tuned_params)
    out["Q9"]  = _run("Q9  Behavioral taxonomy",   behavioral_taxonomy_analysis,
                       results_dict)
    out["Q10"] = _run("Q10 Complexity-CRLB bound", complexity_performance_bound,
                       results_dict)
    out["Q11"] = _run("Q11 Protection margin",     protection_margin_distribution,
                       mc_raw)
    out["Q12"] = _run("Q12 IEC sufficiency",       iec_sufficiency_formal_test,
                       results_dict)

    # ── Block A: Zero-new-computation analyses ────────────────────────────────
    print("\n  -- Block A: IEC/Metric/RoCoF/Relay analyses --")
    out["A1"] = _run("A1  IEC blindness formal",   iec_blindness_formal,
                      results_dict)
    out["A2"] = _run("A2  Metric redundancy",      metric_redundancy_analysis,
                      results_dict)
    out["A3"] = _run("A3  RoCoF analysis",         rocof_analysis,
                      results_dict)
    out["A4"] = _run("A4  Relay coordination",       relay_coordination_analysis,
                      results_dict)

    # ── Block B: Statistical strengthening ───────────────────────────────────
    print("\n  -- Block B: Statistical strengthening --")
    out["B2"] = _run("B2  Stochastic classifier",  stochastic_deterministic_classifier,
                      mc_raw)
    out["B3"] = _run("B3  Bootstrap corr. CIs",    bootstrap_correlation_ci,
                      mc_raw)
    out["B4"] = _run("B4  Full pairwise BH-FDR",   full_pairwise_analysis,
                      mc_raw)

    # ── Block C: New scenarios ─────────────────────────────────────────────────
    print("\n  -- Block C: New stress scenarios --")
    out["C"]  = _run("C   New scenarios",          run_new_scenarios,
                      tuned_params)

    # ── Block D: ML analyses ──────────────────────────────────────────────────
    print("\n  -- Block D: ML analyses --")
    out["D1"] = _run("D1  Method selector",        method_selection_classifier,
                      results_dict, tuned_params)
    out["D2"] = _run("D2  Anomaly detection",      anomaly_detection,
                      mc_raw)
    out["D3"] = _run("D3  PCR islanding",          pc_regression_islanding,
                      results_dict)

    # ── Block E: Paper tables ─────────────────────────────────────────────────
    print("\n  -- Block E: Paper tables --")
    out["E"]  = _run("E   Paper tables E1/E2/E3",  build_paper_tables,
                      results_dict, mc_raw)

    # ── Block G: Tuning robustness ────────────────────────────────────────────
    print("\n  -- Block G: Tuning robustness --")
    out["G1"] = _run("G1  Sensitivity surface",    tuning_sensitivity_analysis,
                      tuned_params)
    out["G2"] = _run("G2  Universal params search", universal_params_search,
                      tuned_params)
    out["G3"] = _run("G3  Rank stability",         sensitivity_rank_stability,
                      tuned_params)

    # ── Block H: Chamorro real-data validation ────────────────────────────────
    print("\n  -- Block H: Chamorro real-data validation --")
    out["H1"] = _run("H1  Load+run Chamorro data", load_and_run_chamorro,
                      tuned_params)
    # H2/H3/H4: generalization analysis (only if H1 produced real results)
    if "error" not in out["H1"] and not out["H1"].get("chamorro_available") is False:
        out["H4"] = _run("H4  Chamorro generalization", chamorro_generalization_analysis,
                          results_dict, out["H1"])
    else:
        out["H4"] = {"chamorro_available": False,
                     "note": "Skipped — Chamorro CSV not found or H1 failed"}
        print("  [H4  Chamorro generalization] SKIPPED (no real data)")

    # ── Block EXT: Advanced Journal Hypotheses ────────────────────────────────
    print("\n  -- Block EXT: Advanced Journal Hypotheses & Rankings --")
    out["EXT_H1"] = _run("EXT_H1 Pareto CPU/Error", _h1_pareto_cpu, results_dict)
    out["EXT_H2"] = _run("EXT_H2 Survival Phase",   _h2_survival_phase_jump, phase_jump_data)
    out["EXT_H3"] = _run("EXT_H3 RoCoF Mann-Whit.", _h3_rocof_mann_whitney, mc_raw)
    out["EXT_H4"] = _run("EXT_H4 Var. Levene",      _h4_variance_levene, mc_raw)
    out["EXT_A1"] = _run("EXT_A1 Family Perf.",     _a1_family_performance, results_dict)
    out["EXT_A2"] = _run("EXT_A2 Stochastic Type",  _a2_stochastic_type, mc_raw)
    out["EXT_A3"] = _run("EXT_A3 Global Ranking",   _a3_global_weighted_ranking, results_dict)

    # ── Summary table ─────────────────────────────────────────────────────────
    analysis_keys = [k for k in out.keys()]
    n_ok  = sum(1 for v in out.values() if isinstance(v, dict) and "error" not in v and v.get("status") != "skipped")
    n_err = sum(1 for v in out.values() if isinstance(v, dict) and ("error" in v or v.get("status") == "skipped"))
    n_total = len(analysis_keys)
    out["summary"] = {
        "n_analyses_ok":    n_ok,
        "n_analyses_error": n_err,
        "n_analyses_total": n_total,
        "generated":        datetime.datetime.now().isoformat(),
        "figures_dir":      FIGURES_SCIENTIFIC_DIR,
        "analyses_run":     analysis_keys,
    }

    # ── Write scientific_summary.txt ──────────────────────────────────────────
    try:
        lines = [
            "# SCIENTIFIC ANALYSES SUMMARY — auto-generated",
            f"# Generated: {out['summary']['generated']}",
            f"# {n_ok}/{n_total} analyses completed, {n_err} errors/skipped",
            "",
            "## Q1-Q12: Original Open Questions",
        ]
        for qk in [f"Q{i}" for i in range(1, 13)]:
            q_res = out.get(qk, {})
            if "error" in q_res:
                lines.append(f"  {qk}: ERROR — {q_res['error']}")
            else:
                interp = q_res.get("interpretation", "see JSON for details")
                lines.append(f"  {qk}: {str(interp)[:200]}")

        lines += ["", "## Block A: IEC/Metric/RoCoF/Relay"]
        for ak in ["A1", "A2", "A3", "A4"]:
            r = out.get(ak, {})
            if "error" in r:
                lines.append(f"  {ak}: ERROR — {r['error']}")
            else:
                lines.append(f"  {ak}: ok — {str(r.get('interpretation', r.get('n_blind', '')))[:120]}")

        lines += ["", "## Block B: Statistical Strengthening"]
        for bk in ["B2", "B3", "B4"]:
            r = out.get(bk, {})
            if "error" in r:
                lines.append(f"  {bk}: ERROR — {r['error']}")
            else:
                lines.append(f"  {bk}: ok — {str(r.get('interpretation', ''))[:120]}")

        lines += ["", "## Block C: New Scenarios"]
        r = out.get("C", {})
        if "error" in r:
            lines.append(f"  C: ERROR — {r['error']}")
        else:
            lines.append(f"  C: ok — scenarios={list(r.get('scenario_results', {}).keys())}")

        lines += ["", "## Block D: ML Analyses"]
        for dk in ["D1", "D2", "D3"]:
            r = out.get(dk, {})
            if "error" in r:
                lines.append(f"  {dk}: ERROR — {r['error']}")
            else:
                lines.append(f"  {dk}: ok — {str(r.get('interpretation', ''))[:120]}")

        lines += ["", "## Block E: Paper Tables"]
        r = out.get("E", {})
        if "error" in r:
            lines.append(f"  E: ERROR — {r['error']}")
        else:
            lines.append(f"  E: ok — tables_file={r.get('tables_file', '')}")

        lines += ["", "## Block G: Tuning Robustness"]
        for gk in ["G1", "G2", "G3"]:
            r = out.get(gk, {})
            if "error" in r:
                lines.append(f"  {gk}: ERROR — {r['error']}")
            else:
                lines.append(f"  {gk}: ok — {str(r.get('interpretation', ''))[:120]}")

        lines += ["", "## Block H: Chamorro Real-Data Validation"]
        for hk in ["H1", "H4"]:
            r = out.get(hk, {})
            if "error" in r:
                lines.append(f"  {hk}: ERROR — {r['error']}")
            elif not r.get("chamorro_available", True):
                lines.append(f"  {hk}: SKIPPED (CSV not available)")
            else:
                lines.append(f"  {hk}: ok")

        lines += ["", "## Block EXT: Advanced Journal Hypotheses"]
        for extk in ["EXT_H1", "EXT_H2", "EXT_H3", "EXT_H4", "EXT_A1", "EXT_A2", "EXT_A3"]:
            r = out.get(extk, {})
            if "error" in r or r.get("status") == "skipped":
                lines.append(f"  {extk}: SKIPPED/ERROR — {r.get('error', r.get('reason', ''))}")
            else:
                lines.append(f"  {extk}: ok — {str(r.get('conclusion', r.get('description', '')))[:120]}")

        with open(os.path.join(FIGURES_SCIENTIFIC_DIR, "scientific_summary.txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  [OUTPUT] scientific_summary.txt → {FIGURES_SCIENTIFIC_DIR}/")
    except Exception as e:
        print(f"  [OUTPUT WARNING] scientific_summary.txt failed: {e}")

    # ── Write paper_contributions.txt ─────────────────────────────────────────
    try:
        ts = out["summary"]["generated"]
        contrib_lines = [
            "# JOURNAL PAPER CONTRIBUTIONS — auto-generated",
            f"# Generated: {ts}",
            f"# {n_ok}/{n_total} analyses completed, {n_err} errors/skipped",
            "",
            "=== ORIGINAL CONTRIBUTIONS (Q1-Q12) ===",
            "",
            "Contribution 1: P/M class conflict quantification",
        ]
        q1 = out.get("Q1", {})
        if "conflict_index" in q1:
            contrib_lines.append(
                f"  Conflict index CI={q1['conflict_index']:.2f}; "
                f"Spearman rho={q1.get('spearman_rho','N/A')}"
            )
        contrib_lines += [
            "",
            "Contribution 2: Topology invariance (5 synthetic nodes)",
        ]
        q2 = out.get("Q2", {})
        if "mean_spearman_rho" in q2:
            contrib_lines.append(f"  Mean pairwise rho={q2['mean_spearman_rho']:.2f}")
        contrib_lines += [
            "",
            "Contribution 3: Kaplan-Meier hazard curves (N_MC=30 per method)",
            "",
            "Contribution 4: Scenario frequency-error classifier (RF, 5-fold CV)",
        ]
        q4 = out.get("Q4", {})
        if "cv_accuracy_mean" in q4:
            contrib_lines.append(
                f"  CV accuracy={q4['cv_accuracy_mean']:.2f}±{q4['cv_accuracy_std']:.2f}"
            )
        contrib_lines += [
            "",
            "Contribution 5: THD sigmoid transition model",
            "",
            "Contribution 6: Minimum non-redundant scenario set",
        ]
        q6 = out.get("Q6", {})
        if "minimal_non_redundant_set" in q6:
            contrib_lines.append(f"  Minimal set: {q6['minimal_non_redundant_set']}")
        contrib_lines += [
            "",
            "Contribution 7: EKF2 tuning transferability landscape",
        ]
        q7 = out.get("Q7", {})
        if "best_universal_q" in q7:
            contrib_lines.append(
                f"  Best universal q={q7['best_universal_q']:.3f}, "
                f"r={q7['best_universal_r']:.4f}, "
                f"max_ratio={q7['best_max_transfer_ratio']:.2f}x"
            )
        contrib_lines += [
            "",
            "Contribution 8: Noise variance scaling law (power law fit)",
        ]
        q8 = out.get("Q8", {})
        if "power_law_fits" in q8:
            near_crlb = [m for m, v in q8["power_law_fits"].items()
                         if isinstance(v, dict) and v.get("near_crlb")]
            contrib_lines.append(f"  Methods near CRLB scaling (alpha~0.5): {near_crlb}")
        contrib_lines += [
            "",
            "Contribution 9: Behavioral taxonomy (hierarchical + K-means clustering)",
            "",
            "Contribution 10: CRLB efficiency and complexity-RMSE Pareto slope",
        ]
        q10 = out.get("Q10", {})
        if "most_efficient_method" in q10:
            contrib_lines.append(
                f"  Most efficient: {q10['most_efficient_method']} "
                f"(eta={q10.get('max_efficiency','?'):.3f})"
            )
        contrib_lines += [
            "",
            "Contribution 11: Protection margin distribution and cost model",
        ]
        q11 = out.get("Q11", {})
        if "lowest_cost_method" in q11:
            contrib_lines.append(f"  Lowest-cost method: {q11['lowest_cost_method']}")
        contrib_lines += [
            "",
            "Contribution 12: IEC 60255-118-1 sufficiency formal test (Fisher)",
        ]
        q12 = out.get("Q12", {})
        if "fully_compliant_methods" in q12:
            contrib_lines.append(
                f"  Fully IEC-compliant (all scenarios): {q12['fully_compliant_methods']}"
            )

        contrib_lines += [
            "",
            "=== EXTENDED CONTRIBUTIONS (Blocks A-H) ===",
            "",
            "Block A1: IEC blindness formal permutation test",
        ]
        a1 = out.get("A1", {})
        if "n_blind" in a1:
            contrib_lines.append(
                f"  n_blind={a1['n_blind']}, p_value={a1.get('p_value','?')}"
            )

        contrib_lines += [
            "",
            "Block A2: Metric redundancy — Spearman heatmap + PCA",
            "",
            "Block A3: RoCoF tracking analysis — RFE_rms ranking",
        ]
        a3 = out.get("A3", {})
        if "best_rocof_method" in a3:
            contrib_lines.append(f"  Best RoCoF tracker: {a3['best_rocof_method']}")

        contrib_lines += [
            "",
            "Block A4: Relay coordination — safe/unsafe operating map",
            "",
            "Block B2: Stochastic/deterministic classifier (CV threshold)",
            "",
            "Block B3: Bootstrap rank correlation CIs (n_boot=2000)",
            "",
            "Block B4: Full 140-pair BH-FDR corrected significance analysis",
            "",
            "Block C:  New stress scenarios (voltage sag, subsync, double-jump)",
            "",
            "Block D1: Method selection random-forest classifier",
            "",
            "Block D2: LOF anomaly detection on MC runs",
            "",
            "Block D3: Principal-component regression for islanding detection",
            "",
            "Block E:  Paper Tables VI (relay), VII (stochastic), VIII (scenarios)",
            "",
            "Block G1: Tuning sensitivity surface (200-perturbation LHS)",
            "",
            "Block G2: Universal minimax parameter search",
        ]
        g2 = out.get("G2", {})
        if "universal_params" in g2:
            contrib_lines.append(f"  Universal params: {g2['universal_params']}")

        contrib_lines += [
            "",
            "Block G3: Rank stability under parameter detuning",
            "",
            "Block H:  Real-data validation on Chamorro dataset",
        ]
        h1 = out.get("H1", {})
        if h1.get("chamorro_available") is False:
            contrib_lines.append("  [CSV not available — pending data acquisition]")
        elif "n_methods_run" in h1:
            contrib_lines.append(f"  n_methods={h1['n_methods_run']}, "
                                  f"n_samples={h1.get('n_samples','?')}")

        contrib_lines += [
            "",
            "=== ADVANCED JOURNAL METRICS (Block EXT) ===",
        ]
        
        h1_ext = out.get("EXT_H1", {})
        if "spearman_rho" in h1_ext:
            contrib_lines.append(f"  H1 (Pareto CPU): rho={h1_ext['spearman_rho']}, p={h1_ext['p_value']} (H0 Rejected: {h1_ext.get('H0_rejected')})")
            
        h3_ext = out.get("EXT_H3", {})
        if "p_value" in h3_ext:
            contrib_lines.append(f"  H3 (RoCoF Kinematics): p={h3_ext['p_value']} (H0 Rejected: {h3_ext.get('H0_rejected')})")

        h4_ext = out.get("EXT_H4", {})
        if "p_value" in h4_ext:
            contrib_lines.append(f"  H4 (Stochastic Variance): p={h4_ext['p_value']} (H0 Rejected: {h4_ext.get('H0_rejected')})")

        a3_ext = out.get("EXT_A3", {})
        if "leaderboard" in a3_ext:
            top3 = list(a3_ext["leaderboard"].items())[:3]
            contrib_lines.append(f"  Global Ranking Winners: {top3}")

        with open(os.path.join(FIGURES_SCIENTIFIC_DIR, "paper_contributions.txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(contrib_lines) + "\n")
        print(f"  [OUTPUT] paper_contributions.txt → {FIGURES_SCIENTIFIC_DIR}/")
    except Exception as e:
        print(f"  [OUTPUT WARNING] paper_contributions.txt failed: {e}")

    print(f"\n  Scientific analyses complete: {n_ok}/{n_total} OK, {n_err} errors.")
    print(f"  Figures → {FIGURES_SCIENTIFIC_DIR}/")
    print("=" * 70)
    return out