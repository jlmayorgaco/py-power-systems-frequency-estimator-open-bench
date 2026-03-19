#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANDES Q1 Story (A/B/C/D) — Deterministic “Hidden Margin” (robust) + MEGA PLOT
============================================================================

What this script does (deterministic, no MC):
A) PRE-STRESS: hs=1.0, no fault
B) POST-STRESS (still "quiet"): hs=hsB, no fault, hsB chosen by PF constraint Vmin>=vmin_quiet_pu
C) STRESS + FAULT: hs=hsC (first failing just above boundary) with fixed 3φ fault [t_fault, t_clear]
D) POST-FAULT / REPAIR: same as C but with inertia bump on a chosen generator (search); if no rescue,
   returns best-improvement (min score)

Fixes included:
(1) Degenerate boundary hs_star_safe == hsC_fail:
    - If hsB already fails under the fixed fault, we do a DOWNSEARCH to find a safe hs_safe0 first,
      then bracket upward and bisection works.

(2) “I don’t see the fault scenario”:
    - All panels mark the fault window [t_fault, t_clear] (vlines + shaded span)
    - Rotor angle sep plot includes δ_crit line.

(3) MEGA PLOT requested:
    - 4 rows (A,B,C,D) × 6 cols:
        [Vbus, Vmin_sys, df, rocof, freq_summary, delta_sep]
    - freq_summary cell shows Δf(t) + lines (nadir/peak) + text metrics

CRITICAL ANDES RULE:
- No ss.add() after ss.setup(). Fault must be added BEFORE setup and tracked by actual idx.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import andes
from andes.utils.paths import get_case


# ============================================================
# Plot style
# ============================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "figure.dpi": 200,
    }
)


# ============================================================
# Utilities
# ============================================================
def _try_alter(model, src: str, idx, value: float) -> bool:
    try:
        model.alter(src=src, idx=idx, value=float(value))
        return True
    except Exception:
        return False


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.array(x).squeeze())


def _ensure_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape((-1, 1))
    return a


# ============================================================
# Event hygiene
# ============================================================
def disable_nonfault_events(ss, tf: float):
    """Push non-fault embedded events far away."""
    future_t = float(tf) + 1e6
    time_fields = ["t", "tf", "tc", "t1", "t2"]
    for ev_name in ["Toggle", "Breaker", "Trip", "ShuntSwitch"]:
        if not hasattr(ss, ev_name):
            continue
        ev = getattr(ss, ev_name)
        if getattr(ev, "n", 0) <= 0:
            continue
        try:
            ids = list(ev.idx.v)
        except Exception:
            continue
        for idx in ids:
            for f in time_fields:
                _try_alter(ev, f, idx, future_t)


def push_faults_away_except(ss, tf: float, keep_fault_idx: Optional[str]):
    """Push ALL Fault events away except keep_fault_idx (string)."""
    if not hasattr(ss, "Fault"):
        return
    ev = ss.Fault
    if getattr(ev, "n", 0) <= 0:
        return

    future_t = float(tf) + 1e6
    time_fields = ["t", "tf", "tc", "t1", "t2"]

    try:
        ids = [str(x) for x in list(ev.idx.v)]
    except Exception:
        return

    for idx in ids:
        if (keep_fault_idx is not None) and (str(idx) == str(keep_fault_idx)):
            continue
        for f in time_fields:
            _try_alter(ev, f, idx, future_t)


# ============================================================
# Load scaling (global stress dial hs)
# ============================================================
def scale_all_pq_loads(ss, factor: float):
    if not hasattr(ss, "PQ") or getattr(ss.PQ, "n", 0) == 0:
        raise RuntimeError("No PQ loads found in this case.")
    pq_ids = list(ss.PQ.idx.v)

    def _vec(src):
        try:
            return np.array(ss.PQ.get(src, pq_ids, attr="vin"), dtype=float)
        except Exception:
            return np.array(ss.PQ.get(src, pq_ids, attr="v"), dtype=float)

    p0 = _vec("p0")
    q0 = _vec("q0")

    for i, dev in enumerate(pq_ids):
        _try_alter(ss.PQ, "p0", dev, p0[i] * factor)
        _try_alter(ss.PQ, "q0", dev, q0[i] * factor)
        # best-effort PF/TDS fields
        _try_alter(ss.PQ, "Ppf", dev, p0[i] * factor)
        _try_alter(ss.PQ, "Qpf", dev, q0[i] * factor)


def pf_vmin_at_pq(ss) -> Tuple[float, Optional[int]]:
    pq_ids = list(ss.PQ.idx.v) if hasattr(ss, "PQ") and ss.PQ.n > 0 else []
    if not pq_ids:
        return np.nan, None
    v_list, bus_list = [], []
    for dev in pq_ids:
        bus = int(ss.PQ.get("bus", dev, attr="v"))
        v = float(ss.Bus.get("v", bus, attr="v"))
        v_list.append(v)
        bus_list.append(bus)
    k = int(np.argmin(v_list))
    return float(np.min(v_list)), int(bus_list[k])


# ============================================================
# Rotor model detection
# ============================================================
def find_rotor_models(ss) -> List[Tuple[str, Any, List[Any]]]:
    out: List[Tuple[str, Any, List[Any]]] = []
    for name in dir(ss):
        if not name.isupper() or name.startswith("_"):
            continue
        obj = getattr(ss, name, None)
        if obj is None or not hasattr(obj, "n"):
            continue
        try:
            if int(obj.n) <= 0:
                continue
        except Exception:
            continue
        if hasattr(obj, "delta") and hasattr(obj, "omega"):
            try:
                ids = list(obj.idx.v)
                if ids:
                    out.append((name, obj, ids))
            except Exception:
                pass
    return out


def pick_rotor_model_prefer_genrou(ss) -> Tuple[str, Any, List[Any]]:
    models = find_rotor_models(ss)
    if not models:
        raise RuntimeError("No rotor model with (delta, omega) found.")
    models_sorted = sorted(models, key=lambda x: (x[0] != "GENROU", x[0]))
    return models_sorted[0]


def detect_inertia_param(model_obj) -> str:
    for p in ["M", "H"]:
        try:
            _ = model_obj.get(p, model_obj.idx.v[0], attr="v")
            return p
        except Exception:
            pass
    for p in ["M", "H"]:
        try:
            _ = model_obj.get(p, model_obj.idx.v[0], attr="vin")
            return p
        except Exception:
            pass
    raise RuntimeError("Could not find inertia parameter (M or H).")


def match_gen_id(existing_ids: List[Any], hint: Any) -> Any:
    if hint is None:
        return existing_ids[0]
    s_hint = str(hint)
    for x in existing_ids:
        if str(x) == s_hint:
            return x
    for x in existing_ids:
        sx = str(x)
        if sx.endswith(s_hint) or (s_hint in sx):
            return x
    # numeric fallback
    try:
        i_hint = int(float(s_hint))
        for x in existing_ids:
            sx = str(x).replace("GENROU_", "").replace("GENCLS_", "").strip()
            try:
                if int(float(sx)) == i_hint:
                    return x
            except Exception:
                continue
    except Exception:
        pass
    return existing_ids[0]


def bump_inertia(
    ss, rotor_model_name: str, inertia_param: str, gen_id: Any, factor: float
):
    if abs(factor - 1.0) < 1e-12:
        return
    model = getattr(ss, rotor_model_name)
    try:
        x0 = _safe_float(model.get(inertia_param, gen_id, attr="v"))
    except Exception:
        x0 = _safe_float(model.get(inertia_param, gen_id, attr="vin"))
    if not _try_alter(model, inertia_param, gen_id, x0 * factor):
        raise RuntimeError(
            f"Failed to alter {rotor_model_name}.{inertia_param} for gen {gen_id}"
        )


# ============================================================
# Fault injection (pre-setup) — robustly track the real idx
# ============================================================
def add_bus_fault_clear_pre_setup(
    ss, bus: int, t_fault: float, t_clear: float, desired_idx: str
) -> str:
    """
    Add a 3φ fault BEFORE setup. Different ANDES versions accept different time fields.
    We try multiple dicts. If idx is rejected, add without idx and capture the created idx.
    Returns the actual fault idx in the model.
    """
    before_ids: List[str] = []
    if hasattr(ss, "Fault") and getattr(ss.Fault, "n", 0) > 0:
        try:
            before_ids = [str(x) for x in list(ss.Fault.idx.v)]
        except Exception:
            before_ids = []

    trials = [
        dict(idx=desired_idx, bus=bus, tf=float(t_fault), tc=float(t_clear), u=1),
        dict(idx=desired_idx, bus=bus, t1=float(t_fault), t2=float(t_clear), u=1),
        dict(idx=desired_idx, bus=bus, tf=float(t_fault), t1=float(t_clear), u=1),
        dict(idx=desired_idx, bus=bus, t=float(t_fault), tf=float(t_clear), u=1),
        dict(idx=desired_idx, bus=bus, t=float(t_fault), tc=float(t_clear), u=1),
    ]
    trials_noidx = [
        dict(bus=bus, tf=float(t_fault), tc=float(t_clear), u=1),
        dict(bus=bus, t1=float(t_fault), t2=float(t_clear), u=1),
        dict(bus=bus, tf=float(t_fault), t1=float(t_clear), u=1),
        dict(bus=bus, t=float(t_fault), tf=float(t_clear), u=1),
        dict(bus=bus, t=float(t_fault), tc=float(t_clear), u=1),
    ]

    last = None
    for d in trials:
        try:
            ss.add("Fault", d)
            if hasattr(ss, "Fault") and getattr(ss.Fault, "n", 0) > 0:
                ids = [str(x) for x in list(ss.Fault.idx.v)]
                if desired_idx in ids:
                    return desired_idx
                new_ids = [x for x in ids if x not in before_ids]
                if new_ids:
                    return str(new_ids[-1])
                return str(ids[-1])
            return desired_idx
        except Exception as e:
            last = e

    for d in trials_noidx:
        try:
            ss.add("Fault", d)
            if hasattr(ss, "Fault") and getattr(ss.Fault, "n", 0) > 0:
                ids = [str(x) for x in list(ss.Fault.idx.v)]
                new_ids = [x for x in ids if x not in before_ids]
                if new_ids:
                    return str(new_ids[-1])
                return str(ids[-1])
        except Exception as e:
            last = e

    raise RuntimeError(
        f"Could not add fault+clear pre-setup at bus={bus}. Last error: {last}"
    )


# ============================================================
# Time-series extraction + failure
# ============================================================
def extract_ts(
    ss,
    rotor_model_name: str,
    gen_ids_run: List[Any],
    ref_gen_run: Any,
    f0_hz: float,
    plot_bus: int,
) -> Dict[str, Any]:

    t = np.array(ss.dae.ts.t, dtype=float)

    # |V| at selected bus
    vbus = np.array(ss.dae.ts.get_data(ss.Bus.v, a=[plot_bus]).squeeze(), dtype=float)
    if vbus.ndim != 1:
        vbus = vbus.reshape((-1,))

    # Vmin(t) across all buses (system-wide); fallback to vbus if needed
    try:
        v_all = np.array(ss.dae.ts.get_data(ss.Bus.v), dtype=float)
        v_all = _ensure_2d(v_all)  # (T, Nbus)
        vmin_ts = np.min(v_all, axis=1)  # (T,)
    except Exception:
        vmin_ts = np.copy(vbus)

    # rotor model
    model = getattr(ss, rotor_model_name)
    omega_all = _ensure_2d(np.array(ss.dae.ts.get_data(model.omega), dtype=float))
    delta_all = _ensure_2d(np.array(ss.dae.ts.get_data(model.delta), dtype=float))

    idx_map = {str(g): k for k, g in enumerate(gen_ids_run)}
    col = idx_map.get(str(ref_gen_run), 0)
    col = int(np.clip(col, 0, omega_all.shape[1] - 1))

    omega = omega_all[:, col]
    df_hz = (omega - 1.0) * float(f0_hz)
    rocof = np.gradient(df_hz, t, edge_order=1) if t.size > 1 else np.zeros_like(df_hz)

    deltas_deg = np.rad2deg(delta_all)
    delta_sep = np.max(deltas_deg, axis=1) - np.min(deltas_deg, axis=1)

    # stress proxy
    stress = (
        float(1.0 / max(1e-6, float(np.min(vmin_ts)))) if vmin_ts.size else float("inf")
    )

    # frequency summary metrics
    df_nadir = float(np.min(df_hz)) if df_hz.size else 0.0
    df_peak = float(np.max(df_hz)) if df_hz.size else 0.0
    iae_df = float(np.trapz(np.abs(df_hz), t)) if t.size > 1 else 0.0
    iae_rocof = float(np.trapz(np.abs(rocof), t)) if t.size > 1 else 0.0

    return dict(
        t=t,
        vbus=vbus,
        vmin_ts=vmin_ts,
        df_hz=df_hz,
        rocof_hz_s=rocof,
        delta_sep_deg=delta_sep,
        stress=stress,
        df_nadir=df_nadir,
        df_peak=df_peak,
        iae_df=iae_df,
        iae_rocof=iae_rocof,
    )


def scenario_failed(X: Dict[str, Any], tf: float, delta_fail_deg: float) -> bool:
    t = np.asarray(X["t"], dtype=float)
    t_end = float(t[-1]) if t.size else -np.inf
    early_stop = (t_end < tf - 1e-6) or (not bool(X.get("ok_tds", True)))
    max_dd = (
        float(np.max(X["delta_sep_deg"]))
        if np.asarray(X["delta_sep_deg"]).size
        else float("inf")
    )
    return bool(early_stop or (max_dd > delta_fail_deg))


# ============================================================
# Run case (ANDES-safe)
# ============================================================
def run_case(
    casefile: str,
    tf: float,
    hs: float,
    do_fault: bool,
    fault_bus: int,
    t_fault: float,
    t_clear: float,
    inertia_bump: float,
    rotor_model_hint: str,
    ref_gen_hint: Any,
    bump_gen_hint: Any,
    f0_hz: float,
    plot_bus: int,
    disable_events: bool = True,
) -> Dict[str, Any]:

    ss = andes.load(casefile, setup=False)

    injected_fault_idx = None
    if do_fault:
        injected_fault_idx = add_bus_fault_clear_pre_setup(
            ss, bus=fault_bus, t_fault=t_fault, t_clear=t_clear, desired_idx="Q1FAULT"
        )

    ss.setup()

    if disable_events:
        disable_nonfault_events(ss, tf=tf)
        push_faults_away_except(ss, tf=tf, keep_fault_idx=injected_fault_idx)

    # apply stress BEFORE PF
    scale_all_pq_loads(ss, float(hs))
    ok_pf = bool(ss.PFlow.run())

    # rotor model per run
    rotor_model_name, rotor_model_obj, gen_ids_run = pick_rotor_model_prefer_genrou(ss)
    if rotor_model_hint and hasattr(ss, rotor_model_hint):
        cand = getattr(ss, rotor_model_hint)
        try:
            if (
                hasattr(cand, "delta")
                and hasattr(cand, "omega")
                and int(getattr(cand, "n", 0)) > 0
            ):
                rotor_model_name = rotor_model_hint
                rotor_model_obj = cand
                gen_ids_run = list(cand.idx.v)
        except Exception:
            pass

    inertia_param = detect_inertia_param(rotor_model_obj)
    ref_gen_run = match_gen_id(gen_ids_run, ref_gen_hint)
    bump_gen_run = (
        match_gen_id(gen_ids_run, bump_gen_hint)
        if bump_gen_hint is not None
        else ref_gen_run
    )

    if abs(inertia_bump - 1.0) > 1e-12:
        bump_inertia(
            ss, rotor_model_name, inertia_param, bump_gen_run, float(inertia_bump)
        )

    ss.TDS.config.tf = float(tf)
    try:
        ss.TDS.config.criteria = 0
    except Exception:
        pass

    ok_tds = bool(ss.TDS.run())

    ts = extract_ts(ss, rotor_model_name, gen_ids_run, ref_gen_run, f0_hz, plot_bus)
    ts.update(
        dict(
            ok_pf=ok_pf,
            ok_tds=ok_tds,
            hs=float(hs),
            inertia_bump=float(inertia_bump),
            bump_gen=str(bump_gen_run),
            fault=bool(do_fault),
            fault_bus=int(fault_bus),
            t_fault=float(t_fault),
            t_clear=float(t_clear),
            plot_bus=int(plot_bus),
            tf=float(tf),
            rotor_model=str(rotor_model_name),
            inertia_param=str(inertia_param),
            gen_ids=[str(x) for x in gen_ids_run],
            ref_gen=str(ref_gen_run),
            injected_fault_idx=(
                str(injected_fault_idx) if injected_fault_idx is not None else None
            ),
        )
    )
    return ts


# ============================================================
# Weak generator rule (placeholder deterministic)
# ============================================================
def pick_weak_gen_from_ts(C: Dict[str, Any]) -> str:
    return str(C.get("ref_gen", "1"))


# ============================================================
# Plotting helpers
# ============================================================
def _auto_ylim(y: np.ndarray, pad_frac: float = 0.15):
    y = np.asarray(y, dtype=float)
    lo = float(np.nanpercentile(y, 1))
    hi = float(np.nanpercentile(y, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    if abs(hi - lo) < 1e-12:
        lo -= 1.0
        hi += 1.0
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def _shade_fault(ax, X: Dict[str, Any]):
    if bool(X.get("fault", False)):
        tfault = float(X.get("t_fault", 0.0))
        tclear = float(X.get("t_clear", 0.0))
        ax.axvline(tfault, linewidth=0.9, alpha=0.35)
        ax.axvline(tclear, linewidth=0.9, alpha=0.35)
        ax.axvspan(tfault, tclear, alpha=0.12)


def _row_label(ax, text: str):
    ax.text(
        0.01,
        0.92,
        text,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.9),
    )


def make_figure(A, B, C, D, title: str, out_png: str, delta_fail_deg: float):
    """Classic 2x2 summary plot (kept)"""
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axs.ravel()

    series = [("A", A), ("B", B), ("C", C), ("D", D)]

    for lab, X in series:
        ax1.plot(X["t"], X["df_hz"], label=lab)
    ax1.set_title(r"Frequency deviation $\Delta f$")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Hz")
    ax1.axhline(0.0, linewidth=0.6, alpha=0.35)
    ax1.set_xlim(0.0, float(A.get("tf", 10.0)))
    ax1.set_ylim(*_auto_ylim(np.concatenate([X["df_hz"] for _, X in series])))
    ax1.legend(fontsize=7, loc="best")

    for lab, X in series:
        ax2.plot(X["t"], X["vbus"], label=lab)
    ax2.set_title(rf"Voltage magnitude $|V|$ at Bus {int(A.get('plot_bus', -1))}")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("p.u.")
    ax2.set_xlim(0.0, float(A.get("tf", 10.0)))
    ax2.set_ylim(*_auto_ylim(np.concatenate([X["vbus"] for _, X in series])))

    for lab, X in series:
        ax3.plot(X["t"], X["rocof_hz_s"], label=lab)
    ax3.set_title(r"RoCoF ($d\Delta f/dt$)")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Hz/s")
    ax3.axhline(0.0, linewidth=0.6, alpha=0.35)
    ax3.set_xlim(0.0, float(A.get("tf", 10.0)))
    ax3.set_ylim(*_auto_ylim(np.concatenate([X["rocof_hz_s"] for _, X in series])))

    for lab, X in series:
        ax4.plot(X["t"], X["delta_sep_deg"], label=lab)
    ax4.set_title(r"Rotor angle separation $\max\delta-\min\delta$")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("deg")
    ax4.set_xlim(0.0, float(A.get("tf", 10.0)))
    ax4.set_ylim(*_auto_ylim(np.concatenate([X["delta_sep_deg"] for _, X in series])))

    # show fault window (use C times)
    tfault = float(C.get("t_fault", 1.0))
    tclear = float(C.get("t_clear", 1.24))
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(tfault, linewidth=0.9, alpha=0.35)
        ax.axvline(tclear, linewidth=0.9, alpha=0.35)
        ax.axvspan(tfault, tclear, alpha=0.12)

    ax1.text(
        0.02,
        0.92,
        f"fault [{tfault:.2f},{tclear:.2f}] s",
        transform=ax1.transAxes,
        fontsize=8,
        va="top",
    )

    ax4.axhline(float(delta_fail_deg), linewidth=0.9, alpha=0.35)
    ax4.text(
        0.02,
        0.92,
        rf"$\delta_{{crit}}={float(delta_fail_deg):.0f}^\circ$",
        transform=ax4.transAxes,
        fontsize=8,
        va="top",
    )

    fig.suptitle(title, y=1.02, fontsize=10)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_mega_figure(A, B, C, D, delta_fail_deg: float, title: str, out_png: str):
    """
    4 rows (A,B,C,D) × 6 cols:
      [Vbus, Vmin_sys, df, rocof, freq_summary, delta_sep]
    """
    rows = [
        ("PRE-STRESS (A)", A),
        ("POST-STRESS, no fault (B)", B),
        ("STRESS + FAULT (C)", C),
        ("POST-FAULT / REPAIR (D)", D),
    ]
    col_titles = [
        r"$|V|$ @ plot bus",
        r"$V_{\min}(t)$ system",
        r"$\Delta f(t)$",
        r"RoCoF $(d\Delta f/dt)$",
        r"Freq summary",
        r"$\max\delta-\min\delta$",
    ]

    fig = plt.figure(figsize=(12.8, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(nrows=4, ncols=6)

    for i, (rname, X) in enumerate(rows):
        t = X["t"]
        tf_local = float(X.get("tf", 10.0))

        # Col 1: Vbus
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t, X["vbus"])
        _shade_fault(ax, X)
        ax.set_xlim(0, tf_local)
        if i == 0:
            ax.set_title(col_titles[0])
        if i == 3:
            ax.set_xlabel("Time [s]")
        ax.set_ylabel("p.u.")
        _row_label(ax, rname)

        # Col 2: Vmin_sys
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t, X.get("vmin_ts", X["vbus"]))
        _shade_fault(ax, X)
        ax.set_xlim(0, tf_local)
        if i == 0:
            ax.set_title(col_titles[1])
        if i == 3:
            ax.set_xlabel("Time [s]")
        ax.set_ylabel("p.u.")

        # Col 3: df
        ax = fig.add_subplot(gs[i, 2])
        ax.plot(t, X["df_hz"])
        _shade_fault(ax, X)
        ax.axhline(0.0, linewidth=0.6, alpha=0.35)
        ax.set_xlim(0, tf_local)
        if i == 0:
            ax.set_title(col_titles[2])
        if i == 3:
            ax.set_xlabel("Time [s]")
        ax.set_ylabel("Hz")

        # Col 4: rocof
        ax = fig.add_subplot(gs[i, 3])
        ax.plot(t, X["rocof_hz_s"])
        _shade_fault(ax, X)
        ax.axhline(0.0, linewidth=0.6, alpha=0.35)
        ax.set_xlim(0, tf_local)
        if i == 0:
            ax.set_title(col_titles[3])
        if i == 3:
            ax.set_xlabel("Time [s]")
        ax.set_ylabel("Hz/s")

        # Col 5: frequency summary (df + lines + numbers)
        ax = fig.add_subplot(gs[i, 4])
        ax.plot(t, X["df_hz"], alpha=0.65)
        _shade_fault(ax, X)
        ax.axhline(0.0, linewidth=0.6, alpha=0.35)
        ax.axhline(
            float(X.get("df_nadir", 0.0)), linestyle="--", linewidth=0.8, alpha=0.8
        )
        ax.axhline(
            float(X.get("df_peak", 0.0)), linestyle="--", linewidth=0.8, alpha=0.8
        )
        ax.set_xlim(0, tf_local)
        if i == 0:
            ax.set_title(col_titles[4])
        if i == 3:
            ax.set_xlabel("Time [s]")
        ax.set_ylabel("Hz")

        txt = (
            f"nadir={X.get('df_nadir',0):+.3f} Hz\n"
            f"peak ={X.get('df_peak',0):+.3f} Hz\n"
            f"∫|Δf|dt={X.get('iae_df',0):.3f}\n"
            f"∫|RoCoF|dt={X.get('iae_rocof',0):.3f}"
        )
        ax.text(
            0.02,
            0.02,
            txt,
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
        )

        # Col 6: delta_sep with delta_crit
        ax = fig.add_subplot(gs[i, 5])
        ax.plot(t, X["delta_sep_deg"])
        _shade_fault(ax, X)
        ax.axhline(float(delta_fail_deg), linestyle="--", linewidth=0.9, alpha=0.9)
        ax.set_xlim(0, tf_local)
        if i == 0:
            ax.set_title(col_titles[5])
        if i == 3:
            ax.set_xlabel("Time [s]")
        ax.set_ylabel("deg")

    fig.suptitle(title, fontsize=11)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize(X: Dict[str, Any]) -> Dict[str, Any]:
    t = np.asarray(X["t"], dtype=float)
    vmin_sys = (
        float(np.min(X["vmin_ts"])) if np.asarray(X.get("vmin_ts", [])).size else None
    )
    vmin_bus = float(np.min(X["vbus"])) if np.asarray(X.get("vbus", [])).size else None
    return dict(
        hs=float(X["hs"]),
        inertia_bump=float(X["inertia_bump"]),
        bump_gen=str(X.get("bump_gen")),
        injected_fault_idx=X.get("injected_fault_idx"),
        vmin_sys=vmin_sys,
        vmin_bus=vmin_bus,
        max_abs_df=(
            float(np.max(np.abs(X["df_hz"]))) if np.asarray(X["df_hz"]).size else None
        ),
        max_rocof=(
            float(np.max(np.abs(X["rocof_hz_s"])))
            if np.asarray(X["rocof_hz_s"]).size
            else None
        ),
        df_nadir=float(X.get("df_nadir", 0.0)),
        df_peak=float(X.get("df_peak", 0.0)),
        iae_df=float(X.get("iae_df", 0.0)),
        iae_rocof=float(X.get("iae_rocof", 0.0)),
        max_delta_sep=(
            float(np.max(X["delta_sep_deg"]))
            if np.asarray(X["delta_sep_deg"]).size
            else None
        ),
        stress=float(X["stress"]),
        ok_pf=bool(X["ok_pf"]),
        ok_tds=bool(X["ok_tds"]),
        fault=bool(X["fault"]),
        fault_bus=int(X["fault_bus"]),
        plot_bus=int(X["plot_bus"]),
        t_fault=float(X["t_fault"]),
        t_clear=float(X["t_clear"]),
        tf=float(X["tf"]),
        t_end=float(t[-1]) if t.size else None,
        rotor_model=str(X.get("rotor_model")),
        inertia_param=str(X.get("inertia_param")),
        gen_ids=list(X.get("gen_ids", [])),
        ref_gen=str(X.get("ref_gen")),
    )


# ============================================================
# Deterministic search routines
# ============================================================
def find_hsB_quiet(
    casefile: str, tf: float, vmin_quiet_pu: float, hs_grid: np.ndarray
) -> Tuple[float, int, float]:
    """
    Find maximal hs such that PF ok and Vmin>=vmin_quiet_pu.
    Returns (hsB, crit_bus, vminB).
    """
    feasible: List[Tuple[float, int, float]] = []
    for hs in hs_grid:
        ss = andes.load(casefile, setup=True)
        disable_nonfault_events(ss, tf=tf)
        push_faults_away_except(ss, tf=tf, keep_fault_idx=None)

        scale_all_pq_loads(ss, float(hs))
        if not bool(ss.PFlow.run()):
            continue
        vmin, crit_bus = pf_vmin_at_pq(ss)
        if np.isfinite(vmin) and (crit_bus is not None) and (vmin >= vmin_quiet_pu):
            feasible.append((float(hs), int(crit_bus), float(vmin)))

    if not feasible:
        raise RuntimeError(
            "No hs satisfies quiet PF constraint. Lower vmin_quiet_pu or shrink hs_grid."
        )
    return feasible[-1]


def downsearch_safe_hs(
    casefile: str,
    tf: float,
    hs_start: float,
    hs_min: float,
    hs_step: float,
    fault_bus: int,
    t_fault: float,
    t_clear: float,
    delta_fail_deg: float,
    rotor_model_hint: str,
    ref_gen_hint: Any,
    f0_hz: float,
    plot_bus: int,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    If hs_start already fails under the fixed trigger, search DOWN until we find a safe hs.
    Returns (hs_safe0, X_safe0, log). Raises if none found down to hs_min.
    """
    log: List[Dict[str, Any]] = []
    hs = float(hs_start)

    while hs >= hs_min - 1e-12:
        X = run_case(
            casefile,
            tf,
            hs,
            True,
            fault_bus,
            t_fault,
            t_clear,
            1.0,
            rotor_model_hint,
            ref_gen_hint,
            None,
            f0_hz,
            plot_bus,
            disable_events=True,
        )
        fail = scenario_failed(X, tf=tf, delta_fail_deg=delta_fail_deg)
        log.append(
            dict(
                hs=float(hs), fail=bool(fail), max_dd=float(np.max(X["delta_sep_deg"]))
            )
        )

        if not fail:
            return float(hs), X, log

        hs -= float(hs_step)

    raise RuntimeError(
        "Downsearch could not find a safe hs. Reduce fault severity or expand hs_min."
    )


def bracket_failure_up(
    casefile: str,
    tf: float,
    hs_safe0: float,
    hs_max: float,
    hs_step: float,
    fault_bus: int,
    t_fault: float,
    t_clear: float,
    delta_fail_deg: float,
    rotor_model_hint: str,
    ref_gen_hint: Any,
    f0_hz: float,
    plot_bus: int,
    X_safe0: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Starting from a known-safe hs_safe0, step UP to find first failing hs_fail.
    Returns (hs_safe, hs_fail, X_safe, X_fail, log).
    """
    log: List[Dict[str, Any]] = []
    hs = float(hs_safe0)

    X_prev = (
        X_safe0
        if X_safe0 is not None
        else run_case(
            casefile,
            tf,
            hs,
            True,
            fault_bus,
            t_fault,
            t_clear,
            1.0,
            rotor_model_hint,
            ref_gen_hint,
            None,
            f0_hz,
            plot_bus,
            disable_events=True,
        )
    )
    prev_fail = scenario_failed(X_prev, tf=tf, delta_fail_deg=delta_fail_deg)
    log.append(
        dict(
            hs=float(hs),
            fail=bool(prev_fail),
            max_dd=float(np.max(X_prev["delta_sep_deg"])),
        )
    )

    if prev_fail:
        return hs, hs, X_prev, X_prev, log

    while hs + hs_step <= hs_max + 1e-12:
        hs2 = hs + hs_step
        X2 = run_case(
            casefile,
            tf,
            hs2,
            True,
            fault_bus,
            t_fault,
            t_clear,
            1.0,
            rotor_model_hint,
            ref_gen_hint,
            None,
            f0_hz,
            plot_bus,
            disable_events=True,
        )
        f2 = scenario_failed(X2, tf=tf, delta_fail_deg=delta_fail_deg)
        log.append(
            dict(
                hs=float(hs2), fail=bool(f2), max_dd=float(np.max(X2["delta_sep_deg"]))
            )
        )

        if f2:
            return float(hs), float(hs2), X_prev, X2, log

        hs, X_prev = float(hs2), X2

    return float(hs), float(hs), X_prev, X_prev, log


def bisect_failure(
    casefile: str,
    tf: float,
    hs_lo: float,
    hs_hi: float,
    fault_bus: int,
    t_fault: float,
    t_clear: float,
    delta_fail_deg: float,
    rotor_model_hint: str,
    ref_gen_hint: Any,
    f0_hz: float,
    plot_bus: int,
    tol: float = 1e-3,
    max_iter: int = 25,
    X_lo: Optional[Dict[str, Any]] = None,
    X_hi: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Bisection on hs to find boundary:
      hs_lo survives, hs_hi fails
    Returns (hs_star_safe, hsC_fail, X_safe, X_fail, log)
    """
    log: List[Dict[str, Any]] = []

    X_lo = (
        X_lo
        if X_lo is not None
        else run_case(
            casefile,
            tf,
            hs_lo,
            True,
            fault_bus,
            t_fault,
            t_clear,
            1.0,
            rotor_model_hint,
            ref_gen_hint,
            None,
            f0_hz,
            plot_bus,
            True,
        )
    )
    X_hi = (
        X_hi
        if X_hi is not None
        else run_case(
            casefile,
            tf,
            hs_hi,
            True,
            fault_bus,
            t_fault,
            t_clear,
            1.0,
            rotor_model_hint,
            ref_gen_hint,
            None,
            f0_hz,
            plot_bus,
            True,
        )
    )

    f_lo = scenario_failed(X_lo, tf=tf, delta_fail_deg=delta_fail_deg)
    f_hi = scenario_failed(X_hi, tf=tf, delta_fail_deg=delta_fail_deg)

    if f_lo:
        return hs_lo, hs_lo, X_lo, X_lo, log
    if not f_hi:
        return hs_hi, hs_hi, X_hi, X_hi, log

    for _ in range(max_iter):
        if abs(hs_hi - hs_lo) <= tol:
            break
        hs_mid = 0.5 * (hs_lo + hs_hi)
        X_mid = run_case(
            casefile,
            tf,
            hs_mid,
            True,
            fault_bus,
            t_fault,
            t_clear,
            1.0,
            rotor_model_hint,
            ref_gen_hint,
            None,
            f0_hz,
            plot_bus,
            disable_events=True,
        )
        f_mid = scenario_failed(X_mid, tf=tf, delta_fail_deg=delta_fail_deg)
        log.append(
            dict(
                hs=float(hs_mid),
                fail=bool(f_mid),
                max_dd=float(np.max(X_mid["delta_sep_deg"])),
            )
        )

        if f_mid:
            hs_hi, X_hi = float(hs_mid), X_mid
        else:
            hs_lo, X_lo = float(hs_mid), X_mid

    return float(hs_lo), float(hs_hi), X_lo, X_hi, log


def search_inertia_repair(
    casefile: str,
    tf: float,
    hsC: float,
    fault_bus: int,
    t_fault: float,
    t_clear: float,
    delta_fail_deg: float,
    rotor_model_hint: str,
    ref_gen_hint: Any,
    f0_hz: float,
    plot_bus: int,
    gens_to_try: List[str],
    bump_grid: np.ndarray,
) -> Tuple[Dict[str, Any], str, float, bool, List[Dict[str, Any]]]:
    """
    Try to rescue by inertia-only bump on one generator.
    Returns (best_D, best_gen, best_bump, rescued, log).
    If no rescue exists, returns best-improvement by score:
      score = 1e6*(early_stop) + max_delta_sep
    """
    log: List[Dict[str, Any]] = []

    def score(D_try: Dict[str, Any]) -> float:
        max_dd = (
            float(np.max(D_try["delta_sep_deg"]))
            if np.asarray(D_try["delta_sep_deg"]).size
            else float("inf")
        )
        t = np.asarray(D_try["t"], dtype=float)
        t_end = float(t[-1]) if t.size else -np.inf
        early = (t_end < tf - 1e-6) or (not bool(D_try["ok_tds"]))
        return (1e6 if early else 0.0) + max_dd

    best_sc = float("inf")
    best_D: Optional[Dict[str, Any]] = None
    best_gen = gens_to_try[0] if gens_to_try else str(ref_gen_hint)
    best_bump = 1.0

    for g in gens_to_try:
        for b in bump_grid:
            D_try = run_case(
                casefile,
                tf,
                hsC,
                True,
                fault_bus,
                t_fault,
                t_clear,
                float(b),
                rotor_model_hint,
                ref_gen_hint,
                bump_gen_hint=g,
                f0_hz=f0_hz,
                plot_bus=plot_bus,
                disable_events=True,
            )
            fail = scenario_failed(D_try, tf=tf, delta_fail_deg=delta_fail_deg)
            sc = score(D_try)
            log.append(
                dict(
                    gen=str(g),
                    bump=float(b),
                    fail=bool(fail),
                    max_dd=float(np.max(D_try["delta_sep_deg"])),
                )
            )

            if sc < best_sc:
                best_sc = sc
                best_D = D_try
                best_gen = str(g)
                best_bump = float(b)

            if not fail:
                return D_try, str(g), float(b), True, log

    if best_D is None:
        raise RuntimeError("Repair search failed to produce any D_try.")
    return best_D, best_gen, best_bump, False, log


# ============================================================
# Main
# ============================================================
def main():
    andes.config_logger(stream_level=20)

    # -----------------
    # Case + constants
    # -----------------
    casefile = get_case("kundur/kundur_full.xlsx")
    print(f"[INFO] Using casefile: {casefile}")

    tf = 10.0
    f0_hz = 60.0

    # Quiet PF condition
    vmin_quiet_pu = 0.95

    # Fixed trigger
    t_fault = 1.0
    t_clear = 1.24

    # Failure criterion
    delta_fail_deg = 80.0

    # Search grids
    hs_grid_pf = np.arange(1.00, 1.60 + 1e-12, 0.02)

    hs_max_search = 1.80
    hs_step_bracket = 0.04
    hs_tol = 1e-3

    # downward search if hsB already fails
    hs_min_search = 0.80
    hs_step_down = 0.02

    # Repair search
    bump_grid = np.arange(1.0, 10.0 + 1e-12, 0.5)

    # -----------------
    # Rotor hints
    # -----------------
    tmp = andes.load(casefile, setup=True)
    rotor_model_hint, _, gen_ids_hint = pick_rotor_model_prefer_genrou(tmp)
    ref_gen_hint = gen_ids_hint[0]
    gens_to_try = [str(g) for g in gen_ids_hint]
    print(
        f"[INFO] Rotor hint: {rotor_model_hint} | gen_ids={gens_to_try} | ref_gen={ref_gen_hint}"
    )

    # -----------------
    # 1) Find B: maximal quiet PF point hsB
    # -----------------
    hsB, plot_bus, vminB = find_hsB_quiet(
        casefile=casefile, tf=tf, vmin_quiet_pu=vmin_quiet_pu, hs_grid=hs_grid_pf
    )
    fault_bus = plot_bus
    print(
        f"[INFO] hsB (quiet PF) = {hsB:.3f} | Vmin={vminB:.3f} | plot_bus=fault_bus={plot_bus}"
    )

    # -----------------
    # A/B runs (no fault)
    # -----------------
    A = run_case(
        casefile,
        tf,
        1.00,
        False,
        fault_bus,
        t_fault,
        t_clear,
        1.0,
        rotor_model_hint,
        ref_gen_hint,
        None,
        f0_hz,
        plot_bus,
        disable_events=True,
    )
    B = run_case(
        casefile,
        tf,
        hsB,
        False,
        fault_bus,
        t_fault,
        t_clear,
        1.0,
        rotor_model_hint,
        ref_gen_hint,
        None,
        f0_hz,
        plot_bus,
        disable_events=True,
    )

    # -----------------
    # 2) Boundary under fixed trigger (robust)
    # -----------------
    X_at_hsB = run_case(
        casefile,
        tf,
        hsB,
        True,
        fault_bus,
        t_fault,
        t_clear,
        1.0,
        rotor_model_hint,
        ref_gen_hint,
        None,
        f0_hz,
        plot_bus,
        disable_events=True,
    )
    hsB_fails = scenario_failed(X_at_hsB, tf=tf, delta_fail_deg=delta_fail_deg)

    down_log: List[Dict[str, Any]] = []
    if hsB_fails:
        print(
            "[INFO] hsB already FAILS under fixed fault. Running downward search for a safe hs..."
        )
        hs_safe0, X_safe0, down_log = downsearch_safe_hs(
            casefile=casefile,
            tf=tf,
            hs_start=hsB,
            hs_min=hs_min_search,
            hs_step=hs_step_down,
            fault_bus=fault_bus,
            t_fault=t_fault,
            t_clear=t_clear,
            delta_fail_deg=delta_fail_deg,
            rotor_model_hint=rotor_model_hint,
            ref_gen_hint=ref_gen_hint,
            f0_hz=f0_hz,
            plot_bus=plot_bus,
        )
    else:
        hs_safe0, X_safe0 = float(hsB), X_at_hsB

    hs_safe_br, hs_fail_br, X_safe_br, X_fail_br, br_log = bracket_failure_up(
        casefile=casefile,
        tf=tf,
        hs_safe0=hs_safe0,
        hs_max=hs_max_search,
        hs_step=hs_step_bracket,
        fault_bus=fault_bus,
        t_fault=t_fault,
        t_clear=t_clear,
        delta_fail_deg=delta_fail_deg,
        rotor_model_hint=rotor_model_hint,
        ref_gen_hint=ref_gen_hint,
        f0_hz=f0_hz,
        plot_bus=plot_bus,
        X_safe0=X_safe0,
    )

    if hs_safe_br == hs_fail_br:
        print(
            "[WARN] Could not bracket a failing point up to hs_max_search. Increase hs_max_search or fault severity."
        )
        hs_star = hs_safe_br
        hsC = hs_fail_br
        C = X_fail_br
        bis_log = []
    else:
        hs_star, hsC, X_star, C, bis_log = bisect_failure(
            casefile=casefile,
            tf=tf,
            hs_lo=hs_safe_br,
            hs_hi=hs_fail_br,
            fault_bus=fault_bus,
            t_fault=t_fault,
            t_clear=t_clear,
            delta_fail_deg=delta_fail_deg,
            rotor_model_hint=rotor_model_hint,
            ref_gen_hint=ref_gen_hint,
            f0_hz=f0_hz,
            plot_bus=plot_bus,
            tol=hs_tol,
            max_iter=25,
            X_lo=X_safe_br,
            X_hi=X_fail_br,
        )

    C_fail = scenario_failed(C, tf=tf, delta_fail_deg=delta_fail_deg)
    print(
        f"[INFO] Boundary: hs_star_safe≈{hs_star:.4f}, hsC_fail≈{hsC:.4f} | C_fail={C_fail}"
    )

    # -----------------
    # 3) Repair search at hsC
    # -----------------
    weak_gen = pick_weak_gen_from_ts(C)
    gens_order = [weak_gen] + [g for g in gens_to_try if g != weak_gen]

    D, best_gen, best_bump, rescued, rep_log = search_inertia_repair(
        casefile=casefile,
        tf=tf,
        hsC=hsC,
        fault_bus=fault_bus,
        t_fault=t_fault,
        t_clear=t_clear,
        delta_fail_deg=delta_fail_deg,
        rotor_model_hint=rotor_model_hint,
        ref_gen_hint=ref_gen_hint,
        f0_hz=f0_hz,
        plot_bus=plot_bus,
        gens_to_try=gens_order,
        bump_grid=bump_grid,
    )

    D_fail = scenario_failed(D, tf=tf, delta_fail_deg=delta_fail_deg)
    print(
        f"[INFO] Repair: weak_gen={weak_gen} | chosen_gen={best_gen} | bump={best_bump:.2f} | rescued={rescued} | D_fail={D_fail}"
    )

    # -----------------
    # Save outputs
    # -----------------
    out_png = os.path.abspath("story_abcd_kundur_hidden_margin.png")
    out_png_mega = os.path.abspath("story_abcd_kundur_hidden_margin_MEGA.png")
    out_json = os.path.abspath("story_abcd_kundur_hidden_margin_summary.json")

    title = (
        f"Kundur hidden-margin (deterministic) | fixed fault [{t_fault:.2f},{t_clear:.2f}]s @bus{fault_bus} | "
        f"hsB={hsB:.3f} | hs*={hs_star:.3f} | hsC={hsC:.3f} | repair gen={best_gen} bump={best_bump:.2f}"
    )

    # Classic 2x2
    make_figure(A, B, C, D, title, out_png, delta_fail_deg=delta_fail_deg)

    # MEGA 4x6
    make_mega_figure(
        A, B, C, D, delta_fail_deg=delta_fail_deg, title=title, out_png=out_png_mega
    )

    summary = dict(
        casefile=casefile,
        andes_version=getattr(andes, "__version__", "unknown"),
        rotor_model_hint=str(rotor_model_hint),
        gen_ids=gens_to_try,
        ref_gen=str(ref_gen_hint),
        plot_bus=int(plot_bus),
        fault_bus=int(fault_bus),
        trigger=dict(t_fault=float(t_fault), t_clear=float(t_clear)),
        quiet_pf=dict(
            vmin_quiet_pu=float(vmin_quiet_pu), hsB=float(hsB), vminB=float(vminB)
        ),
        dynamic_boundary=dict(
            delta_fail_deg=float(delta_fail_deg),
            hs_star_safe=float(hs_star),
            hsC_fail=float(hsC),
            hsB_fails_under_fault=bool(hsB_fails),
            downsearch=dict(
                hs_min=float(hs_min_search), hs_step=float(hs_step_down), log=down_log
            ),
            bracketing_log=br_log,
            bisection_log=bis_log,
        ),
        repair=dict(
            weak_gen=str(weak_gen),
            chosen_gen=str(best_gen),
            bump=float(best_bump),
            rescued=bool(rescued),
            repair_log=rep_log,
        ),
        scenarios=dict(A=summarize(A), B=summarize(B), C=summarize(C), D=summarize(D)),
        outputs=dict(png=out_png, mega_png=out_png_mega, json=out_json),
        note=(
            "Deterministic Q1 protocol: (i) define one global stress dial hs, "
            "(ii) pick hsB from quiet PF, (iii) with fixed fault, if hsB already fails do downsearch "
            "to find a safe hs, then bracket+bisect to locate hs* boundary, (iv) pick C at first failing hs, "
            "(v) attempt inertia-only repair for D. Figures mark fault window explicitly. MEGA plot is 4x6."
        ),
    )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Outputs ===")
    print(f"Saved figure:      {out_png}")
    print(f"Saved MEGA figure: {out_png_mega}")
    print(f"Saved JSON:        {out_json}")


if __name__ == "__main__":
    main()
