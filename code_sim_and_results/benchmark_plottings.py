#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_plottings.py  —  SGSMA 2026 publication figures (v6)
==============================================================
Generates:
  Fig1_Scenarios_Final.png  — 5-scenario stress-test suite  (paper Fig. 1)
  Fig2_Mega_Dashboard.png   — 8-panel performance dashboard (paper Fig. 2)

v6 changes vs v5:
  - All panel titles left-aligned (loc="left")
  - Heatmap title reduced to "(g)" — compliance criteria moved to LaTeX caption
  - Heatmap §IV-B footnote removed — moved to LaTeX caption
  - Pareto PI-GRU footnote removed — moved to LaTeX caption
  - figsize=(8.0, 6), hspace=0.7, wspace=0.25 (compact layout)
  - Radar legend at bottom-center (bbox_to_anchor=(0.5,-0.12), ncol=3)
  - PDF export added for both Fig1 and Fig2
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATHS
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_RAW_DIR = "results_raw"
DEFAULT_OUT_DIR  = "figures_estimatores_benchmark_test5"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  RC-PARAMS
# ─────────────────────────────────────────────────────────────────────────────
FS_TITLE  = 9.5
FS_LABEL  = 9.0
FS_TICK   = 8.0
FS_LEGEND = 7.5
FS_ANNOT  = 7.5

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         9.5,
    "axes.labelsize":    FS_LABEL,
    "axes.titlesize":    FS_TITLE,
    "xtick.labelsize":   FS_TICK,
    "ytick.labelsize":   FS_TICK,
    "legend.fontsize":   FS_LEGEND,
    "axes.linewidth":    0.7,
    "lines.linewidth":   1.2,
    "grid.linewidth":    0.35,
    "grid.linestyle":    ":",
    "grid.alpha":        0.45,
    "axes.grid":         True,
    "figure.autolayout": False,
})

# ─────────────────────────────────────────────────────────────────────────────
# 2.  METHOD PALETTE
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY = {
    "EKF2":           "RA-EKF",
    "PLL":            "SRF-PLL",
    "Koopman-RKDPmu": "Koopman",
    "RLS-VFF":        "VFF-RLS",
    "IpDFT":          "IpDFT",
    "EKF":            "EKF",
    "SOGI":           "SOGI-FLL",
    "TFT":            "TFT",
    "UKF":            "UKF",
    "RLS":            "RLS",
    "Teager":         "TKEO",
    "LKF":            "LKF\u2020",
    "PI-GRU":         "PI-GRU",
}

COLORS = {
    "IpDFT":          "#1f77b4",
    "PLL":            "#2ca02c",
    "EKF":            "#d62728",
    "EKF2":           "#8b0000",
    "SOGI":           "#e377c2",
    "RLS":            "#17becf",
    "Teager":         "#ff7f0e",
    "TFT":            "#9467bd",
    "RLS-VFF":        "#006e6e",
    "UKF":            "#8c564b",
    "LKF":            "#aaaaaa",
    "Koopman-RKDPmu": "#bcbd22",
    "PI-GRU":         "#444444",
}

LS = {
    "EKF2":           "-",
    "EKF":            "--",
    "PLL":            "-.",
    "IpDFT":          ":",
    "TFT":            (0, (5, 2)),
    "UKF":            (0, (3, 1, 1, 1)),
    "SOGI":           (0, (5, 1, 1, 1)),
    "Koopman-RKDPmu": (0, (4, 2, 1, 2)),
    "RLS":            (0, (2, 2)),
    "RLS-VFF":        (0, (4, 1)),
}
LW_SPECIAL = {"EKF2": 1.7}
DEFAULT_LW = 1.2
ALPHA_M    = {"EKF2": 1.0, "default": 0.85}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SCENARIO CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SC_ORDER = [
    "IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation",
    "IBR_Nightmare", "IBR_MultiEvent",
]
SC_LABEL = {
    "IEEE_Mag_Step":   "(A) Mag. Step (+10%)",
    "IEEE_Freq_Ramp":  "(B) Freq. Ramp (+5\u202fHz/s)",
    "IEEE_Modulation": "(C) AM Mod. (10%, 2\u202fHz)",
    "IBR_Nightmare":   "(D) Composite Islanding",
    "IBR_MultiEvent":  "(E) IBR Multi-Event",
}
V_ZOOM = {
    "IEEE_Mag_Step":   (0.44, 0.56),
    "IEEE_Freq_Ramp":  (0.20, 1.10),
    "IEEE_Modulation": (0.00, 1.00),
    "IBR_Nightmare":   (0.665, 0.730),
    "IBR_MultiEvent":  (2.44, 2.55),
}
F_YLIM_FIG1 = {
    "IEEE_Mag_Step":   (59.4, 60.6),
    "IEEE_Freq_Ramp":  (59.5, 65.5),
    "IEEE_Modulation": (59.4, 60.6),
    "IBR_Nightmare":   (58.0, 63.0),
    "IBR_MultiEvent":  (48.0, 65.5),
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  METHOD SUBSETS
# ─────────────────────────────────────────────────────────────────────────────
PUB_METHODS = [
    "IpDFT", "TFT", "PLL", "SOGI", "EKF",
    "EKF2", "UKF", "Koopman-RKDPmu", "PI-GRU",
]

_A_METHODS = ["EKF2", "EKF", "PLL", "IpDFT", "UKF"]
_B_METHODS = ["EKF2", "EKF", "PLL", "IpDFT", "TFT"]
_C_ZOOM    = ["EKF2", "EKF", "PLL", "IpDFT"]
_D_METHODS = ["PLL", "SOGI", "TFT", "RLS-VFF", "RLS"]
_RADAR_M   = ["EKF2", "PLL", "IpDFT", "UKF", "PI-GRU"]
_STEADY_M  = ["PLL", "EKF2", "IpDFT"]

# Heatmap: code keys (for data lookup) and display names (for y-axis labels)
_HM_CODE = ["EKF2",   "UKF",  "EKF",  "PLL",     "SOGI",
            "IpDFT",  "TFT",  "Koopman-RKDPmu", "PI-GRU"]
_HM_DISP = ["RA-EKF", "UKF",  "EKF",  "SRF-PLL", "SOGI-FLL",
            "IpDFT",  "TFT",  "Koopman",        "PI-GRU"]

PASS_RMSE = 0.05   # Hz
PASS_PEAK = 0.50   # Hz
PASS_TRIP = 0.10   # s

_PARETO_EXCLUDE = {"PI-GRU", "Teager", "LKF", "RLS", "RLS-VFF"}

# ─────────────────────────────────────────────────────────────────────────────
# 5.  I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _fpath(sc, m, raw_dir):
    return os.path.join(raw_dir, sc, f"{sc}__{m}.json")

def load_trace(sc, m, raw_dir=RESULTS_RAW_DIR):
    fp = _fpath(sc, m, raw_dir)
    if not os.path.exists(fp):
        return None
    try:
        rec = json.load(open(fp))
        s   = rec.get("signals", {})
        t   = np.asarray(s.get("t",      []), np.float64)
        if not len(t):
            return None
        return (t,
                np.asarray(s.get("v",      []), np.float64),
                np.asarray(s.get("f_true", []), np.float64),
                np.asarray(s.get("f_hat",  []), np.float64))
    except Exception:
        return None

def load_metrics(sc, m, raw_dir=RESULTS_RAW_DIR):
    fp = _fpath(sc, m, raw_dir)
    if not os.path.exists(fp):
        return {}
    try:
        return json.load(open(fp)).get("metrics", {})
    except Exception:
        return {}

def _dn(m):  return DISPLAY.get(m, m)
def _clr(m): return COLORS.get(m, "#888888")
def _ls(m):  return LS.get(m, "-")
def _lw(m):  return LW_SPECIAL.get(m, DEFAULT_LW)
def _al(m):  return ALPHA_M.get(m, ALPHA_M["default"])

def _passes(met):
    return (met.get("RMSE",          1e9) < PASS_RMSE and
            met.get("MAX_PEAK",      1e9) < PASS_PEAK and
            met.get("TRIP_TIME_0p5", 1e9) < PASS_TRIP)

def _ds(arr, mx=2500):
    s = max(1, len(arr) // mx)
    return arr[::s]

def _legend(ax, loc="best", ncol=1):
    ax.legend(fontsize=FS_LEGEND, loc=loc, framealpha=0.80,
              handlelength=1.6, handletextpad=0.40, ncol=ncol,
              borderpad=0.45, labelspacing=0.28, edgecolor="0.75")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FIG 1 — SCENARIO SUITE
# ─────────────────────────────────────────────────────────────────────────────
def make_fig1(raw_dir=RESULTS_RAW_DIR, out_dir=DEFAULT_OUT_DIR):
    fig = plt.figure(figsize=(7.16, 6.8))
    gs  = gridspec.GridSpec(5, 2, figure=fig,
                            left=0.10, right=0.97, top=0.975, bottom=0.055,
                            hspace=0.70, wspace=0.30)
    REF_PRIO = ["EKF2", "IpDFT", "EKF", "PLL", "UKF", "TFT"]

    for row, sc in enumerate(SC_ORDER):
        data = None
        for r in REF_PRIO:
            data = load_trace(sc, r, raw_dir)
            if data is not None:
                break
        if data is None:
            print(f"  [Fig1] no data for {sc}")
            continue
        t, v, f_true, _ = data

        ax_v = fig.add_subplot(gs[row, 0])
        z0, z1 = V_ZOOM[sc]
        msk    = (t >= z0) & (t <= z1)
        t_ms   = _ds(t[msk]) * 1000
        v_ds   = _ds(v[msk])
        ax_v.plot(t_ms, v_ds, lw=0.9, color="#2060a0")

        if sc == "IBR_Nightmare":
            ax_v.axvline(700, color="red", lw=0.9, ls="--", alpha=0.75)
            ax_v.text(0.54, 0.82, "Phase\nJump", transform=ax_v.transAxes,
                      fontsize=7.0, color="red", ha="center", va="center")
        elif sc == "IBR_MultiEvent":
            ax_v.axvline(2500, color="red", lw=0.9, ls="--", alpha=0.75)
            ax_v.text(0.54, 0.82, "+80°\nJump", transform=ax_v.transAxes,
                      fontsize=7.0, color="red", ha="center", va="center")

        ax_v.set_ylabel("V [pu]",    fontsize=FS_LABEL)
        ax_v.set_xlabel("Time [ms]", fontsize=FS_LABEL)
        ax_v.set_xlim(t_ms[0], t_ms[-1])
        ax_v.tick_params(labelsize=FS_TICK)
        ax_v.set_title(SC_LABEL[sc], fontsize=FS_TITLE, fontweight="bold", pad=2)

        ax_f = fig.add_subplot(gs[row, 1])
        t_ds  = _ds(t) * 1000
        ft_ds = _ds(f_true)
        ax_f.plot(t_ds, ft_ds, lw=1.1, color="black")

        y0, y1 = F_YLIM_FIG1[sc]
        ax_f.set_ylim(y0, y1)
        ax_f.set_ylabel("$f$ [Hz]",  fontsize=FS_LABEL)
        ax_f.set_xlabel("Time [ms]", fontsize=FS_LABEL)
        ax_f.tick_params(labelsize=FS_TICK)

        ann = {
            "IEEE_Mag_Step":   ("Freq. unchanged",               "black"),
            "IEEE_Freq_Ramp":  ("+5\u202fHz/s ramp",             "black"),
            "IEEE_Modulation": ("AM-FM modulation",              "black"),
            "IBR_Nightmare":   ("Phase-step frequency transient", "red"),
            "IBR_MultiEvent":  ("1. +40° Jump\n2. Neg. Ramp\n3. Ring-down", "black"),
        }
        txt, col = ann.get(sc, ("", "black"))
        ax_f.text(0.04, 0.95, txt, transform=ax_f.transAxes,
                  fontsize=7.5, color=col, va="top")

        if sc == "IBR_Nightmare":
            ax_f.axvline(700, color="red", lw=0.7, ls="--", alpha=0.5)

    out_path = os.path.join(out_dir, "Fig1_Scenarios_Final.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.join(out_dir, "Fig1_Scenarios_Final.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Fig1_Scenarios_Final.png / .pdf")
    return out_path

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FIG 2 — PANEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _plot_traces(ax, sc, methods, raw_dir,
                 zoom_t=None, f_ylim=None, clip_f=(45, 80),
                 title="", ref_alpha=0.35,
                 legend_loc="best", legend_ncol=1):
    ref_done = False
    for m in methods:
        data = load_trace(sc, m, raw_dir)
        if data is None:
            continue
        t, _, f_true, f_hat = data
        msk = (t >= zoom_t[0]) & (t <= zoom_t[1]) if zoom_t else np.ones(len(t), bool)
        t_p  = _ds(t[msk])  * 1000
        fh_p = np.clip(_ds(f_hat[msk]),  clip_f[0], clip_f[1])
        ft_p = _ds(f_true[msk])
        if not ref_done:
            ax.plot(t_p, ft_p, color="black", lw=0.9, alpha=ref_alpha,
                    label="Ref", zorder=1)
            ref_done = True
        ax.plot(t_p, fh_p, color=_clr(m), ls=_ls(m), lw=_lw(m),
                alpha=_al(m), label=_dn(m),
                zorder=5 if m == "EKF2" else 2)

    ax.set_title(title, fontsize=FS_TITLE, pad=3)
    ax.set_ylabel("$f$ [Hz]",  fontsize=FS_LABEL)
    ax.set_xlabel("Time [ms]", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    if f_ylim:
        ax.set_ylim(*f_ylim)
    _legend(ax, loc=legend_loc, ncol=legend_ncol)


def _plot_pareto(ax, results, raw_dir):
    sc_met = results.get("IBR_MultiEvent", {}).get("methods", {})
    # FIG-2: always use IBR_MultiEvent CPU values (cpu_authoritative section)
    cpu_auth = results.get("cpu_authoritative", {}).get("values_us", {})
    # FIG-3: MC30 Ttrip std for error bars
    mc_methods = results.get("monte_carlo", {}).get("methods", {})

    pts = []
    for m in PUB_METHODS:
        if m in _PARETO_EXCLUDE:
            continue
        met  = sc_met.get(m) or load_metrics("IBR_MultiEvent", m, raw_dir)
        if not met:
            continue
        # FIG-2: prefer cpu_authoritative for x-axis consistency with Table V
        cost = float(cpu_auth.get(m, met.get("TIME_PER_SAMPLE_US", 0)) or 0)
        risk = float(met.get("TRIP_TIME_0p5",       0) or 0)
        # FIG-3: Ttrip std from MC30 for error bars
        trip_std = float(mc_methods.get(m, {}).get("IBR_MultiEvent", {}).get("TRIP_std", 0) or 0)
        if cost > 0:
            pts.append((m, cost, max(risk, 5e-4), trip_std))

    def on_pareto(i):
        xi, yi = pts[i][1], pts[i][2]  # cost, risk (indices 1,2 regardless of tuple length)
        return not any(pts[j][1] <= xi and pts[j][2] <= yi and
                       (pts[j][1] < xi or pts[j][2] < yi)
                       for j in range(len(pts)) if j != i)

    # FIG-3: error bars — unpack trip_std
    def _get_trip_std(i):
        return pts[i][3] if len(pts[i]) > 3 else 0.0

    # Per-method label offsets (dx_pts, dy_pts) — tuned to prevent overlap
    LABEL_OFF = {
        "EKF2":           (  5,  4),
        "EKF":            (-24, -9),
        "PLL":            (-24,  4),
        "IpDFT":          (  5, -9),
        "TFT":            (  5,  4),
        "UKF":            (  5,  4),
        "Koopman-RKDPmu": (  5, -9),
        "SOGI":           (  5,  4),
    }

    # Draw Pareto staircase frontier connecting Pareto-optimal points
    pareto_pts = sorted(
        [(pt[1], pt[2]) for i, pt in enumerate(pts) if on_pareto(i)],
        key=lambda p: p[0]
    )
    if len(pareto_pts) >= 2:
        px = [p[0] for p in pareto_pts]
        py = [p[1] for p in pareto_pts]
        ax.plot(px, py, color="#555555", lw=0.8, ls="--", alpha=0.55, zorder=2,
                label="Pareto front")

    for i, (m, *rest) in enumerate(pts):
        cost, risk = rest[0], rest[1]
        trip_std = _get_trip_std(i)
        ax.scatter(cost, risk,
                   s=50 if on_pareto(i) else 30,
                   color=_clr(m),
                   edgecolors="black" if on_pareto(i) else "none",
                   linewidths=0.9, zorder=4 if on_pareto(i) else 3,
                   marker="o")
        # FIG-3: add ±1σ Ttrip error bars from MC30 data
        if trip_std > 0 and risk > 1e-6:
            ax.errorbar(cost, risk, yerr=trip_std,
                        fmt="none", color=_clr(m), alpha=0.5,
                        capsize=2, capthick=0.7, lw=0.7, zorder=3)
        dx, dy = LABEL_OFF.get(m, (5, 3))
        ax.annotate(_dn(m), (cost, risk),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=FS_ANNOT - 0.5, color=_clr(m))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CPU time / sample [\u00b5s]", fontsize=FS_LABEL)
    ax.set_ylabel("Trip-Risk [s]",               fontsize=FS_LABEL)
    ax.set_title("(e) Cost vs. Risk  [Multi-Event]", fontsize=FS_TITLE, pad=3, loc="left")
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.03, 0.10, "Efficient & Safe", transform=ax.transAxes,
            fontsize=FS_ANNOT - 0.5, color="#2ca02c", va="bottom")
    ax.text(0.58, 0.95, "Costly & Risky", transform=ax.transAxes,
            fontsize=FS_ANNOT - 0.5, color="#d62728", va="top")
    # FIG-2: footnote documenting CPU baseline (consistent with Table V)
    ax.text(0.01, -0.18, "CPU: Python baseline, IBR Multi-Event scenario",
            transform=ax.transAxes, fontsize=FS_ANNOT - 1.5,
            color="#666666", va="top", style="italic")


def _plot_radar(ax, raw_dir):
    dims = ["Accuracy", "Noise", "Speed", "Safety", "Transient"]
    n    = len(dims)
    ang  = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angc = ang + ang[:1]

    raw = {}
    for m in _RADAR_M:
        rmse_all = [load_metrics(sc, m, raw_dir).get("RMSE", np.nan)
                    for sc in SC_ORDER]
        mn  = load_metrics("IBR_Nightmare",   m, raw_dir)
        me  = load_metrics("IBR_MultiEvent",  m, raw_dir)
        mmo = load_metrics("IEEE_Modulation", m, raw_dir)
        # FIX M2: use Multi-Event CPU (representative stressed cost) not Step CPU
        try:
            cpu_f = float(me.get("TIME_PER_SAMPLE_US", np.nan))
        except (TypeError, ValueError):
            cpu_f = np.nan
        raw[m] = {
            "Accuracy":  np.nanmean(rmse_all),
            "Transient": mn.get("RMSE", np.nan),
            "Safety":    me.get("TRIP_TIME_0p5", np.nan),
            "Speed":     max(cpu_f, 1e-3) if np.isfinite(cpu_f) else np.nan,
            "Noise":     mmo.get("RMSE", np.nan),
        }

    for dim in dims:
        vals = [v[dim] for v in raw.values() if np.isfinite(v.get(dim, np.nan))]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        if dim == "Speed":
            lo = np.log10(lo + 1e-12)
            hi = np.log10(hi + 1e-12)
        for m in raw:
            v = raw[m][dim]
            if not np.isfinite(v):
                raw[m][dim] = 0.5
                continue
            if dim == "Speed":
                v = np.log10(max(v, 1e-12))
            raw[m][dim] = 1.0 - (v - lo) / (hi - lo + 1e-12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(20)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"],
                       fontsize=6.5, color="gray")
    ax.set_xticks(ang)
    ax.set_xticklabels(dims, fontsize=FS_LEGEND - 0.5)
    # FIX: push labels outward so "Transient" doesn't overlap the ring
    ax.tick_params(axis="x", pad=7)

    for m in _RADAR_M:
        vals = [raw[m][d] for d in dims] + [raw[m][dims[0]]]
        ax.plot(angc, vals, lw=_lw(m), color=_clr(m), ls=_ls(m),
                label=_dn(m), alpha=0.9)
        ax.fill(angc, vals, alpha=0.07, color=_clr(m))

    # Legend at bottom-center — uses the white space below the radar ring
    ax.legend(fontsize=FS_LEGEND - 0.5,
              loc="upper center",
              bbox_to_anchor=(0.5, -0.12),   # below the polar axes
              ncol=3,                          # 3 columns: 5 items → row of 3 + row of 2
              frameon=False,
              handlelength=1.4,
              columnspacing=0.8,
              labelspacing=0.20)


def _verify_heatmap_consistency(results, raw_dir):
    """
    FIG-4: Check heatmap P/F cells against current JSON data.
    Prints any inconsistencies. Returns list of issues found.
    """
    issues = []
    for sc in SC_ORDER:
        sc_data = results.get(sc, {}).get("methods", {})
        for mk_code in _HM_CODE:
            met = sc_data.get(mk_code) or load_metrics(sc, mk_code, raw_dir) or {}
            if not met:
                continue
            ok = _passes(met)
            rmse = met.get("RMSE", float("nan"))
            peak = met.get("MAX_PEAK", float("nan"))
            trip = met.get("TRIP_TIME_0p5", float("nan"))
            # Check if the cell status disagrees with the JSON iec_compliance flag
            iec = met.get("iec_compliance", {})
            if iec:
                json_pass = iec.get("pass", ok)
                if json_pass != ok:
                    issues.append(
                        f"  INCONSISTENCY: {mk_code}/{sc}: "
                        f"heatmap={('P' if ok else 'F')}, "
                        f"JSON.iec_compliance={('P' if json_pass else 'F')} — "
                        f"RMSE={rmse:.4f}, Peak={peak:.4f}, Ttrip={trip:.4f}"
                    )
            # Special check: Koopman in Step scenario
            if mk_code == "Koopman-RKDPmu" and sc == "IEEE_Mag_Step":
                status = "P" if ok else "F"
                issues.append(
                    f"  FIG-4 CHECK: Koopman/{sc}: "
                    f"RMSE={rmse:.4f}Hz (th={PASS_RMSE}), "
                    f"Peak={peak:.4f}Hz (th={PASS_PEAK}), "
                    f"Ttrip={trip:.4f}s (th={PASS_TRIP}) → {status}"
                )
    if issues:
        print("  [FIG-4] Heatmap consistency check:")
        for issue in issues:
            print(issue)
    else:
        print("  [FIG-4] Heatmap consistency: no inconsistencies found.")
    return issues


def _plot_heatmap(ax, results, raw_dir):
    # FIG-4: verify consistency before drawing
    _verify_heatmap_consistency(results, raw_dir)

    sc_short = {
        "IEEE_Mag_Step":   "Step",
        "IEEE_Freq_Ramp":  "Ramp",
        "IEEE_Modulation": "Mod",
        "IBR_Nightmare":   "Isl",
        "IBR_MultiEvent":  "Multi",
    }

    for ci, sc in enumerate(SC_ORDER):
        sc_data = results.get(sc, {}).get("methods", {})
        for ri, (mk_code, mk_disp) in enumerate(zip(_HM_CODE, _HM_DISP)):
            met = sc_data.get(mk_code) or load_metrics(sc, mk_code, raw_dir) or {}
            if met:
                ok       = _passes(met)
                marginal = (mk_code == "EKF2" and sc == "IBR_Nightmare" and not ok)
                fc       = "#2ca02c" if ok else "#c0392b"
                txt      = "P" if ok else ("F*" if marginal else "F")
            else:
                fc, txt = "#cccccc", "\u2014"
            ax.add_patch(plt.Rectangle([ci - 0.5, ri - 0.5], 1, 1,
                                        color=fc, alpha=0.90, zorder=1))
            ax.text(ci, ri, txt, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white", zorder=2)

    ax.set_xlim(-0.5, len(SC_ORDER) - 0.5)
    ax.set_ylim(-0.5, len(_HM_CODE) - 0.5)
    ax.set_xticks(range(len(SC_ORDER)))
    ax.set_xticklabels([sc_short[s] for s in SC_ORDER], fontsize=FS_TICK)
    ax.set_yticks(range(len(_HM_DISP)))
    ax.set_yticklabels(_HM_DISP, fontsize=FS_TICK)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(labelsize=FS_TICK, length=2)

    # FIX: title + footnote as separate text calls; §IV-B corrected
    ax.set_title("(g)", fontsize=FS_TITLE, pad=14, loc="left")
    ax.grid(False)


def _plot_steady_error(ax, raw_dir):
    any_data = False
    for m in _STEADY_M:
        data = load_trace("IEEE_Modulation", m, raw_dir)
        if data is None:
            continue
        t, _, f_true, f_hat = data
        err  = np.clip(np.abs(f_hat - f_true), 1e-5, None)
        t_ds = _ds(t) * 1000
        e_ds = _ds(err)
        ax.semilogy(t_ds, e_ds, color=_clr(m), ls=_ls(m), lw=_lw(m),
                    alpha=min(_al(m), 0.95), label=_dn(m),
                    solid_capstyle="round")
        any_data = True

    if not any_data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center")
        return

    ax.set_xlabel("Time [ms]",        fontsize=FS_LABEL)
    ax.set_ylabel("log$|e(t)|$ [Hz]", fontsize=FS_LABEL)
    ax.set_title("(h) Steady-State Error (Scen. C)", fontsize=FS_TITLE, pad=3, loc="left")
    ax.set_ylim(1e-5, 1.0)
    ax.tick_params(labelsize=FS_TICK)
    _legend(ax, loc="upper right")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  FIG 2 — MEGA DASHBOARD  (8.0" × 9.2")
# ─────────────────────────────────────────────────────────────────────────────
def make_fig2(json_export, raw_dir=RESULTS_RAW_DIR, out_dir=DEFAULT_OUT_DIR):
    fig = plt.figure(figsize=(8.0,7))
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                            left=0.07, right=0.96, top=0.97, bottom=0.05,
                            hspace=0.7, wspace=0.25)

    results = json_export.get("results", {})

    # ── (a) Phase Jump ───────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_traces(ax_a, "IBR_Nightmare", _A_METHODS, raw_dir,
                 zoom_t=(0.645, 1.10), f_ylim=(59.3, 63.1), clip_f=(45, 80),
                 title="(a) Phase Jump Robustness (Scen. D)",
                 legend_loc="upper right")
    ax_a.axvline(700, color="red", lw=0.8, ls="--", alpha=0.60)
    ax_a.text(703, 62.85, "60° Jump",
              fontsize=FS_ANNOT - 0.5, color="red", va="top", ha="left")
    # FIX: use transAxes so annotation never goes out of bounds
    ax_a.text(0.36, 0.06,
              "RA-EKF: TripTime\u202f=\u202f0.6\u202fms",
              transform=ax_a.transAxes, fontsize=FS_ANNOT - 0.5,
              color=_clr("EKF2"), style="italic")

    # ── (b) Ramp Lag ─────────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_traces(ax_b, "IEEE_Freq_Ramp", _B_METHODS, raw_dir,
                 zoom_t=(0.25, 1.48), f_ylim=(59.8, 65.8), clip_f=(45, 80),
                 title="(b) Ramp Tracking Lag (Scen. B)",
                 legend_loc="upper left")
    ax_b.text(0.97, 0.05,
              "IpDFT: end-of-window\nref. ($N_c$/2 lag)",
              transform=ax_b.transAxes, fontsize=FS_ANNOT - 0.5,
              color=_clr("IpDFT"), va="bottom", ha="right", style="italic",
              bbox=dict(boxstyle="round,pad=0.22", fc="white", alpha=0.80,
                        ec=_clr("IpDFT"), lw=0.5))

    # ── (c) Multi-Event — +80° jump recovery zoom ────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    _T0, _T1  = 2.420, 2.720
    THRESH    = 0.5
    ref_done  = False

    for m in _C_ZOOM:
        data = load_trace("IBR_MultiEvent", m, raw_dir)
        if data is None:
            continue
        t, _, f_true, f_hat = data
        msk  = (t >= _T0) & (t <= _T1)
        t_p  = _ds(t[msk], mx=1500) * 1000
        fh_p = _ds(f_hat[msk], mx=1500)
        ft_p = _ds(f_true[msk], mx=1500)
        if not ref_done:
            ax_c.plot(t_p, ft_p, color="black", lw=1.0, alpha=0.40,
                      label="Ref", zorder=1)
            ref_done = True
        ax_c.plot(t_p, fh_p, color=_clr(m), ls=_ls(m), lw=_lw(m),
                  alpha=_al(m), label=_dn(m),
                  zorder=5 if m == "EKF2" else 2)

    ax_c.axvline(2500, color="red", lw=0.9, ls="--", alpha=0.65)
    # FIX: fixed y-coord (74.0) so annotation is stable before draw
    ax_c.text(2501, 74.0, "+80° Jump",
              fontsize=FS_ANNOT - 0.5, color="red", va="top", ha="left",
              clip_on=True)

    ax_c.axhspan(60 - THRESH, 60 + THRESH, alpha=0.08, color="green", zorder=0)
    ax_c.axhline(60 + THRESH, color="green", lw=0.55, ls=":", alpha=0.70)
    ax_c.axhline(60 - THRESH, color="green", lw=0.55, ls=":", alpha=0.70)
    ax_c.text(2422, 60 + THRESH + 0.30,
              "\u00b10.5\u202fHz safe band",
              fontsize=6.5, color="green", va="bottom")

    ax_c.annotate("RA-EKF\nrecovers\nfastest",
                  xy=(2530, 60.25), xytext=(2590, 63.5),
                  fontsize=6.5, color=_clr("EKF2"),
                  arrowprops=dict(arrowstyle="->", color=_clr("EKF2"),
                                  lw=0.7, connectionstyle="arc3,rad=-0.2"))
    ax_c.set_ylim(48, 75)
    ax_c.set_ylabel("$f$ [Hz]",  fontsize=FS_LABEL)
    ax_c.set_xlabel("Time [ms]", fontsize=FS_LABEL)
    ax_c.tick_params(labelsize=FS_TICK)
    ax_c.set_title("(c) Multi-Event\u2014Phase Jump Recovery (+80°)",
                   fontsize=FS_TITLE, pad=3)
    ax_c.text(0.02, 0.03,
              "f ramping at t<2.5\u202fs (see Scen.\u202fE full trace)",
              transform=ax_c.transAxes, fontsize=6.0, color="gray",
              va="bottom", style="italic")
    _legend(ax_c, loc="lower right", ncol=2)

    # ── (d) Multi-Event Unstable ──────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    _plot_traces(ax_d, "IBR_MultiEvent", _D_METHODS, raw_dir,
                 zoom_t=None, f_ylim=(46.0, 70.0), clip_f=(42, 75),
                 title="(d) Multi-Event \u2014 Unstable (TKEO excl.)",
                 legend_loc="upper right")
    ax_d.axhline(60, color="gray", lw=0.45, ls=":", alpha=0.50)
    ax_d.text(0.02, 0.03,
              "TKEO excl. (RMSE\u202f\u22658.4\u202fHz, off-scale)",
              transform=ax_d.transAxes, fontsize=FS_ANNOT - 1,
              color=_clr("Teager"), style="italic")

    # ── (g) Compliance Heatmap ────────────────────────────────────────────────
    ax_g = fig.add_subplot(gs[2, 0])
    _plot_heatmap(ax_g, results, raw_dir)

    # ── (h) Steady-State Error ────────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[2, 1])
    _plot_steady_error(ax_h, raw_dir)

    # ── (e) Cost vs. Risk ─────────────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[3, 0])
    _plot_pareto(ax_e, results, raw_dir)

    # ── (f) Balance Radar ─────────────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[3, 1], polar=True)
    _plot_radar(ax_f, raw_dir)
    ax_f.set_title("(f) Balance Profile", fontsize=FS_TITLE, pad=16, loc="left")
    ax_f.title.set_position((-0.25, 1.18))   # ← ajusta x e y aquí

    out_path = os.path.join(out_dir, "Fig2_Mega_Dashboard.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.join(out_dir, "Fig2_Mega_Dashboard.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Fig2_Mega_Dashboard.png / .pdf")
    return out_path

# ─────────────────────────────────────────────────────────────────────────────
# 9.  SUPPLEMENTARY — per-scenario diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def make_per_scenario_diagnostics(raw_dir=RESULTS_RAW_DIR, out_dir=DEFAULT_OUT_DIR):
    for sc in SC_ORDER:
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.4))
        fig.suptitle(SC_LABEL[sc], fontsize=FS_TITLE, fontweight="bold")
        ax_tr, ax_err, ax_trip, ax_rmse = (
            axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])

        bars_t, bars_r, bcolors, blabels = [], [], [], []
        ref_done = False

        for m in PUB_METHODS:
            data = load_trace(sc, m, raw_dir)
            if data is None:
                continue
            t, _, f_true, f_hat = data
            t_ds  = _ds(t) * 1000
            fh_ds = np.clip(_ds(f_hat), 40, 80)
            ft_ds = _ds(f_true)
            if not ref_done:
                ax_tr.plot(t_ds, ft_ds, "k-", lw=0.8, alpha=0.35, label="Ref")
                ref_done = True
            ax_tr.plot(t_ds, fh_ds, color=_clr(m), ls=_ls(m),
                       lw=_lw(m), alpha=_al(m), label=_dn(m))
            err = np.abs(f_hat - f_true)
            ax_err.plot(t_ds, _ds(err), color=_clr(m),
                        ls=_ls(m), lw=_lw(m), alpha=_al(m))
            met = load_metrics(sc, m, raw_dir)
            bars_t.append(met.get("TRIP_TIME_0p5", 0))
            bars_r.append(met.get("RMSE", 0))
            bcolors.append(_clr(m))
            blabels.append(_dn(m))

        ax_tr.set_ylabel("$f$ [Hz]",  fontsize=FS_LABEL)
        ax_tr.set_xlabel("Time [ms]", fontsize=FS_LABEL)
        ax_tr.tick_params(labelsize=FS_TICK)
        ax_tr.legend(fontsize=FS_LEGEND - 1, ncol=2, framealpha=0.75)

        ax_err.axhline(0.5, color="red", lw=0.9, ls="--", alpha=0.7)
        ax_err.set_ylabel("|Error| [Hz]", fontsize=FS_LABEL)
        ax_err.set_xlabel("Time [ms]",   fontsize=FS_LABEL)
        ax_err.set_ylim(0, 5.0)
        ax_err.tick_params(labelsize=FS_TICK)

        x = np.arange(len(blabels))
        if blabels:
            ax_trip.bar(x, bars_t, color=bcolors, alpha=0.85, edgecolor="none")
            ax_trip.set_xticks(x)
            ax_trip.set_xticklabels(blabels, rotation=40, ha="right",
                                    fontsize=FS_TICK - 1)
            ax_trip.set_ylabel("Trip-Risk [s]", fontsize=FS_LABEL)
            ax_trip.tick_params(labelsize=FS_TICK)

            ax_rmse.bar(x, bars_r, color=bcolors, alpha=0.85, edgecolor="none")
            ax_rmse.set_xticks(x)
            ax_rmse.set_xticklabels(blabels, rotation=40, ha="right",
                                    fontsize=FS_TICK - 1)
            ax_rmse.set_ylabel("RMSE [Hz]", fontsize=FS_LABEL)
            ax_rmse.set_yscale("symlog", linthresh=0.01)
            ax_rmse.tick_params(labelsize=FS_TICK)

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"Diagnostic_{sc}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Diagnostic_{sc}.png saved")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────
def generate_publication_figures(json_export,
                                  results_raw_dir=RESULTS_RAW_DIR,
                                  out_dir=DEFAULT_OUT_DIR,
                                  also_diagnostics=True):
    os.makedirs(out_dir, exist_ok=True)
    print("\n[PLOTTING] Generating publication figures ...")

    try:
        make_fig1(raw_dir=results_raw_dir, out_dir=out_dir)
    except Exception as exc:
        print(f"  [WARNING] Fig1 failed: {exc}")
        import traceback; traceback.print_exc()

    try:
        make_fig2(json_export, raw_dir=results_raw_dir, out_dir=out_dir)
    except Exception as exc:
        print(f"  [WARNING] Fig2 failed: {exc}")
        import traceback; traceback.print_exc()

    if also_diagnostics:
        try:
            make_per_scenario_diagnostics(raw_dir=results_raw_dir, out_dir=out_dir)
        except Exception as exc:
            print(f"  [WARNING] diagnostics failed: {exc}")

    print(f"[PLOTTING] All figures -> {out_dir}\n")


def main():
    """Standalone: regenerate figures from disk without re-running benchmark."""
    json_export = {"results": {}}
    for cand in [os.path.join(DEFAULT_OUT_DIR, "benchmark_results.json"),
                 "benchmark_results.json"]:
        if os.path.exists(cand):
            try:
                json_export = json.load(open(cand))
                print(f"[INFO] Loaded {cand}")
                break
            except Exception:
                pass
    else:
        print("[WARNING] benchmark_results.json not found — "
              "panels (e)/(g) metrics loaded directly from results_raw/")

    generate_publication_figures(json_export,
                                  results_raw_dir=RESULTS_RAW_DIR,
                                  out_dir=DEFAULT_OUT_DIR,
                                  also_diagnostics=True)


if __name__ == "__main__":
    main()