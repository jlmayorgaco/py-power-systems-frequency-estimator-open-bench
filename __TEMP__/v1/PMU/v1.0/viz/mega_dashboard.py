#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dashboards/physics_dashboard.py
IEEE-journal style comparative + per-scenario dashboards (clean, consistent, print-ready)

Q1 BULLETPROOF FIXES (THIS VERSION):
  1) Adaptive row spacing (hspace) based on number of rows per page.
  2) No scientific offset on y-axis (ScalarFormatter, no +6e1).
  3) No double legend overlay: hard-purge all legend artists before adding one legend.
  4) Deterministic grouped styling: per-trace linestyles consistent across V and f + legend matches.
  5) Flat-frequency auto switch: plot Δf(t)=f(t)-60 and show nominal band annotation.
  6) Optional brace annotations kept, code is syntactically complete.
  7) Phase jump annotation (PURE) now appears in BOTH:
        - per-scenario dashboards
        - grouped dashboards
     using explicit input t0 from scenario config (NO estimation).
  8) Correct interpretation: do NOT draw a frequency spike; instead show symbolic impulse term.

Usage:
  python dashboards/physics_dashboard.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.legend import Legend
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D


# ============================================================
# IEEE Journal styling (Matplotlib)
# ============================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    }
)

VOLT_COLOR = "#0072B2"  # Okabe-Ito blue
FREQ_COLOR = "#D55E00"  # Okabe-Ito vermillion


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class DashboardStyle:
    volt_color: str = VOLT_COLOR
    freq_color: str = FREQ_COLOR

    lw_v: float = 0.95
    lw_f: float = 1.25

    tick_labelsize: int = 7
    nx_v: int = 4
    ny_v: int = 4
    nx_f: int = 5
    ny_f: int = 4

    title_fontsize: int = 8
    label_fontsize: int = 8
    title_pad: int = 3

    ref60_color: str = "0.35"
    ref60_lw: float = 0.85
    ref60_ls: str = ":"
    ref60_alpha: float = 0.45

    anno_fontsize: int = 7

    # frequency presentation
    f0_hz: float = 60.0
    flat_tol_hz: float = 0.02  # peak-to-peak threshold for "nominal/flat"

    # decimation for display only (0 or negative => NO decimation)
    max_points_v: int = 0
    max_points_f: int = 0
    max_points_v_dense: int = 0

    # x labels
    show_xlabel_every_subplot: bool = True
    xlabel_text: str = "Time [s]"

    # legend style (solid white box)
    legend_frame: bool = True
    legend_facecolor: str = "white"
    legend_edgecolor: str = "0.55"
    legend_alpha: float = 1.0
    legend_linewidth: float = 0.6
    legend_fancybox: bool = False


@dataclass(frozen=True)
class LayoutConfig:
    scenarios_per_page: int = 8

    figsize: Tuple[float, float] = (7.2, 10.8)
    width_ratios: Tuple[float, float] = (1.0, 1.15)

    hspace: float = 0.999
    wspace: float = 0.22

    left: float = 0.10
    right: float = 0.99
    bottom: float = 0.055
    top: float = 0.915
    suptitle_y: float = 0.988

    # Comparative
    comp_figsize: Tuple[float, float] = (7.8, 10.5)
    comp_width_ratios: Tuple[float, float] = (1.0, 1.15)

    comp_hspace: float = 0.99
    comp_wspace: float = 0.22

    comp_left: float = 0.10
    comp_right: float = 0.99
    comp_bottom: float = 0.055
    comp_top: float = 0.95
    comp_suptitle_y: float = 0.985


# ============================================================
# Utilities
# ============================================================
def _log(msg: str) -> None:
    print(msg, file=sys.stdout, flush=True)


def _set_plain_ticks(ax) -> None:
    """Force plain tick labels (no +offset, no scientific notation)."""
    fmt = ScalarFormatter(useOffset=False, useMathText=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    try:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    except Exception:
        pass


def _decimate_xy(
    t: np.ndarray, y: np.ndarray, max_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stride decimation for plotting (keeps endpoints).
    If max_points <= 0 => NO decimation.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    n = int(t.size)
    if max_points <= 0 or n <= max_points:
        return t, y
    stride = int(np.ceil(n / max_points))
    idx = np.arange(0, n, stride, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return t[idx], y[idx]


def robust_ylim(
    y: np.ndarray, pad_frac: float = 0.10, q: Tuple[float, float] = (1, 99)
) -> Optional[List[float]]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None
    lo, hi = np.percentile(y, q)
    if lo == hi:
        lo -= 1e-6
        hi += 1e-6
    pad = (hi - lo) * float(pad_frac)
    return [float(lo - pad), float(hi + pad)]


def event_window(
    t: np.ndarray,
    x: np.ndarray,
    default_center: float = 2.0,
    default_width: float = 0.10,
    width: Tuple[float, float] = (0.06, 0.25),
    thr_sigma: float = 4.0,
) -> List[float]:
    """
    Robust event window guess using median/MAD on |dx/dt|.
    If no clear event, returns default centered window.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(t) & np.isfinite(x)
    t, x = t[m], x[m]
    if t.size < 10:
        return [default_center - default_width / 2, default_center + default_width / 2]

    dt = np.diff(t)
    dx = np.diff(x)
    dt[dt == 0] = np.nan

    g = np.abs(dx / dt)
    g = g[np.isfinite(g)]
    if g.size < 5:
        return [default_center - default_width / 2, default_center + default_width / 2]

    med = np.median(g)
    mad = np.median(np.abs(g - med)) + 1e-12

    score = (np.abs(np.diff(x) / np.diff(t)) - med) / mad
    score = np.where(np.isfinite(score), score, 0.0)

    idx = int(np.argmax(score))
    if score[idx] < thr_sigma:
        return [default_center - default_width / 2, default_center + default_width / 2]

    t_event = float(t[idx])
    w = float(np.clip(default_width, width[0], width[1]))
    t0, t1 = t_event - w / 2, t_event + w / 2

    t0 = max(t0, float(t.min()))
    t1 = min(t1, float(t.max()))

    if (t1 - t0) < width[0]:
        mid = 0.5 * (t0 + t1)
        t0 = max(mid - width[0] / 2, float(t.min()))
        t1 = min(mid + width[0] / 2, float(t.max()))

    return [float(t0), float(t1)]


def _purge_all_legends(ax) -> None:
    """Hard-remove ANY legend artists (prevents double-legend overlay)."""
    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.remove()
        except Exception:
            pass
    for child in list(ax.get_children()):
        if isinstance(child, Legend):
            try:
                child.remove()
            except Exception:
                pass


def annotate_phase_jump(
    ax_v,
    ax_f,
    t0: float,
    delta_phi_rad: float | None = None,
    color: str = "crimson",
    ls: str = "--",
    lw: float = 0.9,
    alpha: float = 0.95,
    fontsize: int = 7,
) -> None:
    """
    Q1-correct phase-jump annotation.
    - Voltage: show event time + Δφ label (phase is visible in v(t)).
    - Frequency: keep f(t) continuous; show symbolic impulse term (do NOT draw a spike).
    """
    ax_v.axvline(t0, color=color, ls=ls, lw=lw, alpha=alpha, zorder=6)
    ax_f.axvline(t0, color=color, ls=ls, lw=lw, alpha=alpha, zorder=6)

    vtxt = (
        r"$\Delta\phi$ (phase jump)"
        if delta_phi_rad is None
        else rf"$\Delta\phi={float(delta_phi_rad):+.2f}\,$rad"
    )
    ftxt = r"$f(t)=f_0+\frac{\Delta\phi}{2\pi}\,\delta(t-t_0)$"

    ax_v.annotate(
        vtxt,
        xy=(t0, 0.88),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=fontsize,
        color=color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
        zorder=7,
    )

    ax_f.annotate(
        ftxt,
        xy=(t0, 0.88),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=fontsize,
        color=color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
        zorder=7,
    )


# ============================================================
# Scenario config (UPDATED)
# ============================================================
def get_scenario_config() -> Dict[str, Dict[str, Any]]:
    def_v = [1.95, 2.05]
    def_f = [0.0, 5.0]
    return {
        "G1_E1_Pure_60Hz": {"t_v": def_v, "t_f": def_f, "f_lims": [59.9, 60.1]},
        "G1_E2_Gaussian_Noise_1pct": {
            "t_v": def_v,
            "t_f": def_f,
            "f_lims": [59.5, 60.5],
        },
        "G1_E3_Gaussian_Noise_5pct": {
            "t_v": [1.98, 2.02],
            "t_f": def_f,
            "f_lims": [58.5, 61.5],
        },
        "G2_E4_Voltage_Mag_Step_1pct": {
            "t_v": [1.995, 2.020],
            "t_f": [1.98, 2.02],
            "f_lims": [59.9, 60.1],
        },
        "G2_E5_Voltage_Mag_Step_10pct": {
            "t_v": [1.995, 2.020],
            "t_f": [1.98, 2.02],
            "f_lims": [59.9, 60.1],
        },
        "G2_E6_Freq_Step_60_to_59p5": {
            "t_v": [1.98, 2.02],
            "t_f": [1.98, 2.02],
            "f_lims": [59.4, 60.1],
        },
        "G2_E7_Freq_Step_60_to_55": {
            "t_v": [1.98, 2.02],
            "t_f": [1.98, 2.02],
            "f_lims": [54.5, 60.5],
        },
        "G2_E8_Fast_Ramp_plus5Hzs": {
            "t_v": [1.5, 5.0],
            "t_f": [1.5, 5.0],
            "f_lims": [59.5, 80.5],
        },
        "G2_E9_Slow_Ramp_minus0p5Hzs": {
            "t_v": def_v,
            "t_f": [0.5, 5.0],
            "f_lims": [57.0, 60.5],
        },
        "G3_E10_AM_Modulation": {
            "t_v": [0.0, 2.0],
            "t_f": [0.0, 5.0],
            "f_lims": [59.9, 60.1],
        },
        "G3_E11_FM_Modulation": {
            "t_v": [0.0, 1.2],
            "t_f": [0.0, 5.0],
            "f_lims": [58.5, 61.5],
        },
        # Pure phase jump: YOU provide t0 (same units as CSV time!)
        "G3_E12_Phase_Jump": {
            "t_v": [0.95, 1.05],
            "t_f": [0.0, 5.0],
            "f_lims": [59.9, 60.1],
            "phase_jump_t0": 1.000,  # if CSV time is seconds
            "phase_jump_dphi": None,  # optional (rad); e.g., 1.57
        },
        # Composite islanding (example)
        "G3_E12_Composite_Islanding": {
            "t_v": [0.95, 2.0],
            "t_f": [0.0, 5.0],
            "f_lims": [59.9, 60.1],
        },
        "G3_E13_Impulsive_Outliers": {
            "t_v": [0.0, 5.0],
            "t_f": [0.0, 5.0],
            "f_lims": [59.9, 60.1],
        },
        "G3_E14_Noise_Harmonics": {
            "t_v": [0.0, 0.25],
            "t_f": [0.0, 5.0],
            "f_lims": [59.9, 60.1],
        },
        "G3_E15_Multi_Event_Profile": {
            "t_v": [0.9, 3.4],
            "t_f": [0.9, 3.4],
            "f_lims": [59.5, 64.8],
        },
        "G4_E16_Chamorro_Event": {
            "t_v": [2.40, 2.60],
            "t_f": def_f,
            "f_lims": [59.5, 60.5],
        },
    }


def parse_label(sc_id: str) -> Tuple[str, str]:
    parts = sc_id.split("_")
    code = f"{parts[0]}-{parts[1]}"
    desc = (
        " ".join(parts[2:]).replace("plus", "+").replace("p", ".").replace("pct", "%")
    )
    return code, desc


def annotate_freq_step_brace(
    ax,
    t_step: float,
    f_top: float,
    f_bottom: float,
    f0: float = 60.0,
    brace_dx_frac: float = 0.03,
    text_dx_frac: float = 0.012,
    arrow_dx_frac: float = 0.06,
    fontsize: int = 7,
    color: str = "k",
    lw: float = 0.9,
    alpha: float = 0.95,
) -> None:
    """Vertical brace between step levels + Δf label + arrows to each level."""
    xmin, xmax = ax.get_xlim()
    dx = xmax - xmin

    x_brace = t_step - brace_dx_frac * dx
    x_text = x_brace - text_dx_frac * dx
    x_arrow = t_step - arrow_dx_frac * dx

    y0 = float(min(f_top, f_bottom))
    y1 = float(max(f_top, f_bottom))

    ax.annotate(
        "",
        xy=(x_brace, y1),
        xytext=(x_brace, y0),
        arrowprops=dict(
            arrowstyle="|-|", color=color, lw=lw, alpha=alpha, shrinkA=0, shrinkB=0
        ),
        zorder=6,
    )

    df_mag = abs(f_top - f_bottom)
    y_mid = 0.5 * (y0 + y1)
    ax.text(
        x_text,
        y_mid,
        rf"$\Delta f = {df_mag:.1f}\,\mathrm{{Hz}}$",
        fontsize=fontsize,
        color=color,
        alpha=alpha,
        va="center",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        zorder=7,
    )

    df_top = f_top - f0
    ax.annotate(
        rf"$\Delta f = {df_top:+.1f}\,\mathrm{{Hz}}$",
        xy=(t_step, f_top),
        xytext=(x_arrow, f_top),
        textcoords="data",
        fontsize=fontsize,
        color=color,
        alpha=alpha,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=alpha),
        zorder=7,
    )

    df_bot = f_bottom - f0
    ax.annotate(
        rf"$\Delta f = {df_bot:+.1f}\,\mathrm{{Hz}}$",
        xy=(t_step, f_bottom),
        xytext=(x_arrow, f_bottom),
        textcoords="data",
        fontsize=fontsize,
        color=color,
        alpha=alpha,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=alpha),
        zorder=7,
    )


def add_horizontal_brace(
    ax,
    x0: float,
    x1: float,
    y: float,
    h: float,
    text: Optional[str] = None,
    color: str = "crimson",
    lw: float = 2.0,
    alpha: float = 0.95,
    text_offset: float = 0.02,
    fontsize: int = 9,
    zorder: int = 8,
) -> None:
    """Draw a horizontal curly brace from x0 to x1 at baseline y with height h."""
    if x1 < x0:
        x0, x1 = x1, x0

    xm = 0.5 * (x0 + x1)
    dx = x1 - x0

    a = 0.10 * dx
    b = 0.20 * dx

    verts = [
        (x0, y),
        (x0, y + 0.35 * h),
        (x0 + a, y + 0.90 * h),
        (x0 + 2 * a, y + h),
        (xm - b, y + h),
        (xm - b, y + 0.10 * h),
        (xm, y),
        (xm + b, y + 0.10 * h),
        (xm + b, y + h),
        (x1 - 2 * a, y + h),
        (x1 - a, y + 0.90 * h),
        (x1, y + 0.35 * h),
        (x1, y),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    patch = PathPatch(
        Path(verts, codes),
        fill=False,
        color=color,
        lw=lw,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)

    if text:
        yr = ax.get_ylim()
        yspan = yr[1] - yr[0]
        dy_txt = text_offset * yspan * (1.0 if h > 0 else -1.0)
        ax.text(
            xm,
            y + h + dy_txt,
            text,
            color=color,
            fontsize=fontsize,
            ha="center",
            va="bottom" if h > 0 else "top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
            zorder=zorder + 1,
        )


# ============================================================
# Plotter (base)
# ============================================================
class PhysicsDashboardPlotter:
    def __init__(
        self,
        base_path: str = "artifacts/results_mc/waveforms",
        out_dir: str = "artifacts/dashboards",
        style: DashboardStyle = DashboardStyle(),
        layout: LayoutConfig = LayoutConfig(),
        conf_master: Optional[Dict[str, Dict[str, Any]]] = None,
        strict: bool = True,
        verbose: bool = True,
    ) -> None:
        self.base_path = base_path
        self.out_dir = out_dir
        self.style = style
        self.layout = layout
        self.conf_master = (
            conf_master if conf_master is not None else get_scenario_config()
        )
        self.strict = bool(strict)
        self.verbose = bool(verbose)

        os.makedirs(self.out_dir, exist_ok=True)

        if not os.path.isdir(self.base_path):
            raise RuntimeError(f"Base path does not exist: {self.base_path}")

        if self.verbose:
            _log(f"[INFO] base_path = {os.path.abspath(self.base_path)}")
            _log(f"[INFO] out_dir   = {os.path.abspath(self.out_dir)}")
            _log(f"[INFO] strict   = {self.strict}")

    @staticmethod
    def _ieee_axes(ax) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

    def _adaptive_hspace(self, nrows: int, base: float) -> float:
        n = max(1, int(nrows))
        extra = 0.10 + 0.015 * max(0, n - 6)
        return float(min(0.95, base + extra))

    def _csv_path(self, sc_id: str) -> str:
        return os.path.join(self.base_path, sc_id, "ground_truth.csv")

    def _load_waveform_strict(self, sc_id: str) -> pd.DataFrame:
        path = self._csv_path(sc_id)
        if not os.path.exists(path):
            raise RuntimeError(f"[ERROR] Missing CSV for scenario '{sc_id}': {path}")

        df = pd.read_csv(path)

        required = ("t", "real_v", "real_f")
        for c in required:
            if c not in df.columns:
                raise RuntimeError(
                    f"[ERROR] CSV missing required column '{c}' for '{sc_id}': {path}"
                )
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=list(required))
        if df.empty:
            raise RuntimeError(
                f"[ERROR] CSV has no valid rows after cleaning (NaNs) for '{sc_id}': {path}"
            )

        df = df.sort_values("t", kind="mergesort").reset_index(drop=True)

        # CRITICAL FIX: collapse duplicated timestamps (prevents vertical spikes)
        if df["t"].duplicated(keep=False).any():
            df = (
                df.groupby("t", as_index=False, sort=True)[["real_v", "real_f"]]
                .mean()
                .sort_values("t", kind="mergesort")
                .reset_index(drop=True)
            )

        if df.shape[0] < 4:
            raise RuntimeError(
                f"[ERROR] CSV has too few samples after cleaning for '{sc_id}': {path}"
            )

        t = df["t"].to_numpy(dtype=float)
        if np.any(~np.isfinite(t)):
            raise RuntimeError(
                f"[ERROR] Non-finite time values after cleaning for '{sc_id}': {path}"
            )

        return df

    def _force_ticks(self, ax, nx: int, ny: int) -> None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nx))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=ny))
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=self.style.tick_labelsize,
            direction="out",
            length=3.0,
            width=0.6,
            pad=2.6,
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelbottom=True,
        )
        _set_plain_ticks(ax)

    def _set_xlabel_always(self, ax) -> None:
        if self.style.show_xlabel_every_subplot:
            ax.set_xlabel(
                self.style.xlabel_text, fontsize=self.style.label_fontsize, labelpad=1.8
            )
        else:
            ax.set_xlabel("")

    def _auto_conf(self, df: pd.DataFrame, sc_id: str) -> Dict[str, Any]:
        manual = dict(self.conf_master.get(sc_id, {}))
        t = df["t"].to_numpy(dtype=float)
        v = df["real_v"].to_numpy(dtype=float)
        f = df["real_f"].to_numpy(dtype=float)

        if "t_v" not in manual:
            manual["t_v"] = event_window(
                t, v, default_center=2.0, default_width=0.10, width=(0.06, 0.20)
            )
        if "t_f" not in manual:
            manual["t_f"] = event_window(
                t, f, default_center=2.0, default_width=3.0, width=(2.0, 5.0)
            )

        manual["v_lims"] = robust_ylim(v, pad_frac=0.10) or [-1.2, 1.2]
        if "f_lims" not in manual:
            manual["f_lims"] = robust_ylim(f, pad_frac=0.10) or [59.5, 60.5]
        return manual

    @staticmethod
    def _ideal_60hz(t: np.ndarray) -> np.ndarray:
        return np.sin(2.0 * np.pi * 60.0 * t)

    def _overlay_60hz(self, ax, t: np.ndarray) -> None:
        ax.plot(
            t,
            self._ideal_60hz(t),
            color=self.style.ref60_color,
            lw=self.style.ref60_lw,
            linestyle=self.style.ref60_ls,
            alpha=self.style.ref60_alpha,
            label="_nolegend_",
            zorder=2,
        )

    def _is_flat_frequency(self, f: np.ndarray) -> bool:
        f = np.asarray(f, dtype=float)
        f = f[np.isfinite(f)]
        if f.size < 5:
            return False
        return (np.nanmax(f) - np.nanmin(f)) < float(self.style.flat_tol_hz)

    def _prep_frequency_series(
        self, t: np.ndarray, f: np.ndarray, mode: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        if mode == "delta":
            return t, (f - self.style.f0_hz), r"$\Delta f(t)$ [Hz]"
        return t, f, r"$f(t)$ [Hz]"

    @staticmethod
    def _is_dense_voltage_scenario(sc_id: str) -> bool:
        return sc_id in {
            "G2_E8_Fast_Ramp_plus5Hzs",
            "G2_E9_Slow_Ramp_minus0p5Hzs",
            "G3_E10_AM_Modulation",
            "G3_E11_FM_Modulation",
        }

    def _solid_legend(self, ax, handles, labels, loc="upper right", bbox=None) -> None:
        _purge_all_legends(ax)
        if not handles:
            return
        leg = ax.legend(
            handles,
            labels,
            loc=loc,
            bbox_to_anchor=bbox,
            ncol=1,
            fontsize=self.style.anno_fontsize,
            frameon=self.style.legend_frame,
            fancybox=self.style.legend_fancybox,
            framealpha=self.style.legend_alpha,
            facecolor=self.style.legend_facecolor,
            edgecolor=self.style.legend_edgecolor,
            borderaxespad=0.0 if bbox else 0.5,
            handlelength=2.2,
            labelspacing=0.25,
        )
        try:
            leg.get_frame().set_linewidth(self.style.legend_linewidth)
        except Exception:
            pass

    # ---------- plotting (per-scenario pages) ----------
    def _plot_voltage(
        self, fig, gs, row_idx: int, df: pd.DataFrame, sc_id: str, conf: Dict[str, Any]
    ) -> None:
        code, _ = parse_label(sc_id)
        ax = fig.add_subplot(gs[row_idx, 0])

        t = df["t"].to_numpy(dtype=float)
        v = df["real_v"].to_numpy(dtype=float)

        maxp = (
            self.style.max_points_v_dense
            if self._is_dense_voltage_scenario(sc_id)
            else self.style.max_points_v
        )
        tt, vv = _decimate_xy(t, v, max_points=maxp)

        ax.plot(tt, vv, color=self.style.volt_color, lw=self.style.lw_v, zorder=3)
        ax.set_xlim(conf["t_v"])
        ax.set_ylim(conf["v_lims"])

        ax.set_ylabel(r"$v(t)$ [p.u.]", fontsize=self.style.label_fontsize)
        self._set_xlabel_always(ax)
        ax.set_title(
            f"{code}: Voltage",
            pad=self.style.title_pad,
            fontsize=self.style.title_fontsize,
        )

        # Phase jump (per-scenario)
        if "phase_jump_t0" in conf:
            t0 = float(conf["phase_jump_t0"])
            dphi = conf.get("phase_jump_dphi", None)
            # Voltage panel annotation only here; frequency panel will add the full symbolic note
            ax.axvline(t0, color="crimson", ls="--", lw=0.9, alpha=0.95, zorder=6)
            ax.annotate(
                (
                    r"$\Delta\phi$ (phase jump)"
                    if dphi is None
                    else rf"$\Delta\phi={float(dphi):+.2f}\,$rad"
                ),
                xy=(t0, 0.88),
                xycoords=("data", "axes fraction"),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=self.style.anno_fontsize,
                color="crimson",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
                zorder=7,
            )

        self._force_ticks(ax, nx=self.style.nx_v, ny=self.style.ny_v)
        self._ieee_axes(ax)

    def _plot_frequency(
        self, fig, gs, row_idx: int, df: pd.DataFrame, sc_id: str, conf: Dict[str, Any]
    ) -> None:
        code, desc = parse_label(sc_id)
        ax = fig.add_subplot(gs[row_idx, 1])

        t = df["t"].to_numpy(dtype=float)
        f = df["real_f"].to_numpy(dtype=float)

        freq_mode = "delta" if self._is_flat_frequency(f) else "abs"
        tt, ff, ylabel = self._prep_frequency_series(t, f, mode=freq_mode)
        tt2, ff2 = _decimate_xy(tt, ff, max_points=self.style.max_points_f)

        ax.plot(tt2, ff2, color=self.style.freq_color, lw=self.style.lw_f, zorder=3)
        ax.set_xlim(conf["t_f"])

        if freq_mode == "delta":
            span = float(max(0.01, np.nanmax(np.abs(ff2)) * 1.25))
            ax.set_ylim([-span, +span])
            ax.axhline(0.0, color="0.35", lw=0.75, ls=":", alpha=0.6, zorder=2)
            ax.text(
                0.02,
                0.10,
                r"Nominal: $|\Delta f|<0.02$ Hz",
                transform=ax.transAxes,
                fontsize=self.style.anno_fontsize,
                color="0.35",
                alpha=0.85,
                ha="left",
                va="bottom",
            )
        else:
            ax.set_ylim(conf["f_lims"])

        # Phase jump (per-scenario) — draw correctly (NO spike)
        if "phase_jump_t0" in conf:
            t0 = float(conf["phase_jump_t0"])
            dphi = conf.get("phase_jump_dphi", None)
            ax.axvline(t0, color="crimson", ls="--", lw=0.9, alpha=0.95, zorder=6)
            ax.annotate(
                r"$f(t)=f_0+\frac{\Delta\phi}{2\pi}\,\delta(t-t_0)$",
                xy=(t0, 0.88),
                xycoords=("data", "axes fraction"),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=self.style.anno_fontsize,
                color="crimson",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
                zorder=7,
            )
            # optional tiny tag with dphi
            if dphi is not None:
                ax.text(
                    0.02,
                    0.70,
                    rf"$\Delta\phi={float(dphi):+.2f}$ rad",
                    transform=ax.transAxes,
                    fontsize=self.style.anno_fontsize,
                    color="crimson",
                    alpha=0.95,
                    ha="left",
                    va="top",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.1),
                    zorder=7,
                )

        ax.set_ylabel(ylabel, fontsize=self.style.label_fontsize)
        self._set_xlabel_always(ax)
        ax.set_title(
            f"{code}: Frequency ({desc})",
            pad=self.style.title_pad,
            fontsize=self.style.title_fontsize,
        )

        self._force_ticks(ax, nx=self.style.nx_f, ny=self.style.ny_f)
        self._ieee_axes(ax)

    @staticmethod
    def _batches(items: List[str], batch_size: int) -> List[List[str]]:
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    def _make_figure(self, nrows: int) -> Tuple[plt.Figure, gridspec.GridSpec]:
        fig = plt.figure(figsize=self.layout.figsize)
        hs = self._adaptive_hspace(nrows=nrows, base=self.layout.hspace)
        gs = gridspec.GridSpec(
            nrows,
            2,
            figure=fig,
            width_ratios=list(self.layout.width_ratios),
            hspace=hs,
            wspace=self.layout.wspace,
        )
        return fig, gs

    def _finalize_and_save(self, fig: plt.Figure, page_idx: int, title: str) -> None:
        fig.suptitle(title, y=self.layout.suptitle_y, fontsize=10, fontweight="bold")
        fig.subplots_adjust(
            left=self.layout.left,
            right=self.layout.right,
            bottom=self.layout.bottom,
            top=self.layout.top,
        )

        base = os.path.join(self.out_dir, f"Dashboard_Physics_P{page_idx}")
        fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(f"{base}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
        plt.close(fig)

    def _validate_all_scenarios_exist(self, scenario_ids: List[str]) -> None:
        missing = [sc for sc in scenario_ids if not os.path.exists(self._csv_path(sc))]
        if missing:
            lines = ["[ERROR] Missing required scenarios (CSV not found):"]
            lines += [f"  - {sc}: {self._csv_path(sc)}" for sc in missing]
            raise RuntimeError("\n".join(lines))

    def run(self) -> None:
        sc_list = list(self.conf_master.keys())
        if self.strict:
            self._validate_all_scenarios_exist(sc_list)

        batches = self._batches(sc_list, self.layout.scenarios_per_page)
        total_loaded = 0

        for p_idx, batch in enumerate(batches, start=1):
            fig, gs = self._make_figure(nrows=len(batch))
            for row_idx, sc_id in enumerate(batch):
                df = self._load_waveform_strict(sc_id)
                total_loaded += 1
                conf = self._auto_conf(df, sc_id)

                self._plot_voltage(fig, gs, row_idx, df, sc_id, conf)
                self._plot_frequency(fig, gs, row_idx, df, sc_id, conf)

            self._finalize_and_save(
                fig,
                page_idx=p_idx,
                title=f"Benchmark Excitation Scenarios — Physics Layer (Page {p_idx})",
            )

        _log(
            f"✅ Physics dashboard (per-scenario) generated in {self.out_dir} (loaded={total_loaded})"
        )


# ============================================================
# Comparative (Grouped) Plotter
# ============================================================
class ComparativePhysicsDashboardPlotter(PhysicsDashboardPlotter):
    ROW_SPECS_FIXED: List[Dict[str, Any]] = [
        {
            "scenario_ids": [
                "G1_E3_Gaussian_Noise_5pct",
                "G1_E2_Gaussian_Noise_1pct",
                "G1_E1_Pure_60Hz",
            ],
            "title": "Group G1 (E3 → E2 → E1)",
            "show_ref60": False,
            "x": {"t_v": [1.98, 2.02], "t_f": [1.95, 2.05]},
            "legend": {"loc": "upper right"},
        },
        {
            "scenario_ids": [
                "G2_E5_Voltage_Mag_Step_10pct",
                "G2_E4_Voltage_Mag_Step_1pct",
            ],
            "title": "Group G2 (E5 → E4) Voltage Steps",
            "show_ref60": True,
            "x": {"t_v": [2.00, 2.04], "t_f": [1.98, 2.02]},
            "y": {"v_lims": [-1.1, 1.1]},
            "legend": {"loc": "upper right"},
        },
        {
            "scenario_ids": ["G2_E7_Freq_Step_60_to_55", "G2_E6_Freq_Step_60_to_59p5"],
            "title": "Group G2 (E7 → E6) Frequency Steps",
            "show_ref60": True,
            "x": {"t_v": [2.00, 2.04], "t_f": [1.98, 2.02]},
            "y": {"f_lims": [52.5, 61.5]},
            "legend": {"loc": "upper right"},
        },
        {
            "scenario_ids": ["G2_E8_Fast_Ramp_plus5Hzs", "G2_E9_Slow_Ramp_minus0p5Hzs"],
            "title": "Group G2 (E8 → E9) Frequency Ramps",
            "show_ref60": True,
            "x": {"t_v": [0.90, 1.5], "t_f": [0.5, 5.0]},
            "y": {"f_lims": [57.0, 80.5]},
            "legend": {"loc": "upper right"},
        },
    ]

    ROW_SPECS_SINGLES: List[Dict[str, Any]] = [
        {
            "scenario_ids": ["G3_E10_AM_Modulation"],
            "title": "Single: G3-E10 AM",
            "show_ref60": True,
            "x": {"t_v": [0.0, 2.0], "t_f": [0.0, 5.0]},
            "y": {"f_lims": [59.9, 60.1]},
        },
        {
            "scenario_ids": ["G3_E11_FM_Modulation"],
            "title": "Single: G3-E11 FM",
            "show_ref60": True,
            "x": {"t_v": [0.0, 1.2], "t_f": [0.0, 5.0]},
            "y": {"f_lims": [58.5, 61.5]},
        },
        {
            "scenario_ids": ["G3_E12_Phase_Jump"],
            "title": "Single: G3-E12 Phase Jump (Pure)",
            "show_ref60": True,
            "x": {"t_v": [0.95, 1.05], "t_f": [0.0, 5.0]},
            "y": {"f_lims": [59.9, 60.1]},
        },
        {
            "scenario_ids": ["G3_E12_Composite_Islanding"],
            "title": "Single: G3-E12 Composite Islanding (Realistic)",
            "show_ref60": True,
            "x": {"t_v": [0.95, 2.0], "t_f": [0.0, 5.0]},
            "y": {"f_lims": [59.9, 60.1]},
        },
        {
            "scenario_ids": ["G3_E13_Impulsive_Outliers"],
            "title": "Single: G3-E13 Impulsive Outliers",
            "show_ref60": True,
            "x": {"t_v": [1.85, 2.25], "t_f": [1.85, 2.25]},
            "y": {"f_lims": [59.9, 60.1]},
        },
        {
            "scenario_ids": ["G3_E14_Noise_Harmonics"],
            "title": "Single: G3-E14 Harmonics + Noise",
            "show_ref60": True,
            "x": {"t_v": [0.0, 0.25], "t_f": [0.0, 5.0]},
            "y": {"f_lims": [59.9, 60.1]},
        },
        {
            "scenario_ids": ["G3_E15_Multi_Event_Profile"],
            "title": "Single: G3-E15 Multi-Event",
            "show_ref60": True,
            "x": {"t_v": [0.9, 3.4], "t_f": [0.9, 3.4]},
            "y": {"f_lims": [59.5, 64.8]},
        },
        {
            "scenario_ids": ["G4_E16_Chamorro_Event"],
            "title": "Single: G4-E16 Chamorro",
            "show_ref60": False,
            "x": {"t_v": [2.40, 2.60], "t_f": [0.0, 5.0]},
            "y": {"f_lims": [59.5, 60.5]},
        },
    ]

    GROUP_LINESTYLES: Tuple[str, ...] = ("-", ":", "--", "-.")

    def _apply_manual_limits(
        self, ax_v, ax_f, row_spec: Dict[str, Any], confs: List[Dict[str, Any]]
    ) -> None:
        x = row_spec.get("x", {}) or {}
        y = row_spec.get("y", {}) or {}

        if "t_v" in x:
            ax_v.set_xlim(list(x["t_v"]))
        elif confs and "t_v" in confs[0]:
            ax_v.set_xlim(list(confs[0]["t_v"]))

        if "t_f" in x:
            ax_f.set_xlim(list(x["t_f"]))
        elif confs and "t_f" in confs[0]:
            ax_f.set_xlim(list(confs[0]["t_f"]))

        if "v_lims" in y:
            ax_v.set_ylim(list(y["v_lims"]))
        else:
            ax_v.set_ylim(self._merge_ylims(confs, "v_lims", fallback=[-1.2, 1.2]))

        if "f_lims" in y:
            ax_f.set_ylim(list(y["f_lims"]))
        else:
            ax_f.set_ylim(self._merge_ylims(confs, "f_lims", fallback=[59.5, 60.5]))

    @staticmethod
    def _merge_ylims(
        confs: List[Dict[str, Any]], key: str, fallback: List[float]
    ) -> List[float]:
        lows, highs = [], []
        for c in confs:
            if key in c and c[key] is not None:
                lows.append(float(c[key][0]))
                highs.append(float(c[key][1]))
        if not lows or not highs:
            return fallback
        return [min(lows), max(highs)]

    def _labels_for_group(self, sc_ok: List[str]) -> List[str]:
        s = set(sc_ok)
        if s == {
            "G1_E1_Pure_60Hz",
            "G1_E2_Gaussian_Noise_1pct",
            "G1_E3_Gaussian_Noise_5pct",
        }:
            m = {
                "G1_E1_Pure_60Hz": "Noise = 0%",
                "G1_E2_Gaussian_Noise_1pct": "Noise = 1%",
                "G1_E3_Gaussian_Noise_5pct": "Noise = 5%",
            }
            return [m[x] for x in sc_ok]
        if s == {"G2_E4_Voltage_Mag_Step_1pct", "G2_E5_Voltage_Mag_Step_10pct"}:
            m = {
                "G2_E4_Voltage_Mag_Step_1pct": r"$\Delta V=-1\%$",
                "G2_E5_Voltage_Mag_Step_10pct": r"$\Delta V=-10\%$",
            }
            return [m[x] for x in sc_ok]
        if s == {"G2_E6_Freq_Step_60_to_59p5", "G2_E7_Freq_Step_60_to_55"}:
            m = {
                "G2_E6_Freq_Step_60_to_59p5": r"$\Delta f=-0.5\,\mathrm{Hz}$",
                "G2_E7_Freq_Step_60_to_55": r"$\Delta f=-5.0\,\mathrm{Hz}$",
            }
            return [m[x] for x in sc_ok]
        if s == {"G2_E8_Fast_Ramp_plus5Hzs", "G2_E9_Slow_Ramp_minus0p5Hzs"}:
            m = {
                "G2_E8_Fast_Ramp_plus5Hzs": r"RoCoF $=+5.0\,\mathrm{Hz/s}$",
                "G2_E9_Slow_Ramp_minus0p5Hzs": r"RoCoF $=-0.5\,\mathrm{Hz/s}$",
            }
            return [m[x] for x in sc_ok]
        return [parse_label(x)[0] for x in sc_ok]

    def _build_group_handles(self, labels: List[str]) -> List[Line2D]:
        handles: List[Line2D] = []
        for i in range(len(labels)):
            ls = self.GROUP_LINESTYLES[i % len(self.GROUP_LINESTYLES)]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=self.style.volt_color,
                    lw=self.style.lw_v,
                    linestyle=ls,
                )
            )
        return handles

    def _plot_row(
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        row_idx: int,
        row_spec: Dict[str, Any],
    ) -> None:
        group = list(row_spec["scenario_ids"])
        title = str(row_spec.get("title", "Grouped"))
        show_ref60 = bool(row_spec.get("show_ref60", False))

        if self.strict:
            self._validate_all_scenarios_exist(group)

        dfs: List[pd.DataFrame] = []
        confs: List[Dict[str, Any]] = []
        for sc_id in group:
            df = self._load_waveform_strict(sc_id)
            dfs.append(df)
            confs.append(self._auto_conf(df, sc_id))

        flat_flags = [
            self._is_flat_frequency(df["real_f"].to_numpy(dtype=float)) for df in dfs
        ]
        row_freq_mode = "delta" if all(flat_flags) else "abs"

        ax_v = fig.add_subplot(gs[row_idx, 0])
        ax_f = fig.add_subplot(gs[row_idx, 1])

        _purge_all_legends(ax_v)
        _purge_all_legends(ax_f)

        for i, (sc_id, df) in enumerate(zip(group, dfs)):
            ls = self.GROUP_LINESTYLES[i % len(self.GROUP_LINESTYLES)]
            t = df["t"].to_numpy(dtype=float)

            v = df["real_v"].to_numpy(dtype=float)
            maxp = (
                self.style.max_points_v_dense
                if self._is_dense_voltage_scenario(sc_id)
                else self.style.max_points_v
            )
            tt, vv = _decimate_xy(t, v, max_points=maxp)
            ax_v.plot(
                tt,
                vv,
                color=self.style.volt_color,
                lw=self.style.lw_v,
                linestyle=ls,
                zorder=3,
            )

            f = df["real_f"].to_numpy(dtype=float)
            tf, ff, _ = self._prep_frequency_series(t, f, mode=row_freq_mode)
            tf2, ff2 = _decimate_xy(tf, ff, max_points=self.style.max_points_f)
            ax_f.plot(
                tf2,
                ff2,
                color=self.style.freq_color,
                lw=self.style.lw_f,
                linestyle=ls,
                zorder=3,
            )

            if show_ref60:
                self._overlay_60hz(ax_v, t)

        ax_v.set_ylabel(r"$v(t)$ [p.u.]", fontsize=self.style.label_fontsize)
        ax_f.set_ylabel(
            (r"$\Delta f(t)$ [Hz]" if row_freq_mode == "delta" else r"$f(t)$ [Hz]"),
            fontsize=self.style.label_fontsize,
        )

        self._set_xlabel_always(ax_v)
        self._set_xlabel_always(ax_f)

        ax_v.set_title(
            f"{title} — Voltage",
            fontsize=self.style.title_fontsize,
            pad=self.style.title_pad,
        )
        ax_f.set_title(
            f"{title} — Frequency",
            fontsize=self.style.title_fontsize,
            pad=self.style.title_pad,
        )

        self._apply_manual_limits(ax_v, ax_f, row_spec=row_spec, confs=confs)

        if (("Frequency Steps" in title) or title.startswith("Group G2 (E7")) and (
            row_freq_mode == "abs"
        ):
            annotate_freq_step_brace(
                ax=ax_f,
                t_step=1.99,
                f_top=59.5,
                f_bottom=55.0,
                f0=self.style.f0_hz,
                fontsize=self.style.anno_fontsize,
                color="k",
                lw=0.9,
                alpha=0.95,
            )

        if row_freq_mode == "delta":
            all_ff = np.concatenate(
                [df["real_f"].to_numpy(dtype=float) - self.style.f0_hz for df in dfs]
            )
            span = float(max(0.01, np.nanpercentile(np.abs(all_ff), 99) * 1.25))
            ax_f.set_ylim([-span, +span])
            ax_f.axhline(0.0, color="0.35", lw=0.75, ls=":", alpha=0.6, zorder=2)
            ax_f.text(
                0.02,
                0.10,
                r"Nominal: $|\Delta f|<0.02$ Hz",
                transform=ax_f.transAxes,
                fontsize=self.style.anno_fontsize,
                color="0.35",
                alpha=0.85,
                ha="left",
                va="bottom",
            )

        # ============================================================
        # PHASE JUMP ANNOTATION (GROUPED)  ✅ FIX
        # ============================================================
        pj_conf = None
        for c in confs:
            if isinstance(c, dict) and ("phase_jump_t0" in c):
                pj_conf = c
                break

        if pj_conf is not None:
            t0 = float(pj_conf["phase_jump_t0"])
            dphi = pj_conf.get("phase_jump_dphi", None)
            annotate_phase_jump(
                ax_v=ax_v,
                ax_f=ax_f,
                t0=t0,
                delta_phi_rad=None if dphi is None else float(dphi),
                fontsize=self.style.anno_fontsize,
            )

        self._force_ticks(ax_v, nx=self.style.nx_v, ny=self.style.ny_v)
        self._force_ticks(ax_f, nx=self.style.nx_f, ny=self.style.ny_f)
        self._ieee_axes(ax_v)
        self._ieee_axes(ax_f)

        labels_pretty = self._labels_for_group(group)
        handles = self._build_group_handles(labels_pretty)

        leg_cfg = row_spec.get("legend", {}) or {}
        loc = leg_cfg.get("loc", "upper right")
        bbox = leg_cfg.get("bbox_to_anchor", None)

        if len(labels_pretty) > 1:
            self._solid_legend(ax_v, handles, labels_pretty, loc=loc, bbox=bbox)

    def run(self) -> None:
        all_specs = self.ROW_SPECS_FIXED + self.ROW_SPECS_SINGLES
        rows_per_page = 11
        batches = [
            all_specs[i : i + rows_per_page]
            for i in range(0, len(all_specs), rows_per_page)
        ]

        if self.strict:
            required: List[str] = sorted(
                {sc for rs in all_specs for sc in rs["scenario_ids"]}
            )
            self._validate_all_scenarios_exist(required)

        total_rows = 0
        for p_idx, batch in enumerate(batches, start=1):
            nrows = len(batch)
            fig = plt.figure(figsize=self.layout.comp_figsize)

            hs = self._adaptive_hspace(nrows=nrows, base=self.layout.comp_hspace)
            gs = gridspec.GridSpec(
                nrows,
                2,
                figure=fig,
                width_ratios=list(self.layout.comp_width_ratios),
                hspace=hs,
                wspace=self.layout.comp_wspace,
            )

            for row_idx, row_spec in enumerate(batch):
                total_rows += 1
                self._plot_row(fig, gs, row_idx, row_spec)

            fig.suptitle(
                f"Benchmark Excitation Scenarios — Physics Layer (Grouped) (Page {p_idx})",
                y=self.layout.comp_suptitle_y,
                fontsize=10,
                fontweight="bold",
            )
            fig.subplots_adjust(
                left=self.layout.comp_left,
                right=self.layout.comp_right,
                bottom=self.layout.comp_bottom,
                top=self.layout.comp_top,
            )

            base = os.path.join(self.out_dir, f"Dashboard_Physics_Comparative_P{p_idx}")
            fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)
            fig.savefig(f"{base}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
            plt.close(fig)

        _log(
            f"✅ Comparative dashboard generated in {self.out_dir} (rows={total_rows})"
        )


# ============================================================
# Entry points
# ============================================================
def create_physics_dashboard() -> None:
    PhysicsDashboardPlotter().run()
    ComparativePhysicsDashboardPlotter().run()


def create_comparative_physics_dashboard() -> None:
    ComparativePhysicsDashboardPlotter().run()


if __name__ == "__main__":
    create_physics_dashboard()
