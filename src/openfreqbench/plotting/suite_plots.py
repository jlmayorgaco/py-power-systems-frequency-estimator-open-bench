"""
openfreqbench/plotting/suite_plots.py

IEEE-grade suite-level benchmark figures.

Public API:
    plot_scenario_overview(scenarios, out_path)
        2-row x N-col panel: v(t) waveform + f_true(t) profile with event annotations.

    plot_benchmark_grid(grid_data, scenario_ids, estimator_ids, out_path)
        N_scenarios x N_estimators tracking grid with MC p5-p95 band + stats annotation.

Design choices:
    - apply_ieee_style() sets global rcParams (serif-like fonts, no top/right spines)
    - Okabe-Ito colorblind-safe palette from styles.py
    - Event markers (step, ramp onset) annotated with vertical dashed lines
    - Each grid cell annotates RMSE mean ± std and p95 error
    - Data down-sampled to <= 4000 pts for vector-safe PDF rendering
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from openfreqbench.plotting.styles import (
    C, apply_ieee_style, annotate_box, color_for_estimator,
    DOUBLE_COL_W, FIG_H_UNIT,
)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_MAX_PTS = 4_000   # max data points per line (for readable vector output)


def _thin(arr: np.ndarray, max_pts: int = _MAX_PTS) -> Tuple[np.ndarray, slice]:
    n = len(arr)
    if n <= max_pts:
        return arr, slice(None)
    step = max(1, n // max_pts)
    return arr[::step], slice(None, None, step)


def _ms(t: np.ndarray) -> np.ndarray:
    return t * 1e3


def _mc_matrix(
    f_hat_all: List[np.ndarray],
    n: int,
    latency: int,
) -> np.ndarray:
    """Stack MC runs into (n_runs, n) matrix with NaN padding."""
    L = max(0, latency)
    mat = np.full((len(f_hat_all), n), np.nan)
    for k, fh in enumerate(f_hat_all):
        fh = np.asarray(fh, dtype=float)
        m  = min(len(fh) - L, n)
        if m > 0:
            mat[k, :m] = fh[L: L + m]
    return mat


def _save(fig: plt.Figure, path: Path, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: infer event annotations from scenario schema
# ─────────────────────────────────────────────────────────────────────────────

def _event_markers(schema: dict) -> List[Tuple[float, str]]:
    """
    Return list of (time_s, label) for key scenario events.

    Reads fields from scenario state.schema.
    """
    markers: List[Tuple[float, str]] = []
    sid = str(schema.get("scenario_id", ""))
    if "step_time_s" in schema:
        t_step = float(schema["step_time_s"])
        f1     = schema.get("f1_hz", "?")
        f2     = schema.get("f2_hz", "?")
        markers.append((t_step, f"f: {f1}→{f2} Hz"))
    if "onset_time_s" in schema:
        t_on   = float(schema["onset_time_s"])
        rate   = schema.get("ramp_rate_hzs", "?")
        markers.append((t_on, f"ramp {rate} Hz/s"))
    return markers


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scenario Overview — publication-grade
# ─────────────────────────────────────────────────────────────────────────────

def plot_scenario_overview(
    scenarios: List[dict],
    out_path: Path,
    *,
    n_cycles_waveform: int = 4,
    dpi: int = 300,
) -> Path:
    """
    2-row x N-col panel.

    Row 0: v(t) — first n_cycles_waveform cycles, with zero-crossing dashed line.
    Row 1: f_true(t) — full duration, with f_nom reference, event markers.

    Each entry in `scenarios` is a dict:
        id       : str
        t        : np.ndarray
        v        : np.ndarray
        f_true   : np.ndarray
        fs_hz    : float
        f_nom_hz : float
        schema   : dict   (optional — for event markers)
    """
    apply_ieee_style()

    n_scen  = len(scenarios)
    if n_scen == 0:
        return Path(out_path)

    fig_w   = max(DOUBLE_COL_W, DOUBLE_COL_W / 3 * n_scen)
    fig_h   = FIG_H_UNIT * 2.4
    fig, axes = plt.subplots(
        2, n_scen,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
    )

    for col, scen in enumerate(scenarios):
        t        = np.asarray(scen["t"],      dtype=float)
        v        = np.asarray(scen["v"],      dtype=float)
        f_true   = np.asarray(scen["f_true"], dtype=float)
        fs_hz    = float(scen.get("fs_hz",    10_000.0))
        f_nom    = float(scen.get("f_nom_hz", 60.0))
        label    = str(scen.get("id", f"S{col}"))
        schema   = scen.get("schema", {})
        markers  = _event_markers(schema)

        t_thin, sl = _thin(t)
        t_ms_thin  = _ms(t_thin)

        # ── Row 0: voltage waveform ──────────────────────────────────────────
        ax0     = axes[0, col]
        n_show  = min(len(t_thin),
                      int(n_cycles_waveform / max(f_nom, 1.0) * fs_hz))
        ax0.plot(t_ms_thin[:n_show], v[sl][:n_show],
                 color=C.EST_0, linewidth=0.9, zorder=3)
        ax0.axhline(0, color=C.ZERO_LINE, linewidth=0.5, linestyle="--", zorder=2)
        ax0.set_title(label, fontweight="bold", pad=4)
        ax0.set_ylabel("v(t) [pu]" if col == 0 else "")
        ax0.set_xlabel("Time [ms]")
        ax0.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        _despine(ax0)

        # ── Row 1: f_true profile ─────────────────────────────────────────────
        ax1 = axes[1, col]
        ax1.plot(t_ms_thin, f_true[sl],
                 color=C.TRUE, linewidth=1.0, zorder=3, label="$f_{true}$")
        ax1.axhline(f_nom, color=C.ZERO_LINE, linewidth=0.5,
                    linestyle="--", zorder=2, label=f"$f_{{nom}}$={f_nom} Hz")

        # Event markers
        for t_ev, ev_label in markers:
            t_ev_ms = t_ev * 1e3
            ax1.axvline(t_ev_ms, color=C.EVENT, linewidth=0.8,
                        linestyle=":", zorder=4)
            ax1.text(t_ev_ms, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else f_nom + 0.5,
                     ev_label, rotation=90, va="top", ha="right",
                     fontsize=6, color=C.EVENT, zorder=5)

        ax1.set_ylabel("$f_{true}$ [Hz]" if col == 0 else "")
        ax1.set_xlabel("Time [ms]")
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        _despine(ax1)

        # Add event labels after y-limits are set (re-annotate)
        if markers:
            y_hi = ax1.get_ylim()[1]
            for t_ev, ev_label in markers:
                ax1.texts[-len(markers)].set_y(y_hi)   # skip if already correct

    # Global title
    fig.suptitle(
        "OpenFreqBench — Scenario Overview",
        fontsize=9, fontweight="bold", y=1.02,
    )
    fig.align_ylabels(axes[:, 0])
    return _save(fig, Path(out_path), dpi=dpi)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Benchmark Grid — publication-grade
# ─────────────────────────────────────────────────────────────────────────────

def plot_benchmark_grid(
    grid_data: List[List[Optional[dict]]],
    scenario_ids: List[str],
    estimator_ids: List[str],
    out_path: Path,
    *,
    dpi: int = 300,
) -> Path:
    """
    N_scenarios-row x N_estimators-col frequency-tracking grid.

    Each cell dict:
        t         : np.ndarray
        f_true    : np.ndarray
        f_hat_ref : np.ndarray  (reference trace, seed=0)
        f_hat_all : List[np.ndarray]  (all MC traces)
        latency   : int
        rmse_mean : float    (mean over MC runs)
        rmse_std  : float    (std over MC runs)
        p95_err   : float    (p95 of |error| over MC runs)
        f_nom     : float
        schema    : dict     (optional, for event markers)
    or None if pair was skipped.
    """
    apply_ieee_style()

    n_rows  = len(scenario_ids)
    n_cols  = len(estimator_ids)
    if n_rows == 0 or n_cols == 0:
        return Path(out_path)

    fig_w   = max(DOUBLE_COL_W, DOUBLE_COL_W / 3 * n_cols)
    fig_h   = FIG_H_UNIT * n_rows + 0.6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"hspace": 0.55, "wspace": 0.30},
    )

    for row, scen_id in enumerate(scenario_ids):
        for col, est_id in enumerate(estimator_ids):
            ax    = axes[row, col]
            color = color_for_estimator(est_id)
            cell  = grid_data[row][col]

            if cell is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color=C.ZERO_LINE)
                ax.set_axis_off()
                continue

            t         = np.asarray(cell["t"],         dtype=float)
            f_true    = np.asarray(cell["f_true"],     dtype=float)
            f_hat_ref = np.asarray(cell["f_hat_ref"],  dtype=float)
            f_hat_all = cell.get("f_hat_all", [])
            latency   = int(cell.get("latency", 0))
            rmse_mean = float(cell.get("rmse_mean", float("nan")))
            rmse_std  = float(cell.get("rmse_std",  float("nan")))
            p95_err   = float(cell.get("p95_err",   float("nan")))
            f_nom     = float(cell.get("f_nom",     60.0))
            schema    = cell.get("schema", {})
            markers   = _event_markers(schema)

            n = len(t)
            L = max(0, latency)
            t_thin, sl = _thin(t)
            t_ms = _ms(t_thin)

            # Ground truth
            ax.plot(t_ms, f_true[sl],
                    color=C.TRUE, linewidth=0.9, linestyle="-",
                    label="$f_{true}$", zorder=2)

            # MC p5-p95 band
            if f_hat_all:
                mat = _mc_matrix(f_hat_all, n, latency)
                with np.errstate(all="ignore"):
                    p5  = np.nanpercentile(mat, 5,  axis=0)
                    p95 = np.nanpercentile(mat, 95, axis=0)
                ax.fill_between(t_ms, p5[sl], p95[sl],
                                color=color, alpha=0.15, linewidth=0, zorder=3)

            # Reference trace (seed=0, causal-shifted)
            ref = np.full(n, np.nan)
            if len(f_hat_ref) > L:
                m = min(len(f_hat_ref) - L, n)
                ref[:m] = f_hat_ref[L: L + m]
            ax.plot(t_ms, ref[sl], color=color, linewidth=1.0,
                    label=est_id, zorder=4)

            # Event markers
            for t_ev, ev_label in markers:
                ax.axvline(_ms(np.array([t_ev]))[0], color=C.EVENT,
                           linewidth=0.7, linestyle=":", zorder=5)

            # Stats annotation
            lines = []
            if np.isfinite(rmse_mean):
                rmse_mhz = rmse_mean * 1e3
                if np.isfinite(rmse_std):
                    lines.append(f"RMSE: {rmse_mhz:.2f}±{rmse_std*1e3:.2f} mHz")
                else:
                    lines.append(f"RMSE: {rmse_mhz:.2f} mHz")
            if np.isfinite(p95_err):
                lines.append(f"p95: {p95_err*1e3:.2f} mHz")
            if lines:
                annotate_box(ax, "\n".join(lines), loc="upper right",
                             fontsize=6, color=color)

            # Axis labels
            if row == 0:
                ax.set_title(est_id, fontweight="bold", pad=3)
            if col == 0:
                ax.set_ylabel(f"{scen_id}\nf [Hz]", fontsize=6.5)
            else:
                ax.set_ylabel("")
            if row == n_rows - 1:
                ax.set_xlabel("Time [ms]")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            _despine(ax)

    # Shared legend
    legend_handles = [
        plt.Line2D([0], [0], color=C.TRUE, linewidth=1.0,
                   linestyle="-", label="$f_{true}$"),
        plt.Line2D([0], [0], color="#888888", linewidth=0.8,
                   linestyle="none", marker="s", markersize=7,
                   alpha=0.3, label="MC p5–p95"),
        plt.Line2D([0], [0], color=C.EVENT, linewidth=0.7,
                   linestyle=":", label="Event onset"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=7,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0,
    )

    fig.suptitle(
        "OpenFreqBench — Frequency Tracking Grid (MC p5–p95 band)",
        fontsize=9, fontweight="bold", y=1.06,
    )
    return _save(fig, Path(out_path), dpi=dpi)
