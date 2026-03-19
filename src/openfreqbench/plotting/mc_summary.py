"""
openfreqbench/plotting/mc_summary.py

Monte Carlo statistical summary plots for the benchmark.

plot_mc_distributions(result, out_path)
    For each scenario: one row of violin plots showing RMSE distribution
    across estimators.  Each violin shows the full MC distribution.

plot_metric_heatmap(result, metric, out_path)
    N_scenarios x N_estimators colour-coded heatmap of a scalar metric
    (e.g. mean RMSE, p95 error, CV).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

from openfreqbench.plotting.styles import (
    C, apply_ieee_style, annotate_box, color_for_estimator,
    DOUBLE_COL_W, FIG_H_UNIT,
)


def _save(fig: plt.Figure, path: Path, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# 1. RMSE violin / box strip plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_mc_distributions(
    result,               # SmokeResult
    out_path: Path,
    *,
    metric:   str  = "RMSE_HZ",
    unit_mhz: bool = True,
    dpi:      int  = 300,
) -> Path:
    """
    Grouped violin plots: one row per scenario, one violin per estimator.

    Shows the full Monte Carlo distribution of `metric` across all seeds.

    Parameters
    ----------
    result : SmokeResult
    metric : str — metric key from TraceResult.metrics (e.g. "RMSE_HZ")
    unit_mhz : if True, convert Hz → mHz for display
    """
    apply_ieee_style()

    scenarios   = result.scenarios
    estimators  = result.estimators
    n_scen      = len(scenarios)
    n_est       = len(estimators)

    if n_scen == 0 or n_est == 0:
        return Path(out_path)

    scale  = 1e3 if unit_mhz else 1.0
    ylabel = f"{metric} [mHz]" if unit_mhz else f"{metric} [Hz]"

    fig_h = FIG_H_UNIT * n_scen + 0.6
    fig, axes = plt.subplots(
        n_scen, 1,
        figsize=(DOUBLE_COL_W, fig_h),
        squeeze=False,
        gridspec_kw={"hspace": 0.55},
    )

    positions = list(range(n_est))

    for row, scen_id in enumerate(scenarios):
        ax = axes[row, 0]

        all_data: List[np.ndarray] = []
        colors:   List[str]        = []
        labels:   List[str]        = []
        any_data  = False

        for col, est_id in enumerate(estimators):
            pair = result.get_pair(scen_id, est_id)
            color = color_for_estimator(est_id)
            colors.append(color)
            labels.append(est_id)

            if pair is None or not pair.traces:
                all_data.append(np.array([]))
                continue

            vals = []
            for tr in pair.traces:
                entry = tr.metrics.get(metric)
                if isinstance(entry, dict):
                    v = entry.get("value")
                else:
                    v = entry
                if v is not None and np.isfinite(float(v)):
                    vals.append(float(v) * scale)
            arr = np.array(vals, dtype=float)
            all_data.append(arr)
            if arr.size > 0:
                any_data = True

        if not any_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color=C.ZERO_LINE)
            ax.set_axis_off()
            continue

        # Violin plot — only for distributions with >= 4 points
        for pos, (arr, color, lbl) in enumerate(zip(all_data, colors, labels)):
            if arr.size >= 4:
                parts = ax.violinplot(
                    arr, positions=[pos],
                    showmeans=False, showmedians=False, showextrema=False,
                    widths=0.7,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_edgecolor(color)
                    pc.set_alpha(0.35)

            if arr.size > 0:
                # Median line
                ax.hlines(np.median(arr), pos - 0.3, pos + 0.3,
                          color=color, linewidth=1.5, zorder=4)
                # IQR box
                q25, q75 = np.percentile(arr, [25, 75])
                ax.vlines(pos, q25, q75, color=color, linewidth=3.5,
                          alpha=0.6, zorder=3)
                # Individual points (jittered)
                rng = np.random.default_rng(42)
                jitter = rng.uniform(-0.08, 0.08, size=arr.size)
                ax.scatter(pos + jitter, arr,
                           color=color, s=8, alpha=0.45, zorder=5,
                           linewidths=0)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(scen_id, fontweight="bold", fontsize=8, pad=3)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(True)
        ax.yaxis.get_major_formatter().set_powerlimits((-2, 3))
        _despine(ax)

        # Log scale if range spans > 2 decades
        finite_vals = [v for a in all_data for v in a if np.isfinite(v) and v > 0]
        if finite_vals:
            vmin, vmax = min(finite_vals), max(finite_vals)
            if vmax > 0 and vmin > 0 and vmax / vmin > 100:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(mticker.LogFormatter())

    fig.suptitle(
        f"Monte Carlo Distribution — {metric}",
        fontsize=9, fontweight="bold", y=1.02,
    )
    return _save(fig, Path(out_path), dpi=dpi)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metric heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_heatmap(
    result,
    out_path: Path,
    *,
    metric:   str   = "RMSE_HZ",
    stat:     str   = "mean",
    unit_mhz: bool  = True,
    dpi:      int   = 300,
) -> Path:
    """
    N_scenarios x N_estimators heatmap of a scalar metric statistic.

    Lower is better colour scheme (viridis_r).
    Each cell shows the value as text.

    Parameters
    ----------
    stat : aggregated stat key — "mean", "p95", "std", "median", etc.
    """
    apply_ieee_style()

    scenarios  = result.scenarios
    estimators = result.estimators
    n_rows     = len(scenarios)
    n_cols     = len(estimators)

    if n_rows == 0 or n_cols == 0:
        return Path(out_path)

    scale    = 1e3 if unit_mhz else 1.0
    unit_str = "mHz" if unit_mhz else "Hz"

    # Build matrix
    mat = np.full((n_rows, n_cols), np.nan)
    for row, scen_id in enumerate(scenarios):
        for col, est_id in enumerate(estimators):
            pair = result.get_pair(scen_id, est_id)
            if pair is None:
                continue
            val = pair.aggregated.get(metric, {}).get(stat)
            if val is not None and np.isfinite(float(val)):
                mat[row, col] = float(val) * scale

    fig_w = max(3.5, 1.6 * n_cols)
    fig_h = max(2.0, 1.2 * n_rows + 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Mask NaN for colourmap
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="viridis_r", aspect="auto",
                   vmin=np.nanmin(mat) if not np.all(np.isnan(mat)) else 0,
                   vmax=np.nanmax(mat) if not np.all(np.isnan(mat)) else 1)

    # Colorbar
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(f"{metric} [{stat}] [{unit_str}]", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # Text annotations
    for row in range(n_rows):
        for col in range(n_cols):
            val = mat[row, col]
            txt = f"{val:.3f}" if np.isfinite(val) else "N/A"
            # Choose contrast colour
            norm_val = (val - np.nanmin(mat)) / max(np.nanmax(mat) - np.nanmin(mat), 1e-10)
            txt_col  = "white" if (not np.isnan(norm_val) and norm_val < 0.55) else "#222222"
            ax.text(col, row, txt, ha="center", va="center",
                    fontsize=7, color=txt_col, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(estimators, fontsize=7, rotation=20, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(scenarios, fontsize=7)
    ax.set_title(
        f"{metric} [{stat}] [{unit_str}] — lower is better",
        fontsize=8, fontweight="bold", pad=6,
    )
    _despine(ax)

    fig.tight_layout()
    return _save(fig, Path(out_path), dpi=dpi)
