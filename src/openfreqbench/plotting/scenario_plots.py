"""
openfreqbench/plotting/scenario_plots.py

Generates 3 publication-quality plots for one scenario × method MC run:
  1. v(t)            — raw waveform (first 3 cycles)
  2. f(t) tracking   — f_true vs f_hat (reference + MC band)
  3. error(t)        — f_hat - f_true (reference + MC band)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── matplotlib backend-safe import ────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive, safe on servers / CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Plot style constants ───────────────────────────────────────────────────────
_FIG_W = 10.0
_FIG_H = 3.8
_DPI   = 150
_C_TRUE  = "#d62728"   # red   — ground truth
_C_HAT   = "#1f77b4"   # blue  — estimate
_C_BAND  = "#aec7e8"   # light blue — MC band
_C_GRID  = "#e0e0e0"


def _apply_style(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, color=_C_GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Waveform  v(t) vs t
# ─────────────────────────────────────────────────────────────────────────────

def plot_waveform(
    t: np.ndarray,
    v: np.ndarray,
    out_path: Path,
    scenario_id: str = "",
    n_cycles: int = 3,
    f_nom: float = 60.0,
) -> Path:
    """Plot raw voltage waveform — first n_cycles cycles."""
    fs = 1.0 / float(t[1] - t[0]) if len(t) > 1 else 10_000.0
    samples_to_show = min(len(t), int(n_cycles / f_nom * fs) + 1)
    t_ms = t[:samples_to_show] * 1e3   # seconds → ms

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.plot(t_ms, v[:samples_to_show], color=_C_HAT, linewidth=1.2, zorder=3)
    ax.axhline(0, color="#555555", linewidth=0.6, linestyle="--", zorder=2)

    _apply_style(
        ax,
        xlabel="Time [ms]",
        ylabel="Voltage [pu]",
        title=f"Waveform  v(t)   |   {scenario_id}   (seed=0, first {n_cycles} cycles)",
    )
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    return _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Frequency tracking  f_true(t) and f_hat(t) vs t
# ─────────────────────────────────────────────────────────────────────────────

def plot_frequency_tracking(
    t: np.ndarray,
    f_true: np.ndarray,
    f_hat_ref: np.ndarray,
    f_hat_all: List[np.ndarray],
    latency: int,
    out_path: Path,
    scenario_id: str = "",
    method_id: str = "",
) -> Path:
    """Plot f_true vs f_hat, with MC percentile band."""
    # Trim latency warmup
    start = max(0, latency)
    t_s       = t[start:]
    f_true_s  = f_true[start:]
    f_hat_s   = f_hat_ref[start:]

    # MC band: stack all f_hats (align length)
    n_valid = len(t_s)
    all_hats = np.full((len(f_hat_all), n_valid), np.nan)
    for i, fh in enumerate(f_hat_all):
        fh_trim = fh[start:]
        take = min(len(fh_trim), n_valid)
        all_hats[i, :take] = fh_trim[:take]

    p5  = np.nanpercentile(all_hats, 5,  axis=0)
    p95 = np.nanpercentile(all_hats, 95, axis=0)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    # MC band
    ax.fill_between(t_s, p5, p95, alpha=0.25, color=_C_BAND,
                    label=f"MC p5–p95 (n={len(f_hat_all)})", zorder=2)

    # Ground truth
    ax.plot(t_s, f_true_s, color=_C_TRUE, linewidth=1.8, linestyle="--",
            label="f_true", zorder=4)

    # Reference estimate (seed=0)
    ax.plot(t_s, f_hat_s, color=_C_HAT, linewidth=1.2,
            label=f"{method_id} (seed=0)", zorder=5)

    ax.legend(fontsize=9, loc="upper right")
    _apply_style(
        ax,
        xlabel="Time [s]",
        ylabel="Frequency [Hz]",
        title=f"Frequency Tracking   |   {method_id} x {scenario_id}   (n={len(f_hat_all)} runs)",
    )
    ax.set_xlim(t_s[0], t_s[-1])

    # Tight y-range around nominal
    finite = f_hat_s[np.isfinite(f_hat_s)]
    if finite.size:
        span = max(float(np.nanmax(p95) - np.nanmin(p5)) * 1.5, 0.5)
        mid  = float(np.median(f_true_s))
        ax.set_ylim(mid - span / 2, mid + span / 2)

    fig.tight_layout()
    return _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Error  e(t) = f_hat(t) − f_true(t)  vs t
# ─────────────────────────────────────────────────────────────────────────────

def plot_frequency_error(
    t: np.ndarray,
    f_true: np.ndarray,
    f_hat_ref: np.ndarray,
    f_hat_all: List[np.ndarray],
    latency: int,
    out_path: Path,
    scenario_id: str = "",
    method_id: str = "",
) -> Path:
    """Plot frequency error e(t) = f_hat − f_true, with MC band."""
    start = max(0, latency)
    t_s      = t[start:]
    err_ref  = f_hat_ref[start:] - f_true[start:]

    n_valid = len(t_s)
    all_errs = np.full((len(f_hat_all), n_valid), np.nan)
    for i, fh in enumerate(f_hat_all):
        fh_trim = fh[start:]
        take = min(len(fh_trim), n_valid)
        all_errs[i, :take] = fh_trim[:take] - f_true[start : start + take]

    p5  = np.nanpercentile(all_errs, 5,  axis=0)
    p95 = np.nanpercentile(all_errs, 95, axis=0)
    rms = float(np.sqrt(np.nanmean(all_errs ** 2)))

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    ax.fill_between(t_s, p5, p95, alpha=0.25, color=_C_BAND,
                    label=f"MC p5–p95 (n={len(f_hat_all)})", zorder=2)
    ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--", zorder=3)
    ax.plot(t_s, err_ref, color=_C_HAT, linewidth=1.0,
            label=f"Error seed=0", zorder=5)

    # Annotate RMS
    ax.text(0.02, 0.93, f"RMSE = {rms*1000:.4f} mHz",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.legend(fontsize=9, loc="upper right")
    _apply_style(
        ax,
        xlabel="Time [s]",
        ylabel="Error [Hz]",
        title=f"Frequency Error  e(t) = f_hat - f_true   |   {method_id} x {scenario_id}",
    )
    ax.set_xlim(t_s[0], t_s[-1])

    # Y-range: symmetric, a bit wider than max error
    finite_p5  = p5[np.isfinite(p5)]
    finite_p95 = p95[np.isfinite(p95)]
    if finite_p5.size and finite_p95.size:
        span = max(abs(float(np.min(finite_p5))), abs(float(np.max(finite_p95)))) * 2.5
        span = max(span, 0.05)
        ax.set_ylim(-span, span)

    fig.tight_layout()
    return _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Public orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def generate_scenario_method_plots(
    *,
    t: np.ndarray,
    v: np.ndarray,
    f_true: np.ndarray,
    f_hat_ref: np.ndarray,
    f_hat_all: List[np.ndarray],
    latency: int,
    out_dir: Path,
    ts: str,
    scenario_id: str,
    method_id: str,
    f_nom: float = 60.0,
) -> List[Path]:
    """
    Generate and save the 3 standard plots to out_dir.

    Returns list of saved file paths.
    """
    prefix = f"{method_id}_{ts}"
    paths: List[Path] = []

    paths.append(plot_waveform(
        t=t, v=v,
        out_path=out_dir / f"{prefix}_waveform.png",
        scenario_id=scenario_id,
        f_nom=f_nom,
    ))

    paths.append(plot_frequency_tracking(
        t=t, f_true=f_true,
        f_hat_ref=f_hat_ref, f_hat_all=f_hat_all,
        latency=latency,
        out_path=out_dir / f"{prefix}_tracking.png",
        scenario_id=scenario_id,
        method_id=method_id,
    ))

    paths.append(plot_frequency_error(
        t=t, f_true=f_true,
        f_hat_ref=f_hat_ref, f_hat_all=f_hat_all,
        latency=latency,
        out_path=out_dir / f"{prefix}_error.png",
        scenario_id=scenario_id,
        method_id=method_id,
    ))

    return paths
