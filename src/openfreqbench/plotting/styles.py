"""
openfreqbench/plotting/styles.py

IEEE-grade visual style constants and rcParams configuration.

Usage:
    from openfreqbench.plotting.styles import apply_ieee_style, C, color_for_estimator

    apply_ieee_style()       # sets global rcParams
    ax.plot(..., color=C.TRUE, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (Okabe-Ito  — colorblind-safe + semantic assignments)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Palette:
    # Semantic colours
    TRUE: str = "#333333"  # ground truth — near-black, universal
    BAND: str = "#88BBDD"  # MC uncertainty band — muted blue
    ZERO_LINE: str = "#888888"  # zero-reference dashed line
    GRID: str = "#DDDDDD"  # plot grid colour

    # Estimator colours (Okabe-Ito colorblind-safe 8-color set)
    EST_0: str = "#0072B2"  # blue       — ZeroCrossing / window
    EST_1: str = "#009E73"  # green      — FFTPeak / spectral
    EST_2: str = "#D55E00"  # vermillion — Baseline / passthrough
    EST_3: str = "#CC79A7"  # pink-purple
    EST_4: str = "#56B4E9"  # sky-blue
    EST_5: str = "#E69F00"  # orange
    EST_6: str = "#F0E442"  # yellow
    EST_7: str = "#000000"  # black

    # Event annotation colour
    EVENT: str = "#AA0000"  # dark red — step / ramp onset markers

    @property
    def estimator_cycle(self):
        return [
            self.EST_0,
            self.EST_1,
            self.EST_2,
            self.EST_3,
            self.EST_4,
            self.EST_5,
            self.EST_6,
            self.EST_7,
        ]

    def for_estimator(self, idx: int) -> str:
        cycle = self.estimator_cycle
        return cycle[idx % len(cycle)]


C = _Palette()

# IEEE column widths in inches (based on standard 8.5"x11" two-column format)
SINGLE_COL_W = 3.487  # 88.5 mm — IEEE single column
DOUBLE_COL_W = 7.165  # 182 mm  — IEEE double column
FIG_H_UNIT = 2.0  # height per subplot row


# ─────────────────────────────────────────────────────────────────────────────
# rcParams presets
# ─────────────────────────────────────────────────────────────────────────────


def apply_ieee_style(font_size: float = 8.0) -> None:
    """
    Apply IEEE Transactions-appropriate matplotlib rcParams globally.
    Call once at module level or before generating figures.
    """
    plt.rcParams.update(
        {
            # Font — sans-serif matches most IEEE figures
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "legend.title_fontsize": font_size,
            "legend.framealpha": 0.85,
            "legend.edgecolor": "#CCCCCC",
            # Lines
            "lines.linewidth": 1.2,
            "lines.markersize": 3.0,
            # Axes — remove top/right spines globally
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.linewidth": 0.4,
            "grid.color": C.GRID,
            "grid.alpha": 1.0,
            # Ticks
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.autolayout": False,
            # Rendering
            "path.simplify": True,
            "path.simplify_threshold": 0.5,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Axis helpers
# ─────────────────────────────────────────────────────────────────────────────


def despine(ax) -> None:
    """Remove top and right spines from an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_box(
    ax,
    text: str,
    loc: str = "upper right",
    fontsize: float = 7.0,
    color: str = "#222222",
) -> None:
    """
    Place text with a white background box inside an axis.

    loc: e.g. "upper right", "lower left", "upper left"
    """
    x_anchor = {"left": 0.03, "right": 0.97, "center": 0.50}
    y_anchor = {"upper": 0.95, "lower": 0.05, "center": 0.50}
    parts = loc.lower().split()
    xpos = x_anchor.get(parts[-1] if len(parts) > 1 else "right", 0.97)
    ypos = y_anchor.get(parts[0] if len(parts) > 1 else "upper", 0.95)
    ha = "right" if xpos > 0.5 else "left"
    va = "top" if ypos > 0.5 else "bottom"
    ax.text(
        xpos,
        ypos,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=color,
        ha=ha,
        va=va,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="#CCCCCC",
            alpha=0.85,
            linewidth=0.5,
        ),
    )


def format_hz(val: float) -> str:
    """Human-readable Hz / mHz formatting."""
    if abs(val) >= 0.5:
        return f"{val:.4f} Hz"
    return f"{val * 1000:.4f} mHz"


# ─────────────────────────────────────────────────────────────────────────────
# Estimator → colour registry (consistent across all figures)
# ─────────────────────────────────────────────────────────────────────────────

_EST_COLOR_CACHE: dict[str, str] = {}


def color_for_estimator(name: str) -> str:
    """Return a consistent, deterministic colour for a given estimator name."""
    if name not in _EST_COLOR_CACHE:
        idx = len(_EST_COLOR_CACHE)
        _EST_COLOR_CACHE[name] = C.for_estimator(idx)
    return _EST_COLOR_CACHE[name]
