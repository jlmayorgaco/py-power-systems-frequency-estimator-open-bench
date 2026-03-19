from __future__ import annotations
import matplotlib.pyplot as plt


def apply_ieee_style(single_col: bool = True) -> None:
    """
    IEEE-ish default styling (safe for conference/journal).
    Avoid hardcoded colors: let matplotlib defaults handle it.
    """
    figsize = (3.5, 2.8) if single_col else (7.16, 2.8)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.figsize": figsize,
            "lines.linewidth": 1.0,
            "savefig.dpi": 600,
        }
    )
