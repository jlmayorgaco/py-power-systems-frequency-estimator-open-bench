# viz/plots_risk.py
from __future__ import annotations
from typing import Dict
import matplotlib.pyplot as plt

from .export import save_fig


def plot_risk(
    out_dir: str,
    title: str,
    points: Dict[str, Dict[str, float]],
    x_key: str = "MAX_PEAK",
    y_key: str = "TRIP_TIME_0p5",
) -> None:
    """
    Risk plot: safety-oriented trade-off (peak error vs trip time by default).
    """
    plt.figure()
    for method, m in points.items():
        x = float(m.get(x_key, 0.0))
        y = float(m.get(y_key, 0.0))
        plt.scatter([x], [y])
        plt.text(x, y, method, fontsize=6)

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    save_fig(f"{out_dir}/risk/{title.replace(' ', '_')}__risk")
