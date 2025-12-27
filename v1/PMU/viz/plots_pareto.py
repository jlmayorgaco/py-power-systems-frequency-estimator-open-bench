from __future__ import annotations
from typing import Dict, Any
import matplotlib.pyplot as plt

from .export import save_fig


def plot_pareto(
    out_dir: str,
    title: str,
    points: Dict[str, Dict[str, float]],
    x_key: str = "TIME_PER_SAMPLE_US",
    y_key: str = "RMSE",
) -> None:
    """
    Generic Pareto scatter (e.g., RMSE vs CPU).
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
    plt.xscale("log")  # CPU often spans orders of magnitude
    save_fig(f"{out_dir}/pareto/{title.replace(' ', '_')}__pareto")
