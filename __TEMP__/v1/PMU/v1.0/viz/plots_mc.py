from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

from .export import save_fig


def plot_mc_boxplots(
    out_dir: str,
    scenario: str,
    metric_name: str,
    per_method_values: Dict[str, List[float]],
) -> None:
    """
    Boxplots across MC seeds for each method (metric distributions).
    """
    methods = list(per_method_values.keys())
    data = [per_method_values[m] for m in methods]

    plt.figure(figsize=(7.16, 2.8))
    plt.boxplot(data, labels=methods, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric_name)
    plt.title(f"{scenario} — MC Distribution: {metric_name}")
    save_fig(f"{out_dir}/mc/{scenario}__{metric_name}__boxplot")


def plot_mc_errorbands(
    out_dir: str,
    scenario: str,
    t: np.ndarray,
    f_true: np.ndarray,
    per_method_traces: Dict[str, np.ndarray],
    percentile_bands=(5, 95),
) -> None:
    """
    If you store per-seed traces and want uncertainty bands.
    per_method_traces expects stacked arrays shape (n_seeds, T).
    """
    plt.figure()
    plt.plot(t, f_true, label="True", linewidth=1.5)

    lo, hi = percentile_bands
    for method, stacked in per_method_traces.items():
        q_lo = np.percentile(stacked, lo, axis=0)
        q_hi = np.percentile(stacked, hi, axis=0)
        q_med = np.percentile(stacked, 50, axis=0)
        plt.plot(t, q_med, label=f"{method} median")
        plt.fill_between(t, q_lo, q_hi, alpha=0.2)

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(f"{scenario} — MC Error Bands")
    plt.legend(loc="best", ncol=2)
    save_fig(f"{out_dir}/mc/{scenario}__errorbands")
