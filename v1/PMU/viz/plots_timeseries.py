from __future__ import annotations
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from .export import save_fig


def plot_scenario_traces(
    out_dir: str,
    scenario_name: str,
    t: np.ndarray,
    f_true: np.ndarray,
    traces: Dict[str, np.ndarray],
    max_methods: int = 8,
) -> None:
    """
    Plot frequency traces for one scenario.
    To keep figures readable, optionally cap number of methods.
    """
    plt.figure()
    plt.plot(t, f_true, label="True", linewidth=1.5)

    # plot only top-k methods by default if too many
    items = list(traces.items())[:max_methods]
    for name, f_hat in items:
        plt.plot(t, f_hat, label=name)

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(f"{scenario_name} — Frequency Estimates")
    plt.legend(loc="best", ncol=2)

    save_fig(f"{out_dir}/timeseries/{scenario_name}__traces")
