# viz/figures.py
from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .style import apply_ieee_style
from .plots_timeseries import plot_scenario_traces
from .plots_pareto import plot_pareto
from .plots_risk import plot_risk


def generate_all_figures(
    out_dir: str,
    scenario_name: str,
    t: np.ndarray,
    f_true: np.ndarray,
    traces: Dict[str, np.ndarray],
    metrics_by_method: Dict[str, Dict[str, Any]],
) -> None:
    """
    One-call wrapper to generate all per-scenario figures.
    Keeps main.py clean.
    """
    apply_ieee_style(single_col=True)

    # Time series traces
    plot_scenario_traces(
        out_dir=out_dir,
        scenario_name=scenario_name,
        t=t,
        f_true=f_true,
        traces=traces,
        max_methods=8,
    )

    # Pareto: RMSE vs CPU
    plot_pareto(
        out_dir=out_dir,
        title=f"{scenario_name}",
        points=metrics_by_method,
        x_key="TIME_PER_SAMPLE_US",
        y_key="RMSE",
    )

    # Risk: peak vs trip-time
    plot_risk(
        out_dir=out_dir,
        title=f"{scenario_name}",
        points=metrics_by_method,
        x_key="MAX_PEAK",
        y_key="TRIP_TIME_0p5",
    )
