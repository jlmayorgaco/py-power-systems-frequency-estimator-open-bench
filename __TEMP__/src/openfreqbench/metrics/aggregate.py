"""
openfreqbench/metrics/aggregate.py

Monte Carlo aggregation: collapse N runs → mean/std/max/p95/p99/n.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from openfreqbench.metrics.frequency import MetricConfig


def aggregate_monte_carlo(
    runs: List[Dict[str, Any]],
    cfg: Optional[MetricConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metric dicts from N Monte Carlo runs.

    Parameters
    ----------
    runs : list of metric dicts (each from compute_metrics())
    cfg  : optional MetricConfig for metadata embedding

    Returns
    -------
    dict: { metric_key: { mean, std, max, p95, p99, n } }
    """
    if not runs:
        return {}

    keys = list(runs[0].keys())
    agg: Dict[str, Dict[str, float]] = {}

    for k in keys:
        vals: List[float] = []
        for r in runs:
            v = r.get(k, {})
            raw = v.get("value") if isinstance(v, dict) else None
            if raw is not None and isinstance(raw, (int, float)) and math.isfinite(float(raw)):
                vals.append(float(raw))

        a = np.asarray(vals, dtype=float)
        if a.size > 0:
            agg[k] = {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "max": float(np.max(a)),
                "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)) if a.size >= 5 else float(np.max(a)),
                "n": int(a.size),
            }
        else:
            agg[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}

    if cfg is not None:
        for key, val in [("_meta_fs_hz", cfg.fs_hz), ("_meta_f_nom", cfg.f_nom)]:
            agg[key] = {"mean": float(val), "std": 0.0, "max": float(val),
                        "p95": float(val), "p99": float(val), "n": 1}

    return agg
