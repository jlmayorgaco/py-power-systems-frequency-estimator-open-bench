"""
openfreqbench/metrics/protection.py

Protection & Relay simulation metrics.
Evaluates Total Contiguous Burst and Absolute Risk times.
"""

from __future__ import annotations

import numpy as np
from typing import Any

def _to_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)

def risk_total_time_s(f_err: Any, fs_hz: float, thr_hz: float = 0.5) -> float:
    """Calculates cumulative time (seconds) estimator spent outside tolerance."""
    err = np.abs(_to_1d(f_err))
    violations = np.sum(err > thr_hz)
    return float(violations / fs_hz)

def max_contiguous_burst_s(f_err: Any, fs_hz: float, thr_hz: float = 0.5) -> float:
    """Calculates the maximum consecutive duration (seconds) estimator remains out of bounds."""
    err = np.abs(_to_1d(f_err))
    mask = err > thr_hz
    if not np.any(mask):
        return 0.0
    
    # Fast vectorized consecutive counting via diffs
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    
    lengths = ends - starts
    if len(lengths) == 0:
        return 0.0
    return float(np.max(lengths) / fs_hz)

def append_protection_metrics(out_dict: dict, f_err: Any, fs_hz: float) -> None:
    """In-place augmentation of the metric payload with Protection Metrics"""
    out_dict["RISK_TOTAL_TRIP_TIME_0p5_S"] = risk_total_time_s(f_err, fs_hz, 0.5)
    out_dict["RISK_MAX_BURST_0p5_S"] = max_contiguous_burst_s(f_err, fs_hz, 0.5)
