"""Time array generation utilities."""
from __future__ import annotations
import numpy as np


def make_time(fs_hz: float, T_s: float) -> np.ndarray:
    """Return evenly-spaced time array [0, T_s) at fs_hz rate."""
    n = max(2, int(round(T_s * fs_hz)))
    return np.arange(n, dtype=float) / float(fs_hz)
