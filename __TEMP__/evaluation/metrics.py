from __future__ import annotations

import numpy as np


def frequency_error(estimate_hz: np.ndarray, truth_hz: np.ndarray) -> float:
    """
    Root-mean-square error between estimated and true frequency, elementwise.
    Both arrays must be 1-D and same length.
    """
    est = np.asarray(estimate_hz, dtype=float).ravel()
    tru = np.asarray(truth_hz, dtype=float).ravel()
    if est.size != tru.size:
        raise ValueError("estimate_hz and truth_hz must have the same length")
    diff = est - tru
    return float(np.sqrt(np.mean(diff * diff)))
