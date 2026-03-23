"""
openfreqbench/reporting/statistics.py

Automated hypothesis testing suite based on scipy.stats non-parametric kernels.
"""

from __future__ import annotations

import numpy as np
from typing import Any
import warnings

try:
    from scipy import stats as _sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

def kruskal_wallis_h(*groups: np.ndarray) -> dict[str, Any]:
    if not _HAS_SCIPY: return {"error": "scipy missing"}
    clean = [g for g in groups if len(g) >= 2]
    if len(clean) < 2: return {"error": "insufficient groups"}
    h, p = _sp_stats.kruskal(*clean)
    return {"statistic": float(h), "p_value": float(p), "reject_H0": bool(p < 0.05)}

def mann_whitney_u(a: np.ndarray, b: np.ndarray, alt: str = "two-sided") -> dict[str, Any]:
    if not _HAS_SCIPY: return {"error": "scipy missing"}
    if len(a) < 2 or len(b) < 2: return {"error": "insufficient samples"}
    u, p = _sp_stats.mannwhitneyu(a, b, alternative=alt)
    return {"statistic": float(u), "p_value": float(p), "reject_H0": bool(p < 0.05)}

def spearman_rho(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if not _HAS_SCIPY: return {"error": "scipy missing"}
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4: return {"error": "insufficient samples"}
    rho, p = _sp_stats.spearmanr(a[mask], b[mask])
    return {"statistic": float(rho), "p_value": float(p), "reject_H0": bool(p < 0.05)}

def test_q1_journal_hypotheses(metric_map: dict[str, dict[str, float]]) -> dict[str, Any]:
    """
    Executes standard hypothesis evaluation against a single-run metric dict.
    metric_map: { estimator_id: { 'RMSE': float, ... } }
    """
    # Helper extracting 1D float arrays dynamically
    res = {}
    return res # Stub for dynamic pipeline linkage
