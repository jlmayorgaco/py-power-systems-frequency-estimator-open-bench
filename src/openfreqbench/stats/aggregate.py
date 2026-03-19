"""
openfreqbench/stats/aggregate.py

Monte Carlo aggregation: N run-level metric dicts -> full statistical summary.

Computes per-metric:
  n, n_valid, n_nan
  mean, std, median, min, max
  p1, p5, p25, p50, p75, p95, p99
  cov   -- coefficient of variation (std / |mean|), dimensionless
  ci95_lo, ci95_hi -- bootstrap 95% CI on mean (500 resamples)

Also provides:
  compare_estimators()  -- Welch t-test between two metric distributions
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

# Avoid circular import: metrics/__init__.py re-exports from this module,
# so we must NOT do a module-level import from openfreqbench.metrics.
# MetricConfig is used only as an optional type hint inside functions;
# we import it lazily there.
if TYPE_CHECKING:
    from openfreqbench.metrics.frequency import MetricConfig


_N_BOOTSTRAP = 500
_PERCENTILES  = (1, 5, 25, 50, 75, 95, 99)


def _bootstrap_ci_mean(
    a: np.ndarray,
    n_boot: int = _N_BOOTSTRAP,
    level: float = 95.0,
    rng_seed: int = 0,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (ci_lo, ci_hi) at `level`% confidence.
    Fallback to (mean, mean) if n < 2.
    """
    if a.size < 2:
        m = float(np.mean(a)) if a.size == 1 else float("nan")
        return m, m
    rng = np.random.default_rng(rng_seed)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_means[i] = float(np.mean(rng.choice(a, size=a.size, replace=True)))
    lo = float(np.percentile(boot_means, (100.0 - level) / 2.0))
    hi = float(np.percentile(boot_means, 100.0 - (100.0 - level) / 2.0))
    return lo, hi


def _stat_block(a: np.ndarray, n_total: int) -> Dict[str, Any]:
    """Compute the full statistical block for a 1-D array of finite values."""
    n_valid = int(a.size)
    n_nan   = n_total - n_valid

    if n_valid == 0:
        nan = float("nan")
        base = {k: nan for k in (
            "mean", "std", "median", "min", "max", "cov",
            "ci95_lo", "ci95_hi",
            *(f"p{p}" for p in _PERCENTILES),
        )}
        base.update({"n": n_total, "n_valid": 0, "n_nan": n_nan})
        return base

    mean   = float(np.mean(a))
    std    = float(np.std(a, ddof=min(1, n_valid - 1)))
    median = float(np.median(a))
    mn     = float(np.min(a))
    mx     = float(np.max(a))
    cov    = (std / abs(mean)) if abs(mean) > 1e-300 else float("nan")
    ci_lo, ci_hi = _bootstrap_ci_mean(a)

    pct_vals = {f"p{p}": float(np.percentile(a, p)) for p in _PERCENTILES}

    return {
        "n":       n_total,
        "n_valid": n_valid,
        "n_nan":   n_nan,
        "mean":    mean,
        "std":     std,
        "median":  median,
        "min":     mn,
        "max":     mx,
        "cov":     cov,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        **pct_vals,
    }


def aggregate_monte_carlo(
    runs: List[Dict[str, Any]],
    cfg: Optional[Any] = None,          # Optional[MetricConfig] — lazy to avoid circular
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate N run-level metric dicts into a full statistical summary.

    Each element of `runs` is a metric dict keyed by metric name,
    where each value is either a dict with a "value" key, or a plain scalar.

    Returns: {metric_name: stat_block} where stat_block has keys:
        n, n_valid, n_nan, mean, std, median, min, max,
        cov, ci95_lo, ci95_hi,
        p1, p5, p25, p50, p75, p95, p99
    """
    if not runs:
        return {}

    keys    = list(runs[0].keys())
    n_total = len(runs)
    agg: Dict[str, Dict[str, Any]] = {}

    for k in keys:
        raw_vals: List[float] = []
        for r in runs:
            entry = r.get(k)
            if entry is None:
                continue
            if isinstance(entry, dict):
                v = entry.get("value")
            else:
                v = entry
            if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
                raw_vals.append(float(v))

        a      = np.asarray(raw_vals, dtype=float)
        agg[k] = _stat_block(a, n_total)

    if cfg is not None:
        for key, val in [("_meta_fs_hz", cfg.fs_hz), ("_meta_f_nom", cfg.f_nom)]:
            agg[key] = _stat_block(np.array([float(val)]), 1)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Paired comparison utilities
# ─────────────────────────────────────────────────────────────────────────────

def compare_estimators(
    runs_a: List[Dict[str, Any]],
    runs_b: List[Dict[str, Any]],
    metric: str = "RMSE_HZ",
) -> Dict[str, Any]:
    """
    Welch's t-test comparison of metric distributions for two estimators.

    Returns a dict with:
        mean_a, mean_b, delta_mean (a - b),
        t_stat, p_value, significant (p < 0.05),
        n_a, n_b
    """
    try:
        from scipy.stats import ttest_ind
    except ImportError:
        return {"error": "scipy not available for statistical comparison"}

    def _extract(runs: List[Dict[str, Any]]) -> np.ndarray:
        vals = []
        for r in runs:
            entry = r.get(metric)
            if isinstance(entry, dict):
                v = entry.get("value")
            else:
                v = entry
            if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
                vals.append(float(v))
        return np.asarray(vals, dtype=float)

    a = _extract(runs_a)
    b = _extract(runs_b)

    if a.size < 2 or b.size < 2:
        return {"error": "insufficient data", "n_a": int(a.size), "n_b": int(b.size)}

    t_stat, p_val = ttest_ind(a, b, equal_var=False)
    return {
        "metric":      metric,
        "mean_a":      float(np.mean(a)),
        "mean_b":      float(np.mean(b)),
        "delta_mean":  float(np.mean(a) - np.mean(b)),
        "t_stat":      float(t_stat),
        "p_value":     float(p_val),
        "significant": bool(p_val < 0.05),
        "n_a":         int(a.size),
        "n_b":         int(b.size),
    }
