from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Callable, Tuple
import numpy as np

from domain.protocols import EstimatorFactory, OnlineEstimator


@dataclass(frozen=True)
class GridSearchResult:
    best_params: Dict[str, Any]
    best_score: float
    best_label: str


def params_label(p: Dict[str, Any]) -> str:
    if not p:
        return "default"
    keys = sorted(p.keys())
    return ",".join([f"{k}={p[k]}" for k in keys])


def grid_search(
    v: np.ndarray,
    f_true: np.ndarray,
    grid: List[Dict[str, Any]],
    factory: EstimatorFactory,
    score_fn: ScoreFn,
) -> GridSearchResult:
    """
    Exhaustive grid search for online estimators (run step-by-step).
    """
    best_p = None
    best_s = float("inf")

    for p in grid:
        est = factory(p)
        trace = np.array([est.step(float(x)) for x in v], dtype=float)
        s = float(score_fn(trace, f_true))
        if s < best_s:
            best_s = s
            best_p = dict(p)

    best_p = best_p or {}
    return GridSearchResult(
        best_params=best_p, best_score=float(best_s), best_label=params_label(best_p)
    )
