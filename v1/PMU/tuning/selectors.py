from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np


@dataclass(frozen=True)
class RobustSelection:
    params: Dict[str, Any]
    label: str
    scores: Dict[str, float]


def _label(p: Dict[str, Any]) -> str:
    if not p:
        return "default"
    return ",".join([f"{k}={p[k]}" for k in sorted(p.keys())])


def select_robust_params_median(
    candidate_params: List[Dict[str, Any]],
    candidate_scores: List[float],
) -> RobustSelection:
    """
    Picks params whose score is closest to the median score.
    Stable and easy to justify in Q1 (robust to outliers).
    """
    s = np.asarray(candidate_scores, dtype=float)
    med = float(np.median(s))
    idx = int(np.argmin(np.abs(s - med)))
    p = dict(candidate_params[idx]) if candidate_params else {}
    return RobustSelection(params=p, label=_label(p), scores={"median": med, "picked": float(s[idx])})


def select_robust_params_minimax(
    param_sets: List[Dict[str, Any]],
    per_seed_scores: List[List[float]],
) -> RobustSelection:
    """
    Minimax selection: choose params minimizing worst-case score across seeds.
    More conservative, good for "safety-critical robustness" claims.
    per_seed_scores: list over param_sets, each is list over seeds.
    """
    worst = [float(np.max(np.asarray(sc, dtype=float))) for sc in per_seed_scores]
    j = int(np.argmin(worst)) if worst else 0
    p = dict(param_sets[j]) if param_sets else {}
    return RobustSelection(params=p, label=_label(p), scores={"worst_case": float(worst[j]) if worst else float("inf")})
