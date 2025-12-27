from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

from domain.metrics_logic import rmse



@dataclass(frozen=True)
class TuningStrategy:
    """
    Defines what metric you tune on and how.
    """
    score_name: str
    score_fn: Callable[[np.ndarray, np.ndarray], float]


def default_tuning_strategy() -> TuningStrategy:
    """
    Q1-default: tune on RMSE over train segment.
    Simple, standard, defensible.
    """
    def score(trace: np.ndarray, f_true: np.ndarray) -> float:
        return float(rmse(trace, f_true))
    return TuningStrategy(score_name="RMSE", score_fn=score)
