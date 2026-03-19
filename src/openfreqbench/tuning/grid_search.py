"""
Grid Search Optimization (GSO) runner — external to BaseEstimator.
Extracted from pfebench/estimators/base.py BaseEstimator.optimize().
"""
from __future__ import annotations
import itertools
from typing import Any, Dict, List, Tuple
from openfreqbench.estimators._base import BaseEstimator


class TuningRunner:
    """
    Runs grid search over tuning_ranges() of an estimator.
    This is the canonical replacement for BaseEstimator.optimize().
    """

    def run(
        self,
        estimator: BaseEstimator,
        v: "np.ndarray",
        f_true: "np.ndarray",
        metric: str = "RMSE_HZ",
    ) -> Tuple[Dict[str, Any], float]:
        import numpy as np
        from openfreqbench.metrics.frequency import rmse

        definitions = estimator.tuning_ranges()
        if not definitions:
            return estimator._params.copy(), float("nan")

        param_names = [p.name for p in definitions]
        param_grids = [p.generate_grid() for p in definitions]
        combos = list(itertools.product(*param_grids))

        best_score = float("inf")
        best_cfg = estimator._params.copy()

        for vals in combos:
            cfg = dict(zip(param_names, vals))
            estimator.set_params(**cfg)
            try:
                f_est = estimator.run(v)
                n = min(len(f_est), len(f_true))
                wu = max(10, n // 10)
                score = float(rmse(f_est[wu:n] - f_true[wu:n]))
                if score < best_score:
                    best_score = score
                    best_cfg = cfg.copy()
            except Exception:
                continue

        estimator.set_params(**best_cfg)
        return best_cfg, best_score
