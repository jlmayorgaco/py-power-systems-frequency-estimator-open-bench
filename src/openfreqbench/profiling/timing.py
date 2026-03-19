"""
openfreqbench/profiling/timing.py

TimingHarness: measures wall-clock time of estimator.run() externally.
Timing must NOT happen inside BaseEstimator._step() or step().
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from openfreqbench.estimators._base import BaseEstimator


@dataclass
class TimingHarness:
    history_us: list[float] = field(default_factory=list)

    def timed_run(self, estimator: BaseEstimator, v_array: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter_ns()
        f_hat = estimator.run(v_array)
        dt_ns = time.perf_counter_ns() - t0
        self.history_us.append(dt_ns / 1e3)
        return f_hat, dt_ns / 1e9

    def mean_us(self) -> float:
        return float(np.mean(self.history_us)) if self.history_us else float("nan")

    def reset_history(self) -> None:
        self.history_us.clear()
