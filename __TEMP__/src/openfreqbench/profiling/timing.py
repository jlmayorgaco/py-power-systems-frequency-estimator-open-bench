"""
openfreqbench/profiling/timing.py

TimingHarness: measures wall-clock time of estimator.run() externally.

Separating timing from the estimator class enforces the SRP:
  - estimator._step() = pure signal processing math
  - TimingHarness = observability concern

Usage
-----
    harness = TimingHarness()
    f_hat, exec_time_s = harness.timed_run(estimator, v_array)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from openfreqbench.estimators._base import BaseEstimator


@dataclass
class TimingHarness:
    """
    Measures execution time for a single estimator.run() call.

    Attributes
    ----------
    history_us : list of per-run durations in microseconds (retained for stats)
    """

    history_us: List[float] = field(default_factory=list)

    def timed_run(
        self, estimator: BaseEstimator, v_array: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Execute estimator.run(v_array) and return (f_hat, exec_time_s).

        Parameters
        ----------
        estimator : BaseEstimator instance (reset() will be called inside run())
        v_array   : voltage samples [V]

        Returns
        -------
        f_hat       : np.ndarray, frequency estimates [Hz]
        exec_time_s : float, wall-clock duration [s]
        """
        t0 = time.perf_counter_ns()
        f_hat = estimator.run(v_array)
        dt_ns = time.perf_counter_ns() - t0
        exec_time_s = dt_ns / 1e9
        self.history_us.append(dt_ns / 1e3)
        return f_hat, exec_time_s

    def mean_us(self) -> float:
        if not self.history_us:
            return float("nan")
        return float(np.mean(self.history_us))

    def reset_history(self) -> None:
        self.history_us.clear()
