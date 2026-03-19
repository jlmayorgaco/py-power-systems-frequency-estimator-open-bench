"""Unit tests for TimingHarness."""

from __future__ import annotations

import numpy as np
from openfreqbench.estimators.baseline_passthrough import BaselinePassthrough
from openfreqbench.profiling.timing import TimingHarness


def test_timed_run_returns_tuple():
    est = BaselinePassthrough()  # default f_nom=60.0
    v = np.sin(2 * np.pi * 60.0 * np.arange(0, 0.1, 1e-4))
    f_hat, elapsed = TimingHarness().timed_run(est, v)
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0
    assert len(f_hat) == len(v)
