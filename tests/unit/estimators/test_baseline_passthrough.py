"""Unit tests for BaselinePassthrough."""
from __future__ import annotations

import numpy as np
from openfreqbench.estimators.baseline_passthrough import BaselinePassthrough


def test_baseline_always_returns_f_nom():
    est = BaselinePassthrough()   # default f_nom=60.0 via params
    v = np.sin(2 * np.pi * 59.5 * np.arange(0, 1.0, 1e-4))
    f_hat = est.run(v)
    assert np.all(f_hat == 60.0)
