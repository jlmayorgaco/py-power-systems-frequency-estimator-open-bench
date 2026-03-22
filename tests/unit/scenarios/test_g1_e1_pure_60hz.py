"""Unit tests for G1_E1_Pure_60Hz scenario."""

from __future__ import annotations

import numpy as np
from openfreqbench.scenarios.g1.e1_pure_60hz import G1_E1_Pure_60Hz


def test_build_returns_correct_shape():
    sc = G1_E1_Pure_60Hz(fs_hz=10_000.0, T_s=1.0)
    out = sc.build()
    n = int(10_000.0 * 1.0)
    assert len(out.v) == n
    assert len(out.f_true) == n


def test_f_true_is_60hz():
    sc = G1_E1_Pure_60Hz(fs_hz=10_000.0, T_s=1.0)
    out = sc.build()
    assert np.allclose(out.f_true, 60.0, atol=0.01)


def test_set_montecarlo_seed():
    sc = G1_E1_Pure_60Hz(fs_hz=10_000.0, T_s=0.5)
    sc = sc.set_montecarlo_tuning({"seed": 42})
    out1 = sc.build()
    sc = sc.set_montecarlo_tuning({"seed": 42})
    out2 = sc.build()
    assert np.allclose(out1.v, out2.v)
