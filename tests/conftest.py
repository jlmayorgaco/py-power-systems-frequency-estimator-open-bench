"""Shared pytest fixtures for OpenFreqBench test suite."""
from __future__ import annotations

import numpy as np
import pytest

from openfreqbench.scenarios.g1.e1_pure_60hz import G1_E1_Pure_60Hz
from openfreqbench.estimators.zero_crossing import ZeroCrossingEstimator
from openfreqbench.core.config_models import BenchmarkConfig


@pytest.fixture
def pure_60hz_scenario():
    return G1_E1_Pure_60Hz(fs_hz=10_000.0, T_s=0.5)


@pytest.fixture
def zero_crossing_estimator():
    return ZeroCrossingEstimator()


@pytest.fixture
def simple_sine_waveform():
    """1-second 60 Hz sine at 10 kHz."""
    fs = 10_000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    v = np.sin(2 * np.pi * 60.0 * t)
    return t, v
