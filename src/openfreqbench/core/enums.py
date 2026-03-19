"""Shared enumerations."""
from __future__ import annotations
from enum import Enum


class BenchmarkMode(int, Enum):
    BEST_CASE = 1        # tuned per-scenario (Mode 1)
    TRANSFER = 2         # generalization across scenarios (Mode 2)
    SENSITIVITY = 3      # parameter sweep (Mode 3)
    PROFILING = 4        # compute cost focus (Mode 4)
    COMPLIANCE = 5       # IEEE C37.118 compliance (Mode 5)


class EstimatorFamily(str, Enum):
    TIME_DOMAIN = "TimeDomain"
    MODEL_BASED = "ModelBased"
    SPECTRAL = "Spectral"
    KALMAN = "Kalman"
    NEURAL = "Neural"
