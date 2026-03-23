"""
openfreqbench/estimators/_base.py  [BACKWARD-COMPAT SHIM]

All canonical definitions have moved to:
  openfreqbench.estimators.common.base   → BaseEstimator
  openfreqbench.estimators.common.types  → TuningParam, TuningSpec

This module re-exports those symbols so that existing imports continue to work
without modification.  New code should import directly from the canonical paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.types import TuningParam, TuningSpec

# ── Kept here for backward compat (not in common/ — it's an instance snapshot) ──


@dataclass
class EstimatorMeta:
    """Lightweight instance metadata snapshot (backward-compat)."""

    name: str
    family: str
    family_path: str
    params: dict[str, Any]
    latency: int
    complexity: str
    latency_type: str


__all__ = [
    "BaseEstimator",
    "EstimatorMeta",
    "TuningParam",
    "TuningSpec",
]
