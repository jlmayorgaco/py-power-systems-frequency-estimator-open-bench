"""
openfreqbench/estimators/_outputs.py  [BACKWARD-COMPAT SHIM]

Canonical definitions have moved to:
  openfreqbench.estimators.common.types  → EstimatorOutput, EstimatorSpec

This module re-exports those symbols so existing imports continue to work.
New code should import directly from the canonical path.
"""
from openfreqbench.estimators.common.types import EstimatorOutput, EstimatorSpec  # noqa: F401

__all__ = ["EstimatorOutput", "EstimatorSpec"]
