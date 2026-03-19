# experiments/__init__.py
"""
Experiment layer: orchestration of scenarios, tuning, benchmark runs, Monte Carlo,
aggregation and export.

Q1 rule: importing this package must NOT require optional components (MC, plotting, etc.)
"""

from __future__ import annotations

from .config import ExperimentConfig, GridConfig  # keep minimal hard imports
from .registry import build_registry, MethodSpec
from .runner import BenchmarkRunner

__all__ = [
    "ExperimentConfig",
    "GridConfig",
    "MethodSpec",
    "build_registry",
    "BenchmarkRunner",
    "MonteCarloConfig",
    "MonteCarloRunner",
]

# ------------------------------------------------------------
# Lazy imports (avoid breaking --mode single if MC is not needed)
# ------------------------------------------------------------


def MonteCarloConfig(*args, **kwargs):  # type: ignore
    from .config import MonteCarloConfig as _MCC

    return _MCC(*args, **kwargs)


def MonteCarloRunner(*args, **kwargs):  # type: ignore
    from .mc import MonteCarloRunner as _MCR

    return _MCR(*args, **kwargs)
