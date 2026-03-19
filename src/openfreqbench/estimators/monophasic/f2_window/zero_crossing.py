"""
Backward-compatible re-export.

ZeroCrossingEstimator canonical location is now:
  openfreqbench.estimators.monophasic.f3_recursive.zero_crossing
"""

from openfreqbench.estimators.monophasic.f3_recursive.zero_crossing import (
    ZeroCrossingEstimator,
)

__all__ = ["ZeroCrossingEstimator"]
