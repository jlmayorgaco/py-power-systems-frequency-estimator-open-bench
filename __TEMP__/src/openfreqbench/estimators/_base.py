"""
openfreqbench/estimators/_base.py

Base class for all frequency estimators.

Design contract:
- _step(v_sample) → float  : pure math, NO timing, NO logging
- reset()                  : reinitialise internal state
- latency_samples          : intrinsic causal delay for metric alignment
- tuning_ranges()          : list of TuningParam for GSO (optional)
- set_params(**kwargs)     : inject tuned params (resets state)

Timing is measured EXTERNALLY by TimingHarness (profiling/timing.py).
GSO is run EXTERNALLY by TuningRunner (tuning/grid_search.py).
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Tuning Parameter Descriptor
# ---------------------------------------------------------------------------


@dataclass
class TuningParam:
    """Describes one degree of freedom available for Grid Search Optimization."""

    name: str
    default: Any
    type: Literal["int", "float", "bool", "categorical"]
    values: Optional[List[Any]] = None       # explicit list  → [1, 5, 10, 20]
    range: Optional[Tuple[float, float, int]] = None  # (min, max, n_steps)
    description: str = ""

    def generate_grid(self) -> List[Any]:
        """Expand the descriptor to a concrete list of candidate values."""
        if self.values is not None:
            return list(self.values)
        if self.range is not None:
            lo, hi, steps = self.range
            if self.type == "int":
                arr = np.linspace(lo, hi, steps)
                return np.unique(np.round(arr)).astype(int).tolist()
            else:
                return np.linspace(lo, hi, steps).tolist()
        return [self.default]


# ---------------------------------------------------------------------------
# Estimator Metadata
# ---------------------------------------------------------------------------


@dataclass
class EstimatorMeta:
    name: str
    family: str
    params: Dict[str, Any]
    latency: int


# ---------------------------------------------------------------------------
# Base Class
# ---------------------------------------------------------------------------


class BaseEstimator(ABC):
    """
    Abstract base for every frequency estimator in the framework.

    Subclass contract
    -----------------
    1. Override NAME and FAMILY as class-level strings.
    2. Implement _step(v_sample: float) -> float  (no side effects beyond internal state).
    3. Implement reset() → None.
    4. Implement latency_samples property.
    5. Optionally override tuning_ranges() to expose hyper-parameters.
    """

    NAME: str = "BaseEstimator"
    FAMILY: str = "Generic"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        # Build params dict from defaults then apply user overrides
        self._params: Dict[str, Any] = {p.name: p.default for p in self.tuning_ranges()}
        if params:
            self._params.update(params)
        self.reset()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _step(self, v_sample: float) -> float:
        """Pure math: consume one voltage sample, return f_hat [Hz]."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Re-initialise internal buffers and state."""

    @property
    @abstractmethod
    def latency_samples(self) -> int:
        """Intrinsic causal delay in samples for metric alignment."""
        return 0

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """Return list of TuningParam descriptors. Default: empty (no tunable params)."""
        return []

    # ------------------------------------------------------------------
    # Param management (used by TuningRunner and Monte Carlo)
    # ------------------------------------------------------------------

    def set_params(self, **kwargs: Any) -> None:
        """Update params and re-initialise state. Used by GSO and MC runners."""
        changed = False
        for k, v in kwargs.items():
            if k in self._params:
                self._params[k] = v
                changed = True
        if changed:
            self.reset()

    # ------------------------------------------------------------------
    # Standard execution helpers (no timing — use TimingHarness externally)
    # ------------------------------------------------------------------

    def step(self, v_sample: float) -> float:
        """Call _step with NaN/inf protection. Does NOT measure timing."""
        try:
            val = float(self._step(float(v_sample)))
        except Exception:
            return float("nan")
        if math.isnan(val) or math.isinf(val):
            return float("nan")
        return val

    def run(self, v_array: np.ndarray) -> np.ndarray:
        """Run estimator over a full voltage array. Resets state first."""
        v = np.asarray(v_array, dtype=float)
        out = np.empty_like(v)
        self.reset()
        for i, x in enumerate(v):
            out[i] = self.step(x)
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def info(self) -> EstimatorMeta:
        return EstimatorMeta(
            name=self.NAME,
            family=self.FAMILY,
            params=self._params.copy(),
            latency=self.latency_samples,
        )

    def __repr__(self) -> str:
        return f"{self.NAME}({self._params!r})"
