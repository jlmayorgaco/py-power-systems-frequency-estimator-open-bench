"""
openfreqbench/estimators/common/types.py  [CANONICAL]

Typed interfaces shared across every estimator in the framework.

  EstimatorOutput  — per-sample result from update()
  EstimatorSpec    — immutable class-level metadata
  TuningParam      — one tunable hyperparameter with its candidate grid
  TuningSpec       — full tuning-space declaration for one estimator class

These types have NO dependency on NumPy or any estimator logic.
They are safe to import from anywhere in the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────────────
# EstimatorOutput — per-sample typed result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EstimatorOutput:
    """
    Typed result returned by BaseEstimator.update() for every voltage sample.

    `frequency_hz` and `valid` are always present.
    The remaining fields are optional; estimators that cannot compute them
    leave them as NaN.
    """

    frequency_hz: float
    valid: bool = True
    rocof_hz_s: float = float("nan")  # rate of change of frequency (Hz/s)
    phase_rad: float = float("nan")  # instantaneous phase (rad)
    amplitude_pu: float = float("nan")  # peak amplitude (per-unit)

    @property
    def is_finite(self) -> bool:
        """True if frequency_hz is a finite IEEE-754 number."""
        return math.isfinite(self.frequency_hz)

    @property
    def is_valid(self) -> bool:
        """True if the estimator considers this sample reliable AND finite."""
        return self.valid and self.is_finite

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency_hz": self.frequency_hz,
            "valid": self.valid,
            "rocof_hz_s": self.rocof_hz_s,
            "phase_rad": self.phase_rad,
            "amplitude_pu": self.amplitude_pu,
        }


# ─────────────────────────────────────────────────────────────────────────────
# EstimatorSpec — frozen class-level metadata
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EstimatorSpec:
    """
    Immutable metadata describing an estimator class.

    Set as a class variable ``SPEC = EstimatorSpec(...)`` on every subclass.
    The framework derives display names, family groupings, and validity bounds
    from this object; never from loose string constants.
    """

    name: str
    family: str
    family_path: str
    complexity: str
    latency_type: str
    nominal_freq_hz: float = 60.0
    min_valid_freq_hz: float = 40.0
    max_valid_freq_hz: float = 80.0
    is_three_phase: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "family_path": self.family_path,
            "complexity": self.complexity,
            "latency_type": self.latency_type,
            "nominal_freq_hz": self.nominal_freq_hz,
            "min_valid_freq_hz": self.min_valid_freq_hz,
            "max_valid_freq_hz": self.max_valid_freq_hz,
            "is_three_phase": self.is_three_phase,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TuningParam — one hyperparameter descriptor
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TuningParam:
    """
    Describes one tunable hyperparameter and its candidate search space.

    Provide either ``values`` (explicit list) or ``range`` (lo, hi, n_steps).
    If neither is provided, only the default value is used.
    """

    name: str
    default: Any
    type: Literal["int", "float", "bool", "categorical"]
    values: list[Any] | None = None
    range: tuple[float, float, int] | None = None
    description: str = ""

    def generate_grid(self) -> list[Any]:
        """Return the sorted candidate list for grid search."""
        if self.values is not None:
            return list(self.values)
        if self.range is not None:
            lo, hi, steps = self.range
            import numpy as np  # local import — types module stays framework-free

            if self.type == "int":
                arr = np.linspace(lo, hi, steps)
                return np.unique(np.round(arr)).astype(int).tolist()
            return np.linspace(lo, hi, steps).tolist()
        return [self.default]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "values": self.values,
            "range": list(self.range) if self.range else None,
            "description": self.description,
            "grid": self.generate_grid(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TuningSpec — full tuning-space declaration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TuningSpec:
    """
    Declares the complete tuning space of one estimator class.

    The estimator declares what can be tuned and what values to try.
    The framework decides *when* and *how* to run the tuning loop.
    """

    params: list[TuningParam] = field(default_factory=list)
    objective: str = "RMSE_HZ"
    method: str = "grid"

    def candidate_grid(self) -> list[dict[str, Any]]:
        """Cartesian product of all param grids → list of candidate dicts."""
        if not self.params:
            return [{}]
        names = [p.name for p in self.params]
        grids = [p.generate_grid() for p in self.params]
        return [dict(zip(names, combo)) for combo in product(*grids)]

    def n_candidates(self) -> int:
        n = 1
        for p in self.params:
            n *= len(p.generate_grid())
        return n

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "method": self.method,
            "n_candidates": self.n_candidates(),
            "params": [p.to_dict() for p in self.params],
        }
