"""
openfreqbench/scenarios/_base.py

Base classes for all benchmark scenarios.

Ported from pfebench/scenarios/base.py (pfebench research system).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _cumtrapz(y: Array, t: Array) -> Array:
    """Cumulative trapezoidal integration (scipy-free)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    n = min(y.size, t.size)
    if n <= 1:
        return np.zeros((n,), dtype=float)
    y, t = y[:n], t[:n]
    dt = np.diff(t)
    ymid = 0.5 * (y[1:] + y[:-1])
    out = np.zeros((n,), dtype=float)
    out[1:] = np.cumsum(ymid * dt)
    return out


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ScenarioState:
    t: Array
    fs_hz: float
    f_nom_hz: float
    seed: int
    f_true: Array
    phi: Array
    A: Array
    v: Array
    schema: Dict[str, Any]


@dataclass
class ScenarioOutput:
    """Standard output container for any scenario."""

    scenario_id: str
    state: ScenarioState

    def __getattr__(self, name: str):
        if "state" in self.__dict__ and hasattr(self.state, name):
            return getattr(self.state, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    @property
    def t(self) -> Array:
        return self.state.t

    @property
    def v(self) -> Array:
        return self.state.v

    @property
    def f_true(self) -> Array:
        return self.state.f_true

    @property
    def meta(self) -> Dict[str, Any]:
        return {"scenario_id": self.scenario_id, "schema": self.state.schema}


# ---------------------------------------------------------------------------
# Mixin: noise generators and signal modifiers
# ---------------------------------------------------------------------------


class ScenarioModifiersMixin:
    """Provides noise generators and signal modification helpers."""

    state: ScenarioState

    def _rng(self, seed_offset: int = 0) -> np.random.Generator:
        if hasattr(self, "state"):
            base = int(self.state.seed)
        else:
            base = int(getattr(self, "seed", 0))
        return np.random.default_rng((base + seed_offset) & 0x7FFF_FFFF)

    def recompute_phi_from_f(self, *, phi0_rad: float = 0.0) -> None:
        integ = _cumtrapz(self.state.f_true, self.state.t)
        self.state.phi = float(phi0_rad) + 2.0 * np.pi * integ

    def recompute_v_from_A_phi(self) -> None:
        self.state.v = np.asarray(self.state.A, dtype=float) * np.sin(
            np.asarray(self.state.phi, dtype=float)
        )

    def add_noise_gaussian(self, shape: tuple, sigma: float, rng: np.random.Generator) -> Array:
        return rng.normal(loc=0.0, scale=sigma, size=shape)

    def add_noise_brown(self, shape: tuple, sigma: float, rng: np.random.Generator) -> Array:
        white = rng.normal(loc=0.0, scale=1.0, size=shape)
        brown = np.cumsum(white.reshape(-1))
        brown -= np.mean(brown)
        rms = float(np.sqrt(np.mean(brown ** 2)))
        return ((brown / rms) * sigma).reshape(shape) if rms > 0 else brown.reshape(shape)

    def add_noise_impulsive(
        self, shape: tuple, sigma: float, rng: np.random.Generator, prob: float = 0.001
    ) -> Array:
        noise = np.zeros(shape, dtype=float).reshape(-1)
        mask = rng.random(noise.size) < prob
        noise[mask] = sigma * rng.choice([-1.0, 1.0], size=noise.size)[mask]
        return noise.reshape(shape)

    def add_noise_uniform(self, shape: tuple, sigma: float, rng: np.random.Generator) -> Array:
        width = sigma * np.sqrt(12.0)
        return rng.uniform(-width / 2.0, width / 2.0, size=shape)


# ---------------------------------------------------------------------------
# Scenario Base
# ---------------------------------------------------------------------------


class ScenarioBase(ScenarioModifiersMixin):
    """
    Abstract base class for benchmark scenarios.

    Subclass contract
    -----------------
    1. Set scenario_id class attribute.
    2. Implement build() → ScenarioOutput.
    3. Optionally define tuning_map for Monte Carlo parameter injection.
    """

    scenario_id: str = "UNKNOWN"
    tuning_map: Dict[str, str] = {}

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("Scenario must implement build()")

    def run(self) -> ScenarioOutput:
        return self.build()

    def configure(self, **kwargs: Any) -> "ScenarioBase":
        """Direct attribute update — for advanced / internal use."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Parameter {k!r} not found in {self.scenario_id}")
        return self

    def set_montecarlo_tuning(self, params: Dict[str, Any]) -> "ScenarioBase":
        """Apply Monte Carlo parameter dict using tuning_map aliases."""
        for alias, value in params.items():
            if alias not in self.tuning_map:
                raise ValueError(
                    f"Alias {alias!r} not in tuning_map for {self.scenario_id}. "
                    f"Allowed: {list(self.tuning_map)}"
                )
            attr = self.tuning_map[alias]
            if not hasattr(self, attr):
                raise AttributeError(f"Internal: tuning_map → missing attr {attr!r}")
            orig_type = type(getattr(self, attr))
            setattr(self, attr, orig_type(value))
        return self
