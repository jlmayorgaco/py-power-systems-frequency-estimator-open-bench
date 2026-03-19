"""
openfreqbench/scenarios/_base.py

ScenarioBase, ScenarioState, ScenarioOutput, and ScenarioModifiersMixin.
Ported from pfebench/scenarios/base.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

Array = np.ndarray


def _cumtrapz(y: Array, t: Array) -> Array:
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    n = min(y.size, t.size)
    if n <= 1:
        return np.zeros((n,), dtype=float)
    y, t = y[:n], t[:n]
    out = np.zeros((n,), dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


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
    scenario_id: str
    state: ScenarioState

    def __getattr__(self, name: str):
        if "state" in self.__dict__ and hasattr(self.state, name):
            return getattr(self.state, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    @property
    def t(self) -> Array: return self.state.t
    @property
    def v(self) -> Array: return self.state.v
    @property
    def f_true(self) -> Array: return self.state.f_true
    @property
    def meta(self) -> Dict[str, Any]:
        return {"scenario_id": self.scenario_id, "schema": self.state.schema}


class ScenarioModifiersMixin:
    state: ScenarioState

    def _rng(self, seed_offset: int = 0) -> np.random.Generator:
        base = int(self.state.seed) if hasattr(self, "state") else int(getattr(self, "seed", 0))
        return np.random.default_rng((base + seed_offset) & 0x7FFF_FFFF)

    def recompute_phi_from_f(self, *, phi0_rad: float = 0.0) -> None:
        self.state.phi = float(phi0_rad) + 2.0 * np.pi * _cumtrapz(self.state.f_true, self.state.t)

    def recompute_v_from_A_phi(self) -> None:
        self.state.v = np.asarray(self.state.A, dtype=float) * np.sin(np.asarray(self.state.phi, dtype=float))

    def add_noise_gaussian(self, shape: tuple, sigma: float, rng: np.random.Generator) -> Array:
        return rng.normal(0.0, sigma, size=shape)

    def add_noise_brown(self, shape: tuple, sigma: float, rng: np.random.Generator) -> Array:
        w = rng.normal(0.0, 1.0, size=shape)
        b = np.cumsum(w.reshape(-1))
        b -= np.mean(b)
        rms = float(np.sqrt(np.mean(b**2)))
        return ((b / rms) * sigma).reshape(shape) if rms > 0 else b.reshape(shape)

    def add_noise_impulsive(self, shape: tuple, sigma: float, rng: np.random.Generator, prob: float = 0.001) -> Array:
        n = np.zeros(shape, dtype=float).reshape(-1)
        mask = rng.random(n.size) < prob
        n[mask] = sigma * rng.choice([-1.0, 1.0], size=n.size)[mask]
        return n.reshape(shape)

    def add_noise_uniform(self, shape: tuple, sigma: float, rng: np.random.Generator) -> Array:
        w = sigma * np.sqrt(12.0)
        return rng.uniform(-w/2, w/2, size=shape)


class ScenarioBase(ScenarioModifiersMixin):
    scenario_id: str = "UNKNOWN"
    tuning_map: Dict[str, str] = {}

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("Scenario must implement build()")

    def run(self) -> ScenarioOutput:
        return self.build()

    def configure(self, **kwargs: Any) -> "ScenarioBase":
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Parameter {k!r} not found in {self.scenario_id}")
        return self

    def set_montecarlo_tuning(self, params: Dict[str, Any]) -> "ScenarioBase":
        for alias, value in params.items():
            if alias not in self.tuning_map:
                raise ValueError(f"Alias {alias!r} not in tuning_map for {self.scenario_id}. Allowed: {list(self.tuning_map)}")
            attr = self.tuning_map[alias]
            if not hasattr(self, attr):
                raise AttributeError(f"tuning_map points to missing attr {attr!r}")
            setattr(self, attr, type(getattr(self, attr))(value))
        return self
