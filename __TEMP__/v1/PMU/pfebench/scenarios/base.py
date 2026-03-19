from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

Array = np.ndarray


# =============================================================================
# 1. Numerical Helpers
# =============================================================================


def _rms(x: Array) -> float:
    """Calculates RMS with epsilon stability."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _cumtrapz(y: Array, t: Array) -> Array:
    """Cumulative trapezoidal integration (scipy-free)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    n = min(y.size, t.size)
    if n <= 1:
        return np.zeros((n,), dtype=float)

    y = y[:n]
    t = t[:n]

    dt = np.diff(t)
    ymid = 0.5 * (y[1:] + y[:-1])
    out = np.zeros((n,), dtype=float)
    out[1:] = np.cumsum(ymid * dt)
    return out


# =============================================================================
# 2. Data Structures
# =============================================================================


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
    """
    Standard output container. Proxies attributes to state.
    """

    scenario_id: str
    state: ScenarioState

    def __getattr__(self, name: str):
        if hasattr(self.state, name):
            return getattr(self.state, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

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
        return {
            "scenario_id": self.scenario_id,
            "schema": self.state.schema,
        }


# =============================================================================
# 3. Mixin (Tools & Noise)
# =============================================================================


class ScenarioModifiersMixin:
    """
    Toolbox mixin for signal modification and noise generation.
    """

    state: ScenarioState

    def _ensure_schema(self) -> None:
        if not isinstance(self.state.schema, dict):
            self.state.schema = {}
        defaults = {
            "scenario_id": getattr(self, "scenario_id", "UNKNOWN"),
            "seed": int(self.state.seed),
            "fs_hz": float(self.state.fs_hz),
            "f_nom_hz": float(self.state.f_nom_hz),
            "modifiers": [],
        }
        for k, v in defaults.items():
            self.state.schema.setdefault(k, v)

    def _append_schema(
        self,
        name: str,
        params: Dict[str, Any],
        *,
        targets: List[str],
        stats: Optional[Dict[str, Any]] = None,
        t0: Optional[float] = None,
        t1: Optional[float] = None,
    ) -> None:
        self._ensure_schema()
        rec = {
            "name": str(name),
            "params": dict(params or {}),
            "targets": list(targets),
        }
        if t0 is not None:
            rec["t0_s"] = float(t0)
        if t1 is not None:
            rec["t1_s"] = float(t1)
        if stats:
            rec["stats"] = dict(stats)

        self.state.schema["modifiers"].append(rec)

    def _rng(self, seed_offset: int = 0) -> np.random.Generator:
        """Returns a deterministic generator based on the scenario seed."""
        # Check if 'state' exists (post-build) or use 'self.seed' directly (during-build)
        if hasattr(self, "state"):
            base_seed = self.state.seed
        else:
            base_seed = getattr(self, "seed", 0)

        seed = int(base_seed) + int(seed_offset)
        return np.random.default_rng(seed & 0x7FFFFFFF)

    def recompute_phi_from_f(self, *, phi0_rad: float = 0.0) -> None:
        """Updates phase (phi) by integrating f_true."""
        integ = _cumtrapz(self.state.f_true, self.state.t)
        self.state.phi = float(phi0_rad) + (2.0 * np.pi) * integ

    def recompute_v_from_A_phi(self) -> None:
        """Updates voltage (v) based on A and phi."""
        self.state.v = np.asarray(self.state.A, dtype=float) * np.sin(
            np.asarray(self.state.phi, dtype=float)
        )

    # --- Noise Generators ---

    def add_noise_gaussian(
        self, shape: tuple, sigma: float, rng: np.random.Generator
    ) -> Array:
        """Standard White Gaussian Noise."""
        return rng.normal(loc=0.0, scale=sigma, size=shape)

    def add_noise_brown(
        self, shape: tuple, sigma: float, rng: np.random.Generator
    ) -> Array:
        """Brownian/Red Noise (Random Walk)."""
        white = rng.normal(loc=0.0, scale=1.0, size=shape)
        brown = np.cumsum(white.reshape(-1))
        brown = brown - np.mean(brown)
        current_rms = np.sqrt(np.mean(brown**2))
        if current_rms > 0:
            brown = (brown / current_rms) * sigma
        return brown.reshape(shape)

    def add_noise_impulsive(
        self, shape: tuple, sigma: float, rng: np.random.Generator, prob: float = 0.001
    ) -> Array:
        """Salt & Pepper / Spikes noise."""
        noise = np.zeros(shape, dtype=float).reshape(-1)
        mask = rng.random(noise.size) < prob
        noise[mask] = sigma * rng.choice([-1.0, 1.0], size=noise.size)[mask]
        return noise.reshape(shape)

    def add_noise_uniform(
        self, shape: tuple, sigma: float, rng: np.random.Generator
    ) -> Array:
        """Uniform Noise (Quantization Error)."""
        width = sigma * np.sqrt(12.0)
        return rng.uniform(low=-width / 2.0, high=width / 2.0, size=shape)


# =============================================================================
# 4. Scenario Base Class
# =============================================================================


class ScenarioBase(ScenarioModifiersMixin):
    scenario_id: str = "UNKNOWN"

    # Mapping for Monte Carlo aliases (Alias -> Internal Attribute)
    tuning_map: Dict[str, str] = {}

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("Scenario must implement build()")

    def run(self) -> ScenarioOutput:
        return self.build()

    def configure(self, **kwargs) -> ScenarioBase:
        """
        Direct parameter update (Internal use / Advanced).
        Updates attributes if they exist.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(
                    f"Parameter '{k}' does not exist in {self.scenario_id}"
                )
        return self

    def set_montecarlo_tuning(self, params: Dict[str, Any]) -> ScenarioBase:
        """
        Monte Carlo Domain Adapter.
        Updates parameters using mapped aliases (Safe & Restricted).

        Args:
            params: Dict { 'alias': value } (e.g. {'f': 59.9, 'seed': 123})
        """
        if not self.tuning_map:
            # If no map is defined, we can't tune safely.
            pass

        for alias, value in params.items():
            # 1. Validate Alias
            if alias not in self.tuning_map:
                raise ValueError(
                    f"Parameter '{alias}' is not allowed for tuning in {self.scenario_id}. "
                    f"Allowed: {list(self.tuning_map.keys())}"
                )

            target_attr = self.tuning_map[alias]

            # 2. Validate Internal Existence
            if not hasattr(self, target_attr):
                raise AttributeError(
                    f"Internal Error: Tuning map points to missing attr '{target_attr}'"
                )

            # 3. Cast & Set
            # Maintain original type (float, int, etc.)
            original_type = type(getattr(self, target_attr))
            setattr(self, target_attr, original_type(value))

        return self
