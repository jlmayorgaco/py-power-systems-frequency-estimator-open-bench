"""
openfreqbench/scenarios/_base.py

ScenarioBase, ScenarioState, ScenarioOutput, and ScenarioModifiersMixin.
Ported from pfebench/scenarios/base.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

Array = np.ndarray


def _cumtrapz(y: Array, t: Array, jumps: list[tuple[float, float]] | None = None) -> Array:
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    n = min(y.size, t.size)
    if n <= 1:
        return np.zeros((n,), dtype=float)
    y, t = y[:n], t[:n]
    out = np.zeros((n,), dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    
    if jumps:
        for t_jump, jump_val in jumps:
            out[t >= t_jump] += jump_val
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
    schema: dict[str, Any]
    roco_f_true: Array | None = None


@dataclass
class ScenarioOutput:
    scenario_id: str
    state: ScenarioState

    def __getattr__(self, name: str) -> Any:
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
    def meta(self) -> dict[str, Any]:
        return {"scenario_id": self.scenario_id, "schema": self.state.schema}


class ScenarioModifiersMixin:
    state: ScenarioState

    def _rng(self, seed_offset: int = 0) -> np.random.Generator:
        base = int(self.state.seed) if hasattr(self, "state") else int(getattr(self, "seed", 0))
        return np.random.default_rng((base + seed_offset) & 0x7FFF_FFFF)

    def recompute_phi_from_f(
        self, *, phi0_rad: float = 0.0, phase_jumps_rad: list[tuple[float, float]] | None = None
    ) -> None:
        cycles_jumps = [(tj, dj / (2.0 * np.pi)) for tj, dj in phase_jumps_rad] if phase_jumps_rad else None
        self.state.phi = float(phi0_rad) + 2.0 * np.pi * _cumtrapz(
            self.state.f_true, self.state.t, jumps=cycles_jumps
        )

    def recompute_v_from_A_phi(self) -> None:
        v_1d = np.asarray(self.state.A, dtype=float) * np.sin(
            np.asarray(self.state.phi, dtype=float),
        )
        self.state.v = v_1d.reshape(-1, 1)

    def recompute_v_from_A_phi_3ph(
        self,
        k_neg: float = 0.0,
        phi_neg_rad: float = 0.0,
        k_zero: float = 0.0,
        phi_zero_rad: float = 0.0,
    ) -> None:
        A = np.asarray(self.state.A, dtype=float)
        phi_pos = np.asarray(self.state.phi, dtype=float)
        
        va_pos = A * np.sin(phi_pos)
        vb_pos = A * np.sin(phi_pos - 2.0 * np.pi / 3.0)
        vc_pos = A * np.sin(phi_pos + 2.0 * np.pi / 3.0)
        
        va_neg = A * k_neg * np.sin(phi_pos + phi_neg_rad)
        vb_neg = A * k_neg * np.sin(phi_pos + phi_neg_rad + 2.0 * np.pi / 3.0)
        vc_neg = A * k_neg * np.sin(phi_pos + phi_neg_rad - 2.0 * np.pi / 3.0)
        
        va_z = A * k_zero * np.sin(phi_pos + phi_zero_rad)
        
        self.state.v = np.column_stack((
            va_pos + va_neg + va_z,
            vb_pos + vb_neg + va_z,
            vc_pos + vc_neg + va_z,
        ))

    def recompute_roco_f_from_f(self) -> None:
        self.state.roco_f_true = np.gradient(self.state.f_true) * self.state.fs_hz

    def apply_awgn(self, signal: Array, snr_db: float, rng: np.random.Generator) -> Array:
        sig_power = float(np.mean(np.square(signal)))
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        sigma = np.sqrt(noise_power)
        noise = rng.normal(0.0, sigma, size=signal.shape)
        return signal + noise

    def apply_colored_noise(self, signal: Array, snr_db: float, color: str, rng: np.random.Generator) -> Array:
        sig_power = float(np.mean(np.square(signal)))
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        
        if color == "brown":
            w = rng.normal(0.0, 1.0, size=signal.shape)
            b = np.cumsum(w.reshape(-1))
            b -= np.mean(b)
            rms = float(np.sqrt(np.mean(b**2)))
            noise = ((b / max(1e-12, rms)) * np.sqrt(noise_power)).reshape(signal.shape)
        elif color == "pink":
            # 1/f noise generation
            N = signal.size
            X = np.fft.rfft(rng.normal(0.0, 1.0, size=N))
            f = np.fft.rfftfreq(N)
            f[0] = 1.0  # robust against div 0
            X_pink = X / np.sqrt(f)
            n_pink = np.fft.irfft(X_pink, n=N)
            n_pink -= np.mean(n_pink)
            rms = float(np.sqrt(np.mean(n_pink**2)))
            noise = ((n_pink / max(1e-12, rms)) * np.sqrt(noise_power)).reshape(signal.shape)
        else:
            raise ValueError(f"Unknown noise color {color!r}")

        return signal + noise

    def apply_impulsive_noise(
        self,
        signal: Array,
        snr_db: float,
        rng: np.random.Generator,
        prob: float = 0.001,
    ) -> Array:
        sig_power = float(np.mean(np.square(signal)))
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        impulse_amp = np.sqrt(noise_power / max(1e-9, prob))
        n = np.zeros_like(signal, dtype=float).reshape(-1)
        mask = rng.random(n.size) < prob
        n[mask] = impulse_amp * rng.choice([-1.0, 1.0], size=np.count_nonzero(mask))
        return signal + n.reshape(signal.shape)



class ScenarioBase(ScenarioModifiersMixin):
    scenario_id: ClassVar[str] = "UNKNOWN"
    tuning_map: ClassVar[dict[str, str]] = {}

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("Scenario must implement build()")

    def run(self) -> ScenarioOutput:
        return self.build()

    def configure(self, **kwargs: Any) -> ScenarioBase:
        from dataclasses import replace

        for k in kwargs:
            if not hasattr(self, k):
                raise AttributeError(f"Parameter {k!r} not found in {self.scenario_id}")
        return replace(self, **kwargs)

    def set_montecarlo_tuning(self, params: dict[str, Any]) -> ScenarioBase:
        from dataclasses import replace

        updates = {}
        for alias, value in params.items():
            if alias not in self.tuning_map:
                raise ValueError(
                    f"Alias {alias!r} not in tuning_map for {self.scenario_id}. Allowed: {list(self.tuning_map)}",
                )
            attr = self.tuning_map[alias]
            if not hasattr(self, attr):
                raise AttributeError(f"tuning_map points to missing attr {attr!r}")
            updates[attr] = type(getattr(self, attr))(value)
        return replace(self, **updates)
