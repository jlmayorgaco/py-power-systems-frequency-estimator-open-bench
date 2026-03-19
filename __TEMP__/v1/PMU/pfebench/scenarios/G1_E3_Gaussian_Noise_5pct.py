from __future__ import annotations

# CORRECCIÓN: Añadido 'field' al import
from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G1_E3_Gaussian_Noise_5pct(ScenarioBase):
    """
    Scenario: Nominal sine with High Gaussian White Noise (5%).

    Signal Model: v(t) = A0 * sin(phi(t)) + noise(t)
    where noise(t) ~ N(0, (noise_rel * A0)^2)
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42
    noise_rel: float = 0.05  # Default 5%

    scenario_id: str = "G1_E3_Gaussian_Noise_5pct"

    # --- Monte Carlo Adapter Map ---
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "Vmax": "A0",
            "phi": "phi0_rad",
            "seed": "seed",
            "noise": "noise_rel",
        }
    )

    def build(self) -> ScenarioOutput:
        fs = float(self.fs_hz)
        T = float(self.T_s)

        n = int(round(T * fs))
        if n <= 1:
            n = 2
        t = np.arange(n, dtype=float) / fs

        # Physics
        f_true = np.full_like(t, float(self.f_nom_hz), dtype=float)
        phi = float(self.phi0_rad) + (2.0 * np.pi * float(self.f_nom_hz) * t)

        A = np.full_like(t, float(self.A0), dtype=float)
        v_pure = A * np.sin(phi)

        # Noise Generation
        rng = self._rng()
        sigma = float(self.A0) * float(self.noise_rel)
        noise = self.add_noise_gaussian(shape=t.shape, sigma=sigma, rng=rng)

        # Final Signal
        v = v_pure + noise

        state = ScenarioState(
            t=t,
            fs_hz=fs,
            f_nom_hz=float(self.f_nom_hz),
            seed=int(self.seed),
            f_true=f_true,
            phi=phi,
            A=A,
            v=v,
            schema={
                "scenario_id": self.scenario_id,
                "seed": int(self.seed),
                "fs_hz": fs,
                "f_nom_hz": float(self.f_nom_hz),
                "A0": float(self.A0),
                "noise_rel": float(self.noise_rel),
                "sigma_expected": sigma,
                "noise_type": "gaussian",
                "modifiers": [],
            },
        )
        self.state = state
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
