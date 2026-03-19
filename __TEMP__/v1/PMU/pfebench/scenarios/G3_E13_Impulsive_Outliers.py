from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G3_E13_Impulsive_Outliers(ScenarioBase):
    """
    Scenario: Impulsive Noise (Outliers / Spikes).

    Signal Model:
      v(t) = A0 * sin(phi(t)) + noise_impulsive(t)

    Spikes occur with probability 'outlier_prob' at each sample.
    Spike magnitude is defined by 'outlier_sigma' (usually large, e.g. 2*A0).

    Critical for testing robust statistics and outlier rejection filters.
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42

    # Outlier Parameters
    outlier_prob: float = 0.12  # 0.1% chance per sample (Sparse)
    outlier_sigma: float = 0.25  # Magnitude of the spike (2.0V if A0=1.0)

    scenario_id: str = "G3_E13_Impulsive_Outliers"

    # --- Monte Carlo Adapter Map ---
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "Vmax": "A0",
            "seed": "seed",
            # Outlier Specific
            "prob": "outlier_prob",  # Density of spikes (0.0 to 1.0)
            "spike_mag": "outlier_sigma",  # Height of spikes
        }
    )

    def build(self) -> ScenarioOutput:
        fs = float(self.fs_hz)
        T = float(self.T_s)

        # 1. Time base
        n = int(round(T * fs))
        if n <= 1:
            n = 2
        t = np.arange(n, dtype=float) / fs

        # 2. Physics: Pure Sine Wave Base
        f_true = np.full_like(t, float(self.f_nom_hz), dtype=float)
        phi = float(self.phi0_rad) + (2.0 * np.pi * float(self.f_nom_hz) * t)
        A = np.full_like(t, float(self.A0), dtype=float)

        v_pure = A * np.sin(phi)

        # 3. Noise: Impulsive Generation
        rng = self._rng()

        # Usamos el mixin helper
        noise = self.add_noise_impulsive(
            shape=t.shape,
            sigma=float(self.outlier_sigma),
            rng=rng,
            prob=float(self.outlier_prob),
        )

        # 4. Final Voltage
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
                "noise_type": "impulsive",
                "outlier_prob": float(self.outlier_prob),
                "outlier_sigma": float(self.outlier_sigma),
                "modifiers": [],
            },
        )

        self.state = state
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
