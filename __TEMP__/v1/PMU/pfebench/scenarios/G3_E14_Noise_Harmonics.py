from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G3_E14_Noise_Harmonics(ScenarioBase):
    """
    Scenario: Harmonic Noise (THD).

    Signal Model:
      v(t) = Fundamental + Harmonics
      Harmonics include 3rd, 5th, and 7th orders.

    The 'thd_pct' parameter scales the magnitude of all harmonics
    to achieve the desired Total Harmonic Distortion relative to A0.

    Standard IEEE C37.118 test for Out-of-Band (OOB) interference rejection.
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42

    # Harmonic Parameters
    thd_pct: float = 0.10  # 10% Total Harmonic Distortion

    scenario_id: str = "G3_E14_Noise_Harmonics"

    # --- Monte Carlo Adapter Map ---
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "Vmax": "A0",
            "phi": "phi0_rad",
            "seed": "seed",
            # Harmonic Specific
            "thd": "thd_pct",  # Total Harmonic Distortion (0.0 to 1.0)
        }
    )

    def build(self) -> ScenarioOutput:
        fs = float(self.fs_hz)
        T = float(self.T_s)

        n = int(round(T * fs))
        if n <= 1:
            n = 2
        t = np.arange(n, dtype=float) / fs

        # 1. Physics: Fundamental Component (The "True" Phasor)
        f_fund = float(self.f_nom_hz)
        f_true = np.full_like(t, f_fund, dtype=float)

        phi_fund = float(self.phi0_rad) + (2.0 * np.pi * f_fund * t)
        v_fund = float(self.A0) * np.sin(phi_fund)

        # 2. Physics: Harmonic Generation
        # We define a relative profile: 3rd is strongest, then 5th, then 7th.
        # Profile: [(Order, Relative_Weight)]
        harmonic_profile: List[Tuple[int, float]] = [
            (3, 1.0),  # 3rd Harmonic (Weight 1.0)
            (5, 0.5),  # 5th Harmonic (Weight 0.5)
            (7, 0.25),  # 7th Harmonic (Weight 0.25)
        ]

        # Calculate scaling factor to match requested THD
        # THD = sqrt(sum(Ah^2)) / A0
        # We want: sqrt(sum((Weight*Scale)^2)) = thd_pct
        # Scale * sqrt(sum(Weight^2)) = thd_pct
        # Scale = thd_pct / sqrt(sum(Weight^2))

        sum_sq_weights = sum(w * w for _, w in harmonic_profile)
        norm_factor = float(self.thd_pct) / np.sqrt(sum_sq_weights)

        # Use RNG for random phase offsets of harmonics (Realistic)
        rng = self._rng()

        v_harmonics = np.zeros_like(t)

        for order, weight in harmonic_profile:
            h_amp = float(self.A0) * weight * norm_factor

            # Harmonic Phase: h * phi_fund + random_offset
            # Standard: Harmonics are integer multiples of fundamental phase
            h_phi = (float(order) * phi_fund) + rng.uniform(0, 2 * np.pi)

            v_harmonics += h_amp * np.sin(h_phi)

        # 3. Final Signal
        v = v_fund + v_harmonics

        state = ScenarioState(
            t=t,
            fs_hz=fs,
            f_nom_hz=f_fund,
            seed=int(self.seed),
            f_true=f_true,
            phi=phi_fund,  # The estimator should track FUNDAMENTAL phase
            A=np.full_like(
                t, float(self.A0)
            ),  # True Amplitude is Fundamental's Amplitude
            v=v,
            schema={
                "scenario_id": self.scenario_id,
                "seed": int(self.seed),
                "fs_hz": fs,
                "f_nom_hz": f_fund,
                "A0": float(self.A0),
                "thd_pct": float(self.thd_pct),
                "harmonic_orders": [h[0] for h in harmonic_profile],
                "modifiers": [],
            },
        )

        self.state = state
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
