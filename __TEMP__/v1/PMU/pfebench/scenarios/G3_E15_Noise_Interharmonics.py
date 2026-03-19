from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G3_E15_Noise_Interharmonics(ScenarioBase):
    """
    Scenario: Interharmonic Noise (OOB Interference).

    Signal Model:
      v(t) = Fundamental + Interharmonic

      Interharmonic is a single tone at 'ih_freq_hz' with magnitude
      determined by 'ihd_pct' relative to the fundamental.

    This simulates Out-of-Band (OOB) interference required by IEEE C37.118
    to test the estimator's anti-aliasing and filtering capabilities.
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42

    # Interharmonic Parameters
    ih_freq_hz: float = 95.0  # Common test: Off-nominal interference
    ihd_pct: float = 0.10  # 10% Interharmonic Distortion

    scenario_id: str = "G3_E15_Noise_Interharmonics"

    # --- Monte Carlo Adapter Map ---
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "Vmax": "A0",
            "phi": "phi0_rad",
            "seed": "seed",
            # Interharmonic Specific
            "ih_freq": "ih_freq_hz",  # Frequency of the interference
            "ih_pct": "ihd_pct",  # Magnitude (0.0 to 1.0)
        }
    )

    def build(self) -> ScenarioOutput:
        fs = float(self.fs_hz)
        T = float(self.T_s)

        n = int(round(T * fs))
        if n <= 1:
            n = 2
        t = np.arange(n, dtype=float) / fs

        # 1. Physics: Fundamental Component
        f_fund = float(self.f_nom_hz)
        f_true = np.full_like(t, f_fund, dtype=float)

        phi_fund = float(self.phi0_rad) + (2.0 * np.pi * f_fund * t)
        v_fund = float(self.A0) * np.sin(phi_fund)

        # 2. Physics: Interharmonic Component
        # Amplitude derived from percentage
        A_ih = float(self.A0) * float(self.ihd_pct)
        f_ih = float(self.ih_freq_hz)

        # Random phase for realism
        rng = self._rng()
        phi_ih_offset = rng.uniform(0, 2 * np.pi)

        # v_ih = A_ih * sin(2*pi*f_ih*t + offset)
        v_interharmonic = A_ih * np.sin((2.0 * np.pi * f_ih * t) + phi_ih_offset)

        # 3. Final Signal
        v = v_fund + v_interharmonic

        state = ScenarioState(
            t=t,
            fs_hz=fs,
            f_nom_hz=f_fund,
            seed=int(self.seed),
            f_true=f_true,  # Truth is ONLY the fundamental
            phi=phi_fund,  # Truth phase is ONLY fundamental phase
            A=np.full_like(t, float(self.A0)),
            v=v,
            schema={
                "scenario_id": self.scenario_id,
                "seed": int(self.seed),
                "fs_hz": fs,
                "f_nom_hz": f_fund,
                "A0": float(self.A0),
                "ihd_pct": float(self.ihd_pct),
                "ih_freq_hz": f_ih,
                "modifiers": [],
            },
        )

        self.state = state
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
