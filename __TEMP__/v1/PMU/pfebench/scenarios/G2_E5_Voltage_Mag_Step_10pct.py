from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G2_E5_Voltage_Mag_Step_10pct(ScenarioBase):
    """
    Scenario: Magnitude Step 10% (Significant Amplitude Change).

    Signal Model:
      v(t) = A(t) * sin(phi(t))

      A(t) = A0               if t < step_time
           = A0 * (1 + pct)   if t >= step_time

    Default step_pct is 0.10 (10%), useful for Sag/Swell detection tests.
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42

    # Event Parameters
    step_time_s: float = 2.5
    step_pct: float = 0.10  # 10% Step (0.1 p.u.)

    scenario_id: str = "G2_E5_Voltage_Mag_Step_10pct"

    # --- Monte Carlo Adapter Map ---
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "Vmax": "A0",
            "phi": "phi0_rad",
            "seed": "seed",
            # Event Specific
            "step_t": "step_time_s",
            "step_pct": "step_pct",
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

        # 2. Physics: Constant Frequency
        f_true = np.full_like(t, float(self.f_nom_hz), dtype=float)
        phi = float(self.phi0_rad) + (2.0 * np.pi * float(self.f_nom_hz) * t)

        # 3. Physics: Amplitude Step logic
        A = np.full_like(t, float(self.A0), dtype=float)

        t_step = float(self.step_time_s)
        step_val = float(self.step_pct)

        # Apply Step
        mask_step = t >= t_step
        A[mask_step] *= 1.0 + step_val

        # 4. Final Voltage
        v = A * np.sin(phi)

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
                "event_type": "magnitude_step",
                "step_time_s": t_step,
                "step_pct": step_val,
                "modifiers": [],
            },
        )

        self.state = state
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
