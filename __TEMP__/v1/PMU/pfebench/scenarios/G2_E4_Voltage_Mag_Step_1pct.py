from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G2_E4_Voltage_Mag_Step_1pct(ScenarioBase):
    """
    Scenario: Magnitude Step (Rectangular change in Amplitude).

    Signal Model:
      v(t) = A(t) * sin(phi(t))

      A(t) = A0               if t < step_time
           = A0 * (1 + pct)   if t >= step_time

    Standard test for Step Response measurement (TVE, Response Time).
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42

    # Event Parameters
    step_time_s: float = 2.5  # Time where the step occurs
    step_pct: float = 0.01  # Step magnitude relative to A0 (0.01 = 1%)

    scenario_id: str = "G2_E4_Voltage_Mag_Step_1pct"

    # --- Monte Carlo Adapter Map ---
    # Maps 'Domain Alias' -> 'Internal Attribute'
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",  # Frecuencia base
            "Vmax": "A0",  # Amplitud base (Pre-step)
            "phi": "phi0_rad",  # Fase inicial
            "seed": "seed",  # Semilla
            # Event Specific Tuning
            "step_t": "step_time_s",  # Momento del evento
            "step_pct": "step_pct",  # Tamaño del escalón (ej. 0.1 para 10%)
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

        # 2. Physics: Frequency (Constant) & Phase (Linear)
        f_true = np.full_like(t, float(self.f_nom_hz), dtype=float)
        phi = float(self.phi0_rad) + (2.0 * np.pi * float(self.f_nom_hz) * t)

        # 3. Physics: Amplitude Step logic
        A = np.full_like(t, float(self.A0), dtype=float)

        t_step = float(self.step_time_s)
        step_val = float(self.step_pct)

        # This creates the Heaviside step function effect
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
