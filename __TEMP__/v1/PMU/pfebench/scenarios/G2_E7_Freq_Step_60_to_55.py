from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G2_E7_Freq_Step_60_to_55(ScenarioBase):
    """
    Scenario: Severe Frequency Collapse (60Hz -> 55Hz).

    The frequency drops by 5.0 Hz at t = step_time.
    This represents a massive generation loss or islanding event.

    Critical for testing numerical stability and 'Last Resort' protection logic.
    Phase is integrated to ensure continuity despite the massive slope change.
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42

    # Event Parameters
    step_time_s: float = 2.5
    step_size_hz: float = -5.0  # Massive drop (60 -> 55)

    scenario_id: str = "G2_E7_Freq_Step_60_to_55"

    # --- Monte Carlo Adapter Map ---
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "Vmax": "A0",
            "phi": "phi0_rad",
            "seed": "seed",
            # Event Specific
            "step_t": "step_time_s",
            "step_hz": "step_size_hz",
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

        # 2. Physics: Frequency Step Construction
        f_true = np.full_like(t, float(self.f_nom_hz), dtype=float)

        t_step = float(self.step_time_s)
        step_val = float(self.step_size_hz)

        # Apply Step
        mask_step = t >= t_step
        f_true[mask_step] += step_val

        # 3. Physics: Initialize State
        dummy_zeros = np.zeros_like(t)
        A = np.full_like(t, float(self.A0), dtype=float)

        state = ScenarioState(
            t=t,
            fs_hz=fs,
            f_nom_hz=float(self.f_nom_hz),
            seed=int(self.seed),
            f_true=f_true,
            phi=dummy_zeros,  # Will be computed via Mixin
            A=A,
            v=dummy_zeros,  # Will be computed via Mixin
            schema={
                "scenario_id": self.scenario_id,
                "seed": int(self.seed),
                "fs_hz": fs,
                "f_nom_hz": float(self.f_nom_hz),
                "A0": float(self.A0),
                "event_type": "frequency_collapse",
                "step_time_s": t_step,
                "step_size_hz": step_val,
                "modifiers": [],
            },
        )

        # Attach state to self (Mixin requirement)
        self.state = state

        # --- DRY: Compute Physics ---
        # 1. Integral(f) -> phi (Continuous phase)
        self.recompute_phi_from_f(phi0_rad=float(self.phi0_rad))

        # 2. A * sin(phi) -> v
        self.recompute_v_from_A_phi()

        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
