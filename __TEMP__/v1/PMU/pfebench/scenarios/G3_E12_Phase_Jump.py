from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G3_E12_Phase_Jump(ScenarioBase):
    """
    Scenario: Phase Angle Jump.

    The phase angle steps instantaneously by 'jump_size_deg' at 'jump_time_s'.
    Frequency magnitude remains constant in the 'f_true' ground truth array,
    though mathematically d(phi)/dt is infinite at the jump.

    Standard IEEE C37.118 test for response time and overshoot.
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42

    # Event Parameters
    jump_time_s: float = 2.5
    jump_size_deg: float = 10.0  # +10 degrees jump

    scenario_id: str = "G3_E12_Phase_Jump"

    # --- Monte Carlo Adapter Map ---
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "Vmax": "A0",
            "phi": "phi0_rad",
            "seed": "seed",
            # Event Specific
            "jump_t": "jump_time_s",
            "jump_deg": "jump_size_deg",
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
        # For benchmarking, we define f_true as constant.
        # The estimator will likely show a spike, which is the error to measure.
        f_true = np.full_like(t, float(self.f_nom_hz), dtype=float)

        # 3. Physics: Phase Construction (Base + Jump)
        # Base linear phase
        phi_base = float(self.phi0_rad) + (2.0 * np.pi * float(self.f_nom_hz) * t)

        # Jump logic
        t_jump = float(self.jump_time_s)
        deg_step = float(self.jump_size_deg)
        rad_step = np.deg2rad(deg_step)

        # Heaviside step for phase
        phi_jump = np.zeros_like(t)
        mask_jump = t >= t_jump
        phi_jump[mask_jump] = rad_step

        # Total Phase
        phi = phi_base + phi_jump

        # 4. Initialize State
        A = np.full_like(t, float(self.A0), dtype=float)
        dummy_v = np.zeros_like(t)  # Will be computed below

        state = ScenarioState(
            t=t,
            fs_hz=fs,
            f_nom_hz=float(self.f_nom_hz),
            seed=int(self.seed),
            f_true=f_true,
            phi=phi,
            A=A,
            v=dummy_v,
            schema={
                "scenario_id": self.scenario_id,
                "seed": int(self.seed),
                "fs_hz": fs,
                "f_nom_hz": float(self.f_nom_hz),
                "A0": float(self.A0),
                "event_type": "phase_jump",
                "jump_time_s": t_jump,
                "jump_size_deg": deg_step,
                "modifiers": [],
            },
        )

        self.state = state

        # --- DRY: Compute Voltage ---
        # We DO NOT use recompute_phi_from_f because we manually built the phase with a jump.
        # We only need to compute voltage from our custom A and phi.
        self.recompute_v_from_A_phi()

        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
