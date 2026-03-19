from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G2_E8_Fast_Ramp_plus5Hzs(ScenarioBase):
    """
    Scenario: Frequency Ramp (ROCOF Event).

    The frequency increases linearly starting at t = ramp_start_time.
    f(t) = f_nom + rate * (t - t_start)

    Default rate is +5.0 Hz/s (Fast Ramp).
    Critical for testing tracking latency and ROCOF error.
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42

    # Event Parameters
    ramp_start_time_s: float = 1.0
    ramp_rate_hz_s: float = 5.0  # ROCOF: 5 Hz per second

    scenario_id: str = "G2_E8_Fast_Ramp_plus5Hzs"

    # --- Monte Carlo Adapter Map ---
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "Vmax": "A0",
            "phi": "phi0_rad",
            "seed": "seed",
            # Event Specific
            "ramp_t": "ramp_start_time_s",  # Cuándo empieza la rampa
            "ramp_slope": "ramp_rate_hz_s",  # Pendiente (Hz/s)
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

        # 2. Physics: Frequency Ramp Construction
        f_true = np.full_like(t, float(self.f_nom_hz), dtype=float)

        t_start = float(self.ramp_start_time_s)
        rate = float(self.ramp_rate_hz_s)

        # Apply Ramp Logic: f(t) increases linearly after t_start
        mask_ramp = t >= t_start

        # delta_f = rate * delta_t
        time_since_start = t[mask_ramp] - t_start
        f_true[mask_ramp] += rate * time_since_start

        # 3. Physics: Initialize State
        dummy_zeros = np.zeros_like(t)
        A = np.full_like(t, float(self.A0), dtype=float)

        state = ScenarioState(
            t=t,
            fs_hz=fs,
            f_nom_hz=float(self.f_nom_hz),
            seed=int(self.seed),
            f_true=f_true,
            phi=dummy_zeros,  # Will be computed via Mixin (Integral)
            A=A,
            v=dummy_zeros,  # Will be computed via Mixin
            schema={
                "scenario_id": self.scenario_id,
                "seed": int(self.seed),
                "fs_hz": fs,
                "f_nom_hz": float(self.f_nom_hz),
                "A0": float(self.A0),
                "event_type": "frequency_ramp",
                "ramp_start_time_s": t_start,
                "ramp_rate_hz_s": rate,
                "modifiers": [],
            },
        )

        self.state = state

        # --- DRY: Compute Physics (Integral) ---
        # Since f(t) is linear (at+b), phi(t) will be quadratic (0.5*a*t^2 + ...)
        # The numerical integrator handles this perfectly.
        self.recompute_phi_from_f(phi0_rad=float(self.phi0_rad))
        self.recompute_v_from_A_phi()

        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
