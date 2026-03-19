"""
openfreqbench/scenarios/g2/e6_freq_step_59p5.py

G2_E6_Freq_Step_60_to_59p5 — abrupt frequency step from 60.0 to 59.5 Hz.

Signal model:
    f(t) = f_before      for t < t_step
           f_after        for t >= t_step   (f_before=60.0, f_after=59.5)

    phi(t) = phi0 + 2*pi * integral_0^t f(tau) d_tau   [phase-continuous]
    v(t)   = A0 * sin(phi(t))

The phase integral is computed exactly to guarantee phase continuity across
the step — no artificial phase discontinuity is introduced.

This -0.5 Hz step represents a mild under-frequency event, typical of a
small generator trip on a well-interconnected grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G2_E6_Freq_Step_60_to_59p5(ScenarioBase):
    """
    Phase-continuous frequency step from 60.0 Hz to 59.5 Hz at t_step seconds.

    The -0.5 Hz step (delta_f = -0.5 Hz) represents a mild under-frequency
    disturbance.  Phase continuity is preserved across the step instant.
    Tests estimator transient response to small frequency deviations.
    """

    fs_hz: float = 10_000.0
    T_s: float = 2.0
    seed: int = 42
    t_step: float = 0.5
    f_before: float = 60.0
    f_after: float = 59.5
    scenario_id: str = "G2_E6_Freq_Step_60_to_59p5"

    tuning_map: dict[str, str] = field(
        default_factory=lambda: {
            "seed": "seed",
            "t_step": "t_step",
        },
    )

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
