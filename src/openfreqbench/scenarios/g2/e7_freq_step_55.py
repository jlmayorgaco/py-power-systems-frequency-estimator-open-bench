"""
openfreqbench/scenarios/g2/e7_freq_step_55.py

G2_E7_Freq_Step_60_to_55 — large abrupt frequency step from 60.0 to 55.0 Hz.

Signal model:
    f(t) = f_before      for t < t_step
           f_after        for t >= t_step   (f_before=60.0, f_after=55.0)

    phi(t) = phi0 + 2*pi * integral_0^t f(tau) d_tau   [phase-continuous]
    v(t)   = A0 * sin(phi(t))

The phase integral is computed exactly to guarantee phase continuity across
the step — no artificial phase discontinuity is introduced.

The −5 Hz step models a severe under-frequency event (e.g., loss of a large
generation block).  This is at the boundary of protective relay trip thresholds
and exercises estimator pull-in range and large-signal tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G2_E7_Freq_Step_60_to_55(ScenarioBase):
    """
    Phase-continuous frequency step from 60.0 Hz to 55.0 Hz at t_step seconds.

    The −5 Hz step (delta_f = −5 Hz) simulates a severe under-frequency event
    at the edge of under-frequency load-shedding thresholds.  Phase continuity
    is preserved across the step instant.  Tests large-signal tracking ability
    and lock-range of PLL-based and filter-based estimators.
    """

    fs_hz:       float = 10_000.0
    T_s:         float = 2.0
    seed:        int   = 42
    t_step:      float = 0.5
    f_before:    float = 60.0
    f_after:     float = 55.0
    scenario_id: str   = "G2_E7_Freq_Step_60_to_55"

    tuning_map: Dict[str, str] = field(default_factory=lambda: {
        "seed":   "seed",
        "t_step": "t_step",
    })

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
