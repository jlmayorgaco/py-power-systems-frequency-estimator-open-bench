"""
openfreqbench/scenarios/g4/e16_composite_islanding.py

G4_E16_Composite_Islanding — composite islanding event with simultaneous voltage sag
and frequency ramp.

Signal model (both effects start simultaneously at t_event):
    A(t) = 1.0                                       for t < t_event
           1.0 - v_sag_pu                             for t >= t_event

    f(t) = 60.0                                      for t < t_event
           60.0 + rocof_hz_s * (t - t_event)         for t >= t_event

    phi(t) = phi0 + 2*pi * integral_0^t f(tau) d_tau   [phase-continuous]
    v(t)   = A(t) * sin(phi(t))

Default: t_event = 0.5 s, v_sag_pu = 0.10 (-10% voltage), rocof_hz_s = 2 Hz/s.

This scenario reproduces the first instants of an islanding event where the
distributed resource loses mains synchronism.  The simultaneous onset of
voltage sag and frequency ramp is the key diagnostic signature targeted by
islanding detection algorithms.
"""

from __future__ import annotations

from typing import ClassVar

from dataclasses import dataclass, field

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G4_E16_Composite_Islanding(ScenarioBase):
    """
    Composite islanding event: voltage sag (-v_sag_pu) and frequency ramp
    (+rocof_hz_s Hz/s) start simultaneously at t_event seconds.

    The co-occurrence of amplitude drop and accelerating frequency deviation
    is the canonical signature of an islanding event in IEEE 1547 and
    IEC 62116 test suites.  Tests whether estimators remain unambiguous
    under combined amplitude-and-frequency disturbance.
    """

    fs_hz: float = 10_000.0
    T_s: float = 3.0
    seed: int = 42
    t_event: float = 0.5
    v_sag_pu: float = 0.10
    rocof_hz_s: float = 2.0
    scenario_id: ClassVar[str] = "G4_E16_Composite_Islanding"

    tuning_map: ClassVar[dict[str, str]] = {
            "seed": "seed",
            "t_event": "t_event",
        }

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
