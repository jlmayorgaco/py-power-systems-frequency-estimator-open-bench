"""
openfreqbench/scenarios/g4/e18_chamorro_event.py

G4_E18_Chamorro_Event — Chamorro near-islanding event profile.

Signal model:
    f(t) = 60.0                              for t < 0
           60.0 + rocof_hz_s * t             for t >= 0   (continuous ramp from t=0)

    A(t) = 1.0 + osc_depth * sin(2*pi*osc_freq_hz*t)   (voltage oscillation throughout)

    phi(t) = phi0 + 2*pi * integral_0^t f(tau) d_tau   [phase-continuous]
    v(t)   = A(t) * sin(phi(t))

The Chamorro event is characterised by a simultaneous frequency ramp and
low-frequency voltage oscillation.  The oscillation at osc_freq_hz = 5 Hz
with osc_depth = 0.05 is calibrated to fall within the near-islanding
detection band defined in the Chamorro et al. (2011) reference.

This scenario constitutes the benchmark's primary scientific claim: ranking
estimator performance under the hardest real-world islanding signature.

Reference:
    Chamorro, H. R. et al. (2011). "A new method for islanding detection
    based on wavelet packet transform." IEEE PES Conference.
"""

from __future__ import annotations

from typing import ClassVar

from dataclasses import dataclass, field

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G4_E18_Chamorro_Event(ScenarioBase):
    """
    Chamorro islanding event: simultaneous frequency ramp (rocof_hz_s) and
    voltage oscillation (osc_freq_hz, osc_depth) from t=0.

    Default rocof_hz_s = 1 Hz/s and osc_depth = 5% at 5 Hz place the
    signal at the near-islanding detection boundary.  This is the
    benchmark's primary scientific claim scenario and should be run for
    all estimators in the final comparative study.
    """

    fs_hz: float = 10_000.0
    T_s: float = 3.0
    seed: int = 42
    rocof_hz_s: float = 1.0
    osc_freq_hz: float = 5.0
    osc_depth: float = 0.05
    scenario_id: ClassVar[str] = "G4_E18_Chamorro_Event"

    tuning_map: ClassVar[dict[str, str]] = {
            "seed": "seed",
            "rocof_hz_s": "rocof_hz_s",
        }

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
