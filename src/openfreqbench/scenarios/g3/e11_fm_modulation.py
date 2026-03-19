"""
openfreqbench/scenarios/g3/e11_fm_modulation.py

G3_E11_FM_Modulation — frequency-modulated 60 Hz signal.

Signal model:
    f(t)   = f_carrier + delta_f_hz * sin(2*pi*f_mod_hz*t)
    phi(t) = phi0 + 2*pi * integral_0^t f(tau) d_tau   [phase-continuous]
    v(t)   = A0 * sin(phi(t))

Default parameters: f_carrier = 60 Hz, delta_f_hz = 0.5 Hz, f_mod_hz = 2 Hz.

The instantaneous frequency oscillates sinusoidally between 59.5 and 60.5 Hz
at 2 Hz, representing inter-area power oscillations or governor-induced
frequency swings.  This is the canonical dynamic frequency tracking test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G3_E11_FM_Modulation(ScenarioBase):
    """
    Frequency-modulated signal: f(t) = f_carrier + delta_f_hz·sin(2π·f_mod_hz·t).

    delta_f_hz = 0.5 Hz and f_mod_hz = 2 Hz produce a sinusoidal frequency
    oscillation representative of inter-area power swings.  The ground-truth
    instantaneous frequency is analytically known, enabling exact RMSE
    calculation.  Phase continuity is maintained by numerical integration.
    """

    fs_hz: float = 10_000.0
    T_s: float = 2.0
    seed: int = 42
    f_carrier: float = 60.0
    delta_f_hz: float = 0.5
    f_mod_hz: float = 2.0
    scenario_id: str = "G3_E11_FM_Modulation"

    tuning_map: dict[str, str] = field(
        default_factory=lambda: {
            "seed": "seed",
            "delta_f_hz": "delta_f_hz",
            "f_mod_hz": "f_mod_hz",
        },
    )

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
