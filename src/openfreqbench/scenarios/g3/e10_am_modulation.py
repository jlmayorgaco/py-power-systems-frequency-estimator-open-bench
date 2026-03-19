"""
openfreqbench/scenarios/g3/e10_am_modulation.py

G3_E10_AM_Modulation — 60 Hz carrier with sinusoidal amplitude modulation.

Signal model:
    A(t) = A0 * [1 + mod_depth * sin(2*pi*mod_freq_hz*t)]
    phi(t) = 2*pi*f_carrier*t         (constant frequency)
    v(t)   = A(t) * sin(phi(t))

True frequency is constant at f_carrier = 60 Hz throughout.  The AM
envelope at mod_freq_hz = 5 Hz with 10% depth exercises estimators that
are sensitive to amplitude fluctuations (e.g., instantaneous frequency
via analytic signal, zero-crossing methods with amplitude weighting).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G3_E10_AM_Modulation(ScenarioBase):
    """
    60 Hz carrier with sinusoidal amplitude modulation at mod_freq_hz, depth mod_depth.

    The true instantaneous frequency is constant at f_carrier = 60 Hz; only
    the envelope oscillates.  mod_freq_hz = 5 Hz and mod_depth = 0.10 (10%)
    represent typical sub-synchronous oscillations.  Tests immunity of
    frequency estimators to amplitude modulation artefacts.
    """

    fs_hz: float = 10_000.0
    T_s: float = 2.0
    seed: int = 42
    f_carrier: float = 60.0
    mod_freq_hz: float = 5.0
    mod_depth: float = 0.10
    scenario_id: str = "G3_E10_AM_Modulation"

    tuning_map: dict[str, str] = field(
        default_factory=lambda: {
            "seed": "seed",
            "mod_freq_hz": "mod_freq_hz",
            "mod_depth": "mod_depth",
        },
    )

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
