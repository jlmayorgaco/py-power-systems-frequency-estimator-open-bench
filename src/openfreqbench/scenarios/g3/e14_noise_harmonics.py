"""
openfreqbench/scenarios/g3/e14_noise_harmonics.py

G3_E14_Noise_Harmonics — 60 Hz fundamental plus selected odd harmonics at THD = 5%.

Signal model:
    v(t) = A0 * sin(2*pi*f_fund*t)
         + sum_{k in harmonics} A_k * sin(2*pi*k*f_fund*t)

    where A_k are chosen such that
        THD = sqrt(sum_k A_k^2) / A0 = thd
    and all harmonic amplitudes are equal: A_k = thd * A0 / sqrt(len(harmonics)).

Default harmonics: [3, 5, 7] (3rd, 5th, 7th) — dominant in power systems.
True frequency is constant at f_fund = 60 Hz throughout.

Harmonic distortion is a pervasive power-quality issue.  DFT-based estimators
may exhibit spectral leakage; zero-crossing estimators may be confused by
waveform distortion.  THD = 5% is the IEC 61000-2-2 planning level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G3_E14_Noise_Harmonics(ScenarioBase):
    """
    60 Hz fundamental plus 3rd, 5th, and 7th harmonics at THD = 5%.

    Harmonic amplitudes are distributed equally across the three harmonic
    orders so that their combined THD equals the specified value.  True
    frequency is constant at f_fund = 60 Hz.  Tests spectral selectivity
    and harmonic immunity of frequency estimators.
    """

    fs_hz: float = 10_000.0
    T_s: float = 2.0
    seed: int = 42
    f_fund: float = 60.0
    thd: float = 0.05
    harmonics: list[int] = field(default_factory=lambda: [3, 5, 7])
    scenario_id: str = "G3_E14_Noise_Harmonics"

    tuning_map: dict[str, str] = field(
        default_factory=lambda: {
            "seed": "seed",
            "thd": "thd",
        },
    )

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
