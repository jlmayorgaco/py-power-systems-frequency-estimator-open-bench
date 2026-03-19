"""
openfreqbench/scenarios/g3/e15_noise_interharmonics.py

G3_E15_Noise_Interharmonics — 60 Hz signal with non-integer interharmonic components.

Signal model:
    v(t) = A0 * sin(2*pi*f_fund*t)
         + sum_{f_ih in interharmonics} inter_amp * sin(2*pi*f_ih*t + phi_ih)

Default interharmonics: [135.0, 210.0] Hz at inter_amp = 0.05 (5% each).

Interharmonics at non-integer multiples of the fundamental arise from cyclo-
converters, arc furnaces, and wind turbines.  Unlike harmonics they do not
lock to the fundamental period, causing DFT bin smearing and beat patterns
in zero-crossing methods.  True frequency is constant at f_fund = 60 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G3_E15_Noise_Interharmonics(ScenarioBase):
    """
    60 Hz signal with interharmonic components at 135 Hz and 210 Hz (5% each).

    The interharmonics are at non-integer multiples of the 60 Hz fundamental,
    producing slowly-varying beat envelopes.  True frequency is constant at
    f_fund = 60 Hz.  inter_amp = 0.05 places each interharmonic at 5% of the
    fundamental amplitude.  Tests estimator immunity to non-synchronous
    spectral interference.
    """

    fs_hz:          float       = 10_000.0
    T_s:            float       = 2.0
    seed:           int         = 42
    f_fund:         float       = 60.0
    interharmonics: List[float] = field(default_factory=lambda: [135.0, 210.0])
    inter_amp:      float       = 0.05
    scenario_id:    str         = "G3_E15_Noise_Interharmonics"

    tuning_map: Dict[str, str] = field(default_factory=lambda: {
        "seed":      "seed",
        "inter_amp": "inter_amp",
    })

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
