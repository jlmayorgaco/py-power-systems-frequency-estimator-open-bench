"""
openfreqbench/scenarios/g1/e3_gaussian_noise_5pct.py

G1_E3_Gaussian_Noise_5pct — pure 60 Hz sine with 5% Gaussian additive noise.

Signal model:
    v(t) = A0 * sin(2*pi*f0*t) + n(t)

    where n(t) ~ N(0, noise_sigma^2) i.i.d., noise_sigma = 0.05 (5% of unit amplitude).

SNR ≈ 26 dB.  Frequency truth is constant at f0 = 60 Hz throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G1_E3_Gaussian_Noise_5pct(ScenarioBase):
    """
    Pure 60 Hz sine corrupted by additive white Gaussian noise at 5% amplitude
    (noise_sigma = 0.05, SNR ≈ 26 dB).

    True frequency is constant at 60 Hz for the entire record.  Compared with
    G1_E2, this scenario stresses estimators under moderate noise and reveals
    bias-variance trade-offs between window-based and recursive methods.
    """

    fs_hz:       float = 10_000.0
    T_s:         float = 2.0
    seed:        int   = 42
    noise_sigma: float = 0.05
    scenario_id: str   = "G1_E3_Gaussian_Noise_5pct"

    tuning_map: Dict[str, str] = field(default_factory=lambda: {
        "seed":        "seed",
        "noise_sigma": "noise_sigma",
    })

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
