"""
openfreqbench/scenarios/g1/e2_gaussian_noise_1pct.py

G1_E2_Gaussian_Noise_1pct — pure 60 Hz sine with 1% Gaussian additive noise.

Signal model:
    v(t) = A0 * sin(2*pi*f0*t) + n(t)

    where n(t) ~ N(0, noise_sigma^2) i.i.d., noise_sigma = 0.01 (1% of unit amplitude).

SNR ≈ 40 dB.  Frequency truth is constant at f0 = 60 Hz throughout.
"""

from __future__ import annotations

from typing import ClassVar

from dataclasses import dataclass, field

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G1_E2_Gaussian_Noise_1pct(ScenarioBase):
    """
    Pure 60 Hz sine corrupted by additive white Gaussian noise at 1% amplitude
    (noise_sigma = 0.01, SNR ≈ 40 dB).

    True frequency is constant at 60 Hz for the entire record.  This scenario
    tests estimator noise floor and baseline precision under mild noise.
    """

    fs_hz: float = 10_000.0
    T_s: float = 2.0
    seed: int = 42
    noise_sigma: float = 0.01
    scenario_id: ClassVar[str] = "G1_E2_Gaussian_Noise_1pct"

    tuning_map: ClassVar[dict[str, str]] = {
            "seed": "seed",
            "noise_sigma": "noise_sigma",
        }

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
