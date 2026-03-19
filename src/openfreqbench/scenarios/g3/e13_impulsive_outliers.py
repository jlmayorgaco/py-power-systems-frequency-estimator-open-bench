"""
openfreqbench/scenarios/g3/e13_impulsive_outliers.py

G3_E13_Impulsive_Outliers — 60 Hz sine with sparse Bernoulli-Gaussian impulsive noise.

Signal model:
    v(t) = A0 * sin(2*pi*60*t) + n_impulse(t)

    n_impulse(t) = 0                       with probability 1 − impulse_prob
                   ±impulse_sigma * e      with probability impulse_prob
                   (e ~ Exp(1), sign = ±1 with equal probability)

Default: impulse_prob = 0.05 (5% of samples contaminated),
         impulse_sigma = 5.0 (5× unit amplitude).

True frequency is constant at 60 Hz.  Impulsive outliers stress robust
estimators (e.g., median-based zero-crossing, M-estimators) vs. non-robust
least-squares methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G3_E13_Impulsive_Outliers(ScenarioBase):
    """
    Pure 60 Hz sine contaminated with sparse Bernoulli-Gaussian impulsive noise.

    impulse_prob = 0.05 means 5% of samples are independently corrupted.
    impulse_sigma = 5.0 gives outlier magnitudes 5× the nominal amplitude.
    True frequency is constant at 60 Hz throughout.  Tests estimator
    robustness to large, rare-event noise typical of communication channel
    errors, arc flashes, and switching transients.
    """

    fs_hz:         float = 10_000.0
    T_s:           float = 2.0
    seed:          int   = 42
    impulse_prob:  float = 0.05
    impulse_sigma: float = 5.0
    scenario_id:   str   = "G3_E13_Impulsive_Outliers"

    tuning_map: Dict[str, str] = field(default_factory=lambda: {
        "seed":          "seed",
        "impulse_prob":  "impulse_prob",
        "impulse_sigma": "impulse_sigma",
    })

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
