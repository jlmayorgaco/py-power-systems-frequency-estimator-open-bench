"""
openfreqbench/scenarios/g5/e20_snr_sweep.py

G5_E20_SNR_Sweep — parametric sweep over noise level (SNR).

Signal model (identical to G1_E2/E3 family):
    v(t) = A0 * sin(2*pi*60*t) + n(t)

    where n(t) ~ N(0, noise_sigma^2) and noise_sigma is set to sweep_value
    by the Monte Carlo tuning harness at runtime.

The tuning_map routes the sweep alias "noise_sigma" → sweep_value.
Running this scenario across a grid of sweep_value entries (e.g.,
[0.001, 0.005, 0.01, 0.05, 0.10, 0.20]) generates RMSE-vs-SNR curves,
which are one of the primary benchmark outputs for noise sensitivity analysis.

True frequency is constant at 60 Hz throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput


@dataclass
class G5_E20_SNR_Sweep(ScenarioBase):
    """
    Parametric sweep scenario for noise-level (SNR) characterisation.

    sweep_value is the MC-injectable noise amplitude (alias "noise_sigma" in
    tuning_map).  At build time noise_sigma is set to sweep_value so that
    each MC trial uses a different noise level.  True frequency is constant
    at 60 Hz throughout.  Enables RMSE-vs-SNR curves across all estimators
    in a single batch run.
    """

    fs_hz:       float = 10_000.0
    T_s:         float = 2.0
    seed:        int   = 42
    noise_sigma: float = 0.01
    sweep_value: float = 0.01    # set by MC tuning via tuning_map
    scenario_id: str   = "G5_E20_SNR_Sweep"

    tuning_map: Dict[str, str] = field(default_factory=lambda: {
        "seed":        "seed",
        "noise_sigma": "sweep_value",
    })

    def build(self) -> ScenarioOutput:
        raise NotImplementedError("To be implemented in next phase")
