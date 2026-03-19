"""
estimators/monophasic/f0_pll/sogi_fll.py  [CANONICAL]

SOGIFLLEstimator — Second-Order Generalized Integrator Frequency-Locked Loop.

A PLL-family adaptive filter that locks on the fundamental frequency using
orthogonal signal generation.  No FFT required; fully streaming, O(1).

State update (discrete-time Euler forward, Ts = 1/fs)
──────────────────────────────────────────────────────
  e[n]   = v[n] − α[n−1]
  α[n]   = α[n−1] + Ts·ω̂[n−1]·(k·e[n] − β[n−1])
  β[n]   = β[n−1] + Ts·ω̂[n−1]·α[n−1]
  εf[n]  = e[n]·β[n]         (FLL error signal)
  ω̂[n]  = ω̂[n−1] − γ·Ts·εf[n]
  f̂[n]  = ω̂[n] / (2π)

Tunable
───────
  k     — SOGI damping; √2 ≈ 1.414 is critically damped.
  gamma — FLL loop gain; larger → faster locking but noisier.

References
──────────
  Ciobotaru, M., Teodorescu, R., Blaabjerg, F. (2006).
  "A new single-phase PLL structure based on second order generalized
  integrator." PESC'06, pp. 1–6.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningParam,
    TuningSpec,
)


class SOGIFLLEstimator(BaseEstimator):
    """
    SOGI-FLL frequency estimator — streaming, O(1) per sample.

    Optimal for smooth frequency variations; sensitive to large initial
    errors if the loop gain ``gamma`` is too low.
    """

    SPEC = EstimatorSpec(
        name="SOGI_FLL",
        family="PLL",
        family_path="monophasic/f0_pll",
        complexity="O(1)",
        latency_type="causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {"fs": 10_000.0, "k": 1.414, "gamma": 50.0}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="k",
                    default=1.414,
                    type="float",
                    values=[0.5, 0.707, 1.0, 1.414, 2.0, 2.828, 4.0],
                    description=(
                        "SOGI damping coefficient.  √2 ≈ 1.414 is critically damped. "
                        "Lower → more oscillatory but faster; higher → overdamped."
                    ),
                ),
                TuningParam(
                    name="gamma",
                    default=50.0,
                    type="float",
                    values=[5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
                    description=(
                        "FLL loop gain γ.  Higher → faster frequency tracking but "
                        "more noise sensitivity.  Tune for the expected rate of "
                        "frequency change."
                    ),
                ),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        self._fs:      float = float(self._config.get("fs",    10_000.0))
        self._Ts:      float = 1.0 / self._fs
        self._k:       float = float(self._config.get("k",     1.414))
        self._gamma:   float = float(self._config.get("gamma", 50.0))
        self._omega:   float = 2.0 * math.pi * self.NOMINAL_FREQ_HZ
        self._alpha:   float = 0.0
        self._beta:    float = 0.0
        self._f_est:   float = self.NOMINAL_FREQ_HZ
        self._n_samples: int = 0

    def structural_latency_samples(self) -> int:
        fs = float(self._config.get("fs", 10_000.0))
        return int(fs / self.NOMINAL_FREQ_HZ)   # ≈ 1 fundamental cycle

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        self._n_samples += 1
        v      = float(voltage)
        Ts     = self._Ts
        omega  = self._omega
        k      = self._k
        gamma  = self._gamma

        # SOGI update
        e        = v - self._alpha
        alpha_n  = self._alpha + Ts * omega * (k * e - self._beta)
        beta_n   = self._beta  + Ts * omega * self._alpha

        # FLL frequency update
        eps_f   = e * beta_n
        omega_n = omega - gamma * Ts * eps_f
        omega_n = float(np.clip(
            omega_n,
            2.0 * math.pi * self.MIN_VALID_FREQ_HZ,
            2.0 * math.pi * self.MAX_VALID_FREQ_HZ,
        ))

        self._alpha = alpha_n
        self._beta  = beta_n
        self._omega = omega_n
        self._f_est = omega_n / (2.0 * math.pi)

        valid = (
            self._n_samples > self.structural_latency_samples()
            and self.MIN_VALID_FREQ_HZ <= self._f_est <= self.MAX_VALID_FREQ_HZ
        )
        return EstimatorOutput(
            frequency_hz = self._f_est,
            valid        = valid,
            phase_rad    = math.atan2(self._beta, self._alpha),
            amplitude_pu = math.sqrt(self._alpha ** 2 + self._beta ** 2),
        )

    # Backward-compat
    def _step(self, v_sample: float) -> float:
        return self.update(float(v_sample)).frequency_hz
