"""
estimators/monophasic/f0_pll/srf_pll.py  [CANONICAL]

SRFPLLEstimator — Synchronous Reference Frame Phase-Locked Loop.

Algorithm
─────────
Single-phase SRF-PLL using a SOGI quadrature signal generator (QSG).

Step 1 — SOGI quadrature signal generator (same as SOGI-FLL)
─────────────────────────────────────────────────────────────
  e[n]   = v[n] − α[n−1]
  α[n]   = α[n−1] + Ts·ω̂[n−1]·(k·e[n] − β[n−1])   (in-phase)
  β[n]   = β[n−1] + Ts·ω̂[n−1]·α[n−1]               (quadrature, 90° lag)

Step 2 — dq-frame transformation (Park transform with estimated angle θ)
────────────────────────────────────────────────────────────────────────
  vd[n] =  α[n]·cos(θ[n]) + β[n]·sin(θ[n])
  vq[n] = −α[n]·sin(θ[n]) + β[n]·cos(θ[n])

  For a balanced sinusoidal signal at phase lock: vd → amplitude, vq → 0.
  The q-component is used as the phase error signal.

Step 3 — PI controller drives vq → 0
──────────────────────────────────────
  ε[n]    = vq[n]  (phase error signal)
  u_p[n]  = Kp · ε[n]
  u_i[n]  = u_i[n−1] + Ki · Ts · ε[n]
  ω̂[n]   = ω_nom + u_p[n] + u_i[n]     (angular frequency estimate)

Step 4 — VCO (voltage-controlled oscillator) integrates phase
─────────────────────────────────────────────────────────────
  θ[n] = (θ[n−1] + Ts·ω̂[n]) mod 2π

The SOGI re-uses ω̂ (feedback), making this a SOGI+SRF-PLL rather than a
pure SRF-PLL. This avoids the circular dependency of using Park transform
before frequency is known.

Anti-windup
───────────
  u_i is clamped to ±ω_max_delta to prevent integrator wind-up during
  large frequency transients.

Tunable
───────
  k       — SOGI damping; √2 ≈ 1.414 is critically damped.
  kp      — PI proportional gain.
  ki      — PI integral gain.

Design of PI gains (Teodorescu 2011)
─────────────────────────────────────
  Bandwidth ωn ≈ kp · A / 2  where A is signal amplitude ≈ 1.
  ωn = 2π · 10 Hz → kp ≈ 125 gives ~10 Hz bandwidth.
  Damping ζ = 1 → ki = kp² / 4.

References
──────────
  Kaura, V. & Blasko, V. (1997). "Operation of a phase locked loop system
  under distorted utility conditions." IEEE Trans. Ind. Appl., 33(1), 58–63.

  Teodorescu, R., Liserre, M., Rodriguez, P. (2011). Grid Converters for
  Photovoltaic and Wind Power Systems. Wiley-IEEE Press. Chapter 4.
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

_TWO_PI     = 2.0 * math.pi
_KP_DEFAULT = 125.0    # ~10 Hz bandwidth
_KI_DEFAULT = 3906.25  # kp² / 4 (critical damping)
_K_DEFAULT  = 1.414    # SOGI critically-damped
# Anti-windup: integral limited to ±Δω_max = ±(20 Hz × 2π)
_UI_MAX     = _TWO_PI * 20.0


class SRFPLLEstimator(BaseEstimator):
    """
    Synchronous Reference Frame PLL with SOGI quadrature signal generator.

    Classical PLL structure for single-phase systems.  Locks phase and
    frequency simultaneously; bandwidth is controlled by Kp/Ki.

    Performance comparison vs SOGI-FLL:
      · Steady-state RMSE comparable (both ~O(1) noise → frequency noise gain).
      · Phase tracking more accurate (explicit θ estimate).
      · Slightly slower to recover from large frequency steps (PI integrator).
    """

    SPEC = EstimatorSpec(
        name="SRF_PLL",
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
        return {
            "fs":  10_000.0,
            "kp":  _KP_DEFAULT,
            "ki":  _KI_DEFAULT,
            "k_sogi": _K_DEFAULT,
        }

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="kp",
                    default=_KP_DEFAULT,
                    type="float",
                    values=[31.4, 62.8, 125.0, 250.0, 500.0],
                    description=(
                        "PI proportional gain.  Sets loop bandwidth: "
                        "ωn ≈ Kp·A/2 → Kp=125 gives ~10 Hz bandwidth at A=1."
                    ),
                ),
                TuningParam(
                    name="ki",
                    default=_KI_DEFAULT,
                    type="float",
                    values=[250.0, 1000.0, 3906.0, 15625.0, 62500.0],
                    description=(
                        "PI integral gain.  Critical damping: Ki = Kp²/4."
                    ),
                ),
                TuningParam(
                    name="k_sogi",
                    default=_K_DEFAULT,
                    type="float",
                    values=[0.5, 0.707, 1.0, 1.414, 2.0],
                    description="SOGI damping factor k (√2 → critically damped).",
                ),
            ],
            objective="RMSE_HZ",
        )

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        fs = float(self._config.get("fs", 10_000.0))
        self._fs    = fs
        self._Ts    = 1.0 / fs
        self._kp    = float(self._config.get("kp",     _KP_DEFAULT))
        self._ki    = float(self._config.get("ki",     _KI_DEFAULT))
        self._k     = float(self._config.get("k_sogi", _K_DEFAULT))

        # SOGI states
        self._omega  = _TWO_PI * self.NOMINAL_FREQ_HZ   # angular freq estimate
        self._alpha  = 0.0   # in-phase SOGI output
        self._beta   = 0.0   # quadrature SOGI output (90° lag)

        # PLL states
        self._theta  = 0.0   # estimated phase (VCO output)
        self._ui     = 0.0   # PI integrator state

        self._f_est  = self.NOMINAL_FREQ_HZ
        self._n_samples: int = 0

    def structural_latency_samples(self) -> int:
        fs = float(self._config.get("fs", 10_000.0))
        return int(2 * fs / self.NOMINAL_FREQ_HZ)   # ≈ 2 fundamental cycles

    # ── public interface ───────────────────────────────────────────────────────

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        self._n_samples += 1
        v = float(voltage)

        Ts     = self._Ts
        omega  = self._omega
        k      = self._k
        kp     = self._kp
        ki     = self._ki

        # ── Step 1: SOGI quadrature signal generator ──────────────────────────
        e       = v - self._alpha
        alpha_n = self._alpha + Ts * omega * (k * e - self._beta)
        beta_n  = self._beta  + Ts * omega * self._alpha

        # ── Step 2: dq transformation (Park transform with estimated θ) ───────
        cos_t = math.cos(self._theta)
        sin_t = math.sin(self._theta)
        # vq is the q-component (phase error signal)
        vq = -alpha_n * sin_t + beta_n * cos_t

        # ── Step 3: PI controller drives vq → 0 ───────────────────────────────
        ui_new  = float(np.clip(self._ui + ki * Ts * vq, -_UI_MAX, _UI_MAX))
        omega_n = _TWO_PI * self.NOMINAL_FREQ_HZ + kp * vq + ui_new
        omega_n = float(np.clip(
            omega_n,
            _TWO_PI * self.MIN_VALID_FREQ_HZ,
            _TWO_PI * self.MAX_VALID_FREQ_HZ,
        ))

        # ── Step 4: VCO integrates phase ──────────────────────────────────────
        theta_n = (self._theta + Ts * omega_n) % _TWO_PI

        # ── Store and return ──────────────────────────────────────────────────
        self._alpha  = alpha_n
        self._beta   = beta_n
        self._ui     = ui_new
        self._omega  = omega_n
        self._theta  = theta_n
        self._f_est  = omega_n / _TWO_PI

        valid = (
            self._n_samples > self.structural_latency_samples()
            and self.MIN_VALID_FREQ_HZ <= self._f_est <= self.MAX_VALID_FREQ_HZ
        )
        return EstimatorOutput(
            frequency_hz = self._f_est,
            valid        = valid,
            phase_rad    = self._theta,
            amplitude_pu = math.sqrt(self._alpha ** 2 + self._beta ** 2),
        )

    # Backward-compat
    def _step(self, v_sample: float) -> float:
        return self.update(float(v_sample)).frequency_hz
