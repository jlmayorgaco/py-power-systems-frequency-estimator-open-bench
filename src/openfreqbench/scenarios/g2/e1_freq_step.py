"""
openfreqbench/scenarios/g2/e1_freq_step.py

G2_E1_FreqStep — pure sine with a single instantaneous frequency step.

Signal model:
    f(t) = f1_hz          for t < t_step
           f2_hz          for t >= t_step

    phi(t) = phi0 + 2*pi * integral_0^t f(tau) d_tau     [phase-continuous]
    v(t)   = A0 * sin(phi(t))

The phase integral is computed exactly (Euler forward, 1-sample resolution):
    phi[0] = phi0_rad
    phi[i] = phi[i-1] + 2*pi * f_true[i-1] / fs

This guarantees phase continuity across the step — no artificial phase jump.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G2_E1_FreqStep(ScenarioBase):
    """
    Pure 60-Hz sine that jumps to f2_hz at t = step_frac * T_s.

    Default: 60 Hz → 61 Hz at mid-signal.
    """

    fs_hz:      float = 10_000.0
    T_s:        float = 1.0          # longer default so step + settling both visible
    f1_hz:      float = 60.0
    f2_hz:      float = 61.0
    step_frac:  float = 0.5          # fraction of T_s at which step occurs
    A0:         float = 1.0
    phi0_rad:   float = 0.0
    seed:       int   = 0
    scenario_id: str  = "G2_E1_FreqStep"

    tuning_map: Dict[str, str] = field(default_factory=lambda: {
        "f1":        "f1_hz",
        "f2":        "f2_hz",
        "step_frac": "step_frac",
        "Vmax":      "A0",
        "phi":       "phi0_rad",
        "seed":      "seed",
    })

    def build(self) -> ScenarioOutput:
        fs = float(self.fs_hz)
        n  = max(2, int(round(self.T_s * fs)))
        t  = np.arange(n, dtype=float) / fs

        # ── Frequency profile: step at step_frac * T_s ──────────────────
        step_sample = int(round(float(self.step_frac) * n))
        step_sample = max(1, min(step_sample, n - 1))   # clamp [1, n-2]

        f_true = np.empty(n, dtype=float)
        f_true[:step_sample] = float(self.f1_hz)
        f_true[step_sample:] = float(self.f2_hz)

        # ── Phase-continuous integration ─────────────────────────────────
        # phi[i] = phi[i-1] + 2*pi * f_true[i-1] / fs
        dphi = 2.0 * np.pi * f_true / fs               # instantaneous phase increment
        phi  = float(self.phi0_rad) + np.concatenate([[0.0], np.cumsum(dphi[:-1])])

        A = np.full(n, float(self.A0))
        v = A * np.sin(phi)

        state = ScenarioState(
            t=t, fs_hz=fs, f_nom_hz=float(self.f1_hz), seed=int(self.seed),
            f_true=f_true, phi=phi, A=A, v=v,
            schema={
                "scenario_id": self.scenario_id,
                "seed":        int(self.seed),
                "fs_hz":       fs,
                "f1_hz":       float(self.f1_hz),
                "f2_hz":       float(self.f2_hz),
                "step_frac":   float(self.step_frac),
                "T_s":         float(self.T_s),
                "A0":          float(self.A0),
                "phi0_rad":    float(self.phi0_rad),
                "step_time_s": float(self.step_frac) * float(self.T_s),
                "delta_f_hz":  float(self.f2_hz) - float(self.f1_hz),
                "modifiers":   [],
            },
        )
        self.state = state
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
