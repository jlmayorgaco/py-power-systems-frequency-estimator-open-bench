"""
openfreqbench/scenarios/g2/e2_freq_ramp.py

G2_E2_FreqRamp — pure sine with a linear frequency ramp.

Signal model:
    f(t) = f0_hz                              for t < onset_frac * T_s
           f0_hz + ramp_rate_hzs * (t - t0)  for t >= onset_frac * T_s
           clipped to [f0_hz, f0_hz + df_max] to avoid unbounded drift

    phi(t) = phi0 + 2*pi * integral_0^t f(tau) d_tau  [phase-continuous]
    v(t)   = A0 * sin(phi(t))

Phase is integrated sample-by-sample (Euler forward), matching G2_E1_FreqStep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openfreqbench.scenarios._base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G2_E2_FreqRamp(ScenarioBase):
    """
    60-Hz sine that begins a linear frequency ramp at onset_frac * T_s.

    Default: ramp starts at t=0.25s, rate=2 Hz/s, lasts until end.
    """

    fs_hz: float = 10_000.0
    T_s: float = 2.0  # seconds; long enough to see ramp settling
    f0_hz: float = 60.0  # pre-ramp frequency
    ramp_rate_hzs: float = 2.0  # Hz/s — positive = ramp up
    onset_frac: float = 0.25  # fraction of T_s at which ramp begins
    df_max_hz: float = 10.0  # max total frequency deviation (clamp)
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 0
    scenario_id: str = "G2_E2_FreqRamp"

    tuning_map: dict[str, str] = field(
        default_factory=lambda: {
            "f0": "f0_hz",
            "ramp_rate": "ramp_rate_hzs",
            "onset_frac": "onset_frac",
            "Vmax": "A0",
            "phi": "phi0_rad",
            "seed": "seed",
        },
    )

    def build(self) -> ScenarioOutput:
        fs = float(self.fs_hz)
        n = max(2, round(self.T_s * fs))
        t = np.arange(n, dtype=float) / fs

        # ── Frequency profile ────────────────────────────────────────────────
        onset_sample = round(float(self.onset_frac) * n)
        onset_sample = max(1, min(onset_sample, n - 1))
        onset_time = onset_sample / fs

        f_true = np.full(n, float(self.f0_hz))
        # Linear ramp from onset onwards
        t_rel = np.maximum(0.0, t - onset_time)
        f_ramp = float(self.f0_hz) + float(self.ramp_rate_hzs) * t_rel
        # Clamp to [f0, f0 + df_max] (handles sign of ramp_rate_hzs)
        f_lo = float(self.f0_hz) + min(0.0, float(self.df_max_hz))
        f_hi = float(self.f0_hz) + max(0.0, float(self.df_max_hz))
        f_ramp = np.clip(f_ramp, f_lo, f_hi)
        f_true[onset_sample:] = f_ramp[onset_sample:]

        # ── Phase-continuous integration ─────────────────────────────────────
        dphi = 2.0 * np.pi * f_true / fs
        phi = float(self.phi0_rad) + np.concatenate([[0.0], np.cumsum(dphi[:-1])])

        A = np.full(n, float(self.A0))
        v = A * np.sin(phi)

        state = ScenarioState(
            t=t,
            fs_hz=fs,
            f_nom_hz=float(self.f0_hz),
            seed=int(self.seed),
            f_true=f_true,
            phi=phi,
            A=A,
            v=v,
            schema={
                "scenario_id": self.scenario_id,
                "seed": int(self.seed),
                "fs_hz": fs,
                "f0_hz": float(self.f0_hz),
                "ramp_rate_hzs": float(self.ramp_rate_hzs),
                "onset_frac": float(self.onset_frac),
                "df_max_hz": float(self.df_max_hz),
                "T_s": float(self.T_s),
                "A0": float(self.A0),
                "phi0_rad": float(self.phi0_rad),
                "onset_time_s": onset_time,
                "modifiers": [],
            },
        )
        self.state = state
        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
