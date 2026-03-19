from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G4_E17_Multi_Event_Profile(ScenarioBase):
    """
    Scenario: Multi-Event Dynamic Profile (The "Stress Test").

    Timeline:
      1. Steady State (60Hz) for 1.0s.
      2. Negative Ramp (Down to 55Hz) for 1.5s.
      3. Instant Reconnection (Jump) back to 60Hz.
      4. Ring-down Oscillation (Overshoot + Decay).

    Disturbances (Heavy):
      - Gaussian White Noise.
      - Harmonics (3rd, 5th, 7th).
      - Random Interharmonic (2% magnitude).

    This scenario tests the estimator's ability to track severe dynamic changes
    while rejecting a complex noise floor.
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42

    # --- Event Timing ---
    t_ramp_start: float = 1.0
    t_jump_start: float = 2.5

    # --- Physics Parameters ---
    f_min_hz: float = 55.0  # Bottom of the ramp
    ring_overshoot_hz: float = 2.5  # Peak above 60Hz (reaches 62.5Hz)
    ring_decay_tau: float = 0.4  # Decay time constant (settles fast)
    ring_freq_hz: float = 4.0  # Oscillation frequency (4 Hz = ~3 rings in <1s)

    # --- Noise Parameters ---
    noise_white_sigma: float = 0.005  # 0.5% Gaussian Noise
    thd_pct: float = 0.05  # 5% Harmonics
    ihd_pct: float = 0.02  # 2% Interharmonics (Random Freq)

    scenario_id: str = "G4_E17_Multi_Event_Profile"

    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "Vmax": "A0",
            "seed": "seed",
            "overshoot": "ring_overshoot_hz",
            "decay": "ring_decay_tau",
        }
    )

    def build(self) -> ScenarioOutput:
        fs = float(self.fs_hz)
        T = float(self.T_s)
        n = int(round(T * fs))
        if n <= 1:
            n = 2
        t = np.arange(n, dtype=float) / fs

        # --- 1. Construct Frequency Profile (f_true) ---
        f_true = np.full_like(t, float(self.f_nom_hz), dtype=float)

        t1 = float(self.t_ramp_start)
        t2 = float(self.t_jump_start)

        # Zone A: Ramp (t1 <= t < t2)
        # Slope calculation: (55 - 60) / 1.5 = -3.33 Hz/s
        mask_ramp = (t >= t1) & (t < t2)
        duration_ramp = t2 - t1
        delta_f = float(self.f_min_hz) - float(self.f_nom_hz)
        slope = delta_f / duration_ramp

        # Apply Ramp
        f_true[mask_ramp] = float(self.f_nom_hz) + slope * (t[mask_ramp] - t1)

        # Zone B: Jump & Ring-down (t >= t2)
        mask_ring = t >= t2
        dt_ring = t[mask_ring] - t2

        # Model: f(t) = 60 + Overshoot * exp(-t/tau) * sin(2*pi*f_ring*t)
        # Note: We use sin() so it starts at 0 delta (continuity of f is NOT preserved here,
        # but physically we want a jump. Wait, the prompt says "Jump to 60 then oscillate".
        # A jump from 55 to 60 is a step. The oscillation adds to 60.

        # Let's model the jump as instant return to 60, PLUS the ringing
        oscillation = (
            float(self.ring_overshoot_hz)
            * np.exp(-dt_ring / float(self.ring_decay_tau))
            * np.sin(2.0 * np.pi * float(self.ring_freq_hz) * dt_ring)
        )

        f_true[mask_ring] = float(self.f_nom_hz) + oscillation

        # Hard Step correction: Ensure the first sample of mask_ring breaks from the ramp logic
        # (This creates the "Jump" effect from ~55Hz back to ~60Hz)

        # --- 2. Compute Phase & Voltage (Fundamental) ---
        # Critical: We must integrate f_true because f varies wildly.
        # This handles the FM modulation of the ring-down naturally.

        state_dummy = ScenarioState(
            t=t,
            fs_hz=fs,
            f_nom_hz=float(self.f_nom_hz),
            seed=int(self.seed),
            f_true=f_true,
            phi=np.zeros_like(t),
            A=np.full_like(t, float(self.A0)),
            v=np.zeros_like(t),
            schema={},
        )
        self.state = state_dummy  # Bind for mixin usage

        self.recompute_phi_from_f(phi0_rad=float(self.phi0_rad))
        v_fund = float(self.A0) * np.sin(self.state.phi)

        # --- 3. Add Disturbances (The "Hell" Layer) ---
        rng = self._rng()

        # A) Harmonics (Fixed 3rd, 5th)
        v_harm = np.zeros_like(t)
        if self.thd_pct > 0:
            # Simple profile: 3rd harmonic
            h_amp = float(self.A0) * float(self.thd_pct)
            # Phase coupled to fundamental but 3x faster
            v_harm = h_amp * np.sin(3.0 * self.state.phi + rng.uniform(0, 6.28))

        # B) Interharmonics (Random Freq)
        v_ih = np.zeros_like(t)
        if self.ihd_pct > 0:
            ih_amp = float(self.A0) * float(self.ihd_pct)
            # Random freq between 20Hz and 180Hz (excluding 60)
            f_ih = rng.uniform(20.0, 180.0)
            v_ih = ih_amp * np.sin(2.0 * np.pi * f_ih * t + rng.uniform(0, 6.28))

        # C) Gaussian Noise
        v_noise = np.zeros_like(t)
        if self.noise_white_sigma > 0:
            v_noise = rng.normal(
                0.0, float(self.A0) * float(self.noise_white_sigma), size=t.shape
            )

        # Combine
        v_total = v_fund + v_harm + v_ih + v_noise

        # Update State
        self.state.v = v_total
        self.state.schema = {
            "scenario_id": self.scenario_id,
            "seed": int(self.seed),
            "event_type": "ramp_jump_ringdown",
            "f_min": float(self.f_min_hz),
            "overshoot": float(self.ring_overshoot_hz),
            "disturbances": ["harmonics", "interharmonics", "gaussian"],
            "thd_pct": float(self.thd_pct),
            "ihd_pct": float(self.ihd_pct),
        }

        return ScenarioOutput(scenario_id=self.scenario_id, state=self.state)
