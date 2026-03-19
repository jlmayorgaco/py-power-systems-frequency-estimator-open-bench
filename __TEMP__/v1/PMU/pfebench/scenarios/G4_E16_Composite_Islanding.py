from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from pfebench.scenarios.base import ScenarioBase, ScenarioOutput, ScenarioState


@dataclass
class G4_E16_Composite_Islanding(ScenarioBase):
    """
    Scenario: Composite Islanding Event.

    Simulates the disconnection of a distributed generator from the main grid.
    This is a multi-physics event combining:
      1. Frequency Ramp (ROCOF) due to power mismatch.
      2. Voltage Magnitude Step due to impedance change.
      3. Phase Angle Jump (Vector Shift) at the instant of separation.
      4. Background Gaussian Noise.

    Signal Model:
      t < t_island: Normal grid (f0, A0, phi0)
      t >= t_island:
         f(t) = f0 + rocof * (t - t_island)
         A(t) = A0 * (1 + V_change_pct)
         phi(t) = Integral(f) + phi_jump
    """

    fs_hz: float = 10_000.0
    T_s: float = 5.0
    f_nom_hz: float = 60.0
    A0: float = 1.0
    phi0_rad: float = 0.0
    seed: int = 42

    # --- Event Parameters ---
    island_time_s: float = 2.0

    # 1. Frequency Drift (ROCOF)
    island_rocof_hz_s: float = 2.0  # Hz/s (e.g. Generation > Load)

    # 2. Voltage Step
    island_volt_chg_pct: float = -0.10  # -10% Voltage Drop

    # 3. Phase Jump (Vector Shift)
    island_phase_jump_deg: float = 10.0  # +10 deg shift

    # 4. Background Noise
    noise_level_rel: float = 0.001  # 0.1% Noise

    scenario_id: str = "G4_E16_Composite_Islanding"

    # --- Monte Carlo Adapter Map ---
    tuning_map: Dict[str, str] = field(
        default_factory=lambda: {
            "f": "f_nom_hz",
            "Vmax": "A0",
            "seed": "seed",
            # Composite Tuning
            "t_event": "island_time_s",
            "rocof": "island_rocof_hz_s",  # Frequency Slope
            "v_step": "island_volt_chg_pct",  # Voltage Step
            "phi_jump": "island_phase_jump_deg",  # Vector Shift
        }
    )

    def build(self) -> ScenarioOutput:
        fs = float(self.fs_hz)
        T = float(self.T_s)

        n = int(round(T * fs))
        if n <= 1:
            n = 2
        t = np.arange(n, dtype=float) / fs

        t_ev = float(self.island_time_s)
        mask_event = t >= t_ev

        # 1. Physics: Frequency (Ramp)
        f_true = np.full_like(t, float(self.f_nom_hz), dtype=float)
        rocof = float(self.island_rocof_hz_s)

        # Apply Ramp: f = f0 + R*(t - t_ev)
        dt_ramp = t[mask_event] - t_ev
        f_true[mask_event] += rocof * dt_ramp

        # 2. Physics: Amplitude (Step)
        A = np.full_like(t, float(self.A0), dtype=float)
        v_step = float(self.island_volt_chg_pct)
        A[mask_event] *= 1.0 + v_step

        # 3. Initialize State (Phase will be computed sequentially)
        dummy = np.zeros_like(t)

        state = ScenarioState(
            t=t,
            fs_hz=fs,
            f_nom_hz=float(self.f_nom_hz),
            seed=int(self.seed),
            f_true=f_true,
            phi=dummy,  # TBD
            A=A,
            v=dummy,  # TBD
            schema={
                "scenario_id": self.scenario_id,
                "event_type": "islanding",
                "island_time_s": t_ev,
                "rocof": rocof,
                "v_step_pct": v_step,
                "phase_jump_deg": float(self.island_phase_jump_deg),
                "modifiers": ["noise"],
            },
        )
        self.state = state

        # --- COMPLEX PHYSICS ASSEMBLY ---

        # A. Integrate Frequency to get Continuous Phase (Base)
        # phi_cont = Integral(f_true)
        self.recompute_phi_from_f(phi0_rad=float(self.phi0_rad))

        # B. Apply Vector Shift (Discontinuous Jump) manually
        # phi_final = phi_cont + jump
        p_jump_rad = np.deg2rad(float(self.island_phase_jump_deg))
        self.state.phi[mask_event] += p_jump_rad

        # C. Compute Pure Voltage (A * sin(phi_final))
        self.recompute_v_from_A_phi()

        # D. Add Noise
        rng = self._rng()
        sigma = float(self.A0) * float(self.noise_level_rel)
        noise = self.add_noise_gaussian(shape=t.shape, sigma=sigma, rng=rng)

        self.state.v += noise

        return ScenarioOutput(scenario_id=self.scenario_id, state=state)
