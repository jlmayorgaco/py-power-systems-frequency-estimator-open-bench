# scenarios/ibg_events.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
import numpy as np


# =============================================================================
# Helpers (KEEP: public behavior / numerical conventions)
# =============================================================================
def _generate_phase(f_hz: np.ndarray, fs_hz: float) -> np.ndarray:
    """
    Continuous phase by integrating instantaneous frequency [Hz].
    phi[n] = 2π * Σ_{k<=n} f[k] / fs
    """
    f_hz = np.asarray(f_hz, dtype=float)
    fs_hz = float(fs_hz)
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")
    return 2.0 * np.pi * np.cumsum(f_hz) / fs_hz


def _snr_db_for_noise(
    noise_std: float, signal_rms: float = 1.0 / np.sqrt(2.0)
) -> float:
    """Approx SNR in dB for additive white noise on a unit sine."""
    noise_std = float(noise_std)
    if noise_std <= 0:
        return float("inf")
    return float(20.0 * np.log10(float(signal_rms) / noise_std))


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x) + 1e-18))


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    """C1 smoothstep on [0,1]."""
    x = _clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def _window_smooth(
    t: np.ndarray, t_on: float, t_off: float, edge: float = 0.008
) -> np.ndarray:
    """
    Smooth rectangular window ~1 inside [t_on, t_off], 0 outside.
    edge: smoothing time [s] at each edge.
    """
    t = np.asarray(t, dtype=float)
    e = float(max(1e-6, edge))
    a = _smoothstep01((t - t_on) / e)
    b = _smoothstep01((t_off - t) / e)
    return a * b


def _exp_recover(u: np.ndarray, tau: float) -> np.ndarray:
    """1 - exp(-u/tau) for u>=0, else 0."""
    u = np.asarray(u, dtype=float)
    tau = float(max(1e-6, tau))
    return np.where(u > 0.0, 1.0 - np.exp(-u / tau), 0.0)


# =============================================================================
# OOP core
# =============================================================================
DETUNE_RANGE_HZ = 1e-6  # +/- 1 microHz

# NOTE:
# - We inserted a new scenario "G3_E12_Phase_Jump" (pure phase jump).
# - Composite islanding remains "G3_E12_Composite_Islanding" (mode=GFL/GFM).
# - Multi-event stress test remains "G3_E15_Multi_Event_Profile" (default mode=GFL).
_DETUNE_SET = {
    "G1_E2_Gaussian_Noise_1pct",
    "G1_E3_Gaussian_Noise_5pct",
    "G2_E4_Voltage_Mag_Step_1pct",
    "G2_E5_Voltage_Mag_Step_10pct",
    "G3_E10_AM_Modulation",
    "G3_E12_Phase_Jump",
    "G3_E12_Composite_Islanding",
    "G3_E13_Impulsive_Outliers",
    "G3_E14_Noise_Harmonics",
    "G3_E15_Multi_Event_Profile",  # NEW: detune multi-event too (avoids degenerate cases)
    "G4_E16_Chamorro_Event",
}


@dataclass
class ScenarioContext:
    """Shared buffers/state used by all scenarios."""

    scenario_id: str
    fs: float
    T: float
    seed: int
    t: np.ndarray
    rng: np.random.Generator
    f0_nominal: float

    # Core signals
    f: np.ndarray
    a: np.ndarray
    phase_offset: np.ndarray

    # Disturbances
    v_dist: np.ndarray
    v_noise: np.ndarray

    # knobs/meta
    description: str = ""
    noise_std: float = 0.0
    impulse_prob: float = 0.0
    impulse_scale: float = 0.0
    event_times: Dict[str, float] = field(default_factory=dict)

    # scenario-specific params (e.g., mode select)
    params: Dict[str, Any] = field(default_factory=dict)


class ScenarioBase:
    """Base class for scenarios (DRY/KISS)."""

    scenario_id: str
    description: str

    def apply(self, ctx: ScenarioContext) -> None:
        """Modify ctx.{f,a,phase_offset, noise_std,...} and optionally ctx.v_dist/ctx.v_noise knobs."""
        raise NotImplementedError

    def inject_structured(self, ctx: ScenarioContext, phase: np.ndarray) -> None:
        """Optional: add structured disturbances after phase construction."""
        return

    def inject_noise(self, ctx: ScenarioContext) -> None:
        """Optional: add noise/outliers (final stage)."""
        # Gaussian noise
        if ctx.noise_std > 0:
            ctx.v_noise = ctx.v_noise + ctx.rng.normal(
                0.0, float(ctx.noise_std), size=ctx.v_noise.size
            )

        # Impulsive outliers
        if ctx.impulse_prob > 0 and ctx.impulse_scale > 0:
            n = ctx.v_noise.size
            mask = ctx.rng.random(n) < float(ctx.impulse_prob)
            k = int(np.sum(mask))
            if k > 0:
                spikes = ctx.rng.laplace(
                    loc=0.0, scale=float(ctx.impulse_scale), size=k
                )
                ctx.v_noise = ctx.v_noise.copy()
                ctx.v_noise[mask] += spikes
            # keep bounded but still heavy-tailed
            ctx.v_noise = np.clip(ctx.v_noise, -1.25, +1.25)


# =============================================================================
# Scenario implementations
# =============================================================================
class G1_E1_Pure_60Hz(ScenarioBase):
    scenario_id = "G1_E1_Pure_60Hz"
    description = "Ideal 60Hz reference"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description


class G1_E2_Gaussian_Noise_1pct(ScenarioBase):
    scenario_id = "G1_E2_Gaussian_Noise_1pct"
    description = "60Hz with 1% Gaussian Noise"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.noise_std = 0.01


class G1_E3_Gaussian_Noise_5pct(ScenarioBase):
    scenario_id = "G1_E3_Gaussian_Noise_5pct"
    description = "60Hz with 5% Gaussian Noise"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.noise_std = 0.05


class G2_E4_Voltage_Mag_Step_1pct(ScenarioBase):
    scenario_id = "G2_E4_Voltage_Mag_Step_1pct"
    description = "Voltage Magnitude Step (-1%)"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.a = np.where(ctx.t < 2.0, 1.0, 0.99).astype(float)
        ctx.event_times["mag_step"] = 2.0


class G2_E5_Voltage_Mag_Step_10pct(ScenarioBase):
    scenario_id = "G2_E5_Voltage_Mag_Step_10pct"
    description = "Voltage Magnitude Step (-10%)"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.a = np.where(ctx.t < 2.0, 1.0, 0.90).astype(float)
        ctx.event_times["mag_step"] = 2.0


class G2_E6_Freq_Step_60_to_59p5(ScenarioBase):
    scenario_id = "G2_E6_Freq_Step_60_to_59p5"
    description = "Frequency Step (-0.5 Hz)"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.f = np.where(ctx.t < 2.0, ctx.f0_nominal, ctx.f0_nominal - 0.5).astype(
            float
        )
        ctx.event_times["freq_step"] = 2.0


class G2_E7_Freq_Step_60_to_55(ScenarioBase):
    scenario_id = "G2_E7_Freq_Step_60_to_55"
    description = "Extreme Frequency Step (-5.0 Hz)"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.f = np.where(ctx.t < 2.0, ctx.f0_nominal, ctx.f0_nominal - 5.0).astype(
            float
        )
        ctx.event_times["freq_step"] = 2.0


class G2_E8_Fast_Ramp_plus5Hzs(ScenarioBase):
    scenario_id = "G2_E8_Fast_Ramp_plus5Hzs"
    description = "High RoCoF Positive Ramp (+5 Hz/s)"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.f = (ctx.f0_nominal + 5.0 * np.clip(ctx.t - 1.0, 0.0, None)).astype(float)
        ctx.event_times["ramp_start"] = 1.0


class G2_E9_Slow_Ramp_minus0p5Hzs(ScenarioBase):
    scenario_id = "G2_E9_Slow_Ramp_minus0p5Hzs"
    description = "Standard Low RoCoF Negative Ramp (-0.5 Hz/s)"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.f = (ctx.f0_nominal - 0.5 * np.clip(ctx.t - 1.0, 0.0, None)).astype(float)
        ctx.event_times["ramp_start"] = 1.0


class G3_E10_AM_Modulation(ScenarioBase):
    scenario_id = "G3_E10_AM_Modulation"
    description = "Amplitude Modulation (2Hz, 10% Depth)"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.a = (1.0 + 0.1 * np.sin(2.0 * np.pi * 2.0 * ctx.t)).astype(float)


class G3_E11_FM_Modulation(ScenarioBase):
    scenario_id = "G3_E11_FM_Modulation"
    description = "Frequency Modulation (1.5Hz, 1Hz Deviation)"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.f = (ctx.f0_nominal + 1.0 * np.sin(2.0 * np.pi * 1.5 * ctx.t)).astype(float)


# -----------------------------------------------------------------------------
# NEW: Pure Phase Jump (clean, no harmonics, no ringdown)
# -----------------------------------------------------------------------------
class G3_E12_Phase_Jump(ScenarioBase):
    scenario_id = "G3_E12_Phase_Jump"
    description = "Pure Phase Jump (60° at 1.0s)"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.phase_offset = np.where(ctx.t < 1.0, 0.0, np.deg2rad(60.0)).astype(float)
        ctx.event_times["phase_jump"] = 1.0


# -----------------------------------------------------------------------------
# UPDATED: Composite Islanding (IBR-realistic) with MODE input: "GFL" or "GFM"
# - SAME ID (pipeline stable)
# -----------------------------------------------------------------------------
class G3_E12_Composite_Islanding(ScenarioBase):
    scenario_id = "G3_E12_Composite_Islanding"
    description = "Composite Islanding (IBR): Mode-selectable GFL/GFM"

    def apply(self, ctx: ScenarioContext) -> None:
        mode = str(ctx.params.get("mode", "GFM")).upper().strip()
        if mode not in {"GFL", "GFM"}:
            raise ValueError(
                f"{self.scenario_id}: mode must be 'GFL' or 'GFM', got {mode!r}"
            )
        ctx.params["mode"] = mode

        t0 = float(ctx.params.get("t0", 1.0))
        ctx.event_times["islanding"] = t0
        ctx.event_times["phase_jump"] = t0

        dphi_deg = float(ctx.params.get("dphi_deg", 60.0))
        ctx.phase_offset = np.where(ctx.t < t0, 0.0, np.deg2rad(dphi_deg)).astype(float)

        sag_depth = float(ctx.params.get("sag_depth", 0.18))
        sag_dur = float(ctx.params.get("sag_dur", 0.12))
        sag_edge = float(ctx.params.get("sag_edge", 0.010))
        rec_tau = float(ctx.params.get("rec_tau", 0.60))
        a_min = float(max(0.40, 1.0 - sag_depth))

        w_sag = _window_smooth(ctx.t, t0, t0 + sag_dur, edge=sag_edge)
        u = np.clip(ctx.t - (t0 + sag_dur), 0.0, None)
        rec = _exp_recover(u, tau=rec_tau)

        a_recovered = float(ctx.params.get("a_recovered", 0.95))
        ctx.a = (1.0 - w_sag * (1.0 - a_min)).astype(float)
        ctx.a = np.where(
            ctx.t <= (t0 + sag_dur), ctx.a, (a_min + (a_recovered - a_min) * rec)
        ).astype(float)

        f0 = float(ctx.f0_nominal)
        t = ctx.t
        u0 = np.clip(t - t0, 0.0, None)

        if mode == "GFM":
            rocof = float(ctx.params.get("rocof_hz_s", -4.0))
            rocof_dur = float(ctx.params.get("rocof_dur", 0.15))
            _ = rocof * np.clip(u0, 0.0, rocof_dur)

            f_osc = float(ctx.params.get("ring_f_hz", 1.2))
            zeta = float(ctx.params.get("ring_zeta", 0.15))
            A_df = float(ctx.params.get("ring_df_hz", 0.25))

            u1 = np.clip(u0 - rocof_dur, 0.0, None)
            ring = (
                A_df
                * np.exp(-zeta * 2.0 * np.pi * f_osc * u1)
                * np.sin(2.0 * np.pi * f_osc * u1)
            )

            f_settle = float(ctx.params.get("f_settle_hz", f0 - 0.20))
            df_end = rocof * rocof_dur
            base = (f0 + df_end) + (f_settle - (f0 + df_end)) * _exp_recover(
                u1, tau=float(ctx.params.get("settle_tau", 0.60))
            )

            ctx.f = np.where(t < t0, f0, base + ring).astype(float)
            ctx.description = f"Composite Islanding (GFM): sag+rocof+nadir+ringdown, Δφ={dphi_deg:.0f}° @ {t0:.2f}s"

        else:
            df_peak = float(ctx.params.get("pll_df_peak_hz", +1.00))
            df_nadir = float(ctx.params.get("pll_df_nadir_hz", -0.60))
            t_up = float(ctx.params.get("pll_t_up", 0.08))
            t_down = float(ctx.params.get("pll_t_down", 0.22))
            t_rec = float(ctx.params.get("pll_t_rec", 0.70))

            uu = u0
            s1 = _smoothstep01(uu / max(1e-6, t_up))
            s2 = _smoothstep01((uu - t_up) / max(1e-6, (t_down - t_up)))
            s3 = _smoothstep01((uu - t_down) / max(1e-6, (t_rec - t_down)))

            df_transient = (
                (df_peak * s1) * (uu <= t_up)
                + (df_peak + (df_nadir - df_peak) * s2) * ((uu > t_up) & (uu <= t_down))
                + (df_nadir * (1.0 - s3)) * ((uu > t_down) & (uu <= t_rec))
                + 0.0 * (uu > t_rec)
            )

            f_hunt = float(ctx.params.get("pll_hunt_hz", 6.0))
            zeta = float(ctx.params.get("pll_zeta", 0.20))
            A_df = float(ctx.params.get("pll_hunt_df_hz", 0.12))
            hunt = (
                A_df
                * np.exp(-zeta * 2.0 * np.pi * f_hunt * uu)
                * np.sin(2.0 * np.pi * f_hunt * uu)
            )

            f_settle = float(ctx.params.get("f_settle_hz", f0 - 0.05))
            base = f_settle + (f0 - f_settle) * np.exp(
                -uu / float(ctx.params.get("settle_tau", 0.45))
            )

            ctx.f = np.where(t < t0, f0, base + df_transient + hunt).astype(float)
            ctx.description = f"Composite Islanding (GFL): sag+PLL hunting, Δφ={dphi_deg:.0f}° @ {t0:.2f}s"

        ctx.event_times["current_limit_on"] = t0
        ctx.event_times["current_limit_off"] = t0 + sag_dur
        ctx.noise_std = float(ctx.params.get("noise_std", 0.002))

    def inject_structured(self, ctx: ScenarioContext, phase: np.ndarray) -> None:
        t = ctx.t
        t0 = float(ctx.event_times.get("islanding", 1.0))
        t1 = float(ctx.event_times.get("current_limit_off", t0 + 0.12))
        w_lim = _window_smooth(t, t0, t1, edge=0.010)

        u = np.clip(t - t0, 0.0, None)
        f_ring_v = float(ctx.params.get("v_ring_hz", 3.0))
        zeta_v = float(ctx.params.get("v_ring_zeta", 0.25))
        A_v = float(ctx.params.get("v_ring_amp", 0.03))
        v_ring = (
            A_v
            * np.exp(-zeta_v * 2.0 * np.pi * f_ring_v * u)
            * np.sin(2.0 * np.pi * f_ring_v * u)
        )
        ctx.v_dist += np.where(t < t0, 0.0, v_ring).astype(float)

        h3 = float(ctx.params.get("h3", 0.025))
        h5 = float(ctx.params.get("h5", 0.015))
        h7 = float(ctx.params.get("h7", 0.010))
        for k, h_amp in ((3, h3), (5, h5), (7, h7)):
            ctx.v_dist += (w_lim * h_amp * np.sin(float(k) * phase)).astype(float)

        ih1_f = float(ctx.params.get("ih1_f_hz", 72.0))
        ih1_a = float(ctx.params.get("ih1_a", 0.010))
        ih2_f = float(ctx.params.get("ih2_f_hz", 90.0))
        ih2_a = float(ctx.params.get("ih2_a", 0.007))
        ctx.v_dist += (w_lim * ih1_a * np.sin(2.0 * np.pi * ih1_f * t)).astype(float)
        ctx.v_dist += (w_lim * ih2_a * np.sin(2.0 * np.pi * ih2_f * t)).astype(float)

        beat_hz = float(ctx.params.get("negseq_beat_hz", 2.0))
        beat_a = float(ctx.params.get("negseq_beat_a", 0.006))
        ctx.v_dist += (
            w_lim * beat_a * np.sin(2.0 * np.pi * beat_hz * t) * np.sin(phase)
        ).astype(float)


class G3_E13_Impulsive_Outliers(ScenarioBase):
    scenario_id = "G3_E13_Impulsive_Outliers"

    def apply(self, ctx: ScenarioContext) -> None:
        mode = "frequent_small"  # keep behavior identical to your current code

        if mode == "rare_large":
            ctx.description = "Nominal 60Hz + 0.5% Gaussian Noise + Rare Large Impulsive Outliers (heavy-tailed)"
            ctx.noise_std = 0.015
            ctx.impulse_prob = 0.015
            ctx.impulse_scale = 0.05

        elif mode == "frequent_small":
            ctx.description = "Nominal 60Hz + 0.5% Gaussian Noise + Frequent Small Impulsive Outliers (heavy-tailed)"
            ctx.noise_std = 0.015
            ctx.impulse_prob = 0.05
            ctx.impulse_scale = 0.05
        else:
            raise ValueError("Invalid mode for impulsive outliers.")


class G3_E14_Noise_Harmonics(ScenarioBase):
    scenario_id = "G3_E14_Noise_Harmonics"
    description = "Harmonics (3,5,7) + 1% Gaussian Noise"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.noise_std = 0.01

    def inject_structured(self, ctx: ScenarioContext, phase: np.ndarray) -> None:
        for k, h_amp in zip((3, 5, 7), (0.03, 0.02, 0.01)):
            ctx.v_dist += float(h_amp) * np.sin(float(k) * phase)


class G3_E15_Multi_Event_Profile(ScenarioBase):
    """
    Q1-grade multi-event stress test for IBR frequency estimators.
    Default mode: GFL (PLL hunting signature).
    """

    scenario_id = "G3_E15_Multi_Event_Profile"
    description = "Multi-event IBR Stress Test (default: GFL)"

    def apply(self, ctx: ScenarioContext) -> None:
        # MODE (default = GFL)
        mode = str(ctx.params.get("mode", "GFL")).upper().strip()
        if mode not in {"GFL", "GFM"}:
            raise ValueError(
                f"{self.scenario_id}: mode must be 'GFL' or 'GFM', got {mode!r}"
            )
        ctx.params["mode"] = mode

        # Master event timing
        t0 = float(ctx.params.get("t0", 1.00))
        ctx.event_times["multi_event_t0"] = t0
        ctx.event_times["phase_jump"] = t0
        ctx.event_times["islanding"] = t0

        # (1) Explicit phase jump
        dphi_deg = float(ctx.params.get("dphi_deg", 40.0))
        ctx.phase_offset = np.where(ctx.t < t0, 0.0, np.deg2rad(dphi_deg)).astype(float)

        # (2) Voltage sag + recovery
        sag_depth = float(ctx.params.get("sag_depth", 0.22))
        sag_dur = float(ctx.params.get("sag_dur", 0.16))
        sag_edge = float(ctx.params.get("sag_edge", 0.010))
        rec_tau = float(ctx.params.get("rec_tau", 0.55))
        a_rec = float(ctx.params.get("a_recovered", 0.96))

        a_min = float(max(0.40, 1.0 - sag_depth))
        w_sag = _window_smooth(ctx.t, t0, t0 + sag_dur, edge=sag_edge)
        u_rec = np.clip(ctx.t - (t0 + sag_dur), 0.0, None)
        rec = _exp_recover(u_rec, tau=rec_tau)

        a_sag = 1.0 - w_sag * (1.0 - a_min)
        ctx.a = np.where(
            ctx.t <= (t0 + sag_dur), a_sag, (a_min + (a_rec - a_min) * rec)
        ).astype(float)

        ctx.event_times["current_limit_on"] = t0
        ctx.event_times["current_limit_off"] = t0 + sag_dur

        # (3) Frequency truth f(t)
        f0 = float(ctx.f0_nominal)
        t = ctx.t
        u = np.clip(t - t0, 0.0, None)

        if mode == "GFM":
            rocof = float(ctx.params.get("rocof_hz_s", -3.5))
            rocof_dur = float(ctx.params.get("rocof_dur", 0.18))
            df_end = rocof * rocof_dur

            f_settle = float(ctx.params.get("f_settle_hz", f0 - 0.18))
            settle_tau = float(ctx.params.get("settle_tau", 0.60))
            u1 = np.clip(u - rocof_dur, 0.0, None)
            base = (f0 + df_end) + (f_settle - (f0 + df_end)) * _exp_recover(
                u1, tau=settle_tau
            )

            ring_f = float(ctx.params.get("ring_f_hz", 1.4))
            ring_z = float(ctx.params.get("ring_zeta", 0.18))
            ring_A = float(ctx.params.get("ring_df_hz", 0.22))
            ring = (
                ring_A
                * np.exp(-ring_z * 2.0 * np.pi * ring_f * u1)
                * np.sin(2.0 * np.pi * ring_f * u1)
            )

            ctx.f = np.where(t < t0, f0, base + ring).astype(float)
            ctx.description = (
                f"Multi-event IBR Stress (GFM): sag+rocof+nadir+ringdown + harmonics/outliers, "
                f"Δφ={dphi_deg:.0f}° @ {t0:.2f}s"
            )

        else:
            df_peak = float(ctx.params.get("pll_df_peak_hz", +0.90))
            df_nadir = float(ctx.params.get("pll_df_nadir_hz", -0.80))
            t_up = float(ctx.params.get("pll_t_up", 0.07))
            t_down = float(ctx.params.get("pll_t_down", 0.24))
            t_rec = float(ctx.params.get("pll_t_rec", 0.85))

            s1 = _smoothstep01(u / max(1e-6, t_up))
            s2 = _smoothstep01((u - t_up) / max(1e-6, (t_down - t_up)))
            s3 = _smoothstep01((u - t_down) / max(1e-6, (t_rec - t_down)))

            df_transient = (
                (df_peak * s1) * (u <= t_up)
                + (df_peak + (df_nadir - df_peak) * s2) * ((u > t_up) & (u <= t_down))
                + (df_nadir * (1.0 - s3)) * ((u > t_down) & (u <= t_rec))
                + 0.0 * (u > t_rec)
            )

            hunt_f = float(ctx.params.get("pll_hunt_hz", 6.0))
            hunt_z = float(ctx.params.get("pll_zeta", 0.22))
            hunt_A = float(ctx.params.get("pll_hunt_df_hz", 0.14))
            hunt = (
                hunt_A
                * np.exp(-hunt_z * 2.0 * np.pi * hunt_f * u)
                * np.sin(2.0 * np.pi * hunt_f * u)
            )

            f_settle = float(ctx.params.get("f_settle_hz", f0 - 0.06))
            settle_tau = float(ctx.params.get("settle_tau", 0.55))
            base = f_settle + (f0 - f_settle) * np.exp(-u / settle_tau)

            ctx.f = np.where(t < t0, f0, base + df_transient + hunt).astype(float)
            ctx.description = (
                f"Multi-event IBR Stress (GFL): phase jump + sag + PLL hunting + harmonics/outliers, "
                f"Δφ={dphi_deg:.0f}° @ {t0:.2f}s"
            )

        # (4) Base noise + (5) heavy-tail impulsive window
        ctx.noise_std = float(ctx.params.get("noise_std", 0.006))

        imp_on = float(ctx.params.get("imp_on", t0 + 0.90))
        imp_off = float(ctx.params.get("imp_off", t0 + 1.20))
        ctx.event_times["bad_data_on"] = imp_on
        ctx.event_times["bad_data_off"] = imp_off

        # store (for reproducibility / meta)
        ctx.params["imp_on"] = imp_on
        ctx.params["imp_off"] = imp_off

        # BULLETPROOF: precompute bad-data mask here (do not depend on inject_structured)
        ctx.params["bad_data_mask"] = _window_smooth(ctx.t, imp_on, imp_off, edge=0.010)

        ctx.impulse_prob = float(ctx.params.get("impulse_prob", 0.03))
        ctx.impulse_scale = float(ctx.params.get("impulse_scale", 0.05))

    def inject_structured(self, ctx: ScenarioContext, phase: np.ndarray) -> None:
        t = ctx.t
        t0 = float(ctx.event_times.get("multi_event_t0", 1.0))
        t1 = float(ctx.event_times.get("current_limit_off", t0 + 0.16))

        # Voltage ringdown (small) after islanding
        u = np.clip(t - t0, 0.0, None)
        v_ring_f = float(ctx.params.get("v_ring_hz", 3.2))
        v_ring_z = float(ctx.params.get("v_ring_zeta", 0.25))
        v_ring_A = float(ctx.params.get("v_ring_amp", 0.03))
        v_ring = (
            v_ring_A
            * np.exp(-v_ring_z * 2.0 * np.pi * v_ring_f * u)
            * np.sin(2.0 * np.pi * v_ring_f * u)
        )
        ctx.v_dist += np.where(t < t0, 0.0, v_ring).astype(float)

        # Harmonics + interharmonics only during current limiting window
        w_lim = _window_smooth(
            t, t0, t1, edge=float(ctx.params.get("harm_edge", 0.010))
        )

        h3 = float(ctx.params.get("h3", 0.030))
        h5 = float(ctx.params.get("h5", 0.018))
        h7 = float(ctx.params.get("h7", 0.012))
        for k, h_amp in ((3, h3), (5, h5), (7, h7)):
            ctx.v_dist += (w_lim * h_amp * np.sin(float(k) * phase)).astype(float)

        ih1_f = float(ctx.params.get("ih1_f_hz", 72.0))
        ih1_a = float(ctx.params.get("ih1_a", 0.010))
        ih2_f = float(ctx.params.get("ih2_f_hz", 90.0))
        ih2_a = float(ctx.params.get("ih2_a", 0.007))
        ctx.v_dist += (w_lim * ih1_a * np.sin(2.0 * np.pi * ih1_f * t)).astype(float)
        ctx.v_dist += (w_lim * ih2_a * np.sin(2.0 * np.pi * ih2_f * t)).astype(float)

        # Gentle AM flicker after event (breaks stationarity)
        am_on = float(ctx.params.get("am_on", t0 + 0.25))
        am_off = float(ctx.params.get("am_off", t0 + 2.50))
        w_am = _window_smooth(
            t, am_on, am_off, edge=float(ctx.params.get("am_edge", 0.02))
        )
        am_f = float(ctx.params.get("am_f_hz", 2.0))
        am_depth = float(ctx.params.get("am_depth", 0.05))
        ctx.v_dist += (
            w_am * am_depth * np.sin(2.0 * np.pi * am_f * t) * np.sin(phase)
        ).astype(float)

        # Optional: keep mask aligned with any updated imp_on/off (if user overrides later)
        # (Still safe because apply() already created it.)
        imp_on = float(ctx.params.get("imp_on", t0 + 0.90))
        imp_off = float(ctx.params.get("imp_off", t0 + 1.20))
        ctx.params["bad_data_mask"] = _window_smooth(t, imp_on, imp_off, edge=0.010)

    def inject_noise(self, ctx: ScenarioContext) -> None:
        # 1) base gaussian
        if ctx.noise_std > 0:
            ctx.v_noise = ctx.v_noise + ctx.rng.normal(
                0.0, float(ctx.noise_std), size=ctx.v_noise.size
            )

        # 2) gated impulsive outliers (heavy-tail) ONLY inside bad-data window
        if ctx.impulse_prob > 0 and ctx.impulse_scale > 0:
            gate = np.asarray(
                ctx.params.get("bad_data_mask", np.ones_like(ctx.v_noise)), dtype=float
            )

            n = ctx.v_noise.size
            u = ctx.rng.random(n)
            mask = (u < float(ctx.impulse_prob)) & (gate > 0.5)

            k = int(np.sum(mask))
            if k > 0:
                spikes = ctx.rng.laplace(
                    loc=0.0, scale=float(ctx.impulse_scale), size=k
                )
                ctx.v_noise = ctx.v_noise.copy()
                ctx.v_noise[mask] += spikes

            ctx.v_noise = np.clip(ctx.v_noise, -1.25, +1.25)


class G4_E16_Chamorro_Event(ScenarioBase):
    scenario_id = "G4_E16_Chamorro_Event"
    description = "Real-world Event: Chamorro CSV (Mocked as constant 60Hz-ish)"

    def apply(self, ctx: ScenarioContext) -> None:
        ctx.description = self.description
        ctx.f = np.full_like(ctx.t, ctx.f0_nominal, dtype=float)


# Registry: scenario_id -> instance (singletons are fine; they hold no state)
_SCENARIOS: Dict[str, ScenarioBase] = {
    cls.scenario_id: cls()
    for cls in [
        G1_E1_Pure_60Hz,
        G1_E2_Gaussian_Noise_1pct,
        G1_E3_Gaussian_Noise_5pct,
        G2_E4_Voltage_Mag_Step_1pct,
        G2_E5_Voltage_Mag_Step_10pct,
        G2_E6_Freq_Step_60_to_59p5,
        G2_E7_Freq_Step_60_to_55,
        G2_E8_Fast_Ramp_plus5Hzs,
        G2_E9_Slow_Ramp_minus0p5Hzs,
        G3_E10_AM_Modulation,
        G3_E11_FM_Modulation,
        G3_E12_Phase_Jump,
        G3_E12_Composite_Islanding,  # mode-selectable via scenario_params
        G3_E13_Impulsive_Outliers,
        G3_E14_Noise_Harmonics,
        G3_E15_Multi_Event_Profile,
        G4_E16_Chamorro_Event,
    ]
}


def _choose_f0(scenario_id: str, rng: np.random.Generator) -> float:
    if scenario_id == "G1_E1_Pure_60Hz":
        return 60.0
    if scenario_id in _DETUNE_SET:
        return 60.0 + float(rng.uniform(-DETUNE_RANGE_HZ, DETUNE_RANGE_HZ))
    return 60.0


# =============================================================================
# Public API (KEEP CONTRACTS)
# =============================================================================
def get_test_signal(
    scenario_id: str,
    fs: float = 2400.0,
    T: float = 5.0,
    seed: int = 123,
    # NEW (backwards-compatible): optional per-scenario params, e.g.
    #   scenario_params={"mode":"GFL","t0":1.0,"dphi_deg":40}
    scenario_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Public factory (contract preserved):
    returns (t, v, f_true, meta)

    NEW (optional): scenario_params dict to select GFL/GFM for composite islanding.
    """
    fs = float(fs)
    T = float(T)
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")

    t = np.arange(0.0, T, 1.0 / fs, dtype=float)
    n = int(t.size)
    if n < 4:
        raise ValueError("T and fs produce too few samples.")

    rng = np.random.default_rng(int(seed))

    if scenario_id not in _SCENARIOS:
        raise ValueError(f"Scenario {scenario_id} not defined.")

    f0_nominal = _choose_f0(scenario_id, rng)

    # init buffers
    f = np.full_like(t, f0_nominal, dtype=float)
    a = np.ones_like(t, dtype=float)
    phase_offset = np.zeros_like(t, dtype=float)
    v_dist = np.zeros_like(t, dtype=float)
    v_noise = np.zeros_like(t, dtype=float)

    ctx = ScenarioContext(
        scenario_id=scenario_id,
        fs=fs,
        T=T,
        seed=int(seed),
        t=t,
        rng=rng,
        f0_nominal=float(f0_nominal),
        f=f,
        a=a,
        phase_offset=phase_offset,
        v_dist=v_dist,
        v_noise=v_noise,
        params=dict(scenario_params or {}),
    )

    scen = _SCENARIOS[scenario_id]

    # 1) define dynamics + knobs
    scen.apply(ctx)

    # 2) build phase + clean fundamental
    base_phase = _generate_phase(ctx.f, fs)
    phase = base_phase + ctx.phase_offset
    v_clean = ctx.a * np.sin(phase)

    # 3) structured disturbances (post-phase)
    scen.inject_structured(ctx, phase)

    # 4) noise/outliers last
    scen.inject_noise(ctx)

    # final waveform
    v = v_clean + ctx.v_dist + ctx.v_noise

    # SNR meta
    snr_db: float = float("inf")
    if ctx.noise_std > 0:
        sig_rms = _rms(v_clean + ctx.v_dist)
        snr_db = _snr_db_for_noise(noise_std=ctx.noise_std, signal_rms=sig_rms)

    meta: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "description": ctx.description,
        "fs": float(fs),
        "T": float(T),
        "seed": int(seed),
        "f0": float(ctx.f0_nominal),
        "f_init": float(ctx.f[0]),
        "detune_range_hz": float(DETUNE_RANGE_HZ),
    }

    if ctx.params:
        meta["scenario_params"] = dict(ctx.params)

    if ctx.event_times:
        meta["event_times"] = dict(ctx.event_times)

    if np.isfinite(snr_db):
        meta["snr_db"] = float(snr_db)

    if ctx.impulse_prob > 0:
        meta["impulsive_prob"] = float(ctx.impulse_prob)
    if ctx.impulse_scale > 0:
        meta["impulsive_scale"] = float(ctx.impulse_scale)

    return t.astype(float), v.astype(float), ctx.f.astype(float), meta


def get_all_scenario_names() -> List[str]:
    """List of scenario IDs to iterate in the runner (contract preserved)."""
    return [
        "G1_E1_Pure_60Hz",
        "G1_E2_Gaussian_Noise_1pct",
        "G1_E3_Gaussian_Noise_5pct",
        "G2_E4_Voltage_Mag_Step_1pct",
        "G2_E5_Voltage_Mag_Step_10pct",
        "G2_E6_Freq_Step_60_to_59p5",
        "G2_E7_Freq_Step_60_to_55",
        "G2_E8_Fast_Ramp_plus5Hzs",
        "G2_E9_Slow_Ramp_minus0p5Hzs",
        "G3_E10_AM_Modulation",
        "G3_E11_FM_Modulation",
        "G3_E12_Phase_Jump",
        "G3_E12_Composite_Islanding",  # mode-selectable via scenario_params
        "G3_E13_Impulsive_Outliers",
        "G3_E14_Noise_Harmonics",
        "G3_E15_Multi_Event_Profile",
        "G4_E16_Chamorro_Event",
    ]
