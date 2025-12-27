# scenarios/ibg_events.py
from __future__ import annotations

from typing import Tuple, Dict, Any, List
import numpy as np


# -----------------------------
# Helpers
# -----------------------------
def _generate_phase(f_hz: np.ndarray, fs_hz: float) -> np.ndarray:
    """
    Continuous phase by integrating instantaneous frequency [Hz].

    phi[n] = 2π * Σ_{k<=n} f[k] / fs
    """
    f_hz = np.asarray(f_hz, dtype=float)
    fs_hz = float(fs_hz)
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")
    # Discrete-time integral: phi[n] = phi[n-1] + 2π f[n]/fs
    return 2.0 * np.pi * np.cumsum(f_hz) / fs_hz


def _snr_db_for_noise(noise_std: float, signal_rms: float = 1.0 / np.sqrt(2.0)) -> float:
    """
    Approx SNR in dB for additive white noise on a unit sine.
    Default RMS of sin() with amplitude=1 is 1/sqrt(2).
    """
    noise_std = float(noise_std)
    if noise_std <= 0:
        return float("inf")
    return float(20.0 * np.log10(float(signal_rms) / noise_std))


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x) + 1e-18))


# -----------------------------
# Main factory
# -----------------------------
def get_test_signal(
    scenario_id: str,
    fs: float = 2400.0,
    T: float = 5.0,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Master scenario factory for Journal-grade benchmark.
    Naming: G{Group}_E{Event}_{Name}

    Key detail (CRITICAL for DFT-based methods):
    - If nominal f is exactly 60.0 Hz AND analysis windows span an integer number of cycles,
      DFT methods can become unrealistically perfect (RMSE ~ 0).
    - We apply a microscopic, seed-reproducible detune to constant-60Hz-ish scenarios,
      while keeping the ideal reference exactly at 60.0 Hz for sanity checks.
    """
    fs = float(fs)
    T = float(T)
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")

    # time vector
    t = np.arange(0.0, T, 1.0 / fs, dtype=float)
    n = int(t.size)
    if n < 4:
        raise ValueError("T and fs produce too few samples.")

    rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------
    # Seed-reproducible microscopic detune (NOT 0.02 Hz!)
    # ------------------------------------------------------------
    DETUNE_RANGE_HZ = 1e-6  # +/- 1 microHz

    detune_set = {
        "G1_E2_Gaussian_Noise_1pct",
        "G1_E3_Gaussian_Noise_5pct",
        "G2_E4_Voltage_Mag_Step_1pct",
        "G2_E5_Voltage_Mag_Step_10pct",
        "G3_E10_AM_Modulation",
        "G3_E12_Composite_Islanding",
        "G3_E13_Impulsive_Outliers",
        "G3_E14_Noise_Harmonics",
        "G4_E16_Chamorro_Event",  # mocked -> avoid "0 error forever"
    }

    def _nominal_f0() -> float:
        # keep the truly ideal scenario exactly 60.0 Hz
        if scenario_id == "G1_E1_Pure_60Hz":
            return 60.0
        # detune only constant-ish ones
        if scenario_id in detune_set:
            return 60.0 + float(rng.uniform(-DETUNE_RANGE_HZ, DETUNE_RANGE_HZ))
        return 60.0

    f0_nominal = _nominal_f0()

    # Defaults
    f = np.full_like(t, f0_nominal, dtype=float)     # instantaneous frequency [Hz]
    a = np.ones_like(t, dtype=float)                 # amplitude envelope
    phase_offset = np.zeros_like(t, dtype=float)     # additive phase offset [rad]
    v_noise = np.zeros_like(t, dtype=float)          # additive broadband noise
    v_dist = np.zeros_like(t, dtype=float)           # additive structured disturbance
    description = ""
    snr_db: float = float("inf")

    # Keep some explicit knobs for meta
    noise_std: float = 0.0
    impulse_prob: float = 0.0
    impulse_scale: float = 0.0

    # ============================================================
    # GROUP 1: BASELINES
    # ============================================================
    if scenario_id == "G1_E1_Pure_60Hz":
        description = "Ideal 60Hz reference"

    elif scenario_id == "G1_E2_Gaussian_Noise_1pct":
        description = "60Hz with 1% Gaussian Noise"
        noise_std = 0.01
        v_noise = rng.normal(0.0, noise_std, size=n)

    elif scenario_id == "G1_E3_Gaussian_Noise_5pct":
        description = "60Hz with 5% Gaussian Noise"
        noise_std = 0.05
        v_noise = rng.normal(0.0, noise_std, size=n)

    # ============================================================
    # GROUP 2: STANDARD EVENT SENSITIVITY
    # ============================================================
    elif scenario_id == "G2_E4_Voltage_Mag_Step_1pct":
        description = "Voltage Magnitude Step (-1%)"
        a = np.where(t < 2.0, 1.0, 0.99).astype(float)

    elif scenario_id == "G2_E5_Voltage_Mag_Step_10pct":
        description = "Voltage Magnitude Step (-10%)"
        a = np.where(t < 2.0, 1.0, 0.90).astype(float)

    elif scenario_id == "G2_E6_Freq_Step_60_to_59p5":
        description = "Frequency Step (-0.5 Hz)"
        f = np.where(t < 2.0, f0_nominal, f0_nominal - 0.5).astype(float)

    elif scenario_id == "G2_E7_Freq_Step_60_to_55":
        description = "Extreme Frequency Step (-5.0 Hz)"
        f = np.where(t < 2.0, f0_nominal, f0_nominal - 5.0).astype(float)

    elif scenario_id == "G2_E8_Fast_Ramp_plus5Hzs":
        description = "High RoCoF Positive Ramp (+5 Hz/s)"
        # ramp starts at t=1.0
        f = (f0_nominal + 5.0 * np.clip(t - 1.0, 0.0, None)).astype(float)

    elif scenario_id == "G2_E9_Slow_Ramp_minus0p5Hzs":
        description = "Standard Low RoCoF Negative Ramp (-0.5 Hz/s)"
        f = (f0_nominal - 0.5 * np.clip(t - 1.0, 0.0, None)).astype(float)

    # ============================================================
    # GROUP 3: ROBUSTNESS & MODULATIONS
    # ============================================================
    elif scenario_id == "G3_E10_AM_Modulation":
        description = "Amplitude Modulation (2Hz, 10% Depth)"
        a = (1.0 + 0.1 * np.sin(2.0 * np.pi * 2.0 * t)).astype(float)

    elif scenario_id == "G3_E11_FM_Modulation":
        description = "Frequency Modulation (1.5Hz, 1Hz Deviation)"
        f = (f0_nominal + 1.0 * np.sin(2.0 * np.pi * 1.5 * t)).astype(float)

    elif scenario_id == "G3_E12_Composite_Islanding":
        description = "Composite: Phase Jump + Harmonics + Inter-harmonics"
        # Phase jump +60 deg at t=1.0
        phase_offset = np.where(t < 1.0, 0.0, np.deg2rad(60.0)).astype(float)
        # Structured disturbances are injected AFTER phase is computed (so harmonics follow the jump)

    elif scenario_id == "G3_E13_Impulsive_Outliers":
        description = "Nominal 60Hz + Gaussian Noise + Rare Impulsive Outliers"
        noise_std = 0.005
        v_noise = rng.normal(0.0, noise_std, size=n)
        impulse_prob = 0.002
        impulse_scale = 0.1  # absolute voltage amplitude
        mask = rng.random(n) < impulse_prob
        if np.any(mask):
            v_noise[mask] += rng.normal(0.0, impulse_scale, size=int(np.sum(mask)))

    elif scenario_id == "G3_E14_Noise_Harmonics":
        description = "Harmonics (3,5,7) + 1% Gaussian Noise"
        noise_std = 0.01
        v_noise = rng.normal(0.0, noise_std, size=n)
        # harmonics injected after phase computed

    elif scenario_id == "G3_E15_Multi_Event_Profile":
        description = "Multi-event: Ramp (+2Hz/s for 2s) then Hold + 1% Noise"
        # ramp from t=1 to t=3 reaching +4 Hz, then hold
        f = (
            f0_nominal
            + np.where(t < 1.0, 0.0, np.where(t < 3.0, 2.0 * (t - 1.0), 4.0))
        ).astype(float)
        noise_std = 0.01
        v_noise = rng.normal(0.0, noise_std, size=n)

    # ============================================================
    # GROUP 4: REAL DATA (placeholder)
    # ============================================================
    elif scenario_id == "G4_E16_Chamorro_Event":
        description = "Real-world Event: Chamorro CSV (Mocked as constant 60Hz-ish)"
        f = np.full_like(t, f0_nominal, dtype=float)

    else:
        raise ValueError(f"Scenario {scenario_id} not defined.")

    # ------------------------------------------------------------
    # Build voltage signal with consistent phase definition
    # ------------------------------------------------------------
    base_phase = _generate_phase(f, fs)
    phase = base_phase + phase_offset

    # Fundamental (unit sine) with amplitude envelope
    v = a * np.sin(phase)

    # Structured disturbances that should follow the same phase (incl. jumps)
    if scenario_id == "G3_E12_Composite_Islanding":
        # Harmonics (5th: 5%, 7th: 3%) aligned with phase
        v_dist += 0.05 * np.sin(5.0 * phase)
        v_dist += 0.03 * np.sin(7.0 * phase)
        # Inter-harmonic (83.5 Hz, 2%) absolute-frequency tone
        v_dist += 0.02 * np.sin(2.0 * np.pi * 83.5 * t)

    if scenario_id == "G3_E14_Noise_Harmonics":
        # Standard harmonics aligned with phase
        for k, h_amp in zip((3, 5, 7), (0.03, 0.02, 0.01)):
            v_dist += float(h_amp) * np.sin(float(k) * phase)

    # Add noise last (so SNR corresponds to final waveform reasonably)
    v = v + v_dist + v_noise

    # SNR meta: if we used explicit Gaussian noise_std, compute approximate SNR
    if noise_std > 0:
        # Use RMS of the (clean) fundamental+disturbance as "signal"
        sig_rms = _rms(a * np.sin(phase) + v_dist)
        snr_db = _snr_db_for_noise(noise_std=noise_std, signal_rms=sig_rms)

    meta: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "description": description,
        "fs": float(fs),
        "T": float(T),
        "seed": int(seed),
        # nominal label (after detune policy) — useful for reproducibility
        "f0": float(f0_nominal),
        # actual starting frequency
        "f_init": float(f[0]),
        "detune_range_hz": float(DETUNE_RANGE_HZ),
    }

    if np.isfinite(snr_db):
        meta["snr_db"] = float(snr_db)

    # extra provenance for outliers (optional, but helps paper)
    if impulse_prob > 0:
        meta["impulsive_prob"] = float(impulse_prob)
    if impulse_scale > 0:
        meta["impulsive_scale"] = float(impulse_scale)

    return t.astype(float), v.astype(float), f.astype(float), meta


def get_all_scenario_names() -> List[str]:
    """List of scenario IDs to iterate in the runner."""
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
        "G3_E12_Composite_Islanding",
        "G3_E13_Impulsive_Outliers",
        "G3_E14_Noise_Harmonics",
        "G3_E15_Multi_Event_Profile",
        "G4_E16_Chamorro_Event",
    ]
