"""
tests/test_e4_if_dphi.py

Test Suite for InstantaneousFrequencyPhaseIncrementEstimator (IF-Δφ).
Verifies analytic-signal phase-increment behavior, dynamic response, robustness, and edge cases.

REALISTIC EXPECTATIONS:
- This estimator is sample-by-sample after warm-up, but has FIR Hilbert latency.
- Accuracy depends on Hilbert length; do not set overly-strict tolerances for short FIR.
- Tests focus on meaningful properties:
  * converges near nominal (with realistic tolerance)
  * fractional frequency is distinguishable
  * step response is directional
  * filtering (input MA and/or dphi smoothing) reduces variance under noise
  * reports latency consistent with configured delays
  * no crashes on silence/NaNs
"""

import numpy as np
import pytest

from pfebench.estimators.e4_if_dphi import InstantaneousFrequencyPhaseIncrementEstimator


# =============================================================================
# 0. HELPER: REAL-TIME SIMULATION LOOP
# =============================================================================


def run_simulation(estimator, signal):
    """
    Real-time step-by-step simulation (do NOT use estimator.run()).
    """
    estimator.reset()
    n = len(signal)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        out[i] = estimator.step(float(signal[i]))
    return out


# =============================================================================
# 1. FIXTURES
# =============================================================================


@pytest.fixture
def estimator_pure():
    """
    IF-Δφ "puro":
      - no MA
      - Hilbert FIR moderate
      - no Δφ smoothing
      - ema_alpha=1.0 (no extra lag)
    """
    return InstantaneousFrequencyPhaseIncrementEstimator(
        {
            "fs": 1000.0,
            "filter_win": 1,
            "hilbert_len": 31,
            "dphi_win": 1,
            "ema_alpha": 1.0,
            "min_abs": 1e-6,
            "min_freq_hz": 40.0,
            "max_freq_hz": 80.0,
        }
    )


@pytest.fixture
def estimator_filt():
    """
    IF-Δφ con suavizado:
      - MA de entrada
      - Δφ smoothing
      - ema_alpha moderado
    """
    return InstantaneousFrequencyPhaseIncrementEstimator(
        {
            "fs": 1000.0,
            "filter_win": 5,
            "hilbert_len": 31,
            "dphi_win": 5,
            "ema_alpha": 0.35,
            "min_abs": 1e-6,
            "min_freq_hz": 40.0,
            "max_freq_hz": 80.0,
        }
    )


@pytest.fixture
def sine_60hz():
    fs = 1000.0
    t = np.arange(int(fs)) / fs
    return np.sin(2 * np.pi * 60.0 * t)


@pytest.fixture
def sine_605hz():
    fs = 1000.0
    t = np.arange(int(fs)) / fs
    return np.sin(2 * np.pi * 60.5 * t)


@pytest.fixture
def sine_step():
    """Escalón: 0.5s a 60Hz -> 0.5s a 61Hz con fase continua."""
    fs = 1000.0
    t = np.arange(int(fs)) / fs
    f_inst = np.where(t < 0.5, 60.0, 61.0)
    phi = 2 * np.pi * np.cumsum(f_inst) / fs
    return np.sin(phi)


@pytest.fixture
def low_amp_60hz():
    fs = 1000.0
    t = np.arange(int(fs)) / fs
    return 0.05 * np.sin(2 * np.pi * 60.0 * t)


# =============================================================================
# 2. BASIC BEHAVIOR (REALISTIC)
# =============================================================================


def test_converges_near_nominal(estimator_pure, sine_60hz):
    """
    IF-Δφ should converge near 60 Hz after warm-up.
    """
    out = run_simulation(estimator_pure, sine_60hz)

    # Warm-up: FIR Hilbert delay ~ (hilbert_len-1)/2 = 15 samples, plus a bit.
    steady = out[200:]
    mean_val = float(np.nanmean(steady))

    assert mean_val == pytest.approx(60.0, abs=0.20), f"Mean={mean_val:.4f} Hz"


def test_fractional_frequency_is_distinguishable(estimator_pure, sine_60hz, sine_605hz):
    """
    60.5 Hz should produce a higher mean than 60 Hz in steady-state.
    """
    out60 = run_simulation(estimator_pure, sine_60hz)[200:]
    out605 = run_simulation(estimator_pure, sine_605hz)[200:]

    m60 = float(np.nanmean(out60))
    m605 = float(np.nanmean(out605))

    assert m605 > m60 + 0.15, f"No distingue: mean60={m60:.4f}, mean60.5={m605:.4f}"


def test_output_is_finite_most_of_time(estimator_pure, sine_60hz):
    """
    For a clean sine, the estimator should not produce many NaNs/Infs.
    """
    out = run_simulation(estimator_pure, sine_60hz)
    frac_bad = float(np.mean(~np.isfinite(out)))
    assert frac_bad < 0.02, f"Demasiados NaNs/Infs: {frac_bad:.3f}"


# =============================================================================
# 3. DYNAMICS
# =============================================================================


def test_step_response_directional(estimator_pure, sine_step):
    """
    Must reflect 60->61 Hz change directionally.
    """
    out = run_simulation(estimator_pure, sine_step)

    pre = float(np.nanmean(out[200:400]))  # 0.2-0.4 s
    post = float(np.nanmean(out[650:850]))  # 0.65-0.85 s

    assert post > pre + 0.30, f"No sube suficiente: pre={pre:.3f}, post={post:.3f}"
    assert post == pytest.approx(61.0, abs=0.35)  # reasonably tight for IF-Δφ


def test_latency_reporting_consistent():
    """
    latency_samples ≈ delay(MA) + delay(Hilbert) + delay(dphi MA)
    For filter_win=1, hilbert_len=31, dphi_win=1:
      0 + 15 + 0 = 15
    """
    est = InstantaneousFrequencyPhaseIncrementEstimator(
        {
            "fs": 1000.0,
            "filter_win": 1,
            "hilbert_len": 31,
            "dphi_win": 1,
        }
    )
    lat = int(est.latency_samples)
    assert 12 <= lat <= 18, f"latency_samples inesperada: {lat}"


def test_latency_reporting_with_smoothing():
    """
    For filter_win=5, hilbert_len=31, dphi_win=5:
      delay = 2 + 15 + 2 = 19
    """
    est = InstantaneousFrequencyPhaseIncrementEstimator(
        {
            "fs": 1000.0,
            "filter_win": 5,
            "hilbert_len": 31,
            "dphi_win": 5,
        }
    )
    lat = int(est.latency_samples)
    assert 16 <= lat <= 23, f"latency_samples inesperada: {lat}"


# =============================================================================
# 4. ROBUSTNESS & NOISE
# =============================================================================


def test_noise_rejection_filter_helps():
    """
    Compare raw vs filtered under noise.
    Filtering should reduce variance of the estimate.
    """
    fs = 1000.0
    t = np.arange(int(fs)) / fs
    clean = np.sin(2 * np.pi * 60.0 * t)

    rng = np.random.default_rng(42)
    noisy = clean + 0.10 * rng.normal(size=len(t))

    est_raw = InstantaneousFrequencyPhaseIncrementEstimator(
        {
            "fs": fs,
            "filter_win": 1,
            "hilbert_len": 31,
            "dphi_win": 1,
            "ema_alpha": 1.0,
            "min_abs": 1e-6,
            "min_freq_hz": 40.0,
            "max_freq_hz": 80.0,
        }
    )
    out_raw = run_simulation(est_raw, noisy)
    std_raw = float(np.nanstd(out_raw[250:]))

    est_filt = InstantaneousFrequencyPhaseIncrementEstimator(
        {
            "fs": fs,
            "filter_win": 5,
            "hilbert_len": 31,
            "dphi_win": 5,
            "ema_alpha": 0.35,
            "min_abs": 1e-6,
            "min_freq_hz": 40.0,
            "max_freq_hz": 80.0,
        }
    )
    out_filt = run_simulation(est_filt, noisy)
    std_filt = float(np.nanstd(out_filt[250:]))

    assert (
        std_filt < std_raw
    ), f"Filtro no reduce STD: filt={std_filt:.3f} raw={std_raw:.3f}"


def test_low_amplitude_fragility(estimator_pure, low_amp_60hz):
    """
    Low amplitude can cause noisy phase increments; we only require stability.
    """
    out = run_simulation(estimator_pure, low_amp_60hz)
    frac_bad = float(np.mean(~np.isfinite(out)))
    assert frac_bad < 0.10, f"Demasiados NaNs/Infs en baja amplitud: {frac_bad:.3f}"


def test_bounds_reject_outliers():
    """
    If bounds are tight, out-of-range updates should be rejected and output should
    remain near nominal.
    """
    fs = 1000.0
    t = np.arange(int(fs)) / fs
    sig_62 = np.sin(2 * np.pi * 62.0 * t)

    est = InstantaneousFrequencyPhaseIncrementEstimator(
        {
            "fs": fs,
            "filter_win": 1,
            "hilbert_len": 31,
            "dphi_win": 1,
            "ema_alpha": 1.0,
            "min_abs": 1e-6,
            "min_freq_hz": 59.5,
            "max_freq_hz": 60.2,  # tight: should reject 62 Hz
        }
    )

    out = run_simulation(est, sig_62)
    mean_val = float(np.nanmean(out[250:]))

    assert mean_val < 60.4, f"No parece rechazar outliers: mean={mean_val:.3f} Hz"


# =============================================================================
# 5. EDGE CASES
# =============================================================================


def test_silence_handling():
    """
    Zero input should not crash; should hold nominal.
    """
    est = InstantaneousFrequencyPhaseIncrementEstimator(
        {
            "fs": 1000.0,
            "filter_win": 1,
            "hilbert_len": 31,
            "dphi_win": 1,
            "ema_alpha": 1.0,
        }
    )
    silence = np.zeros(200, dtype=float)
    out = run_simulation(est, silence)
    assert np.all(out == InstantaneousFrequencyPhaseIncrementEstimator.NOMINAL_FREQ_HZ)


def test_nans_input_does_not_raise(estimator_pure):
    """
    NaN input should not raise an exception.
    We do not demand perfect NaN handling, only no crash.
    """
    fs = 1000.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * 60.0 * t)
    sig[50] = np.nan
    sig[51] = np.nan

    try:
        out = run_simulation(estimator_pure, sig)
        assert out.size == sig.size
    except Exception as e:
        pytest.fail(f"El estimador crasheó con NaNs: {e}")
