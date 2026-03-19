"""
tests/test_e3_pm.py

Test Suite for PeriodMeasurementEstimator (PM).
Verifies full-cycle period measurement behavior (rising-to-rising), dynamic response,
filter robustness, bounds/outlier rejection, and edge cases.

NOTES (important for realistic expectations):
- PM updates only on RISING zero-crossings and holds (ZOH) between them.
- With fs=1000 Hz and ~60 Hz signal, updates happen about every 16-17 samples (one cycle).
- Mean-of-output is meaningful only after warm-up; do NOT expect ultra-low bias at baseline settings.
- latency_samples in PM includes: MA group delay + fs/(2*f_nom) (half nominal period).
"""

import numpy as np
import pytest

from pfebench.estimators.e3_pm import PeriodMeasurementEstimator


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
    """PM puro: sin filtro MA (filter_win=1)."""
    return PeriodMeasurementEstimator({"fs": 1000.0, "filter_win": 1, "f_nom_hz": 60.0})


@pytest.fixture
def estimator_filt():
    """PM con filtro pequeño para comparar robustez."""
    return PeriodMeasurementEstimator({"fs": 1000.0, "filter_win": 5, "f_nom_hz": 60.0})


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
# 2. TESTS DE COMPORTAMIENTO BÁSICO (REALISTAS PARA PM)
# =============================================================================


def test_converges_near_nominal(estimator_pure, sine_60hz):
    """
    PM puro debe converger cerca de 60Hz (tolerancia realista).
    """
    out = run_simulation(estimator_pure, sine_60hz)

    # Ignorar warm-up y primeros cruces
    steady = out[200:]
    mean_val = float(np.nanmean(steady))

    assert mean_val == pytest.approx(60.0, abs=0.25), f"Mean={mean_val:.4f} Hz"


def test_fractional_frequency_is_distinguishable(estimator_pure, sine_60hz, sine_605hz):
    """
    60.5Hz debe producir una media mayor que 60Hz de forma distinguible.
    """
    out60 = run_simulation(estimator_pure, sine_60hz)[200:]
    out605 = run_simulation(estimator_pure, sine_605hz)[200:]

    m60 = float(np.nanmean(out60))
    m605 = float(np.nanmean(out605))

    assert m605 > m60 + 0.10, f"No distingue: mean60={m60:.4f}, mean60.5={m605:.4f}"


def test_output_is_finite_most_of_time(estimator_pure, sine_60hz):
    """
    Para senoide limpia, PM no debería producir NaNs/Infs masivos.
    """
    out = run_simulation(estimator_pure, sine_60hz)
    frac_bad = float(np.mean(~np.isfinite(out)))
    assert frac_bad < 0.01, f"Demasiados NaNs/Infs: {frac_bad:.3f}"


# =============================================================================
# 3. TESTS DE DINÁMICA
# =============================================================================


def test_step_response_directional(estimator_pure, sine_step):
    """
    Debe reflejar el cambio 60->61Hz al menos de forma direccional.
    No exigimos respuesta exacta ni instantánea (baseline por cruces).
    """
    out = run_simulation(estimator_pure, sine_step)

    pre = float(np.nanmean(out[200:400]))  # 0.2-0.4s
    post = float(np.nanmean(out[650:850]))  # 0.65-0.85s

    assert post > pre + 0.20, f"No sube suficiente: pre={pre:.3f}, post={post:.3f}"
    assert post == pytest.approx(61.0, abs=0.50)  # laxo: baseline


def test_latency_reporting_pure(estimator_pure):
    """
    En PM puro:
      latency ≈ (filter_win-1)/2 + fs/(2*f_nom)
      filter_win=1 => 0 + 1000/(120)=8.33 => ~8 samples
    """
    lat = int(estimator_pure.latency_samples)
    assert 6 <= lat <= 12, f"latency_samples inesperada: {lat}"


def test_latency_reporting_with_filter(estimator_filt):
    """
    Con filtro win=5:
      group delay = 2
      + 8.33 => ~10 samples
    """
    lat = int(estimator_filt.latency_samples)
    assert 8 <= lat <= 14, f"latency_samples inesperada: {lat}"


# =============================================================================
# 4. TESTS DE ROBUSTEZ Y RUIDO
# =============================================================================


def test_noise_rejection_filter_helps():
    """
    Comparar PM sin filtro vs con filtro ante ruido.
    El filtro debe reducir la varianza del estimado.
    """
    fs = 1000.0
    t = np.arange(int(fs)) / fs
    clean = np.sin(2 * np.pi * 60.0 * t)

    rng = np.random.default_rng(42)
    noisy = clean + 0.10 * rng.normal(size=len(t))

    est_raw = PeriodMeasurementEstimator({"fs": fs, "filter_win": 1, "f_nom_hz": 60.0})
    out_raw = run_simulation(est_raw, noisy)
    std_raw = float(np.nanstd(out_raw[250:]))

    est_filt = PeriodMeasurementEstimator({"fs": fs, "filter_win": 5, "f_nom_hz": 60.0})
    out_filt = run_simulation(est_filt, noisy)
    std_filt = float(np.nanstd(out_filt[250:]))

    assert (
        std_filt < std_raw
    ), f"Filtro no reduce STD: filt={std_filt:.3f} raw={std_raw:.3f}"


def test_dc_offset_degrades_but_survives(estimator_filt):
    """
    DC offset no debe crashear. Puede introducir jitter/sesgo.
    """
    fs = 1000.0
    t = np.arange(int(fs)) / fs
    sig = np.sin(2 * np.pi * 60.0 * t) + 0.10

    out = run_simulation(estimator_filt, sig)
    mean_val = float(np.nanmean(out[250:]))

    assert np.isfinite(mean_val)
    assert mean_val == pytest.approx(60.0, abs=1.50)


def test_low_amplitude_fragility(estimator_pure, low_amp_60hz):
    """
    Baja amplitud cerca de cero es difícil: no exigimos precisión, solo estabilidad.
    """
    out = run_simulation(estimator_pure, low_amp_60hz)
    frac_bad = float(np.mean(~np.isfinite(out)))
    assert frac_bad < 0.10, f"Demasiados NaNs/Infs en baja amplitud: {frac_bad:.3f}"


def test_bounds_reject_high_frequency_updates():
    """
    Si configuramos max_freq_hz muy bajo, el estimador debe rechazar updates > max_freq_hz
    y tender a quedarse cerca del nominal (ZOH).
    """
    fs = 1000.0
    t = np.arange(int(fs)) / fs
    sig_61 = np.sin(2 * np.pi * 61.0 * t)

    est = PeriodMeasurementEstimator(
        {
            "fs": fs,
            "filter_win": 1,
            "f_nom_hz": 60.0,
            "min_freq_hz": 59.5,
            "max_freq_hz": 60.1,  # muy bajo: 61Hz debería ser rechazado
            "ema_alpha": 1.0,
        }
    )

    out = run_simulation(est, sig_61)
    mean_val = float(np.nanmean(out[250:]))

    # Debe quedarse cerca del nominal (no seguir 61Hz)
    assert mean_val < 60.3, f"No parece rechazar outliers: mean={mean_val:.3f} Hz"


# =============================================================================
# 5. EDGE CASES
# =============================================================================


def test_silence_handling(estimator_pure):
    """
    Entrada cero no debe crashear; debe mantener nominal.
    """
    silence = np.zeros(200, dtype=float)
    out = run_simulation(estimator_pure, silence)
    assert np.all(out == PeriodMeasurementEstimator.NOMINAL_FREQ_HZ)


def test_nans_input_does_not_raise(estimator_pure):
    """
    Entrada con NaNs: no debe lanzar excepción.
    No exigimos "handling perfecto", solo que no crashee.
    """
    fs = 1000.0
    t = np.arange(200) / fs
    sig = np.sin(2 * np.pi * 60.0 * t)
    sig[50] = np.nan

    try:
        out = run_simulation(estimator_pure, sig)
        assert out.size == sig.size
    except Exception as e:
        pytest.fail(f"El estimador crasheó con NaNs: {e}")
