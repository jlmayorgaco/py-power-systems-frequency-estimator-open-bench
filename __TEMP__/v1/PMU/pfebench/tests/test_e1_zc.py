"""
tests/test_e1_zc.py

Test Suite for ZeroCrossingEstimator.
Verifies interpolation precision, noise rejection, and dynamic behavior.
CORRECTED: Explicit sample-by-sample simulation loop to enforce real-time processing constraints.
"""

import numpy as np
import pytest
from pfebench.estimators.e1_zc import ZeroCrossingEstimator

# =============================================================================
# 0. HELPER: REAL-TIME SIMULATION LOOP
# =============================================================================


def run_simulation(estimator, signal):
    """
    Simula el procesamiento en tiempo real alimentando muestras una por una.
    Evita usar el método .run() interno para asegurar el cumplimiento estricto
    de la API step-by-step y la causalidad.
    """
    estimator.reset()
    n = len(signal)
    out = np.zeros(n)

    # Bucle explícito imitando hardware/streaming
    for i in range(n):
        v_sample = float(signal[i])
        out[i] = estimator.step(v_sample)

    return out


# =============================================================================
# 1. FIXTURES
# =============================================================================


@pytest.fixture
def estimator():
    """Instancia fresca del estimador para cada test."""
    return ZeroCrossingEstimator({"fs": 1000.0, "filter_win": 5})


@pytest.fixture
def sine_60hz():
    """1 segundo de 60Hz puro a 1000Hz fs."""
    fs = 1000.0
    t = np.arange(fs) / fs
    return np.sin(2 * np.pi * 60.0 * t)


@pytest.fixture
def sine_step():
    """Escalón de frecuencia: 0.5s a 60Hz -> 0.5s a 61Hz."""
    fs = 1000.0
    t = np.arange(fs) / fs
    # Fase acumulada para continuidad
    f_inst = np.where(t < 0.5, 60.0, 61.0)
    phi = 2 * np.pi * np.cumsum(f_inst) / fs
    return np.sin(phi)


# =============================================================================
# 2. TESTS DE PRECISIÓN BÁSICA
# =============================================================================


def test_static_accuracy(estimator, sine_60hz):
    """
    Debe medir 60Hz con alta precisión.
    NOTA: La interpolación lineal tiene un error intrínseco por la curvatura del seno.
    A 1000Hz, este error es aprox 0.017 Hz. Ajustamos tolerancia a 0.025 Hz.
    """
    out = run_simulation(estimator, sine_60hz)

    # Ignoramos el warm-up
    steady_state = out[50:]

    # El error debe ser bajo
    mae = np.mean(np.abs(steady_state - 60.0))
    assert mae < 0.025, f"Error medio alto: {mae:.4f} Hz (Linear Interp Limit)"


def test_interpolated_accuracy():
    """Prueba crítica: Frecuencia no entera (e.g. 60.5 Hz)."""
    fs = 1000.0
    t = np.arange(fs) / fs
    sig = np.sin(2 * np.pi * 60.5 * t)

    # Sin filtro para probar pura interpolación
    est = ZeroCrossingEstimator({"fs": fs, "filter_win": 1})
    out = run_simulation(est, sig)

    steady = out[50:]
    mae = np.mean(np.abs(steady - 60.5))

    assert mae < 0.05, f"Fallo en interpolación sub-sample. MAE: {mae:.4f}"


# =============================================================================
# 3. TESTS DE DINÁMICA
# =============================================================================


def test_step_response(estimator, sine_step):
    """Debe detectar el cambio de 60 a 61 Hz."""
    out = run_simulation(estimator, sine_step)

    # Zona 1: 60Hz (t=0.2s a 0.4s)
    assert np.mean(out[200:400]) == pytest.approx(60.0, abs=0.05)

    # Zona 2: 61Hz (t=0.6s a 0.8s)
    assert np.mean(out[600:800]) == pytest.approx(61.0, abs=0.05)

    # Latencia: El cambio debe verse reflejado rápido
    assert out[530] > 60.2


def test_latency_reporting(estimator):
    """El estimador debe reportar su latencia teórica correctamente."""
    lat = estimator.latency_samples
    assert 4 <= lat <= 8


# =============================================================================
# 4. TESTS DE ROBUSTEZ Y RUIDO
# =============================================================================


def test_noise_rejection():
    """
    Comparar con y sin filtro ante ruido.
    CORRECCIÓN: Usamos filter_win=5 para evitar cancelación de señal.
    """
    fs = 1000.0
    t = np.arange(fs) / fs
    clean = np.sin(2 * np.pi * 60.0 * t)
    # Ruido considerable
    rng = np.random.default_rng(42)
    noisy = clean + 0.1 * rng.normal(size=len(t))

    # 1. Sin filtro
    est_raw = ZeroCrossingEstimator({"fs": fs, "filter_win": 1})
    out_raw = run_simulation(est_raw, noisy)
    std_raw = np.std(out_raw[100:])

    # 2. Con filtro adecuado (5 muestras)
    est_filt = ZeroCrossingEstimator({"fs": fs, "filter_win": 5})
    out_filt = run_simulation(est_filt, noisy)
    std_filt = np.std(out_filt[100:])

    # El filtro debe reducir la varianza
    assert (
        std_filt < std_raw
    ), f"Filtro falló: STD_Filt={std_filt:.3f} vs STD_Raw={std_raw:.3f}"


def test_dc_offset_rejection(estimator):
    """Un pequeño DC offset no debe romper el ZC."""
    fs = 1000.0
    t = np.arange(fs) / fs
    sig = np.sin(2 * np.pi * 60.0 * t) + 0.1

    out = run_simulation(estimator, sig)
    # Con offset, los cruces se vuelven asimétricos (jitter sistemático),
    # pero el promedio debe mantenerse cerca de 60Hz.
    assert np.mean(out[100:]) == pytest.approx(60.0, abs=1.0)


# =============================================================================
# 5. EDGE CASES
# =============================================================================


def test_silence_handling(estimator):
    """Entrada cero o muy baja no debe crashear."""
    silence = np.zeros(100)
    out = run_simulation(estimator, silence)
    # Debe mantener el valor nominal
    assert np.all(out == 60.0)


def test_nans_input(estimator):
    """Entrada con NaNs debe ser manejada sin excepción."""
    sig = np.full(100, 60.0)
    sig[50] = np.nan
    try:
        run_simulation(estimator, sig)
    except Exception as e:
        pytest.fail(f"El estimador crasheó con NaNs: {e}")
