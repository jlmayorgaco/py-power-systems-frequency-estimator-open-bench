from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G1_E1_Pure_60Hz import G1_E1_Pure_60Hz

# =============================================================================
# TEST 1: Basic Physics & Schema (Unit Test)
# =============================================================================


def test_G1_E1_Pure_60Hz_basic_invariants():
    """
    Verifica que una ejecución simple genere una onda sinusoidal perfecta
    y respete el schema de datos definido.
    """
    # 1. Configuración
    fs = 10_000.0
    sc = G1_E1_Pure_60Hz(
        fs_hz=fs,
        T_s=5.0,
        f_nom_hz=60.0,
        A0=1.0,
        phi0_rad=0.0,
        seed=123,
    )
    out = sc.run()

    # 2. Extracción de datos (Flat arrays)
    t = out.t
    v = out.v
    f = out.f_true

    # 3. Validaciones de Estructura (Shapes & NaNs)
    assert t.size == v.size == f.size
    assert t.size > 10
    assert np.all(np.isfinite(v))

    # 4. Validaciones Temporales
    dt = float(np.median(np.diff(t)))
    assert dt == pytest.approx(1.0 / fs, rel=1e-12), "Time step (dt) mismatch"
    assert (t[-1] - t[0]) == pytest.approx(5.0, abs=2 * dt), "Duration mismatch"

    # 5. Validaciones Físicas (Waveform)
    # Frecuencia constante
    assert np.mean(f) == pytest.approx(60.0, rel=1e-12)

    # Sin DC offset
    assert np.mean(v) == pytest.approx(0.0, abs=1e-3)

    # Amplitud Pico a Pico ~ 2*A0
    v_pp = np.ptp(v)  # Peak-to-peak
    assert v_pp == pytest.approx(2.0, rel=0.01)

    # 6. Validación Espectral (FFT simple)
    # Quitamos la media y aplicamos ventana simple para evitar fugas espectrales graves en el test
    x = v - np.mean(v)
    freqs = np.fft.rfftfreq(len(x), d=dt)
    mag = np.abs(np.fft.rfft(x))
    f_peak = float(freqs[np.argmax(mag[1:]) + 1])  # Ignore DC bin 0

    assert f_peak == pytest.approx(60.0, abs=0.2), "Dominant frequency is not 60Hz"

    # 7. Validación de Schema y Metadatos
    assert isinstance(out.meta, dict)
    assert out.meta["scenario_id"] == "G1_E1_Pure_60Hz"

    schema = out.meta["schema"]
    assert isinstance(schema, dict)
    assert schema["f_nom_hz"] == 60.0
    assert isinstance(schema.get("modifiers"), list)


# =============================================================================
# TEST 2: Monte Carlo & Tuning Adapter (Integration Test)
# =============================================================================


def test_G1_E1_MonteCarlo_Sweep_Integrity():
    """
    Verifica la robustez del método 'set_montecarlo_tuning'.

    Objetivos:
    1. Validar que el 'tuning_map' funcione (Aliases 'f' y 'Vmax' alteren el estado).
    2. Validar que la generación aleatoria cubra el rango solicitado (Uniformidad).
    3. Validar fidelidad física (Lo que pido es lo que obtengo).
    """
    N_ITER = 1000

    # Rangos de prueba
    f_range = (58.0, 62.0)
    v_range = (0.9, 1.1)

    # Historiales
    inputs_f, inputs_v = [], []
    outputs_f_mean, outputs_v_peak = [], []

    # Instancia única (Patrón Fluent) con tiempo corto para velocidad
    sc = G1_E1_Pure_60Hz(fs_hz=5000.0, T_s=0.1)

    for i in range(N_ITER):
        # A. Sampling Controlado
        rng = np.random.default_rng(seed=i)
        target_f = rng.uniform(*f_range)
        target_v = rng.uniform(*v_range)

        # B. Tuning (Test del Adaptador)
        sc.set_montecarlo_tuning({"f": target_f, "Vmax": target_v, "seed": i})

        # C. Ejecución
        out = sc.run()

        # D. Recolección
        inputs_f.append(target_f)
        inputs_v.append(target_v)

        # Medimos la realidad generada
        outputs_f_mean.append(np.mean(out.f_true))
        outputs_v_peak.append(np.max(np.abs(out.v)))

    # Convertimos a arrays para numpy math
    in_f_arr = np.array(inputs_f)
    out_f_arr = np.array(outputs_f_mean)
    in_v_arr = np.array(inputs_v)
    out_v_arr = np.array(outputs_v_peak)

    # --- VALIDACIONES ---

    # 1. Integridad Lógica (El adaptador funcionó)
    # La frecuencia en 'f_true' debe ser idéntica a la solicitada 'f'
    np.testing.assert_allclose(
        in_f_arr,
        out_f_arr,
        rtol=1e-12,
        err_msg="Tuning Adapter Failure: Request frequency != Generated frequency",
    )

    # 2. Fidelidad Física
    # El pico de voltaje generado debe coincidir con Vmax solicitado
    # Tolerancia pequeña por errores de muestreo discreto (peak finding en discreto)
    # Con fs=5000 y 60Hz, el error es despreciable.
    np.testing.assert_allclose(
        in_v_arr,
        out_v_arr,
        rtol=1e-3,
        err_msg="Physics Failure: Generated Amplitude does not match Vmax request",
    )

    # 3. Relevancia Estadística (Distribución)
    # Validamos que el MC exploró el rango completo y no se quedó sesgado

    # Cobertura de Frecuencia
    assert np.min(in_f_arr) < (
        f_range[0] + 0.1
    ), "Poor Coverage: Lower bound not reached"
    assert np.max(in_f_arr) > (
        f_range[1] - 0.1
    ), "Poor Coverage: Upper bound not reached"

    # Media centrada (Uniforme)
    expected_mean = sum(f_range) / 2
    assert np.mean(in_f_arr) == pytest.approx(
        expected_mean, abs=0.1
    ), "Statistical Bias: Random sampling is not centered"

    # Desviación Estándar (Uniforme = rango / sqrt(12))
    expected_std = (f_range[1] - f_range[0]) / np.sqrt(12)
    assert np.std(in_f_arr) == pytest.approx(
        expected_std, rel=0.05
    ), "Distribution Error: Sampling does not appear Uniform"
