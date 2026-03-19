from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G1_E2_Gaussian_Noise_1pct import G1_E2_Gaussian_Noise_1pct

# =============================================================================
# UNIT TESTS: Basic Logic & Physics
# =============================================================================


def test_G1_E2_Noise_Statistics():
    """
    Verifica que el ruido añadido tenga las propiedades estadísticas correctas
    (Media ~ 0 y Std ~ 1% de A0).
    """
    # Configuramos un caso con amplitud 100 y 1% ruido -> Std debe ser 1.0
    A0_val = 100.0
    noise_pct = 0.01
    seed_val = 12345

    sc = G1_E2_Gaussian_Noise_1pct(
        fs_hz=10_000.0,
        T_s=2.0,  # Aquí N es grande (20,000), la varianza será baja
        f_nom_hz=60.0,
        A0=A0_val,
        noise_rel=noise_pct,
        seed=seed_val,
    )
    out = sc.run()

    # 1. Recuperar datos
    t = out.t
    v_noisy = out.v

    # 2. Reconstruir la señal pura teórica
    phi_theoretical = 2.0 * np.pi * 60.0 * t
    v_pure = A0_val * np.sin(phi_theoretical)

    # 3. Extraer el ruido (Residuo)
    noise_extracted = v_noisy - v_pure

    # 4. Validar estadística del ruido

    # a) La media debe ser cercana a 0
    noise_mean = np.mean(noise_extracted)
    assert abs(noise_mean) < 0.05, f"Mean bias detected: {noise_mean}"

    # b) La desviación estándar debe ser cercana a A0 * 1% = 1.0
    noise_std = np.std(noise_extracted)
    expected_std = A0_val * noise_pct

    # Tolerancia del 5%
    error_rel = abs(noise_std - expected_std) / expected_std
    assert (
        error_rel < 0.05
    ), f"Incorrect Noise Power. Std={noise_std}, Expected={expected_std}"


def test_G1_E2_Determinism():
    """Verifica que la semilla (seed) asegure repetibilidad exacta."""
    # Mismo seed
    sc1 = G1_E2_Gaussian_Noise_1pct(seed=555)
    out1 = sc1.run()

    sc2 = G1_E2_Gaussian_Noise_1pct(seed=555)
    out2 = sc2.run()

    # Distinto seed
    sc3 = G1_E2_Gaussian_Noise_1pct(seed=999)
    out3 = sc3.run()

    assert np.array_equal(out1.v, out2.v), "Same seed must yield identical data"
    assert not np.array_equal(
        out1.v, out3.v
    ), "Different seeds must yield different data"


def test_G1_E2_Schema_Metadata():
    """Verifica que los metadatos del ruido se guarden en el schema."""
    sc = G1_E2_Gaussian_Noise_1pct(noise_rel=0.05)
    out = sc.run()

    schema = out.meta["schema"]
    assert schema["noise_rel"] == 0.05
    assert schema["scenario_id"] == "G1_E2_Gaussian_Noise_1pct"
    assert "sigma_expected" in schema


# =============================================================================
# INTEGRATION TEST: Monte Carlo Sweep
# =============================================================================


def test_G1_E2_MonteCarlo_Noise_Linearity():
    """
    Monte Carlo Sweep para G1_E2.

    Objetivo:
    Validar que el nivel de ruido generado escale linealmente con Vmax y noise_rel
    al variar Frecuencia, Amplitud y Ruido simultáneamente usando el Tuning Adapter.
    """
    N_ITER = 1000

    # Sweep Ranges
    f_range = (58.0, 62.0)
    v_range = (50.0, 150.0)
    noise_range = (0.001, 0.05)  # 0.1% a 5.0%

    # Instancia única (fs baja y tiempo corto para velocidad)
    # T_s = 0.1s @ 5000Hz = 500 muestras.
    # El error estadístico esperado para N=500 es ~3.1% (1/sqrt(2*(N-1)))
    sc = G1_E2_Gaussian_Noise_1pct(fs_hz=5000.0, T_s=0.1)

    errors_rel = []

    for i in range(N_ITER):
        # A. Sampling
        rng = np.random.default_rng(seed=i)
        target_f = rng.uniform(*f_range)
        target_v = rng.uniform(*v_range)
        target_noise = rng.uniform(*noise_range)

        # B. Tuning (Test del mapa: 'f', 'Vmax', 'noise')
        sc.set_montecarlo_tuning(
            {"f": target_f, "Vmax": target_v, "noise": target_noise, "seed": i}
        )

        # C. Run
        out = sc.run()

        # D. Análisis Físico
        # Reconstruir onda pura para aislar el ruido generado
        t = out.t
        v_pure = target_v * np.sin(2.0 * np.pi * target_f * t)
        residual = out.v - v_pure

        # Medir Sigma real vs esperada
        meas_sigma = np.std(residual)
        exp_sigma = target_v * target_noise

        # Error relativo
        if exp_sigma > 1e-9:
            errors_rel.append(abs(meas_sigma - exp_sigma) / exp_sigma)
        else:
            errors_rel.append(0.0)

    # --- VALIDACIONES ESTADÍSTICAS DEL SWEEP ---

    arr_err = np.array(errors_rel)
    mean_err = np.mean(arr_err)
    max_err = np.max(arr_err)

    # 1. Bias promedio (Ajustado a estadística de muestras pequeñas N=500)
    # Esperamos ~2.6% de error puramente estadístico.
    # Ponemos el límite en 4.0% para detectar fallos reales sin falsos positivos.
    assert (
        mean_err < 0.04
    ), f"Monte Carlo Bias: Average noise error is too high ({mean_err:.2%})"

    # 2. Outliers controlados (< 20% en peor caso individual)
    assert max_err < 0.20, f"Monte Carlo Outlier: Max error too high ({max_err:.2%})"
