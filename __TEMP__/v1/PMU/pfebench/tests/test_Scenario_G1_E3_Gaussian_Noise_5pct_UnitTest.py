from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G1_E3_Gaussian_Noise_5pct import G1_E3_Gaussian_Noise_5pct

# =============================================================================
# UNIT TESTS: High Noise Verification
# =============================================================================


def test_G1_E3_High_Noise_Statistics():
    """
    Verifica las propiedades estadísticas a alto nivel de ruido (5%).
    Con A0=100V, esperamos una desviación estándar de ~5V.
    """
    A0_val = 100.0
    target_noise = 0.05
    seed_val = 999

    sc = G1_E3_Gaussian_Noise_5pct(
        fs_hz=10_000.0,
        T_s=2.0,
        f_nom_hz=60.0,
        A0=A0_val,
        noise_rel=target_noise,
        seed=seed_val,
    )
    out = sc.run()

    # 1. Extracción de señal pura y ruido
    t = out.t
    phi_theoretical = 2.0 * np.pi * 60.0 * t
    v_pure = A0_val * np.sin(phi_theoretical)
    noise_extracted = out.v - v_pure

    # 2. Validaciones

    # a) Media ~ 0 (Sin DC Bias significativo)
    noise_mean = np.mean(noise_extracted)
    assert abs(noise_mean) < 0.15, f"High Noise Mean Bias detected: {noise_mean}"

    # b) Std ~ 5.0
    noise_std = np.std(noise_extracted)
    expected_std = A0_val * target_noise  # 5.0

    error_rel = abs(noise_std - expected_std) / expected_std
    assert (
        error_rel < 0.05
    ), f"Noise Power mismatch. Got {noise_std}, Expected {expected_std}"


def test_G1_E3_Schema_Integrity():
    """Verifica que los metadatos reporten el 5% por defecto."""
    sc = G1_E3_Gaussian_Noise_5pct()
    out = sc.run()

    schema = out.meta["schema"]
    assert schema["scenario_id"] == "G1_E3_Gaussian_Noise_5pct"
    assert schema["noise_rel"] == 0.05
    assert schema["noise_type"] == "gaussian"


# =============================================================================
# INTEGRATION TEST: Monte Carlo Sweep (Stress Test)
# =============================================================================


def test_G1_E3_MonteCarlo_Stress_Linearity():
    """
    Monte Carlo Sweep enfocándose en rangos de ruido medio-alto.
    Verifica la linealidad del generador cuando el ruido es considerable.
    """
    N_ITER = 1000

    # Rangos (Exploramos desde 3% hasta 10% de ruido)
    f_range = (58.0, 62.0)
    v_range = (80.0, 120.0)
    noise_range = (0.03, 0.10)  # 3% a 10% (Muy ruidoso)

    # T_s = 0.1s -> N=500 -> Error estadístico esperado ~3%
    sc = G1_E3_Gaussian_Noise_5pct(fs_hz=5000.0, T_s=0.1)

    errors_rel = []

    for i in range(N_ITER):
        # A. Sampling
        rng = np.random.default_rng(seed=i)
        tf = rng.uniform(*f_range)
        tv = rng.uniform(*v_range)
        tn = rng.uniform(*noise_range)

        # B. Tuning
        sc.set_montecarlo_tuning({"f": tf, "Vmax": tv, "noise": tn, "seed": i})

        # C. Run & Extract
        out = sc.run()

        # Reconstruir referencia
        v_pure = tv * np.sin(2.0 * np.pi * tf * out.t)
        residual = out.v - v_pure

        meas_sigma = np.std(residual)
        exp_sigma = tv * tn

        if exp_sigma > 1e-9:
            errors_rel.append(abs(meas_sigma - exp_sigma) / exp_sigma)

    # --- Validaciones Estadísticas ---
    arr_err = np.array(errors_rel)
    mean_err = np.mean(arr_err)
    max_err = np.max(arr_err)

    # El bias promedio debe estar dentro de lo estadísticamente posible para N=500
    # Ajustamos tolerancia a < 4.0%
    assert mean_err < 0.04, f"Bias at high noise: {mean_err:.2%}"

    # Los outliers pueden ser algo mayores debido a la alta varianza
    assert max_err < 0.25, f"Unstable generation at high noise: {max_err:.2%}"
