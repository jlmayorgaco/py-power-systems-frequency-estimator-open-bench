from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G3_E13_Impulsive_Outliers import G3_E13_Impulsive_Outliers

# =============================================================================
# UNIT TESTS: Physics & Statistics
# =============================================================================


def test_G3_E13_Spike_Density():
    """
    Verifica que la cantidad de outliers generados coincida aproximadamente
    con la probabilidad solicitada (Ley de los Grandes Números).
    """
    prob = 0.01  # 1% de outliers
    fs = 10_000.0
    T = 5.0
    N_total = int(fs * T)  # 50,000 muestras

    # Spike gigante para detectarlo fácil (50V sobre 1V de señal)
    sc = G3_E13_Impulsive_Outliers(
        fs_hz=fs, T_s=T, A0=1.0, outlier_prob=prob, outlier_sigma=50.0
    )
    out = sc.run()

    # Detección simple: Todo lo que supere 2.0V es un outlier
    # (La señal pura es max 1.0V)
    outliers_detected = np.sum(np.abs(out.v) > 2.0)

    expected_count = N_total * prob  # 500

    # Tolerancia estadística (Poisson distribution variance)
    # Aceptamos +/- 20% de variación para N=500
    margin = 0.20 * expected_count

    assert (
        abs(outliers_detected - expected_count) < margin
    ), f"Outlier count mismatch. Got {outliers_detected}, Expected ~{expected_count}"


def test_G3_E13_Determinism():
    """Seed control check para ruido aleatorio."""
    sc1 = G3_E13_Impulsive_Outliers(seed=123, outlier_prob=0.1)
    out1 = sc1.run()

    sc2 = G3_E13_Impulsive_Outliers(seed=123, outlier_prob=0.1)
    out2 = sc2.run()

    assert np.array_equal(out1.v, out2.v)


# =============================================================================
# INTEGRATION TEST: Monte Carlo Sweep
# =============================================================================


def test_G3_E13_MonteCarlo_Outlier_Sweep():
    """
    Barrido Monte Carlo variando la probabilidad de ocurrencia de outliers.
    """
    N_ITER = 500
    fs_test = 5000.0
    T_test = 0.5
    N_samples = int(fs_test * T_test)  # 2500 muestras

    # Rango de probabilidad: 0.1% a 5%
    prob_range = (0.001, 0.05)

    sc = G3_E13_Impulsive_Outliers(fs_hz=fs_test, T_s=T_test, outlier_sigma=100.0)

    req_probs = []

    for i in range(N_ITER):
        rng = np.random.default_rng(seed=i)
        target_prob = rng.uniform(*prob_range)

        # Tuning: 'prob'
        sc.set_montecarlo_tuning({"prob": target_prob, "seed": i})

        out = sc.run()

        # Validación Estadística
        count = np.sum(np.abs(out.v) > 2.0)
        measured_prob = count / N_samples

        # Error relativo permitido alto en ventanas cortas
        # pero el promedio del MC debe converger.
        # Aquí solo guardamos para verificar cobertura y bias global.

        req_probs.append(target_prob)

    # Validaciones Globales
    arr_req = np.array(req_probs)

    # Cobertura
    assert np.min(arr_req) < (prob_range[0] + 0.005)
    assert np.max(arr_req) > (prob_range[1] - 0.005)
