from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G3_E14_Noise_Harmonics import G3_E14_Noise_Harmonics

# =============================================================================
# UNIT TESTS: Physics
# =============================================================================


def test_G3_E14_THD_Accuracy():
    """
    Verifica que la energía total de los armónicos generados coincida
    con el porcentaje de THD solicitado.
    """
    A0 = 100.0
    target_thd = 0.10  # 10%
    fs = 10_000.0

    sc = G3_E14_Noise_Harmonics(fs_hz=fs, T_s=2.0, A0=A0, thd_pct=target_thd)
    out = sc.run()

    # 1. Reconstruir Fundamental Pura (Sabemos la fase exacta en out.phi)
    # out.phi contiene la fase de la fundamental (phi0 + w*t)
    v_fund_reconstructed = A0 * np.sin(out.phi)

    # 2. Extraer Armónicos (Residuo)
    v_harmonics = out.v - v_fund_reconstructed

    # 3. Calcular RMS de los armónicos
    rms_harmonics = np.sqrt(np.mean(v_harmonics**2))

    # 4. Calcular RMS de la fundamental
    rms_fund = A0 / np.sqrt(2)

    # 5. THD Medido = RMS_harm / RMS_fund
    # Nota: La definición de THD en el generador fue relativa a la Amplitud Pico (A0)
    # para simplificar la generación A_h = A0 * pct.
    # Verifiquemos cómo lo definimos en build():
    # norm_factor = thd_pct / sqrt(sum_weights).
    # Ah = A0 * weight * norm_factor.
    # Total Harmonic Amp Vector Magnitude = sqrt(sum(Ah^2)) = A0 * thd_pct.
    # Por tanto, el RMS total armónico debe ser (A0 * thd_pct) / sqrt(2).

    expected_rms_harmonics = (A0 * target_thd) / np.sqrt(2)

    # Validamos energías
    assert rms_harmonics == pytest.approx(
        expected_rms_harmonics, rel=0.05
    ), "Generated THD energy mismatch."


def test_G3_E14_Fundamental_Frequency_Unchanged():
    """
    Asegura que f_true siga reportando 60Hz.
    El estimador debe ignorar los armónicos, no seguirlos.
    """
    sc = G3_E14_Noise_Harmonics(thd_pct=0.5)  # 50% THD (Muy sucio)
    out = sc.run()

    assert np.all(out.f_true == 60.0), "Ground truth frequency should ignore harmonics"


# =============================================================================
# INTEGRATION TEST: Monte Carlo THD Sweep
# =============================================================================


def test_G3_E14_MonteCarlo_THD_Sweep():
    """
    Barrido Monte Carlo variando el nivel de distorsión (THD).
    Desde red limpia (0%) hasta red muy sucia (20%).
    """
    N_ITER = 500
    fs_test = 5000.0

    # Rangos
    thd_range = (0.01, 0.20)  # 1% a 20%

    sc = G3_E14_Noise_Harmonics(fs_hz=fs_test, T_s=0.5, A0=100.0)

    req_thds = []

    for i in range(N_ITER):
        rng = np.random.default_rng(seed=i)
        target_thd = rng.uniform(*thd_range)

        # Tuning Adapter ('thd')
        sc.set_montecarlo_tuning({"thd": target_thd, "seed": i})

        out = sc.run()

        # Validación
        v_fund = 100.0 * np.sin(out.phi)
        residue = out.v - v_fund

        rms_residue = np.sqrt(np.mean(residue**2))
        expected_rms = (100.0 * target_thd) / np.sqrt(2)

        # Tolerancia del 5% es suficiente para varianza estocástica de fase
        assert rms_residue == pytest.approx(
            expected_rms, rel=0.05
        ), f"Iter {i}: THD mismatch."

        req_thds.append(target_thd)

    # Cobertura
    arr = np.array(req_thds)
    assert np.min(arr) < (thd_range[0] + 0.02)
    assert np.max(arr) > (thd_range[1] - 0.02)
