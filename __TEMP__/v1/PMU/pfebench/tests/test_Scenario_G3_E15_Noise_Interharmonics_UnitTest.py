from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G3_E15_Noise_Interharmonics import G3_E15_Noise_Interharmonics

# =============================================================================
# UNIT TESTS: Physics
# =============================================================================


def test_G3_E15_Interharmonic_Energy():
    """
    Verifica que la energía (RMS) del componente interarmónico coincida
    con el porcentaje solicitado.
    """
    A0 = 100.0
    pct = 0.15  # 15% Distortion
    f_ih = 30.0  # Sub-harmonic
    fs = 10_000.0

    sc = G3_E15_Noise_Interharmonics(
        fs_hz=fs, T_s=2.0, A0=A0, ihd_pct=pct, ih_freq_hz=f_ih
    )
    out = sc.run()

    # 1. Reconstruir Fundamental
    v_fund = A0 * np.sin(out.phi)

    # 2. Aislar Interarmónico
    residue = out.v - v_fund

    # 3. Medir RMS
    rms_residue = np.sqrt(np.mean(residue**2))
    expected_rms = (A0 * pct) / np.sqrt(2)

    assert rms_residue == pytest.approx(
        expected_rms, rel=0.05
    ), "Interharmonic energy mismatch"


def test_G3_E15_Ground_Truth_Isolation():
    """
    Asegura que f_true y phi reporten SOLO la fundamental (60Hz),
    ignorando la interferencia de 95Hz.
    """
    sc = G3_E15_Noise_Interharmonics(ih_freq_hz=95.0, ihd_pct=0.5)
    out = sc.run()

    # Frecuencia debe ser plana 60Hz
    assert np.all(out.f_true == 60.0)

    # Amplitud verdadera debe ser plana A0
    assert np.all(out.A == 1.0)


# =============================================================================
# INTEGRATION TEST: Monte Carlo Frequency Sweep
# =============================================================================


def test_G3_E15_MonteCarlo_Interference_Sweep():
    """
    Barrido Monte Carlo variando la FRECUENCIA del interarmónico.
    Probamos todo el espectro: desde Sub-sincrónicos (10Hz) hasta Super-sincrónicos (120Hz).
    """
    N_ITER = 500
    fs_test = 5000.0

    # Rangos de interferencia
    # Evitamos 60Hz +/- 1Hz para no confundir la medición con la fundamental en el test simple
    ih_freq_range = (10.0, 120.0)

    sc = G3_E15_Noise_Interharmonics(fs_hz=fs_test, T_s=1.0, A0=100.0, ihd_pct=0.1)

    req_freqs = []

    for i in range(N_ITER):
        rng = np.random.default_rng(seed=i)
        target_freq = rng.uniform(*ih_freq_range)

        # Tuning Adapter ('ih_freq')
        sc.set_montecarlo_tuning({"ih_freq": target_freq, "seed": i})

        out = sc.run()

        # Validación: Análisis Espectral (FFT) para confirmar frecuencia
        # Como restamos la fundamental, el pico dominante del residuo debe ser target_freq
        v_fund = 100.0 * np.sin(out.phi)
        residue = out.v - v_fund

        # FFT simple
        fft_vals = np.abs(np.fft.rfft(residue))
        fft_freqs = np.fft.rfftfreq(len(residue), d=1 / fs_test)

        # Ignorar DC (índice 0)
        idx_peak = np.argmax(fft_vals[1:]) + 1
        measured_freq = fft_freqs[idx_peak]

        # La resolución de FFT es fs/N = 5000/5000 = 1Hz (aprox)
        # Tolerancia de +/- 2Hz es segura
        assert (
            abs(measured_freq - target_freq) < 2.0
        ), f"Iter {i}: Interharmonic frequency wrong. Got {measured_freq}, Expected {target_freq}"

        req_freqs.append(target_freq)

    # Cobertura
    arr = np.array(req_freqs)
    assert np.min(arr) < (ih_freq_range[0] + 5.0)
    assert np.max(arr) > (ih_freq_range[1] - 5.0)
