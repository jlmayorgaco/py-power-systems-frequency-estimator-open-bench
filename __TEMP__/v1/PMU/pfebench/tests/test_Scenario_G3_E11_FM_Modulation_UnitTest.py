from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G3_E11_FM_Modulation import G3_E11_FM_Modulation

# =============================================================================
# UNIT TESTS: Physics
# =============================================================================


def test_G3_E11_FM_Frequency_Bounds():
    """
    Verifica que la frecuencia oscile exactamente entre (f_nom - depth) y (f_nom + depth).
    """
    f0 = 60.0
    depth = 2.0  # Range: 58 Hz to 62 Hz
    fm = 1.0  # 1 cycle per second

    sc = G3_E11_FM_Modulation(
        fs_hz=10_000.0, T_s=2.0, f_nom_hz=f0, mod_depth_hz=depth, mod_freq_hz=fm
    )
    out = sc.run()

    # Validar Máximos y Mínimos de Frecuencia
    f_max_meas = np.max(out.f_true)
    f_min_meas = np.min(out.f_true)

    # Usamos tolerancia pequeña (numpy float precision)
    assert f_max_meas == pytest.approx(f0 + depth, rel=1e-9)
    assert f_min_meas == pytest.approx(f0 - depth, rel=1e-9)

    # Validar forma de onda (Cos) en t=0 (Debe ser máximo)
    assert out.f_true[0] == pytest.approx(f0 + depth, rel=1e-9)

    # En t=0.5s (Media vuelta de 1Hz), cos(pi)=-1 -> debe ser mínimo
    idx_half = int(0.5 * 10_000)
    assert out.f_true[idx_half] == pytest.approx(f0 - depth, rel=1e-9)


def test_G3_E11_Phase_Integration():
    """
    Verifica que la fase sea suave (derivada continua) pese a la oscilación de f.
    """
    sc = G3_E11_FM_Modulation(mod_depth_hz=5.0)  # Heavy modulation
    out = sc.run()

    # Diff de la fase ~ Frecuencia
    # No debe haber saltos bruscos
    d_phi = np.diff(out.phi)
    assert (
        np.max(np.abs(np.diff(d_phi))) < 0.1
    ), "Phase integration instability detected"


# =============================================================================
# INTEGRATION TEST: Monte Carlo FM Sweep
# =============================================================================


def test_G3_E11_MonteCarlo_FM_Sweep():
    """
    Barrido Monte Carlo variando Frecuencia de Modulación y Profundidad (Desviación).
    Simula pruebas de rechazo de interferencia o ancho de banda.
    """
    N_ITER = 500
    fs_test = 5000.0

    # Rangos
    fm_range = (0.1, 5.0)  # Velocidad de modulación
    depth_range = (0.1, 5.0)  # Desviación de frecuencia (Hz)

    sc = G3_E11_FM_Modulation(fs_hz=fs_test, T_s=1.0)

    req_fm = []

    for i in range(N_ITER):
        rng = np.random.default_rng(seed=i)
        target_fm = rng.uniform(*fm_range)
        target_depth = rng.uniform(*depth_range)

        # Tuning Adapter (Alias check: 'fm_freq', 'fm_depth')
        sc.set_montecarlo_tuning(
            {"fm_freq": target_fm, "fm_depth": target_depth, "seed": i}
        )

        out = sc.run()

        # Validación
        # Verificamos un punto aleatorio t
        idx_check = int(0.35 * fs_test)

        # IMPORTANTE: Alineación temporal con el tiempo real de la muestra
        t_sample = out.t[idx_check]
        f_meas = out.f_true[idx_check]

        # Formula teórica: f = 60 + k * cos(2*pi*fm*t)
        f_exp = 60.0 + target_depth * np.cos(2.0 * np.pi * target_fm * t_sample)

        assert f_meas == pytest.approx(
            f_exp, rel=1e-9
        ), f"Iter {i}: FM Profile mismatch."

        req_fm.append(target_fm)

    # Cobertura
    arr = np.array(req_fm)
    assert np.min(arr) < (fm_range[0] + 0.5)
    assert np.max(arr) > (fm_range[1] - 0.5)
