from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G3_E10_AM_Modulation import G3_E10_AM_Modulation

# =============================================================================
# UNIT TESTS: Physics
# =============================================================================


def test_G3_E10_AM_Envelope_Correctness():
    """
    Verifica que la envolvente de amplitud (A) siga la ecuación de modulación:
    A = A0 * (1 + k * cos(2*pi*fm*t))
    """
    A0 = 100.0
    depth = 0.1  # 10%
    fm = 5.0  # 5 Hz modulation
    fs = 10_000.0

    sc = G3_E10_AM_Modulation(fs_hz=fs, T_s=1.0, A0=A0, mod_depth=depth, mod_freq_hz=fm)
    out = sc.run()

    # 1. Validar Frecuencia Portadora (Debe ser constante en AM pura)
    assert np.all(out.f_true == 60.0), "Carrier frequency should not change in AM"

    # 2. Validar Envolvente (Amplitude Array)
    # t=0 -> cos(0)=1 -> A = A0 * 1.1
    assert out.A[0] == pytest.approx(A0 * 1.1, rel=1e-9)

    # t = 1/(2*fm) = 0.1s -> cos(pi)=-1 -> A = A0 * 0.9
    idx_valley = int((1.0 / (2.0 * fm)) * fs)
    # Usamos t real para precisión
    t_val = out.t[idx_valley]
    expected_val = A0 * (1.0 + depth * np.cos(2.0 * np.pi * fm * t_val))

    assert out.A[idx_valley] == pytest.approx(expected_val, rel=1e-9)


# =============================================================================
# INTEGRATION TEST: Monte Carlo AM Sweep
# =============================================================================


def test_G3_E10_MonteCarlo_AM_Sweep():
    """
    Barrido Monte Carlo variando Frecuencia de Modulación y Profundidad.
    Esto simula pruebas de ancho de banda (Bandwidth Testing).
    """
    N_ITER = 500
    fs_test = 5000.0

    # Rangos de prueba
    fm_range = (0.1, 5.0)  # 0.1 Hz a 5 Hz
    depth_range = (0.01, 0.20)  # 1% a 20%

    sc = G3_E10_AM_Modulation(
        fs_hz=fs_test, T_s=1.0
    )  # 1s es suficiente para ver fm=5Hz

    req_fm = []

    for i in range(N_ITER):
        rng = np.random.default_rng(seed=i)
        target_fm = rng.uniform(*fm_range)
        target_depth = rng.uniform(*depth_range)

        # Tuning Adapter (Alias check: 'am_freq', 'am_depth')
        sc.set_montecarlo_tuning(
            {"am_freq": target_fm, "am_depth": target_depth, "seed": i}
        )

        out = sc.run()

        # Validación
        # Verificamos un punto aleatorio en el tiempo para asegurar que la formula
        # se actualizó con los parámetros nuevos.
        idx_check = int(0.5 * fs_test)
        t_check = out.t[idx_check]
        A_meas = out.A[idx_check]

        A_exp = 1.0 * (1.0 + target_depth * np.cos(2.0 * np.pi * target_fm * t_check))

        assert A_meas == pytest.approx(
            A_exp, rel=1e-9
        ), f"Iter {i}: AM Envelope mismatch."

        req_fm.append(target_fm)

    # Cobertura
    arr = np.array(req_fm)
    assert np.min(arr) < (fm_range[0] + 0.5)
    assert np.max(arr) > (fm_range[1] - 0.5)
