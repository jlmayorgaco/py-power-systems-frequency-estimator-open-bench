from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G2_E6_Freq_Step_60_to_59p5 import G2_E6_Freq_Step_60_to_59p5

# =============================================================================
# UNIT TESTS: Physics
# =============================================================================


def test_G2_E6_Default_Behavior():
    """
    Verifica que por defecto ocurra una caída de -0.5 Hz (60.0 -> 59.5).
    """
    t_step = 2.0
    fs = 10_000.0
    sc = G2_E6_Freq_Step_60_to_59p5(
        fs_hz=fs,
        T_s=4.0,
        step_time_s=t_step,
        # step_size_hz default is -0.5
    )
    out = sc.run()

    # Verificar frecuencias
    idx_pre = int((t_step - 0.1) * fs)
    idx_post = int((t_step + 0.1) * fs)

    assert out.f_true[idx_pre] == 60.0
    assert out.f_true[idx_post] == 59.5, "Frequency did not drop to 59.5 Hz"


def test_G2_E6_Phase_Continuity_UnderFrequency():
    """
    Verifica que durante una CAÍDA de frecuencia, la fase siga siendo continua.
    La pendiente de la fase debe disminuir, pero no haber un salto vertical.
    """
    f0 = 60.0
    step_hz = -2.0  # Caída severa a 58Hz para exagerar el efecto
    t_step = 1.0
    fs = 10_000.0

    sc = G2_E6_Freq_Step_60_to_59p5(
        fs_hz=fs, T_s=2.0, f_nom_hz=f0, step_time_s=t_step, step_size_hz=step_hz
    )
    out = sc.run()

    # Análisis de discontinuidad en t_step
    idx_step = int(t_step * fs)

    phi_pre = out.phi[idx_step - 1]
    phi_at = out.phi[idx_step]

    delta_phi = phi_at - phi_pre

    # El cambio esperado es proporcional a la frecuencia en ese instante
    # delta ~ 2*pi * f * dt
    expected_delta = 2.0 * np.pi * f0 * (1.0 / fs)

    # Si hubiera discontinuidad (reinicio de fase), el error sería > 1.0 radianes
    assert delta_phi == pytest.approx(
        expected_delta, rel=0.1
    ), "Phase Jump detected during under-frequency step!"


# =============================================================================
# INTEGRATION TEST: Monte Carlo Sweep
# =============================================================================


def test_G2_E6_MonteCarlo_Negative_Sweep():
    """
    Barrido Monte Carlo enfocado en eventos de BAJA frecuencia (Shedding).
    Probamos rangos de caída desde -0.1 Hz hasta -3.0 Hz (Colapso).
    """
    N_ITER = 500
    fs_test = 5000.0

    # Rangos para UFLS (Under Frequency Load Shedding)
    t_range = (0.2, 0.8)
    # Drops negativos exclusivamente
    drop_range = (-3.0, -0.1)

    sc = G2_E6_Freq_Step_60_to_59p5(fs_hz=fs_test, T_s=1.0)

    req_drops = []

    for i in range(N_ITER):
        rng = np.random.default_rng(seed=i)
        target_t = rng.uniform(*t_range)
        target_drop = rng.uniform(*drop_range)

        # Tuning: 'step_hz' recibe valores negativos
        sc.set_montecarlo_tuning(
            {"step_t": target_t, "step_hz": target_drop, "seed": i}
        )

        out = sc.run()

        # Validación
        idx_check = int((target_t + 0.05) * fs_test)
        f_meas = out.f_true[idx_check]

        f_exp = 60.0 + target_drop

        assert f_meas == pytest.approx(
            f_exp, rel=1e-9
        ), f"Iter {i}: Frequency drop mismatch. Got {f_meas}, Expected {f_exp}"

        req_drops.append(target_drop)

    # Cobertura
    arr_drops = np.array(req_drops)
    assert np.min(arr_drops) < (drop_range[0] + 0.1)
    assert np.max(arr_drops) > (drop_range[1] - 0.1)
