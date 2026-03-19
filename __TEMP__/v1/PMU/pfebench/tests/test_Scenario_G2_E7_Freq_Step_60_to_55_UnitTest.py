from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G2_E7_Freq_Step_60_to_55 import G2_E7_Freq_Step_60_to_55

# =============================================================================
# UNIT TESTS: Severe Physics
# =============================================================================


def test_G2_E7_Default_Collapse():
    """
    Verifica que el escenario por defecto genere una caída masiva a 55 Hz.
    """
    t_step = 2.0
    fs = 10_000.0
    sc = G2_E7_Freq_Step_60_to_55(
        fs_hz=fs,
        T_s=4.0,
        step_time_s=t_step,
        # step_size_hz default is -5.0
    )
    out = sc.run()

    # Verificar frecuencias
    idx_pre = int((t_step - 0.1) * fs)
    idx_post = int((t_step + 0.1) * fs)

    assert out.f_true[idx_pre] == 60.0
    assert out.f_true[idx_post] == 55.0, "Frequency did not collapse to 55.0 Hz"


def test_G2_E7_Phase_Continuity_Severe_Drop():
    """
    Verifica continuidad de fase ante un cambio brusco de -5 Hz.
    Un cambio tan grande podría revelar errores en la integración numérica si no es robusta.
    """
    f0 = 60.0
    step_hz = -5.0
    t_step = 1.0
    fs = 10_000.0

    sc = G2_E7_Freq_Step_60_to_55(
        fs_hz=fs, T_s=2.0, f_nom_hz=f0, step_time_s=t_step, step_size_hz=step_hz
    )
    out = sc.run()

    # Análisis de continuidad
    idx_step = int(t_step * fs)
    phi_pre = out.phi[idx_step - 1]
    phi_at = out.phi[idx_step]

    delta_phi = phi_at - phi_pre

    # Delta esperado basado en f instantánea (aprox f0)
    expected_delta = 2.0 * np.pi * f0 * (1.0 / fs)

    # Tolerancia relativa del 15% (el cambio es masivo, el error de 2do orden aumenta)
    assert delta_phi == pytest.approx(
        expected_delta, rel=0.15
    ), "Phase discontinuity in severe collapse scenario!"


# =============================================================================
# INTEGRATION TEST: Monte Carlo Sweep
# =============================================================================


def test_G2_E7_MonteCarlo_Collapse_Sweep():
    """
    Barrido Monte Carlo para eventos extremos (Severe frequency deviations).
    Probamos caídas desde -3 Hz hasta -10 Hz (Colapso total).
    """
    N_ITER = 500
    fs_test = 5000.0

    # Rangos Extremos
    t_range = (0.2, 0.8)
    drop_range = (-10.0, -3.0)  # Caídas masivas

    sc = G2_E7_Freq_Step_60_to_55(fs_hz=fs_test, T_s=1.0)

    req_drops = []

    for i in range(N_ITER):
        rng = np.random.default_rng(seed=i)
        target_t = rng.uniform(*t_range)
        target_drop = rng.uniform(*drop_range)

        # Tuning
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
        ), f"Iter {i}: Collapse magnitude mismatch."

        req_drops.append(target_drop)

    # Cobertura
    arr_drops = np.array(req_drops)
    assert np.min(arr_drops) < (drop_range[0] + 0.5)
    assert np.max(arr_drops) > (drop_range[1] - 0.5)
