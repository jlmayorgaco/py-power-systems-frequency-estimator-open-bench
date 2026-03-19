from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G2_E8_Fast_Ramp_plus5Hzs import G2_E8_Fast_Ramp_plus5Hzs

# =============================================================================
# UNIT TESTS: Ramp Physics
# =============================================================================


def test_G2_E8_Ramp_Linearity():
    """
    Verifica que la frecuencia aumente linealmente según la tasa especificada.
    Si ROCOF = 5 Hz/s, después de 1 segundo la f debe haber subido 5 Hz.
    """
    f0 = 60.0
    rate = 5.0
    t_start = 1.0
    fs = 10_000.0

    sc = G2_E8_Fast_Ramp_plus5Hzs(
        fs_hz=fs, T_s=3.0, f_nom_hz=f0, ramp_start_time_s=t_start, ramp_rate_hz_s=rate
    )
    out = sc.run()

    # 1. Antes de la rampa: Frecuencia Nominal
    idx_pre = int((t_start - 0.1) * fs)
    assert out.f_true[idx_pre] == f0

    # 2. Durante la rampa (0.5s después del inicio)
    dt_check = 0.5
    # IMPORTANTE: Usamos el tiempo real de la muestra para validar
    idx_check = int((t_start + dt_check) * fs)
    t_actual = out.t[idx_check]

    # Calculamos f esperado usando el tiempo REAL de esa muestra
    expected_f = f0 + (rate * (t_actual - t_start))

    assert out.f_true[idx_check] == pytest.approx(
        expected_f, rel=1e-9
    ), "Frequency did not ramp correctly at t=1.5s"


def test_G2_E8_Phase_Smoothness():
    """
    En una rampa, la frecuencia es continua.
    Verificamos que la integral numérica funcione sin saltos bruscos.
    """
    sc = G2_E8_Fast_Ramp_plus5Hzs(fs_hz=10_000.0, T_s=2.0, ramp_rate_hz_s=10.0)
    out = sc.run()

    phase_diff = np.diff(out.phi)
    # El salto máximo entre muestras no debe ser absurdo
    assert np.max(np.abs(phase_diff)) < 0.1, "Phase integration unstable"


# =============================================================================
# INTEGRATION TEST: Monte Carlo ROCOF Sweep
# =============================================================================


def test_G2_E8_MonteCarlo_ROCOF_Sweep():
    """
    Barrido Monte Carlo variando la pendiente de la rampa (ROCOF).
    Probamos rampas positivas y negativas.
    """
    N_ITER = 500
    fs_test = 5000.0

    # Rangos
    t_range = (0.2, 0.8)
    slope_range = (-5.0, 5.0)

    sc = G2_E8_Fast_Ramp_plus5Hzs(fs_hz=fs_test, T_s=2.0)

    req_slopes = []

    for i in range(N_ITER):
        rng = np.random.default_rng(seed=i)
        target_t = rng.uniform(*t_range)
        target_slope = rng.uniform(*slope_range)

        # Tuning: 'ramp_t' y 'ramp_slope'
        sc.set_montecarlo_tuning(
            {"ramp_t": target_t, "ramp_slope": target_slope, "seed": i}
        )

        out = sc.run()

        # Validación: Medir 0.5s después del inicio teórico
        check_time = target_t + 0.5
        idx_check = int(check_time * fs_test)

        # --- CORRECCIÓN CLAVE AQUÍ ---
        # 1. Obtenemos el tiempo discreto REAL de la muestra seleccionada
        t_sample_actual = out.t[idx_check]

        # 2. Obtenemos el valor medido
        f_meas = out.f_true[idx_check]

        # 3. Calculamos el valor esperado usando t_sample_actual
        # f(t) = f0 + slope * (t_actual - t_start)
        f_exp = 60.0 + target_slope * (t_sample_actual - target_t)

        assert f_meas == pytest.approx(
            f_exp, rel=1e-9
        ), f"Iter {i}: Ramp slope incorrect. Got {f_meas}, Expected {f_exp}"

        req_slopes.append(target_slope)

    # Cobertura
    arr = np.array(req_slopes)
    assert np.min(arr) < (slope_range[0] + 0.5)
    assert np.max(arr) > (slope_range[1] - 0.5)
