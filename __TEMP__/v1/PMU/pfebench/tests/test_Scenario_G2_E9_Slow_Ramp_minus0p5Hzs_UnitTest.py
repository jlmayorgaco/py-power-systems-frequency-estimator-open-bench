from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G2_E9_Slow_Ramp_minus0p5Hzs import G2_E9_Slow_Ramp_minus0p5Hzs

# =============================================================================
# UNIT TESTS: Physics & Defaults
# =============================================================================


def test_G2_E9_Default_Negative_Slope():
    """
    Verifica que por defecto la rampa sea negativa (-0.5 Hz/s).
    """
    f0 = 60.0
    t_start = 1.0
    fs = 10_000.0

    sc = G2_E9_Slow_Ramp_minus0p5Hzs(
        fs_hz=fs,
        T_s=3.0,
        f_nom_hz=f0,
        ramp_start_time_s=t_start,
        # Default ramp_rate_hz_s is -0.5
    )
    out = sc.run()

    # Check 1.0s after start
    dt = 1.0
    check_time = t_start + dt
    idx = int(check_time * fs)

    t_actual = out.t[idx]
    f_meas = out.f_true[idx]

    # Expected: 60 - 0.5 * (t_actual - 1.0)
    expected_f = 60.0 + (-0.5 * (t_actual - t_start))

    assert f_meas == pytest.approx(expected_f, rel=1e-9)
    assert f_meas < f0, "Frequency should be decreasing"


# =============================================================================
# INTEGRATION TEST: Monte Carlo Slow Sweep
# =============================================================================


def test_G2_E9_MonteCarlo_Slow_Sweep():
    """
    Barrido Monte Carlo para rampas lentas (Slow Drifts).
    Probamos pendientes pequeñas entre -1.0 Hz/s y +1.0 Hz/s.
    """
    N_ITER = 500
    fs_test = 5000.0

    t_range = (0.2, 0.8)
    # Rango de rampas lentas
    slope_range = (-1.0, 1.0)

    sc = G2_E9_Slow_Ramp_minus0p5Hzs(fs_hz=fs_test, T_s=2.0)

    req_slopes = []

    for i in range(N_ITER):
        rng = np.random.default_rng(seed=i)
        target_t = rng.uniform(*t_range)
        target_slope = rng.uniform(*slope_range)

        # Tuning
        sc.set_montecarlo_tuning(
            {"ramp_t": target_t, "ramp_slope": target_slope, "seed": i}
        )

        out = sc.run()

        # Validación: Medir 0.5s después del inicio
        check_time = target_t + 0.5
        idx_check = int(check_time * fs_test)

        # CORRECCIÓN DE ALINEACIÓN TEMPORAL (Crucial para tests de rampa)
        t_sample_actual = out.t[idx_check]
        f_meas = out.f_true[idx_check]

        f_exp = 60.0 + target_slope * (t_sample_actual - target_t)

        assert f_meas == pytest.approx(
            f_exp, rel=1e-9
        ), f"Iter {i}: Slow Ramp slope incorrect."

        req_slopes.append(target_slope)

    # Cobertura
    arr = np.array(req_slopes)
    assert np.min(arr) < (slope_range[0] + 0.1)
    assert np.max(arr) > (slope_range[1] - 0.1)
