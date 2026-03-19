from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G2_E5_Voltage_Mag_Step_10pct import G2_E5_Voltage_Mag_Step_10pct

# =============================================================================
# UNIT TESTS: Physics & Defaults
# =============================================================================


def test_G2_E5_Default_10pct_Step():
    """
    Verifica que por defecto el escenario genere un cambio exacto del 10%.
    """
    A0_val = 100.0
    t_step = 1.0

    sc = G2_E5_Voltage_Mag_Step_10pct(
        fs_hz=10_000.0,
        T_s=2.0,
        A0=A0_val,
        step_time_s=t_step,
        # step_pct usa default 0.10
    )
    out = sc.run()

    # Verificar amplitud antes y después
    idx_pre = int((t_step - 0.1) * 10_000)
    idx_post = int((t_step + 0.1) * 10_000)

    amp_pre = out.A[idx_pre]
    amp_post = out.A[idx_post]

    # Pre debe ser 100
    assert amp_pre == pytest.approx(100.0)

    # Post debe ser 110 (100 * 1.10)
    assert amp_post == pytest.approx(110.0)

    # Metadata check
    assert out.meta["schema"]["step_pct"] == 0.10


# =============================================================================
# INTEGRATION TEST: Monte Carlo Sweep
# =============================================================================


def test_G2_E5_MonteCarlo_Sweep_Large_Steps():
    """
    Monte Carlo Sweep variando el escalón en rangos amplios (Sags y Swells).
    Prueba que el Adapter 'step_pct' funcione para cambios positivos y negativos.
    """
    N_ITER = 500
    fs_test = 5000.0

    # Rangos para el sweep
    t_range = (0.2, 0.8)
    # Probamos desde -20% (Sag severo) hasta +20% (Swell severo)
    pct_range = (-0.20, 0.20)

    sc = G2_E5_Voltage_Mag_Step_10pct(fs_hz=fs_test, T_s=1.0, A0=100.0)

    req_pcts = []

    for i in range(N_ITER):
        rng = np.random.default_rng(seed=i)
        target_t = rng.uniform(*t_range)
        target_pct = rng.uniform(*pct_range)

        # Tuning Adapter
        sc.set_montecarlo_tuning(
            {"step_t": target_t, "step_pct": target_pct, "seed": i}
        )

        out = sc.run()

        # Validación Física
        # Usamos out.A para máxima precisión
        idx_check = int((target_t + 0.05) * fs_test)
        amp_measured = out.A[idx_check]

        amp_expected = 100.0 * (1.0 + target_pct)

        assert amp_measured == pytest.approx(
            amp_expected, rel=1e-9
        ), f"Iter {i}: Amplitude Step mismatch."

        req_pcts.append(target_pct)

    # Cobertura
    arr_pct = np.array(req_pcts)
    assert np.min(arr_pct) < (pct_range[0] + 0.05)
    assert np.max(arr_pct) > (pct_range[1] - 0.05)
