from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G3_E12_Phase_Jump import G3_E12_Phase_Jump

# =============================================================================
# UNIT TESTS: Physics
# =============================================================================


def test_G3_E12_Phase_Step_Accuracy():
    """
    Verifica que la fase salte exactamente la magnitud en grados solicitada.
    """
    jump_deg = 45.0
    t_jump = 1.0
    fs = 10_000.0
    f0 = 60.0

    sc = G3_E12_Phase_Jump(
        fs_hz=fs, T_s=2.0, f_nom_hz=f0, jump_time_s=t_jump, jump_size_deg=jump_deg
    )
    out = sc.run()

    # Índices alrededor del evento
    idx_pre = int((t_jump - 0.001) * fs)
    idx_post = int((t_jump + 0.001) * fs)

    # Medir fase
    phi_pre = out.phi[idx_pre]
    phi_post = out.phi[idx_post]

    # Calcular delta teórico debido a la rotación normal (f0)
    # Entre pre y post hay un pequeño delta t
    dt_samples = (idx_post - idx_pre) / fs
    delta_rotation = 2.0 * np.pi * f0 * dt_samples

    # El salto total observado es Rotación + Salto de Evento
    observed_diff = phi_post - phi_pre
    event_jump_rad = observed_diff - delta_rotation

    # Convertir a grados
    event_jump_deg = np.rad2deg(event_jump_rad)

    # Validar
    assert event_jump_deg == pytest.approx(
        jump_deg, rel=1e-5
    ), "Phase jump magnitude incorrect."


def test_G3_E12_Frequency_Is_Constant():
    """
    En este benchmark, la 'frecuencia verdadera' se define constante
    (aunque físicamente haya un Dirac delta). Esto sirve de referencia.
    """
    sc = G3_E12_Phase_Jump(jump_size_deg=90.0)
    out = sc.run()

    # f_true debe ser plano
    assert np.all(out.f_true == 60.0)


# =============================================================================
# INTEGRATION TEST: Monte Carlo Sweep
# =============================================================================


def test_G3_E12_MonteCarlo_Phase_Sweep():
    """
    Barrido Monte Carlo variando el ángulo del salto.
    Desde saltos pequeños (1 deg) hasta grandes (180 deg).
    """
    N_ITER = 500
    fs_test = 5000.0

    # Rangos
    t_range = (0.2, 0.8)
    deg_range = (-180.0, 180.0)  # Saltos en todo el círculo

    sc = G3_E12_Phase_Jump(fs_hz=fs_test, T_s=1.0)

    req_degs = []

    for i in range(N_ITER):
        rng = np.random.default_rng(seed=i)
        target_t = rng.uniform(*t_range)
        target_deg = rng.uniform(*deg_range)

        # Tuning Adapter Check ('jump_t', 'jump_deg')
        sc.set_montecarlo_tuning(
            {"jump_t": target_t, "jump_deg": target_deg, "seed": i}
        )

        out = sc.run()

        # Validación
        # Para evitar calcular la rotación manual, comparamos contra un caso base sin salto?
        # Mejor: Calculamos el salto directamente en la fase "des-rotada"
        # Phase_detrended = phi - 2*pi*f*t

        phi_detrend = out.phi - (2.0 * np.pi * 60.0 * out.t)

        # Medimos antes y después en la fase sin tendencia
        idx_pre = int((target_t - 0.01) * fs_test)
        idx_post = int((target_t + 0.01) * fs_test)

        # Como quitamos la rotación de 60Hz, la diferencia debe ser solo el salto + phi0
        val_pre = phi_detrend[idx_pre]
        val_post = phi_detrend[idx_post]

        meas_jump_deg = np.rad2deg(val_post - val_pre)

        assert meas_jump_deg == pytest.approx(
            target_deg, abs=0.1
        ), f"Iter {i}: Phase jump mismatch."

        req_degs.append(target_deg)

    # Cobertura
    arr = np.array(req_degs)
    assert np.min(arr) < (deg_range[0] + 10.0)
    assert np.max(arr) > (deg_range[1] - 10.0)
