from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G2_E4_Voltage_Mag_Step_1pct import G2_E4_Voltage_Mag_Step_1pct

# =============================================================================
# UNIT TESTS: Basic Physics & Logic
# =============================================================================


def test_magnitude_step_accuracy():
    """
    Verifies that the amplitude changes exactly at step_time_s
    and by the exact step_pct amount.
    """
    # Setup: 10V base, 50% step up at 1.0s
    A0_val = 10.0
    step_pct_val = 0.5
    step_time = 1.0

    sc = G2_E4_Voltage_Mag_Step_1pct(
        fs_hz=10_000.0,
        T_s=2.0,
        A0=A0_val,
        step_time_s=step_time,
        step_pct=step_pct_val,
        f_nom_hz=60.0,  # High freq to ensure peaks in both windows
    )
    out = sc.run()

    t = out.t
    v = out.v

    # 1. Split signal into Pre-Step and Post-Step
    # Use a small epsilon buffer to avoid edge-case ambiguity at the exact sample
    mask_pre = t < step_time
    mask_post = t >= step_time

    v_pre = v[mask_pre]
    v_post = v[mask_post]

    # 2. Check Amplitudes
    # Since it's a pure sine wave (no noise), Max(Abs(v)) should equal Amplitude exactly.

    # Pre-step amplitude should be A0 (10.0)
    max_pre = np.max(np.abs(v_pre))
    assert max_pre == pytest.approx(
        A0_val, rel=1e-4
    ), f"Pre-step amplitude mismatch. Expected {A0_val}, got {max_pre}"

    # Post-step amplitude should be A0 * (1 + 0.5) = 15.0
    expected_post = A0_val * (1.0 + step_pct_val)
    max_post = np.max(np.abs(v_post))
    assert max_post == pytest.approx(
        expected_post, rel=1e-4
    ), f"Post-step amplitude mismatch. Expected {expected_post}, got {max_post}"


def test_frequency_invariance():
    """
    Ensures that a Magnitude Step does NOT alter the true frequency array.
    """
    sc = G2_E4_Voltage_Mag_Step_1pct(f_nom_hz=50.0)
    out = sc.run()

    # f_true should be constantly 50.0 everywhere
    assert np.all(out.f_true == 50.0)

    # Sanity check: Duration
    assert out.t[-1] > 0


def test_metadata_integrity():
    """Checks schema recording of event details."""
    sc = G2_E4_Voltage_Mag_Step_1pct(step_time_s=3.5, step_pct=0.1)
    out = sc.run()

    meta = out.meta["schema"]
    assert meta["event_type"] == "magnitude_step"
    assert meta["step_time_s"] == 3.5
    assert meta["step_pct"] == 0.1


# =============================================================================
# INTEGRATION TEST: Monte Carlo Event Sweep
# =============================================================================


def test_G2_E4_MonteCarlo_Event_Sweep():
    """
    Monte Carlo Sweep para verificar la configuración dinámica de eventos.

    Objetivo:
    Validar que 'step_t' y 'step_pct' mapeen correctamente a través del adapter
    y alteren la señal física en el momento y magnitud solicitados.
    """
    N_ITER = 500

    # Definimos rangos para el barrido
    # T_s será 1.0s para el test, barremos el evento entre 0.2s y 0.8s
    fs_test = 5000.0
    t_event_range = (0.2, 0.8)
    pct_range = (-0.5, 0.5)  # De -50% (sag) a +50% (swell)

    sc = G2_E4_Voltage_Mag_Step_1pct(fs_hz=fs_test, T_s=1.0, A0=100.0)

    # Historiales para validación de cobertura
    req_times = []
    req_pcts = []

    for i in range(N_ITER):
        # A. Sampling
        rng = np.random.default_rng(seed=i)
        target_t = rng.uniform(*t_event_range)
        target_pct = rng.uniform(*pct_range)

        # B. Tuning (Uso de alias 'step_t' y 'step_pct')
        sc.set_montecarlo_tuning(
            {"step_t": target_t, "step_pct": target_pct, "seed": i}
        )

        # C. Ejecución
        out = sc.run()

        # D. Validación Física Directa (Analizando el array de Amplitud 'A')
        # Accedemos directamente a out.A (gracias al property proxy de state)
        # Esto es más preciso que analizar 'v' porque 'A' es la "verdad física".
        A_arr = out.A

        # Definimos índices "antes" y "después" del evento con un margen de seguridad
        # para evitar problemas de redondeo en el índice exacto.
        idx_pre = int((target_t - 0.05) * fs_test)
        idx_post = int((target_t + 0.05) * fs_test)

        val_pre = A_arr[idx_pre]
        val_post = A_arr[idx_post]

        # D1. Validar Pre-Step (Debe ser A0 base, que es 100.0)
        assert val_pre == pytest.approx(
            100.0, rel=1e-9
        ), f"Iter {i}: Pre-step amplitude wrong at t={target_t}"

        # D2. Validar Post-Step (Debe ser A0 * (1+pct))
        expected_post = 100.0 * (1.0 + target_pct)
        assert val_post == pytest.approx(
            expected_post, rel=1e-9
        ), f"Iter {i}: Post-step amplitude wrong. Got {val_post}, expected {expected_post}"

        # Guardar para estadística
        req_times.append(target_t)
        req_pcts.append(target_pct)

    # --- Validaciones de Cobertura (Statistical Relevance) ---
    arr_t = np.array(req_times)
    arr_pct = np.array(req_pcts)

    # Validar que barrimos el tiempo correctamente
    assert np.min(arr_t) < (t_event_range[0] + 0.05)
    assert np.max(arr_t) > (t_event_range[1] - 0.05)

    # Validar que barrimos las magnitudes (Sags y Swells)
    assert np.min(arr_pct) < (pct_range[0] + 0.05)
    assert np.max(arr_pct) > (pct_range[1] - 0.05)
