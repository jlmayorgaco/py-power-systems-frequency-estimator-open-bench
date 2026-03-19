from __future__ import annotations
import numpy as np
import pytest

from pfebench.scenarios.G4_E16_Composite_Islanding import G4_E16_Composite_Islanding

# =============================================================================
# UNIT TESTS: Multi-Physics Verification
# =============================================================================


def test_G4_E16_Physics_Coherence():
    """
    Verifica que los tres fenómenos (Frecuencia, Voltaje, Fase) ocurran
    simultáneamente en t_island.
    """
    fs = 10_000.0
    t_ev = 1.0

    # Configuramos un evento muy notorio
    rocof = 5.0  # +5 Hz/s
    v_drop = -0.5  # -50% Voltaje
    p_jump = 45.0  # +45 grados

    sc = G4_E16_Composite_Islanding(
        fs_hz=fs,
        T_s=2.0,
        island_time_s=t_ev,
        island_rocof_hz_s=rocof,
        island_volt_chg_pct=v_drop,
        island_phase_jump_deg=p_jump,
        noise_level_rel=0.0,  # Sin ruido para validar física exacta
    )
    out = sc.run()

    # Índices Pre y Post evento
    idx_pre = int((t_ev - 0.01) * fs)
    idx_post = int((t_ev + 0.01) * fs)

    # 1. Validar Voltaje (Escalón)
    assert out.A[idx_pre] == pytest.approx(1.0)
    assert out.A[idx_post] == pytest.approx(0.5)  # 1.0 * (1 - 0.5)

    # 2. Validar Frecuencia (Rampa)
    # En t_post (0.01s después), f debe haber subido rocof * dt
    dt = out.t[idx_post] - t_ev
    f_expected = 60.0 + (rocof * dt)
    assert out.f_true[idx_post] == pytest.approx(f_expected, rel=1e-5)

    # 3. Validar Fase (Salto + Rotación)
    # Este es el difícil.
    # Delta observado = (phi_post - phi_pre)
    # Delta fisico = (Rotacion 60Hz) + (Integral Rampa) + (Salto Evento)

    # Simplificación: Verificamos el salto en la fase "des-rotada"
    # Quitamos la componente 60Hz base
    phi_detrend = out.phi - (2.0 * np.pi * 60.0 * out.t)

    # El salto observado en detrend debe ser aprox Jump + pequeño término cuadrático del ROCOF
    jump_rad_meas = phi_detrend[idx_post] - phi_detrend[idx_pre]
    jump_rad_exp = np.deg2rad(p_jump)

    # Tolerancia holgada (0.05 rad) por el efecto cuadrático de la rampa en ese dt
    assert jump_rad_meas == pytest.approx(jump_rad_exp, abs=0.05)


# =============================================================================
# INTEGRATION TEST: Monte Carlo Composite Sweep
# =============================================================================


def test_G4_E16_MonteCarlo_Composite_Sweep():
    """
    Barrido Complejo: Variamos t, ROCOF, Voltaje y Fase simultáneamente.
    Verificamos que el Tuning Adapter mapee todo correctamente.
    """
    N_ITER = 500
    fs_test = 5000.0

    # Rangos
    t_range = (0.2, 0.8)
    rocof_range = (-2.0, 2.0)
    v_range = (-0.2, 0.2)
    phi_range = (-30.0, 30.0)

    sc = G4_E16_Composite_Islanding(fs_hz=fs_test, T_s=1.0)

    for i in range(N_ITER):
        rng = np.random.default_rng(seed=i)

        # Targets
        tg_t = rng.uniform(*t_range)
        tg_rocof = rng.uniform(*rocof_range)
        tg_v = rng.uniform(*v_range)
        tg_phi = rng.uniform(*phi_range)

        # Tuning Composite
        sc.set_montecarlo_tuning(
            {
                "t_event": tg_t,
                "rocof": tg_rocof,
                "v_step": tg_v,
                "phi_jump": tg_phi,
                "seed": i,
            }
        )

        out = sc.run()

        # --- VALIDACIONES PUNTUALES ---

        # Checkpoint: 0.1s después del evento
        idx_check = int((tg_t + 0.1) * fs_test)

        # 1. Check Voltaje
        # Usamos out.A que es la verdad del terreno (Ground Truth)
        # Nota: out.A no tiene ruido, out.v sí tiene ruido. Usamos A.
        meas_A = out.A[idx_check]
        exp_A = 1.0 * (1.0 + tg_v)
        assert meas_A == pytest.approx(exp_A, rel=1e-5), f"Iter {i}: Voltage fail"

        # 2. Check Frecuencia
        # Corregir alineación temporal
        t_sample = out.t[idx_check]
        meas_f = out.f_true[idx_check]
        exp_f = 60.0 + tg_rocof * (t_sample - tg_t)
        assert meas_f == pytest.approx(exp_f, rel=1e-5), f"Iter {i}: Freq fail"

        # 3. Check Metadata (Schema)
        # Asegurar que se guardó el salto de fase solicitado
        assert out.meta["schema"]["phase_jump_deg"] == pytest.approx(tg_phi)
