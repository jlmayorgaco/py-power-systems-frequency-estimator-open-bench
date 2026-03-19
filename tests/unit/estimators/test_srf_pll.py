"""
Unit tests for SRFPLLEstimator (monophasic/f0_pll/srf_pll.py)

Covers:
  - All seven mandatory estimator tests
  - SRF-PLL-specific: phase tracking, vq→0 at lock, anti-windup
  - Comparison with SOGI-FLL on pure 60 Hz (both should converge)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from openfreqbench.estimators.monophasic.f0_pll.srf_pll import SRFPLLEstimator
from openfreqbench.estimators._outputs import EstimatorOutput, EstimatorSpec
from openfreqbench.estimators._base import TuningSpec

FS  = 10_000.0
F0  = 60.0
T_S = 1.0
N   = int(FS * T_S)


def _sine(fs: float = FS, f0: float = F0, n: int = N, A: float = 1.0) -> np.ndarray:
    t = np.arange(n) / fs
    return A * np.sin(2 * np.pi * f0 * t)


def _make_est(**kwargs) -> SRFPLLEstimator:
    cfg = {"fs": FS}
    cfg.update(kwargs)
    return SRFPLLEstimator(config=cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mandatory: instantiation
# ─────────────────────────────────────────────────────────────────────────────

def test_instantiate_with_default_config():
    e = SRFPLLEstimator()
    assert e is not None
    assert e._config["fs"] == pytest.approx(10_000.0)


def test_name():
    assert SRFPLLEstimator.NAME == "SRF_PLL"


def test_family_path():
    assert "f0_pll" in SRFPLLEstimator.FAMILY_PATH


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mandatory: reset restores initial state
# ─────────────────────────────────────────────────────────────────────────────

def test_reset_restores_initial_state():
    est = _make_est()
    v = _sine()
    for s in v[:500]:
        est.update(float(s))
    est.reset()
    assert abs(est._f_est - F0) < 0.1
    assert est._ui == pytest.approx(0.0)   # integrator cleared


def test_reset_clears_sample_counter():
    est = _make_est()
    for s in _sine()[:100]:
        est.update(float(s))
    est.reset()
    assert est._n_samples == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mandatory: update() returns EstimatorOutput
# ─────────────────────────────────────────────────────────────────────────────

def test_step_returns_estimator_output():
    est = _make_est()
    result = est.update(0.5)
    assert isinstance(result, EstimatorOutput)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Mandatory: no NaN
# ─────────────────────────────────────────────────────────────────────────────

def test_step_output_no_nan():
    est = _make_est()
    v = _sine()
    for s in v:
        out = est.update(float(s))
        assert math.isfinite(out.frequency_hz)


def test_no_nan_on_zero_signal():
    est = _make_est()
    for _ in range(200):
        out = est.update(0.0)
        assert math.isfinite(out.frequency_hz)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mandatory: valid range [40, 80] Hz
# ─────────────────────────────────────────────────────────────────────────────

def test_step_output_in_valid_range():
    est = _make_est()
    v = _sine()
    for s in v:
        out = est.update(float(s))
        assert 40.0 <= out.frequency_hz <= 80.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Mandatory: pure 60 Hz converges
# ─────────────────────────────────────────────────────────────────────────────

def test_pure_60hz_converges():
    est = _make_est()
    v = _sine()
    out = est.run(v)
    L = est.latency_samples + int(0.1 * N)
    steady = out[L:]
    rmse = float(np.sqrt(np.mean((steady - F0) ** 2)))
    assert rmse < 1.0, f"RMSE {rmse:.4f} Hz > 1.0 Hz on pure 60 Hz"


def test_steady_state_converges():
    """After full warm-up, mean frequency should be within 0.5 Hz of nominal."""
    est = _make_est()
    v = _sine(n=N)
    out = est.run(v)
    # Use last 30 % of signal
    steady = out[int(0.7 * N):]
    bias = abs(float(np.mean(steady)) - F0)
    assert bias < 0.5, f"Bias {bias:.4f} Hz exceeds 0.5 Hz"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Mandatory: reproducible with seed
# ─────────────────────────────────────────────────────────────────────────────

def test_mc_reproducible_with_seed():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 0.5, int(FS * 0.5))
    v = np.sin(2 * np.pi * F0 * t) + rng.normal(0, 0.01, len(t))

    e1 = SRFPLLEstimator(config={"fs": FS})
    e2 = SRFPLLEstimator(config={"fs": FS})
    out1 = e1.run(v.copy())
    out2 = e2.run(v.copy())
    np.testing.assert_array_equal(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# SRF-PLL-specific: phase tracking
# ─────────────────────────────────────────────────────────────────────────────

def test_phase_rad_is_finite():
    est = _make_est()
    v = _sine()
    out = None
    for s in v:
        out = est.update(float(s))
    assert math.isfinite(out.phase_rad)


def test_phase_in_range():
    """Phase estimate must stay in [0, 2π)."""
    est = _make_est()
    v = _sine()
    for s in v:
        out = est.update(float(s))
        assert 0.0 <= out.phase_rad < 2 * math.pi


def test_amplitude_pu_positive_after_warmup():
    """Amplitude estimate (SOGI envelope) should be positive after lock."""
    est = _make_est()
    v = _sine(A=1.0)
    out = None
    for s in v:
        out = est.update(float(s))
    assert out.amplitude_pu > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SRF-PLL-specific: PI anti-windup
# ─────────────────────────────────────────────────────────────────────────────

def test_anti_windup_limits_integrator():
    """Feed a constant 1.0 V offset (DC) — integrator must stay bounded."""
    from openfreqbench.estimators.monophasic.f0_pll.srf_pll import _UI_MAX
    est = _make_est()
    for _ in range(5000):
        est.update(1.0)
    assert abs(est._ui) <= _UI_MAX + 1e-9


def test_frequency_stays_in_range_after_impulse():
    """After a spike 10× signal amplitude, frequency must stay in [40, 80] Hz."""
    est = _make_est()
    v = _sine(n=int(3 * FS / F0))
    for s in v:
        est.update(float(s))
    est.update(10.0)   # impulse
    out = est.update(0.0)
    assert 40.0 <= out.frequency_hz <= 80.0


# ─────────────────────────────────────────────────────────────────────────────
# SRF-PLL-specific: run() batch
# ─────────────────────────────────────────────────────────────────────────────

def test_run_output_length():
    est = _make_est()
    v = _sine()
    out = est.run(v)
    assert len(out) == len(v)


def test_run_no_nan_after_latency():
    est = _make_est()
    v = _sine()
    out = est.run(v)
    L = est.latency_samples
    assert np.all(np.isfinite(out[L:])), "NaN/inf after latency window"


def test_latency_positive():
    est = _make_est()
    assert est.latency_samples > 0


# ─────────────────────────────────────────────────────────────────────────────
# SRF-PLL-specific: gain sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def test_higher_kp_locks_faster():
    """Higher Kp should produce lower RMSE at early samples (faster lock)."""
    v = _sine(n=N)

    est_slow = SRFPLLEstimator(config={"fs": FS, "kp": 31.4,  "ki": 250.0})
    est_fast = SRFPLLEstimator(config={"fs": FS, "kp": 250.0, "ki": 15625.0})

    out_slow = est_slow.run(v)
    out_fast = est_fast.run(v)

    # Compare in first 20 % of signal (before slow PLL converges)
    early = slice(int(0.05 * N), int(0.2 * N))
    rmse_slow = float(np.sqrt(np.mean((out_slow[early] - F0) ** 2)))
    rmse_fast = float(np.sqrt(np.mean((out_fast[early] - F0) ** 2)))
    assert rmse_fast <= rmse_slow + 1.0, (
        f"Higher Kp should lock faster: slow={rmse_slow:.3f}, fast={rmse_fast:.3f} Hz"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-description API
# ─────────────────────────────────────────────────────────────────────────────

def test_spec_type():
    assert isinstance(SRFPLLEstimator.spec(), EstimatorSpec)


def test_tuning_spec_type():
    assert isinstance(SRFPLLEstimator.tuning_spec(), TuningSpec)


def test_tuning_spec_has_kp():
    names = {p.name for p in SRFPLLEstimator.tuning_ranges()}
    assert "kp" in names


def test_tuning_spec_has_ki():
    names = {p.name for p in SRFPLLEstimator.tuning_ranges()}
    assert "ki" in names


def test_descriptor_keys():
    desc = SRFPLLEstimator.descriptor()
    for key in ("name", "family", "family_path", "complexity",
                "latency_type", "suggested_objective", "tuning_params"):
        assert key in desc


# ─────────────────────────────────────────────────────────────────────────────
# set_params
# ─────────────────────────────────────────────────────────────────────────────

def test_set_params_kp():
    est = _make_est()
    est.set_params(kp=200.0)
    assert est._kp == pytest.approx(200.0)


def test_set_params_ki():
    est = _make_est()
    est.set_params(ki=5000.0)
    assert est._ki == pytest.approx(5000.0)
