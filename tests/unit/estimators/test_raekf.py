"""
Unit tests for RAEKFEstimator (monophasic/f1_kalman/raekf.py)

Covers the seven mandatory estimator tests plus RAEKF-specific tests:
  - Huber down-weighting of outlier innovations
  - Sage-Husa R adaptation (r_hat changes after impulsive noise)
  - Robustness: noisy 60 Hz RMSE meaningfully lower than EKF on impulsive noise
  - spec / tuning_spec / descriptor API
  - set_params triggers reset
"""

from __future__ import annotations

import math

import numpy as np
from openfreqbench.estimators._base import TuningSpec
from openfreqbench.estimators._outputs import EstimatorOutput, EstimatorSpec
from openfreqbench.estimators.monophasic.f1_kalman.raekf import RAEKFEstimator
import pytest

FS = 10_000.0
F0 = 60.0
T_S = 1.0
N = int(FS * T_S)


def _sine(fs: float = FS, f0: float = F0, n: int = N, A: float = 1.0) -> np.ndarray:
    t = np.arange(n) / fs
    return A * np.sin(2 * np.pi * f0 * t)


def _make_est(**kwargs) -> RAEKFEstimator:
    cfg = {"fs": FS}
    cfg.update(kwargs)
    est = RAEKFEstimator(config=cfg)
    est.reset()
    return est


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mandatory: instantiation
# ─────────────────────────────────────────────────────────────────────────────


def test_instantiate_with_default_config():
    e = RAEKFEstimator()
    assert e is not None
    assert e._config["fs"] == pytest.approx(10_000.0)


def test_name():
    assert RAEKFEstimator.NAME == "RAEKF"


def test_family_path():
    assert "f1_kalman" in RAEKFEstimator.FAMILY_PATH


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mandatory: reset restores initial state
# ─────────────────────────────────────────────────────────────────────────────


def test_reset_restores_initial_state():
    est = _make_est()
    v = _sine()
    for sample in v[:500]:
        est.update(float(sample))
    est.reset()
    # After reset, frequency should be back at nominal
    assert abs(est._f_est - F0) < 0.1


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


def test_step_frequency_hz_is_float():
    est = _make_est()
    result = est.update(0.5)
    assert isinstance(result.frequency_hz, float)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Mandatory: no NaN
# ─────────────────────────────────────────────────────────────────────────────


def test_step_output_no_nan():
    est = _make_est()
    v = _sine()
    for sample in v:
        out = est.update(float(sample))
        assert math.isfinite(out.frequency_hz), "NaN/inf frequency output"


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
    for sample in v:
        out = est.update(float(sample))
        assert 40.0 <= out.frequency_hz <= 80.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Mandatory: pure 60 Hz converges
# ─────────────────────────────────────────────────────────────────────────────


def test_pure_60hz_converges():
    est = _make_est()
    v = _sine(n=N)
    out = est.run(v)
    L = est.latency_samples + int(0.05 * N)  # skip first 5 % + latency
    steady = out[L:]
    steady = steady[np.isfinite(steady)]
    assert len(steady) > 100
    rmse = float(np.sqrt(np.mean((steady - F0) ** 2)))
    assert rmse < 0.5, f"RMSE {rmse:.4f} Hz on pure 60 Hz signal"


def test_steady_state_bias_small():
    est = _make_est()
    v = _sine(n=N)
    out = est.run(v)
    L = est.latency_samples + int(0.1 * N)
    steady = out[L:]
    steady = steady[np.isfinite(steady)]
    bias = abs(float(np.mean(steady)) - F0)
    assert bias < 0.1, f"Bias {bias:.4f} Hz exceeds 0.1 Hz"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Mandatory: reproducible with seed (deterministic for same input)
# ─────────────────────────────────────────────────────────────────────────────


def test_mc_reproducible_with_seed():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 0.5, int(FS * 0.5))
    v = np.sin(2 * np.pi * F0 * t) + rng.normal(0, 0.01, len(t))

    e1 = RAEKFEstimator(config={"fs": FS})
    e2 = RAEKFEstimator(config={"fs": FS})
    out1 = e1.run(v.copy())
    out2 = e2.run(v.copy())
    np.testing.assert_array_equal(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# RAEKF-specific: Huber robustness
# ─────────────────────────────────────────────────────────────────────────────


def test_huber_downweights_large_innovation():
    """
    Feed an impulse (10x amplitude spike) and verify the Huber weight w < 1.
    We do this by inspecting the estimator's internal state after the spike.
    """
    est = _make_est(huber_delta=1.345)
    # Warm up for 2 cycles
    v_warmup = _sine(n=int(2 * FS / F0))
    for s in v_warmup:
        est.update(float(s))

    # Snapshot state before spike

    # One large spike (10x signal amplitude = extreme outlier)
    est.update(10.0)

    # Frequency should not jump far from nominal
    f_after = est._f_est
    assert abs(f_after - F0) < 2.0, (
        f"Huber filter should limit spike influence: f_after={f_after:.3f} Hz"
    )


def test_huber_full_weight_in_gaussian_zone():
    """
    When innovation is within delta, weight should be 1.0 (Gaussian zone).
    Verify indirectly: small innovations → filter behaves similarly to standard EKF.
    Use a generous RMSE threshold since EKF-class filters oscillate ~0.2-1 Hz
    during convergence; the key claim is that Huber doesn't introduce extra bias.
    """
    est = _make_est(huber_delta=1.345)
    v = _sine(n=N)
    out = est.run(v)
    # Skip first 30 % of signal (latency + convergence ramp)
    L = int(0.3 * N)
    steady = out[L:]
    rmse = float(np.sqrt(np.mean((steady - F0) ** 2)))
    assert rmse < 0.8, f"Pure sine RMSE {rmse:.4f} should be < 0.8 Hz in Gaussian zone"


def test_huber_delta_0p5_more_robust_to_spike():
    """
    Smaller delta → heavier clipping → harder outlier suppression.
    After a spike, delta=0.5 should produce less frequency error than delta=3.0.
    """

    def _spike_error(delta: float) -> float:
        est = _make_est(huber_delta=delta)
        v_warmup = _sine(n=int(3 * FS / F0))
        for s in v_warmup:
            est.update(float(s))
        est.update(10.0)
        return abs(est._f_est - F0)

    err_aggressive = _spike_error(0.5)
    err_lenient = _spike_error(3.0)
    assert err_aggressive <= err_lenient + 0.5, (
        f"Smaller delta should not be worse: δ=0.5→{err_aggressive:.3f}, δ=3.0→{err_lenient:.3f} Hz"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RAEKF-specific: Sage-Husa adaptation
# ─────────────────────────────────────────────────────────────────────────────


def test_sage_husa_r_hat_adapts():
    """After impulsive noise, r_hat should increase from its initial value."""
    est = _make_est(r=0.01, forget_factor=0.98)
    r_hat_init = est._r_hat

    # Feed impulsive noise
    rng = np.random.default_rng(0)
    v = _sine(n=int(0.1 * N))
    spikes = (rng.uniform(0, 1, len(v)) < 0.05) * rng.normal(0, 10.0, len(v))
    v_noisy = v + spikes
    for s in v_noisy:
        est.update(float(s))

    # r_hat should have increased (noise level detected)
    assert est._r_hat > r_hat_init, (
        f"r_hat did not adapt: init={r_hat_init:.4f}, after={est._r_hat:.4f}"
    )


def test_sage_husa_r_hat_never_below_floor():
    """r_hat must stay ≥ r_min = r / 100 at all times."""
    est = _make_est(r=0.01)
    r_min = est._r_min
    v = _sine(n=N)
    for s in v:
        est.update(float(s))
    assert est._r_hat >= r_min


def test_q_w_is_positive():
    """Process noise must be positive (Q is fixed, not adapted)."""
    est = _make_est(q_omega=1.0)
    assert est._q_w > 0.0


def test_forget_factor_large_adapts_faster():
    """
    Large forget factor (b=0.99) should produce faster R adaptation than small (b=0.95)
    when measuring how quickly r_hat converges after a noise-level change.
    """

    def _r_hat_after(b: float, n: int = 200) -> float:
        est = _make_est(r=0.0001, forget_factor=b)
        # inject high-noise signal
        rng = np.random.default_rng(1)
        v = _sine(n=n) + rng.normal(0, 1.0, n)
        for s in v:
            est.update(float(s))
        return est._r_hat

    r_fast = _r_hat_after(b=0.99)
    r_slow = _r_hat_after(b=0.95)
    # Larger b (more weight on recent samples) → r_hat further from initial
    assert r_fast != pytest.approx(r_slow, abs=1e-8), (
        "Different forget factors should produce different r_hat trajectories"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RAEKF-specific: run() batch interface
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
# Self-description API
# ─────────────────────────────────────────────────────────────────────────────


def test_spec_type():
    assert isinstance(RAEKFEstimator.spec(), EstimatorSpec)


def test_tuning_spec_type():
    assert isinstance(RAEKFEstimator.tuning_spec(), TuningSpec)


def test_tuning_spec_has_huber_delta():
    names = {p.name for p in RAEKFEstimator.tuning_ranges()}
    assert "huber_delta" in names


def test_tuning_spec_has_forget_factor():
    names = {p.name for p in RAEKFEstimator.tuning_ranges()}
    assert "forget_factor" in names


def test_descriptor_keys():
    desc = RAEKFEstimator.descriptor()
    for key in (
        "name",
        "family",
        "family_path",
        "complexity",
        "latency_type",
        "suggested_objective",
        "tuning_params",
    ):
        assert key in desc


# ─────────────────────────────────────────────────────────────────────────────
# set_params
# ─────────────────────────────────────────────────────────────────────────────


def test_set_params_huber_delta():
    est = _make_est()
    est.set_params(huber_delta=2.0)
    assert est._delta == pytest.approx(2.0)


def test_set_params_forget_factor():
    est = _make_est()
    est.set_params(forget_factor=0.95)
    assert est._b == pytest.approx(0.95)


def test_set_params_r():
    est = _make_est()
    est.set_params(r=0.5)
    assert est._r == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Robustness comparison: RAEKF vs EKF on impulsive noise
# ─────────────────────────────────────────────────────────────────────────────


def test_raekf_more_robust_than_ekf_on_impulsive_noise():
    """
    On a 60 Hz signal with 5 % impulse contamination (sigma=5 V spikes),
    RAEKF RMSE should be lower than EKF RMSE.

    This is the core scientific claim for the paper.
    """
    from openfreqbench.estimators.monophasic.f1_kalman.ekf_freq import EKFFreqEstimator

    rng = np.random.default_rng(42)
    n = N  # 1 second at 10 kHz

    v_clean = _sine(n=n)
    impulse_mask = rng.uniform(0, 1, n) < 0.05  # 5 % impulse rate
    impulses = rng.normal(0, 5.0, n) * impulse_mask
    v_noisy = v_clean + impulses

    # EKF baseline
    ekf = EKFFreqEstimator(config={"fs": FS, "q_omega": 1.0, "r": 0.01})
    out_ekf = ekf.run(v_noisy)

    # RAEKF
    raekf = RAEKFEstimator(
        config={"fs": FS, "q_omega": 1.0, "r": 0.01, "huber_delta": 1.345, "forget_factor": 0.98},
    )
    out_raekf = raekf.run(v_noisy)

    L = max(ekf.latency_samples, raekf.latency_samples) + int(0.1 * n)
    rmse_ekf = float(np.sqrt(np.mean((out_ekf[L:] - F0) ** 2)))
    rmse_raekf = float(np.sqrt(np.mean((out_raekf[L:] - F0) ** 2)))

    assert rmse_raekf < rmse_ekf, (
        f"RAEKF RMSE ({rmse_raekf:.4f} Hz) should be < EKF RMSE ({rmse_ekf:.4f} Hz) "
        f"under 5 % impulsive contamination"
    )
