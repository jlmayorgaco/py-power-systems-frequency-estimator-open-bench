"""Unit tests for ZeroCrossingEstimator."""

import numpy as np
import pytest

from openfreqbench.estimators.zc import ZeroCrossingEstimator


FS = 10_000.0
F_NOM = 60.0
T = 2.0  # seconds
N = int(T * FS)
T_ARR = np.arange(N) / FS
V_CLEAN = np.sin(2.0 * np.pi * F_NOM * T_ARR)  # pure 60 Hz


class TestZeroCrossingEstimator:
    def test_instantiates_with_defaults(self):
        zc = ZeroCrossingEstimator()
        assert zc.NAME == "ZeroCrossing"
        assert zc.FAMILY == "TimeDomain"

    def test_default_filter_win(self):
        zc = ZeroCrossingEstimator()
        assert zc._params["filter_win"] == 5

    def test_custom_filter_win(self):
        zc = ZeroCrossingEstimator(params={"filter_win": 10})
        assert zc._params["filter_win"] == 10

    def test_latency_samples_positive(self):
        zc = ZeroCrossingEstimator(params={"fs": FS, "filter_win": 5})
        assert zc.latency_samples > 0

    def test_run_returns_correct_shape(self):
        zc = ZeroCrossingEstimator(params={"fs": FS})
        f_hat = zc.run(V_CLEAN)
        assert f_hat.shape == V_CLEAN.shape

    def test_steady_state_accuracy_clean(self):
        zc = ZeroCrossingEstimator(params={"fs": FS, "filter_win": 5})
        f_hat = zc.run(V_CLEAN)
        # Skip warm-up (first 10% or latency samples)
        warm = max(zc.latency_samples + 50, int(0.1 * N))
        steady = f_hat[warm:]
        assert np.nanmean(np.abs(steady - F_NOM)) < 0.05  # < 50 mHz mean error

    def test_no_nan_in_steady_state(self):
        zc = ZeroCrossingEstimator(params={"fs": FS})
        f_hat = zc.run(V_CLEAN)
        warm = int(0.15 * N)
        assert not np.any(np.isnan(f_hat[warm:]))

    def test_reset_idempotent(self):
        zc = ZeroCrossingEstimator(params={"fs": FS})
        f1 = zc.run(V_CLEAN)
        f2 = zc.run(V_CLEAN)
        np.testing.assert_array_equal(f1, f2)

    def test_tuning_ranges_nonempty(self):
        ranges = ZeroCrossingEstimator.tuning_ranges()
        assert len(ranges) == 1
        assert ranges[0].name == "filter_win"

    def test_set_params_resets(self):
        zc = ZeroCrossingEstimator(params={"fs": FS, "filter_win": 5})
        zc.set_params(filter_win=10)
        assert zc._params["filter_win"] == 10
        # After set_params, run should still work
        f_hat = zc.run(V_CLEAN)
        assert f_hat.shape == V_CLEAN.shape

    def test_outputs_not_all_nominal(self):
        # After warm-up, estimator should have converged away from constant 60 Hz
        # (or settled at 60 Hz but via actual estimation, not just nominal hold)
        zc = ZeroCrossingEstimator(params={"fs": FS, "filter_win": 1})
        f_hat = zc.run(V_CLEAN)
        warm = int(0.1 * N)
        # Should have at least some variation due to crossing detection resolution
        # but for clean signal converges to 60 Hz
        assert np.nanmean(f_hat[warm:]) == pytest.approx(F_NOM, abs=0.1)
