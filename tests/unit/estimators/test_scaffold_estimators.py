"""
tests/unit/estimators/test_scaffold_estimators.py

Parametrized smoke tests for all scaffold estimators.

Each test verifies that:
  1. The class is importable and instantiable with default config.
  2. SPEC.name is a non-empty string.
  3. update() raises NotImplementedError (scaffold contract).
  4. reset() runs without error.
  5. tuning_spec() returns a TuningSpec.
  6. descriptor() returns a dict with required keys.
  7. The class is registered in EstimatorRegistry.

Scaffold estimators are NOT expected to implement signal-processing logic.
update() and _step() must raise NotImplementedError — that is intentional.
These tests only verify the *metadata contract* and *registration*.
"""
from __future__ import annotations

import pytest

from openfreqbench.estimators.common.types import TuningSpec

# ── Scaffold estimator classes ────────────────────────────────────────────────

from openfreqbench.estimators.monophasic.f0_pll.ddsrf_pll        import DDSRFPLLEstimator
from openfreqbench.estimators.monophasic.f0_pll.dsogi_fll        import DSOGIFLLEstimator
from openfreqbench.estimators.monophasic.f0_pll.epll             import EPLLEstimator
from openfreqbench.estimators.monophasic.f0_pll.anf              import ANFEstimator
from openfreqbench.estimators.monophasic.f1_kalman.linear_kalman import LinearKalmanEstimator
from openfreqbench.estimators.monophasic.f1_kalman.ckf           import CKFEstimator
from openfreqbench.estimators.monophasic.f1_kalman.adaptive_ekf  import AdaptiveEKFEstimator
from openfreqbench.estimators.monophasic.f2_window.goertzel      import GoertzelEstimator
from openfreqbench.estimators.monophasic.f2_window.dynamic_phasor import DynamicPhasorEstimator
from openfreqbench.estimators.monophasic.f3_recursive.interp_zero_crossing import InterpolatedZeroCrossingEstimator
from openfreqbench.estimators.monophasic.f3_recursive.windowed_ls import WindowedLeastSquaresEstimator
from openfreqbench.estimators.monophasic.f5_parametric.prony     import PronyEstimator
from openfreqbench.estimators.monophasic.f5_parametric.music     import MUSICEstimator
from openfreqbench.estimators.monophasic.f5_parametric.esprit    import ESPRITEstimator
from openfreqbench.estimators.monophasic.f6_time_frequency.hilbert_freq import HilbertFrequencyEstimator
from openfreqbench.estimators.monophasic.f4_data_driven.gru_regressor      import GRURegressorEstimator
from openfreqbench.estimators.monophasic.f4_data_driven.lstm_regressor     import LSTMRegressorEstimator
from openfreqbench.estimators.monophasic.f4_data_driven.bilstm_regressor   import BiLSTMRegressorEstimator
from openfreqbench.estimators.monophasic.f4_data_driven.temporal_cnn       import TemporalCNNEstimator
from openfreqbench.estimators.monophasic.f4_data_driven.tcn_frequency      import TCNFrequencyEstimator
from openfreqbench.estimators.monophasic.f4_data_driven.transformer_regressor import TransformerRegressorEstimator
from openfreqbench.estimators.monophasic.f4_data_driven.reservoir_echo     import ReservoirEchoStateEstimator
from openfreqbench.estimators.monophasic.f4_data_driven.mlp_window         import MLPWindowRegressorEstimator

SCAFFOLD_CLASSES = [
    # f0_pll
    DDSRFPLLEstimator,
    DSOGIFLLEstimator,
    EPLLEstimator,
    ANFEstimator,
    # f1_kalman
    LinearKalmanEstimator,
    CKFEstimator,
    AdaptiveEKFEstimator,
    # f2_window
    GoertzelEstimator,
    DynamicPhasorEstimator,
    # f3_recursive
    InterpolatedZeroCrossingEstimator,
    WindowedLeastSquaresEstimator,
    # f5_parametric
    PronyEstimator,
    MUSICEstimator,
    ESPRITEstimator,
    # f6_time_frequency
    HilbertFrequencyEstimator,
    # f4_data_driven (ML)
    GRURegressorEstimator,
    LSTMRegressorEstimator,
    BiLSTMRegressorEstimator,
    TemporalCNNEstimator,
    TCNFrequencyEstimator,
    TransformerRegressorEstimator,
    ReservoirEchoStateEstimator,
    MLPWindowRegressorEstimator,
]

_REQUIRED_DESCRIPTOR_KEYS = (
    "name", "family", "family_path", "complexity",
    "latency_type", "suggested_objective", "tuning_params",
)


@pytest.mark.parametrize("cls", SCAFFOLD_CLASSES, ids=lambda c: c.SPEC.name)
class TestScaffoldEstimatorContract:
    """Metadata and stub-safety contract for all scaffold estimators."""

    def test_instantiates_with_default_config(self, cls):
        est = cls()
        assert est is not None

    def test_spec_name_nonempty(self, cls):
        assert isinstance(cls.SPEC.name, str) and len(cls.SPEC.name) > 0

    def test_spec_family_path_nonempty(self, cls):
        assert isinstance(cls.SPEC.family_path, str) and len(cls.SPEC.family_path) > 0

    def test_update_raises_not_implemented(self, cls):
        """Scaffold estimators must raise NotImplementedError from update()."""
        est = cls()
        with pytest.raises(NotImplementedError):
            est.update(0.5)

    def test_reset_does_not_crash(self, cls):
        est = cls()
        est.reset()  # must not raise

    def test_reset_after_update(self, cls):
        est = cls()
        est.update(0.5)
        est.reset()  # must not raise

    def test_tuning_spec_type(self, cls):
        ts = cls.tuning_spec()
        assert isinstance(ts, TuningSpec)

    def test_descriptor_has_required_keys(self, cls):
        desc = cls.descriptor()
        assert isinstance(desc, dict)
        for key in _REQUIRED_DESCRIPTOR_KEYS:
            assert key in desc, f"descriptor() missing key {key!r} for {cls.SPEC.name}"

    def test_in_registry(self, cls):
        """Every scaffold must be findable by name in the EstimatorRegistry."""
        from openfreqbench.estimators.registry import EstimatorRegistry
        names = EstimatorRegistry.list_names()
        assert cls.SPEC.name in names, (
            f"{cls.SPEC.name} not found in EstimatorRegistry"
        )


# ── ML-specific: BaseMLEstimator API ─────────────────────────────────────────

ML_CLASSES = [
    GRURegressorEstimator,
    LSTMRegressorEstimator,
    BiLSTMRegressorEstimator,
    TemporalCNNEstimator,
    TCNFrequencyEstimator,
    TransformerRegressorEstimator,
    ReservoirEchoStateEstimator,
    MLPWindowRegressorEstimator,
]


@pytest.mark.parametrize("cls", ML_CLASSES, ids=lambda c: c.SPEC.name)
class TestMLEstimatorMetadataAPI:
    """ML estimators must expose model_spec / training_spec / data_protocol_spec."""

    def test_model_spec_raises_not_implemented(self, cls):
        with pytest.raises(NotImplementedError):
            cls.model_spec()

    def test_training_spec_raises_not_implemented(self, cls):
        with pytest.raises(NotImplementedError):
            cls.training_spec()

    def test_data_protocol_spec_raises_not_implemented(self, cls):
        with pytest.raises(NotImplementedError):
            cls.data_protocol_spec()


# ── Registry round-trip ───────────────────────────────────────────────────────

def test_all_scaffold_estimators_in_registry():
    from openfreqbench.estimators.registry import EstimatorRegistry
    names = set(EstimatorRegistry.list_names())
    expected = {cls.SPEC.name for cls in SCAFFOLD_CLASSES}
    missing = expected - names
    assert not missing, f"These scaffold estimators are missing from registry: {missing}"


def test_total_registered_estimator_count():
    """Registry must have at least 38 estimators (9 real + 6 stub + 23 scaffold)."""
    from openfreqbench.estimators.registry import EstimatorRegistry
    count = len(EstimatorRegistry.list_names())
    assert count >= 38, f"Expected ≥38 registered estimators, got {count}"
