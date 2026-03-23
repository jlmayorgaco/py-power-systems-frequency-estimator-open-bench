"""Verify all public modules import without error."""


def test_import_openfreqbench():
    import openfreqbench  # noqa: F401


def test_import_estimators():
    from openfreqbench.estimators import _base  # noqa: F401
    from openfreqbench.estimators.zero_crossing import ZeroCrossingEstimator  # noqa: F401


def test_import_scenarios():
    from openfreqbench.scenarios.g1.e1_pure_60hz import G1_E1_Pure_60Hz  # noqa: F401


def test_import_metrics():
    from openfreqbench.metrics.frequency import compute_metrics  # noqa: F401


def test_import_runners():
    from openfreqbench.runners.scenario_method_runner import ScenarioMethodRunner  # noqa: F401
    from openfreqbench.runners.trace_runner import TraceRunner  # noqa: F401


def test_import_core():
    from openfreqbench.core.config_models import BenchmarkConfig, load_config  # noqa: F401
    from openfreqbench.core.registry import EstimatorRegistry, ScenarioRegistry  # noqa: F401
