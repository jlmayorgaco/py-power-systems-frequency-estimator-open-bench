"""
openfreqbench/core/registry.py

Facade that re-exports EstimatorRegistry and ScenarioRegistry.
The canonical registry classes live in estimators/registry.py and scenarios/registry.py.
"""
from openfreqbench.estimators.registry import EstimatorRegistry
from openfreqbench.scenarios.registry import ScenarioRegistry

__all__ = ["EstimatorRegistry", "ScenarioRegistry"]
