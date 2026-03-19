"""
openfreqbench/core/registry.py

Simple name → class registry for estimators and scenarios.
All built-in implementations are pre-registered at module load time.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from openfreqbench.estimators._base import BaseEstimator
from openfreqbench.scenarios._base import ScenarioBase


class EstimatorRegistry:
    _registry: Dict[str, Type[BaseEstimator]] = {}

    @classmethod
    def register(cls, estimator_cls: Type[BaseEstimator]) -> Type[BaseEstimator]:
        cls._registry[estimator_cls.NAME] = estimator_cls
        return estimator_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseEstimator]:
        if name not in cls._registry:
            raise KeyError(
                f"Estimator {name!r} not registered. "
                f"Available: {sorted(cls._registry)}"
            )
        return cls._registry[name]

    @classmethod
    def build(cls, name: str, params: Dict[str, Any] | None = None) -> BaseEstimator:
        return cls.get(name)(params=params or {})

    @classmethod
    def list_names(cls) -> list[str]:
        return sorted(cls._registry)


class ScenarioRegistry:
    _registry: Dict[str, Type[ScenarioBase]] = {}

    @classmethod
    def register(cls, scenario_cls: Type[ScenarioBase]) -> Type[ScenarioBase]:
        cls._registry[scenario_cls.scenario_id] = scenario_cls
        return scenario_cls

    @classmethod
    def get(cls, name: str) -> Type[ScenarioBase]:
        if name not in cls._registry:
            raise KeyError(
                f"Scenario {name!r} not registered. "
                f"Available: {sorted(cls._registry)}"
            )
        return cls._registry[name]

    @classmethod
    def build(cls, name: str, params: Dict[str, Any] | None = None) -> ScenarioBase:
        cls_ = cls.get(name)
        return cls_(**params) if params else cls_()

    @classmethod
    def list_names(cls) -> list[str]:
        return sorted(cls._registry)


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

from openfreqbench.estimators.zc import ZeroCrossingEstimator  # noqa: E402
from openfreqbench.scenarios.g1.e1_pure_60hz import G1_E1_Pure_60Hz  # noqa: E402

EstimatorRegistry.register(ZeroCrossingEstimator)
ScenarioRegistry.register(G1_E1_Pure_60Hz)
