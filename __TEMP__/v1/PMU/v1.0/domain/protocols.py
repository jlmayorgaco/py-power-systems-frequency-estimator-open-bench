from __future__ import annotations
from typing import Protocol, Dict, Any, runtime_checkable
import numpy as np


@runtime_checkable
class OnlineEstimator(Protocol):
    """
    Protocol for online frequency estimators.
    """

    def reset(self) -> None: ...

    def step(self, v_sample: float) -> float:
        """
        Process one voltage sample and return frequency estimate.
        """
        ...

    @property
    def latency_samples(self) -> int:
        """
        Structural latency (in samples).
        Used for fair metric evaluation.
        """
        ...


@runtime_checkable
class EstimatorFactory(Protocol):
    """
    Factory that builds estimators with fixed parameters.
    """

    def __call__(self, params: Dict[str, Any]) -> OnlineEstimator: ...


@runtime_checkable
class HyperparameterTuner(Protocol):
    """
    Hyperparameter tuner (train segment only).
    """

    def tune(self, v: np.ndarray, f: np.ndarray) -> Dict[str, Any]:
        """
        Returns a dict of fixed parameters.
        """
        ...
