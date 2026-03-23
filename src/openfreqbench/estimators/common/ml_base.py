"""
openfreqbench/estimators/common/ml_base.py

BaseMLEstimator — abstract base for ML/data-driven frequency estimators.

Extends BaseEstimator with three ML-specific classmethods:
  model_spec()          → ModelSpec (architecture)
  training_spec()       → TrainingSpec (training config)
  data_protocol_spec()  → DataProtocolSpec (I/O convention)

update() and reset() are still required from BaseEstimator.
No training logic lives here.
"""

from __future__ import annotations

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.ml_types import (
    DataProtocolSpec,
    ModelSpec,
    TrainingSpec,
)


class BaseMLEstimator(BaseEstimator):
    """
    Abstract base for ML/data-driven frequency estimators.

    Subclass contract (in addition to BaseEstimator):
      5. Implement ``model_spec()``          → ModelSpec
      6. Implement ``training_spec()``       → TrainingSpec
      7. Implement ``data_protocol_spec()``  → DataProtocolSpec
    """

    @classmethod
    def model_spec(cls) -> ModelSpec:
        """Return the neural architecture specification for this estimator."""
        raise NotImplementedError("To be implemented in next phase")

    @classmethod
    def training_spec(cls) -> TrainingSpec:
        """Return the default training configuration."""
        raise NotImplementedError("To be implemented in next phase")

    @classmethod
    def data_protocol_spec(cls) -> DataProtocolSpec:
        """Return the input/output data protocol specification."""
        raise NotImplementedError("To be implemented in next phase")
