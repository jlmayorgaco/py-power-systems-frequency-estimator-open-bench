"""
openfreqbench/estimators/common/ml_types.py

Typed metadata models for ML/data-driven estimators.
Used by BaseMLEstimator and all f4_data_driven subclasses.

These are data-only dataclasses — no training logic here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """
    Immutable neural architecture specification.

    Describes the shape and structure of the ML model without
    binding to any specific framework (torch/tf/numpy).
    """

    architecture: str
    """Human-readable architecture name, e.g. 'GRU', 'TCN', 'Transformer'."""

    input_features: int
    """Number of input features per timestep (1 = raw voltage)."""

    sequence_length: int
    """Input window length in samples."""

    output_features: int = 1
    """Number of outputs (default 1 = frequency_hz)."""

    hidden_dim: int = 64
    """Hidden state / channel dimension."""

    num_layers: int = 2
    """Number of recurrent/convolutional layers."""

    dropout: float = 0.0
    """Dropout probability during training."""

    backend: str = "torch"
    """Compute backend: 'torch', 'tensorflow', 'numpy'."""

    pretrained_path: str | None = None
    """Path to pretrained weights file (None = train from scratch)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "input_features": self.input_features,
            "sequence_length": self.sequence_length,
            "output_features": self.output_features,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "backend": self.backend,
            "pretrained_path": self.pretrained_path,
        }


@dataclass
class TrainingSpec:
    """
    Training configuration for ML-based estimators.

    Mutable (not frozen) because training runs may tune these.
    """

    dataset_name: str = "G1_E1_Pure_60Hz"
    n_train_scenarios: int = 500
    n_val_scenarios: int = 100
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    loss: str = "mse"
    lr_scheduler: str = "cosine"
    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    seed: int = 42
    checkpoint_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "n_train_scenarios": self.n_train_scenarios,
            "n_val_scenarios": self.n_val_scenarios,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "early_stopping_patience": self.early_stopping_patience,
            "device": self.device,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class DataProtocolSpec:
    """
    Defines input/output data conventions for ML estimators.

    Consumed by the TraceRunner to know how to feed data into a
    model and interpret its output.
    """

    input_type: str = "raw_voltage"
    """Input representation: 'raw_voltage', 'dft_coeffs', 'stft_frame'."""

    normalization: str = "unit_amplitude"
    """Normalisation applied before model: 'none', 'z-score', 'unit_amplitude'."""

    output_type: str = "frequency_hz"
    """What the model directly outputs: 'frequency_hz', 'phase_rad', 'phasor'."""

    latency_samples: int = 0
    """Additional causal delay introduced by the data windowing (samples)."""

    requires_calibration: bool = False
    """True if the model needs an offline calibration pass before inference."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_type": self.input_type,
            "normalization": self.normalization,
            "output_type": self.output_type,
            "latency_samples": self.latency_samples,
            "requires_calibration": self.requires_calibration,
        }
