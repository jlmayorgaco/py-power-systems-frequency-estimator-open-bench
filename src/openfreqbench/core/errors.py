"""Domain exceptions for openfreqbench."""
from __future__ import annotations


class OpenFreqBenchError(Exception):
    """Base exception."""


class RegistryError(OpenFreqBenchError):
    """Raised when a registry lookup fails."""


class ConfigError(OpenFreqBenchError):
    """Raised for invalid benchmark configuration."""


class NumericalError(OpenFreqBenchError):
    """Raised when a metric computation fails due to bad data."""


class ParityError(OpenFreqBenchError):
    """Raised when numerical parity check fails against pfebench reference."""
