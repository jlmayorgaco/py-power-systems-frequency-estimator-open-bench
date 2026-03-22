"""
openfreqbench/estimators/common/base.py  [CANONICAL]

BaseEstimator — abstract base class for every OpenFreqBench frequency estimator.

Design invariants
─────────────────
  update(voltage, timestamp) → EstimatorOutput   ← ONLY required override
  reset()                                         ← required override
  structural_latency_samples() → int              ← optional override (default 0)
  tuning_spec()              → TuningSpec         ← optional classmethod
  default_config()           → dict               ← optional classmethod

The framework keeps all of the following OUTSIDE the estimator:
  timing · memory profiling · Monte Carlo loops · metric computation
  statistical tests · plotting · artifact storage · reporting

Backward-compatibility shims (kept for existing framework code)
────────────────────────────────────────────────────────────────
  step(v) → float                     wraps update()
  set_params(**kwargs)                 wraps reconfigure()
  latency_samples  (property)          wraps structural_latency_samples()
  NAME / FAMILY / FAMILY_PATH / ...    auto-derived from SPEC via __init_subclass__
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, ClassVar

import numpy as np

from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningParam,
    TuningSpec,
)


class BaseEstimator(ABC):
    """
    Abstract base for every OpenFreqBench frequency estimator.

    Subclass contract
    ─────────────────
      1.  Set ``SPEC = EstimatorSpec(...)`` as a class variable.
      2.  Implement ``reset()`` to re-initialise all internal state.
      3.  Implement ``update(voltage, timestamp) → EstimatorOutput``.
      4.  Override ``default_config()`` to declare defaults (including ``fs``).
      5.  Override ``tuning_spec()`` to declare the tunable parameter space.
      6.  Override ``structural_latency_samples()`` if non-zero.
      7.  Override ``run()`` for block-based (non-causal) estimators only.

    Example
    ───────
    See ``src/openfreqbench/estimators/TEMPLATE.py`` for a complete stub.
    """

    # ── Subclass must set this ────────────────────────────────────────────────
    SPEC: ClassVar[EstimatorSpec]

    # ── Auto-populated from SPEC by __init_subclass__ ─────────────────────────
    NAME: ClassVar[str]
    FAMILY: ClassVar[str]
    FAMILY_PATH: ClassVar[str]
    COMPLEXITY: ClassVar[str]
    LATENCY_TYPE: ClassVar[str]
    NOMINAL_FREQ_HZ: ClassVar[float]
    MIN_VALID_FREQ_HZ: ClassVar[float]
    MAX_VALID_FREQ_HZ: ClassVar[float]

    # ── Auto-derive legacy class vars from SPEC ───────────────────────────────

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "SPEC" in cls.__dict__:
            spec = cls.__dict__["SPEC"]
            cls.NAME = spec.name
            cls.FAMILY = spec.family
            cls.FAMILY_PATH = spec.family_path
            cls.COMPLEXITY = spec.complexity
            cls.LATENCY_TYPE = spec.latency_type
            cls.NOMINAL_FREQ_HZ = spec.nominal_freq_hz
            cls.MIN_VALID_FREQ_HZ = spec.min_valid_freq_hz
            cls.MAX_VALID_FREQ_HZ = spec.max_valid_freq_hz

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        # backward-compat alias accepted but merged into config
        params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise with an optional configuration dict.

        The dict is merged over ``default_config()``; keys not present in
        ``default_config()`` are accepted without restriction.

        Args:
            config: Estimator-specific parameters (e.g. ``{"fs": 10000.0,
                    "window_size": 512}``).  ``None`` → use defaults only.
            params: Backward-compat alias for ``config``.  If both are given,
                    ``config`` takes precedence over ``params``.
        """
        self._config: dict[str, Any] = {
            **self.default_config(),
            **(params or {}),
            **(config or {}),
        }
        # Convenience counter subclasses may use to track validity
        self._n_samples: int = 0
        self.reset()

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def reset(self) -> None:
        """Re-initialise all internal state to construction-time defaults."""

    @abstractmethod
    def update(self, voltage: float | np.ndarray, timestamp: float = 0.0) -> EstimatorOutput:
        """
        Process one voltage sample and return a typed EstimatorOutput.

        This is the PRIMARY public interface for sample-by-sample processing.
        Implement all estimator logic here; the framework calls this per sample.

        Args:
            voltage:   Raw voltage sample (V or per-unit). Usually float, but can be np.ndarray for 3-phase.
            timestamp: Wall-clock time of this sample (seconds).  Monotonically
                       increasing.  Use to derive ``fs`` if not in config.

        Returns:
            EstimatorOutput with at least ``frequency_hz`` and ``valid`` set.
        """

    # ── Optional overrides ────────────────────────────────────────────────────

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """
        Default configuration dict for this estimator class.

        Override to declare estimator-specific defaults.
        Always include ``"fs"`` (sampling frequency in Hz).

        Example::

            @classmethod
            def default_config(cls) -> dict:
                return {"fs": 10_000.0, "window_size": 512}
        """
        return {"fs": 10_000.0}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        """
        Declare the tunable parameter space for this estimator class.

        The estimator declares *what* can be tuned and *what values* to try.
        The framework decides *when* and *how* to run the search.

        Override to return a TuningSpec with one TuningParam per hyperparameter.
        """
        return TuningSpec()

    def structural_latency_samples(self) -> int:
        """
        Estimated causal delay (samples) before output is reliable.

        Override when latency depends on config (e.g. window-based estimators).
        The default is 0 (instant estimators).
        """
        return 0

    # ── Batch processing (override for non-causal estimators) ─────────────────

    def run(self, v_array: np.ndarray) -> np.ndarray:
        """
        Process a full waveform, returning f_hat of the same length.

        Default: calls ``reset()`` then ``update()`` per sample (causal, online).
        Override for block-based estimators (FFT, parametric, etc.) that need
        access to the full signal or use non-causal framing.

        Returns:
            1-D float array of frequency estimates, same length as ``v_array``.
        """
        v = np.asarray(v_array, dtype=float)
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        n = len(v)
        fs = float(self._config.get("fs", 10_000.0))
        Ts = 1.0 / fs
        out = np.empty(n, dtype=float)
        self._n_samples = 0
        self.reset()
        is_multi = v.shape[1] > 1
        for i, x in enumerate(v):
            self._n_samples += 1
            sample = x if is_multi else float(x[0])
            result = self.update(sample, timestamp=i * Ts)
            out[i] = result.frequency_hz
        return out

    # ── Param management ──────────────────────────────────────────────────────

    def reconfigure(self, config: dict[str, Any]) -> None:
        """Update config entries and reset internal state."""
        self._config.update(config)
        self._n_samples = 0
        self.reset()

    # ── Self-description API ──────────────────────────────────────────────────

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        """Full JSON-serialisable self-description of this estimator class."""
        d = cls.SPEC.to_dict()
        ts = cls.tuning_spec()
        d["suggested_objective"] = ts.objective
        d["tuning_method"] = ts.method
        d["tuning_params"] = [p.to_dict() for p in ts.params]
        return d

    # ── Backward-compatibility shims ──────────────────────────────────────────
    # These exist so existing runner/profiling code continues to work while
    # estimators migrate to the new API.  Do NOT add new framework code that
    # relies on these.

    @property
    def _params(self) -> dict[str, Any]:
        """Backward-compat alias for ``_config`` used by legacy runner code."""
        return self._config

    def step(self, v_sample: float) -> float:
        """Single-sample step returning a raw float (backward-compat)."""
        try:
            result = self.update(float(v_sample), timestamp=0.0)
            f = result.frequency_hz
        except Exception:
            return float("nan")
        return float("nan") if not math.isfinite(f) else f

    def set_params(self, **kwargs: Any) -> None:
        """Update config params and reset.  Wraps ``reconfigure()``."""
        self.reconfigure(kwargs)

    @property
    def latency_samples(self) -> int:
        """Backward-compat property.  Prefer ``structural_latency_samples()``."""
        return self.structural_latency_samples()

    # ── Legacy classmethod shims (used by old registry / tests) ───────────────

    @classmethod
    def tuning_ranges(cls) -> list[TuningParam]:
        """Backward-compat: returns the list of TuningParam from tuning_spec()."""
        return cls.tuning_spec().params

    @classmethod
    def spec(cls) -> EstimatorSpec:
        """Return the EstimatorSpec for this class."""
        return cls.SPEC

    @classmethod
    def suggested_objective(cls) -> str:
        """Return the suggested tuning objective string."""
        return cls.tuning_spec().objective

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        name = getattr(self, "NAME", type(self).__name__)
        return f"{name}({self._config!r})"
