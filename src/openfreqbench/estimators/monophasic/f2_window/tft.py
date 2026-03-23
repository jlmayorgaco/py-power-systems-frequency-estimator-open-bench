"""
estimators/monophasic/f2_window/tft.py  [STUB]

TFTEstimator — Taylor-Fourier Transform (TFT / Taylor-Weighted-Least-Squares).

Algorithm sketch
────────────────
  1. Represent the signal as a Taylor-Fourier series centred on the analysis
     window: x(t) = Σ_k Σ_m c_{k,m} · t^m · e^(j k ω₀ t).
  2. Solve a windowed WLS problem for the complex coefficient c_{1,0} and its
     first two time-derivatives (m=0,1,2).
  3. Instantaneous frequency: f = f₀ + Im{ċ_{1,0} / c_{1,0}} / (2π).
  4. ROCOF from second derivative similarly.

Reference: Bertocco M. et al., "A Compressive Sensing Approach for PMU Signal
Compression and Reconstruction in Smart Grids", IEEE TPWRD, 2016.

Status: STUB — outputs nominal frequency with valid=False until implemented.
"""

from __future__ import annotations

from typing import Any

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningParam,
    TuningSpec,
)


class TFTEstimator(BaseEstimator):
    """
    Taylor-Fourier Transform (TFT) frequency estimator — stub.

    Block-based (overrides ``run()``).  ``update()`` is a causal stub.
    All outputs carry ``valid=False`` until implemented.
    """

    SPEC = EstimatorSpec(
        name="TFT",
        family="Spectral",
        family_path="monophasic/f2_window",
        complexity="O(N²)",
        latency_type="semi-causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"fs": 10_000.0, "window_size": 512, "taylor_order": 2}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="window_size",
                    default=512,
                    type="int",
                    values=[128, 256, 512, 1024],
                    description="Analysis window length (samples).",
                ),
                TuningParam(
                    name="taylor_order",
                    default=2,
                    type="int",
                    values=[1, 2, 3],
                    description=(
                        "Polynomial order of the Taylor expansion. "
                        "Order 1 = constant phasor; 2 = linear FM; 3 = quadratic FM."
                    ),
                ),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        self._f_est: float = self.NOMINAL_FREQ_HZ

    def structural_latency_samples(self) -> int:
        return int(self._config.get("window_size", 512)) // 2

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        return EstimatorOutput(frequency_hz=self._f_est, valid=False)

    def _step(self, v_sample: float) -> float:
        return self._f_est
