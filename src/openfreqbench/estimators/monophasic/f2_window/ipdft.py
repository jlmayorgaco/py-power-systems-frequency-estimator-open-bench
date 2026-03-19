"""
estimators/monophasic/f2_window/ipdft.py  [CANONICAL]

IpDFTEstimator — Interpolated DFT with Hann-specific two-point interpolation.

Algorithm
─────────
  1. Divide the signal into overlapping frames (window_size, 50% hop).
  2. Per frame:
       a. Apply a Hann window.
       b. Compute the one-sided DFT via rfft.
       c. Find the peak bin k_max in [f_min, f_max] by magnitude.
       d. Identify the adjacent bin with greater magnitude:
            k_adj = k_max+1  if |X[k_max+1]| ≥ |X[k_max−1]|
                    k_max−1  otherwise
       e. Apply the two-point interpolation:
            sign  = +1 if k_adj > k_max else −1
            δ     = sign · |X[k_adj]| / (|X[k_max]| + |X[k_adj]|)
            f̂    = (k_max + δ) · fs / N
  3. Each output sample is assigned the estimate of the nearest frame centre.

Complexity:  O(N log N) per frame.
Latency:     window_size // 2 samples (semi-causal).

References
──────────
  Grandke, T. (1983). "Interpolation algorithms for discrete Fourier
  transforms of weighted signals." IEEE Trans. Instrum. Meas., 32(2), 350–355.

  Toscani, S., Muscas, C., Pegoraro, P.A. (2012). "Design and performance
  comparison of PMU algorithms for smart grid applications." IEEE Trans.
  Instrum. Meas., 61(8), 2284–2293.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.estimators.common.types import (
    EstimatorOutput,
    EstimatorSpec,
    TuningParam,
    TuningSpec,
)


class IpDFTEstimator(BaseEstimator):
    """
    Hann-windowed Interpolated DFT (IpDFT) frequency estimator.

    Block-based: overrides ``run()`` for frame-level batch processing.
    ``update()`` returns the last frame's estimate (causal stub).
    """

    SPEC = EstimatorSpec(
        name="IpDFT",
        family="Spectral",
        family_path="monophasic/f2_window",
        complexity="O(N log N)",
        latency_type="semi-causal",
        nominal_freq_hz=60.0,
        min_valid_freq_hz=40.0,
        max_valid_freq_hz=80.0,
    )

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {"fs": 10_000.0, "window_size": 1024}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="window_size",
                    default=1024,
                    type="int",
                    values=[256, 512, 1024, 2048, 4096],
                    description=(
                        "DFT frame length (samples).  Larger → finer frequency "
                        "resolution, higher latency, slower transient response. "
                        "At fs=10 kHz: 1024→102.4 ms, 512→51.2 ms."
                    ),
                ),
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        self._window_size: int   = int(self._config.get("window_size", 1024))
        self._f_est:       float = self.NOMINAL_FREQ_HZ

    def structural_latency_samples(self) -> int:
        return int(self._config.get("window_size", 1024)) // 2

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        """Sample-by-sample stub — delegates actual work to run()."""
        return EstimatorOutput(frequency_hz=self._f_est, valid=False)

    def run(self, v_array: np.ndarray) -> np.ndarray:
        """Process full waveform via sliding Hann-windowed IpDFT."""
        self.reset()
        v  = np.asarray(v_array, dtype=float)
        n  = len(v)
        ws = self._window_size
        fs = float(self._config.get("fs", 10_000.0))

        out = np.full(n, self.NOMINAL_FREQ_HZ, dtype=float)
        if n < ws:
            out[:] = self._ipdft_frame(v, fs)
            return out

        hop  = max(1, ws // 2)
        hann = np.hanning(ws)

        centres: List[int]   = []
        freqs:   List[float] = []

        for start in range(0, n - ws + 1, hop):
            frame = v[start: start + ws] * hann
            freqs.append(self._ipdft_frame(frame, fs))
            centres.append(start + ws // 2)

        if not centres:
            return out

        centres_arr = np.array(centres, dtype=int)
        freqs_arr   = np.array(freqs,   dtype=float)
        for i in range(n):
            idx    = int(np.argmin(np.abs(centres_arr - i)))
            out[i] = freqs_arr[idx]
        return out

    # ── Internal algorithm ────────────────────────────────────────────────────

    def _ipdft_frame(self, frame: np.ndarray, fs: float) -> float:
        """Two-point IpDFT on one (already Hann-windowed) frame."""
        spectrum  = np.abs(np.fft.rfft(frame))
        n         = len(frame)
        freq_bins = np.fft.rfftfreq(n, d=1.0 / fs)

        lo = int(np.searchsorted(freq_bins, self.MIN_VALID_FREQ_HZ))
        hi = int(np.searchsorted(freq_bins, self.MAX_VALID_FREQ_HZ))
        if hi <= lo or lo >= len(spectrum):
            return self.NOMINAL_FREQ_HZ

        sub   = spectrum[lo:hi]
        k_max = lo + int(np.argmax(sub))

        left_mag  = spectrum[k_max - 1] if k_max > 0                else 0.0
        right_mag = spectrum[k_max + 1] if k_max < len(spectrum) - 1 else 0.0

        if right_mag >= left_mag:
            sign    = 1.0
            adj_mag = right_mag
        else:
            sign    = -1.0
            adj_mag = left_mag

        denom = float(spectrum[k_max]) + float(adj_mag)
        delta = sign * adj_mag / denom if denom > 1e-12 else 0.0

        f_hat = (k_max + delta) * fs / n
        if not (self.MIN_VALID_FREQ_HZ < f_hat < self.MAX_VALID_FREQ_HZ):
            return self.NOMINAL_FREQ_HZ
        return float(f_hat)

    # Backward-compat
    def _step(self, v_sample: float) -> float:
        return self._f_est
