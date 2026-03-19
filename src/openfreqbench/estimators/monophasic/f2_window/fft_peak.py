"""
estimators/monophasic/f2_window/fft_peak.py  [CANONICAL]

FFTPeakEstimator — Hann-windowed FFT with parabolic peak interpolation.

Algorithm
─────────
  1. Divide the signal into overlapping frames (window_size, 50% hop).
  2. Per frame:
       a. Apply a Hann window.
       b. Compute the one-sided DFT via rfft.
       c. Find the bin with maximum magnitude in [f_min, f_max].
       d. Refine the peak using parabolic (Quinn's) interpolation:
            α = |X[k−1]|,  β = |X[k]|,  γ = |X[k+1]|
            δ = 0.5·(α − γ) / (α − 2β + γ)
            f̂ = (k + δ) · fs / N
  3. Each output sample is assigned the estimate of the nearest frame centre.

Complexity:  O(N log N) per frame.
Latency:     window_size // 2 samples (semi-causal).

References
──────────
  Quinn, B.G. (1994). "Estimating frequency by interpolation using
  Fourier coefficients." IEEE Trans. Signal Process., 42(5), 1264–1268.
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


class FFTPeakEstimator(BaseEstimator):
    """
    Hann-windowed FFT with Quinn's parabolic peak interpolation.

    Block-based: overrides ``run()`` for frame-level batch processing.
    ``update()`` returns the last frame's estimate (causal stub).
    """

    SPEC = EstimatorSpec(
        name="FFTPeak",
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
        return {"fs": 10_000.0, "window_size": 512}

    @classmethod
    def tuning_spec(cls) -> TuningSpec:
        return TuningSpec(
            params=[
                TuningParam(
                    name="window_size",
                    default=512,
                    type="int",
                    values=[64, 128, 256, 512, 1024, 2048],
                    description=(
                        "FFT window length (samples). "
                        "Larger → finer frequency resolution but higher latency "
                        "and slower transient tracking. "
                        "At fs=10 kHz: 512→51.2 ms, 256→25.6 ms."
                    ),
                )
            ],
            objective="RMSE_HZ",
        )

    def reset(self) -> None:
        self._window_size: int   = int(self._config.get("window_size", 512))
        self._f_est:       float = self.NOMINAL_FREQ_HZ

    def structural_latency_samples(self) -> int:
        return int(self._config.get("window_size", 512)) // 2

    def update(self, voltage: float, timestamp: float = 0.0) -> EstimatorOutput:
        """Sample-by-sample stub — delegates actual work to run()."""
        return EstimatorOutput(frequency_hz=self._f_est, valid=False)

    def run(self, v_array: np.ndarray) -> np.ndarray:
        """
        Process full waveform via sliding Hann-windowed FFT.

        Returns f_hat of same length as v_array.
        """
        self.reset()
        v  = np.asarray(v_array, dtype=float)
        n  = len(v)
        ws = self._window_size
        fs = float(self._config.get("fs", 10_000.0))

        out = np.full(n, self.NOMINAL_FREQ_HZ, dtype=float)
        if n < ws:
            out[:] = self._fft_peak(v, fs)
            return out

        hop  = max(1, ws // 2)
        hann = np.hanning(ws)

        centres: List[int]   = []
        freqs:   List[float] = []

        for start in range(0, n - ws + 1, hop):
            frame = v[start: start + ws] * hann
            freqs.append(self._fft_peak(frame, fs))
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

    def _fft_peak(self, frame: np.ndarray, fs: float) -> float:
        """Parabolic-interpolated FFT peak within the valid frequency band."""
        spectrum  = np.abs(np.fft.rfft(frame))
        n         = len(frame)
        freq_bins = np.fft.rfftfreq(n, d=1.0 / fs)

        lo = int(np.searchsorted(freq_bins, self.MIN_VALID_FREQ_HZ))
        hi = int(np.searchsorted(freq_bins, self.MAX_VALID_FREQ_HZ))
        if hi <= lo or lo >= len(spectrum):
            return self.NOMINAL_FREQ_HZ

        sub      = spectrum[lo:hi]
        peak_bin = lo + int(np.argmax(sub))

        if 1 <= peak_bin < len(spectrum) - 1:
            alpha = float(spectrum[peak_bin - 1])
            beta  = float(spectrum[peak_bin])
            gamma = float(spectrum[peak_bin + 1])
            denom = alpha - 2.0 * beta + gamma
            delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
        else:
            delta = 0.0

        f_hat = (peak_bin + delta) * fs / n
        if not (self.MIN_VALID_FREQ_HZ < f_hat < self.MAX_VALID_FREQ_HZ):
            return self.NOMINAL_FREQ_HZ
        return float(f_hat)

    # Backward-compat
    def _step(self, v_sample: float) -> float:
        return self._f_est
