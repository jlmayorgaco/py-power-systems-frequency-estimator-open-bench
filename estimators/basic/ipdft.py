from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from estimators.base import EstimatorBase
from utils.pmu.pmu_input import PMU_Input
from utils.pmu.pmu_output import PMU_Output, PhasorName


class IpDFT(EstimatorBase):
    """
    Interpolated DFT (IpDFT) frequency estimator with a sliding buffer.

    Config keys (with defaults):
      - fs: float                 (required) sampling rate [Hz]
      - frame_len: int            (default 50) samples per frame
      - channel: str              (default "V1") PMU_Input field to read
      - nominal_hz: float         (default 60.0) used before buffer fills
    """

    def __init__(
        self,
        config: Mapping[str, Any] | Any,
        name: str = "ipdft",
        profile: Literal["P", "M"] = "M",
    ) -> None:
        super().__init__(config=config, name=name, profile=profile)

        def g(key: str, default: Any) -> Any:
            return (
                getattr(config, key, default) if hasattr(config, key) else config.get(key, default)
            )

        self.fs: float = float(g("fs", 0.0))
        if self.fs <= 0.0:
            raise ValueError("IpDFT requires config['fs'] > 0")

        self.frame_len: int = int(g("frame_len", 50))
        if self.frame_len <= 2:
            raise ValueError("IpDFT requires frame_len >= 3")

        self.channel: str = str(g("channel", "V1"))
        self.nominal_hz: float = float(g("nominal_hz", 60.0))

        self.buffer: NDArray[np.float64] = np.zeros(self.frame_len, dtype=float)
        self.ptr: int = 0
        self.filled: bool = False

        self._last_freq: float | None = None
        self._last_ts: float | None = None

    def reset(self) -> None:
        self.buffer.fill(0.0)
        self.ptr = 0
        self.filled = False
        self._last_freq = None
        self._last_ts = None
        super().reset()

    def _estimate_freq(self) -> float:
        """Return IpDFT frequency [Hz] from the current full buffer."""
        X: NDArray[np.complex128] = np.fft.fft(self.buffer, n=self.frame_len)
        half = self.frame_len // 2
        mag: NDArray[np.float64] = np.abs(X[:half])
        k: int = int(np.argmax(mag))  # 0..half-1
        if 1 <= k < len(mag) - 1:
            denom = mag[k - 1] - 2.0 * mag[k] + mag[k + 1]
            delta = 0.5 * (mag[k - 1] - mag[k + 1]) / denom if denom != 0.0 else 0.0
        else:
            delta = 0.0
        return float((k + delta) * self.fs / self.frame_len)

    def update(self, measures: PMU_Input) -> PMU_Output:
        # time & sample
        ts: float = float(measures.timestamp)
        x: float = float(getattr(measures, self.channel, 0.0))

        # slide buffer
        self.buffer[self.ptr] = x
        self.ptr = (self.ptr + 1) % self.frame_len
        if not self.filled and self.ptr == 0:
            self.filled = True

        # frequency estimate (use nominal until buffer fills)
        f_hat: float = self._estimate_freq() if self.filled else self.nominal_hz

        # rocof (finite difference when we have a previous estimate with Δt > 0)
        if self._last_freq is not None and self._last_ts is not None:
            dt = ts - self._last_ts
            r_hat = float((f_hat - self._last_freq) / dt) if dt > 0.0 else 0.0
        else:
            r_hat = 0.0

        self._last_freq = f_hat
        self._last_ts = ts

        # phasor placeholders — standard keys to satisfy Literal typings
        phasors: dict[PhasorName, complex] = {
            "V1": complex(float(getattr(measures, "V1", 0.0)), 0.0),
            "V2": complex(float(getattr(measures, "V2", 0.0)), 0.0),
            "V3": complex(float(getattr(measures, "V3", 0.0)), 0.0),
            "I1": complex(float(getattr(measures, "I1", 0.0)), 0.0),
            "I2": complex(float(getattr(measures, "I2", 0.0)), 0.0),
            "I3": complex(float(getattr(measures, "I3", 0.0)), 0.0),
        }

        return PMU_Output(
            timestamp_utc=ts,
            frequency_hz=f_hat,
            rocof_hz_s=r_hat,
            phasors=phasors,
        )
