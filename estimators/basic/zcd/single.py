# estimators/zero_crossing.py

import numpy as np

from __future__ import annotations

from typing import Any, Dict

from estimators.base import EstimatorBase
from utils.pmu.pmu_input import PMU_Input
from utils.pmu.pmu_output import PMU_Output


class ZeroCrossing(EstimatorBase):
    """
    Online Zero-Crossing frequency estimator (neg→pos crossings).

    - Uses linear interpolation between samples to time-stamp the crossing.
    - Keeps state in `self.memory`:
        prev_val, prev_ts, last_cross_ts, prev_cross_ts,
        last_freq, prev_freq
    - ROCOF computed from consecutive crossing-based freq estimates.
    - Returns a PMU_Output on every `update` call (using last-known values
      until a new crossing is observed).
    """

    def __init__(self, config: Any, name: str = "zero_crossing", profile: str = "M") -> None:
        super().__init__(config=config, name=name, profile=profile)
        # Initialize memory/state
        self.memory.update(
            prev_val=None,
            prev_ts=None,
            last_cross_ts=None,
            prev_cross_ts=None,
            last_freq=None,
            prev_freq=None,
        )

    def update(self, measures: PMU_Input) -> PMU_Output:
        if not isinstance(measures, PMU_Input):
            raise TypeError("update() requires PMU_Input (single snapshot).")

        # --- Config knobs (with safe defaults) ---
        # channel to monitor for zero-crossing (e.g., "V1", "I1", etc.)
        chan: str = getattr(self.config, "channel", None) if hasattr(self.config, "channel") \
            else self.config.get("channel", "V1")
        eps: float = getattr(self.config, "epsilon", None) if hasattr(self.config, "epsilon") \
            else self.config.get("epsilon", 0.0)
        nominal_hz: float = getattr(self.config, "nominal_hz", None) if hasattr(self.config, "nominal_hz") \
            else self.config.get("nominal_hz", 60.0)

        # --- Read the selected instantaneous value and timestamp ---
        try:
            x = float(getattr(measures, chan))
        except AttributeError:
            raise AttributeError(f"PMU_Input does not have channel '{chan}'")
        ts = float(getattr(measures, "timestamp"))

        prev_val = self.memory["prev_val"]
        prev_ts = self.memory["prev_ts"]

        # Detect neg→pos crossing with a small deadband `eps`
        crossed = False
        t_cross = None

        if prev_val is not None and prev_ts is not None:
            if (prev_val < -eps) and (x >= eps):
                crossed = True
                # Linear interpolation for zero-cross time
                dx = x - prev_val
                if dx != 0.0:
                    alpha = (-prev_val) / dx  # fraction between prev and current sample
                    t_cross = prev_ts + (ts - prev_ts) * alpha
                else:
                    t_cross = ts  # degenerate case

                # Update frequency from consecutive crossings
                last_cross_ts = self.memory["last_cross_ts"]
                if last_cross_ts is not None:
                    period = t_cross - last_cross_ts
                    if period > 0.0:
                        freq = 1.0 / period
                        # shift frequency history
                        self.memory["prev_freq"] = self.memory["last_freq"]
                        self.memory["last_freq"] = freq
                        # shift cross timestamps history
                        self.memory["prev_cross_ts"] = last_cross_ts
                        self.memory["last_cross_ts"] = t_cross
                    else:
                        # invalid period (identical timestamps); ignore
                        pass
                else:
                    # first detected crossing
                    self.memory["last_cross_ts"] = t_cross

        # Update prev sample state
        self.memory["prev_val"] = x
        self.memory["prev_ts"] = ts

        # --- Determine outputs (use last-known when no new crossing) ---
        last_freq = self.memory["last_freq"]
        prev_freq = self.memory["prev_freq"]
        last_cross_ts = self.memory["last_cross_ts"]
        prev_cross_ts = self.memory["prev_cross_ts"]

        # Frequency estimate
        frequency_hz = float(last_freq) if last_freq is not None else float(nominal_hz)

        # ROCOF estimate (Hz/s) from freq diffs between consecutive crossings
        rocof_hz_s = 0.0
        if (last_freq is not None) and (prev_freq is not None) and (last_cross_ts is not None) and (prev_cross_ts is not None):
            dt = last_cross_ts - prev_cross_ts
            if dt > 0.0:
                rocof_hz_s = float((last_freq - prev_freq) / dt)

        # Timestamp policy: emit the current sample's timestamp.
        # (Optionally, use last_cross_ts or the window center; keep it simple here.)
        timestamp_utc = ts

        # Phasors: this estimator focuses on frequency; provide instantaneous placeholders
        # (Replace with real phasors if you compute them elsewhere.)
        phasors: Dict[str, complex] = {
            "V1": complex(getattr(measures, "V1", 0.0), 0.0),
            "V2": complex(getattr(measures, "V2", 0.0), 0.0),
            "V3": complex(getattr(measures, "V3", 0.0), 0.0),
            "I1": complex(getattr(measures, "I1", 0.0), 0.0),
            "I2": complex(getattr(measures, "I2", 0.0), 0.0),
            "I3": complex(getattr(measures, "I3", 0.0), 0.0),
        }

        return PMU_Output(
            phasors=phasors,
            frequency_hz=frequency_hz,
            rocof_hz_s=rocof_hz_s,
            timestamp_utc=timestamp_utc,
            # status_word=...  # include if your PMU_Output supports it
        )
