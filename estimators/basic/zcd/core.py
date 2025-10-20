# estimators/zcd/core.py
# ---------------------------------------------------------------------
# Reusable Zero-Crossing (ZCD) core utilities for frequency & RoCoF.
# Intended to be imported by:
#   - estimators/zcd/single.py       (single-phase)
#   - estimators/zcd/multi.py        (three-phase aggregation)
#   - estimators/zcd/distributed.py  (multi-node/distributed)
#
# Provides:
#   - ZCDConfig: configuration dataclass
#   - ZCDState:  minimal streaming state
#   - detect_crossing(): robust linear-interp crossing detector
#   - ZCDEstimatorBase: base with update_scalar(value, ts)
#
# Dependencies: numpy (optional), stdlib only otherwise.
# ---------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

try:
    import numpy as np  # optional (only used for sign checks/convenience)
except Exception:  # pragma: no cover
    np = None  # type: ignore


# --------------------------- Config & State ---------------------------

CrossingMode = Literal["neg_to_pos", "pos_to_neg", "either"]


@dataclass
class ZCDConfig:
    """Runtime knobs for ZCD."""
    epsilon: float = 0.0                # deadband around zero
    nominal_hz: float = 60.0            # fallback when no crossings yet
    mode: CrossingMode = "neg_to_pos"   # which crossing to detect
    min_period_s: float = 1e-6          # ignore absurdly small periods
    max_period_s: float = 1.0           # ignore absurdly large periods (outliers)


@dataclass
class ZCDState:
    """Streaming state across samples."""
    prev_val: Optional[float] = None
    prev_ts: Optional[float] = None
    last_cross_ts: Optional[float] = None
    prev_cross_ts: Optional[float] = None
    last_freq: Optional[float] = None
    prev_freq: Optional[float] = None


# ------------------------- Core functionality -------------------------

def _sign(x: float, eps: float) -> int:
    """Signed region with deadband: -1, 0, +1."""
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def detect_crossing(
    prev_val: float,
    prev_ts: float,
    curr_val: float,
    curr_ts: float,
    eps: float = 0.0,
    mode: CrossingMode = "neg_to_pos",
) -> Tuple[bool, Optional[float]]:
    """
    Detect a zero crossing between two samples and linearly interpolate the
    crossing time.

    Returns:
        (crossed, t_cross)
    """
    s0 = _sign(prev_val, eps)
    s1 = _sign(curr_val, eps)

    crossed = False
    if mode == "neg_to_pos":
        crossed = (s0 == -1) and (s1 >= 0)
    elif mode == "pos_to_neg":
        crossed = (s0 == +1) and (s1 <= 0)
    else:  # "either"
        crossed = (s0 != 0) and (s1 != 0) and (s0 != s1)

    if not crossed:
        return False, None

    # Linear interpolation between the two samples:
    dx = curr_val - prev_val
    if dx == 0.0:
        return True, curr_ts  # degenerate; fall back to current timestamp
    alpha = (-prev_val) / dx  # fraction in [0,1] ideally
    t_cross = prev_ts + (curr_ts - prev_ts) * alpha
    return True, float(t_cross)


class ZCDEstimatorBase:
    """
    Minimal helper you can mix into an EstimatorBase subclass.
    It encapsulates the ZCD state machine and returns (freq, rocof, crossed, t_cross)
    when you feed scalar samples through `update_scalar(value, ts)`.

    Typical usage in a concrete estimator:
        self.freq_hz, self.rocof_hz_s, crossed, t_cross = \
            self.zcd.update_scalar(value=v_a, ts=timestamp)

    Then construct and return your PMU_Output as desired.
    """

    def __init__(self, cfg: Optional[ZCDConfig] = None):
        self.cfg = cfg or ZCDConfig()
        self.state = ZCDState()

    def reset(self) -> None:
        self.state = ZCDState()

    # Core streaming update
    def update_scalar(self, value: float, ts: float) -> Tuple[float, float, bool, Optional[float]]:
        """
        Feed one scalar sample with timestamp (seconds).

        Returns:
            freq_hz, rocof_hz_s, crossed, t_cross
        """
        st = self.state
        cfg = self.cfg

        crossed = False
        t_cross = None

        if st.prev_val is not None and st.prev_ts is not None:
            crossed, t_cross = detect_crossing(
                st.prev_val, st.prev_ts, value, ts, eps=cfg.epsilon, mode=cfg.mode
            )
            if crossed:
                if st.last_cross_ts is not None:
                    period = float(t_cross - st.last_cross_ts)
                    # Filter absurd periods:
                    if cfg.min_period_s <= period <= cfg.max_period_s:
                        freq = 1.0 / period
                        st.prev_freq = st.last_freq
                        st.last_freq = freq
                        st.prev_cross_ts = st.last_cross_ts
                        st.last_cross_ts = t_cross
                else:
                    st.last_cross_ts = t_cross

        # advance sample history
        st.prev_val = float(value)
        st.prev_ts = float(ts)

        # frequency: last known or nominal
        freq_hz = float(st.last_freq) if st.last_freq is not None else float(cfg.nominal_hz)

        # rocof via consecutive crossing-based freqs
        rocof_hz_s = 0.0
        if (
            st.last_freq is not None
            and st.prev_freq is not None
            and st.last_cross_ts is not None
            and st.prev_cross_ts is not None
        ):
            dt = float(st.last_cross_ts - st.prev_cross_ts)
            if dt > 0.0:
                rocof_hz_s = float((st.last_freq - st.prev_freq) / dt)

        return freq_hz, rocof_hz_s, crossed, t_cross


# ------------------------ Tiny local smoke test -----------------------

class ZCDCoreTester:
    """Utility to sanity-check the core without PMU wiring."""
    def __init__(self, cfg: Optional[ZCDConfig] = None):
        self.core = ZCDEstimatorBase(cfg)

    def run_sine(self, fs: float = 10000.0, f: float = 60.0, seconds: float = 0.2):
        import math
        n = int(fs * seconds)
        dt = 1.0 / fs
        t0 = 0.0
        out = []
        for k in range(n):
            t = t0 + k * dt
            x = math.sin(2.0 * math.pi * f * t)
            freq, rocof, crossed, t_cross = self.core.update_scalar(x, t)
            out.append((t, x, freq, rocof, crossed, t_cross))
        return out
