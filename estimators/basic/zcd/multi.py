# estimators/zcd/multi.py
# ---------------------------------------------------------------------
# Three-phase ZCD estimator: runs one ZCD core per phase and aggregates.
# Depends on estimators/zcd/core.py (ZCDEstimatorBase, ZCDConfig).
# ---------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List
from estimators.zcd.core import ZCDEstimatorBase, ZCDConfig
from estimators.base import EstimatorBase
from utils.pmu.pmu_input import PMU_Input
from utils.pmu.pmu_output import PMU_Output

def _agg(vals: List[float], mode: str) -> float:
    if not vals:
        return 0.0
    if mode == "mean":
        return float(sum(vals) / len(vals))
    # default: median
    vals_sorted = sorted(vals)
    m = len(vals_sorted) // 2
    return float((vals_sorted[m] if len(vals_sorted) % 2 else (vals_sorted[m - 1] + vals_sorted[m]) / 2))

class ZCDMulti(EstimatorBase):
    """
    ZCD for 3-phase signals (or N selected channels).
    Config keys (optional):
      - channels: list[str] (default ["V1","V2","V3"])
      - epsilon: float deadband around zero (default 0.0)
      - nominal_hz: fallback frequency before first crossing (default 60.0)
      - mode: "neg_to_pos" | "pos_to_neg" | "either" (default "neg_to_pos")
      - agg: "median" | "mean"  (default "median")
    """

    def __init__(self, config, name: str = "zcd_multi", profile: str = "M"):
        super().__init__(config=config, name=name, profile=profile)

        # read config with dict/attr compatibility
        def g(key, default):
            return getattr(config, key, default) if hasattr(config, key) else config.get(key, default)

        self.channels: List[str] = g("channels", ["V1", "V2", "V3"])
        eps = float(g("epsilon", 0.0))
        nominal = float(g("nominal_hz", 60.0))
        mode = g("mode", "neg_to_pos")
        self.agg_mode = g("agg", "median")

        self.cores: Dict[str, ZCDEstimatorBase] = {
            ch: ZCDEstimatorBase(ZCDConfig(epsilon=eps, nominal_hz=nominal, mode=mode))
            for ch in self.channels
        }

    def reset(self) -> None:
        for core in self.cores.values():
            core.reset()
        super().reset()

    def update(self, measures: PMU_Input) -> PMU_Output:
        ts = float(getattr(measures, "timestamp"))

        f_list, r_list = [], []
        for ch, core in self.cores.items():
            if not hasattr(measures, ch):
                continue
            x = float(getattr(measures, ch))
            f, r, _crossed, _tc = core.update_scalar(x, ts)
            f_list.append(f); r_list.append(r)

        # aggregate across phases (robust by default via median)
        f_hat = _agg(f_list, self.agg_mode)
        r_hat = _agg(r_list, self.agg_mode)

        # Optionally pass-through instantaneous values as phasor placeholders
        phasors = {ch: complex(float(getattr(measures, ch, 0.0)), 0.0) for ch in self.channels}

        return PMU_Output(
            phasors=phasors,
            frequency_hz=float(f_hat),
            rocof_hz_s=float(r_hat),
            timestamp_utc=ts,
        )
