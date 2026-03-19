# estimators/zcd/multi.py
# ---------------------------------------------------------------------
# Three-phase ZCD estimator: runs one ZCD core per phase and aggregates.
# Depends on estimators/zcd/core.py (ZCDEstimatorBase, ZCDConfig).
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Literal

from estimators.base import EstimatorBase
from estimators.zcd.core import ZCDConfig, ZCDEstimatorBase
from utils.pmu.pmu_input import PMU_Input
from utils.pmu.pmu_output import PMU_Output, PhasorName, PhasorMap


def _agg(vals: list[float], mode: Literal["median", "mean"]) -> float:
    if not vals:
        return 0.0
    if mode == "mean":
        return float(sum(vals) / len(vals))
    # default: median
    vals_sorted = sorted(vals)
    m = len(vals_sorted) // 2
    return float(
        vals_sorted[m] if len(vals_sorted) % 2 else (vals_sorted[m - 1] + vals_sorted[m]) / 2
    )


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

    def __init__(
        self,
        config: dict[str, Any] | Any,
        name: str = "zcd_multi",
        profile: Literal["P", "M"] = "M",
    ) -> None:
        super().__init__(config=config, name=name, profile=profile)

        # read config with dict/attr compatibility
        def g(key: str, default: Any) -> Any:
            return (
                getattr(config, key, default) if hasattr(config, key) else config.get(key, default)
            )

        # Filtra al conjunto permitido para satisfacer PhasorName
        valid: tuple[PhasorName, ...] = ("V1", "V2", "V3", "I1", "I2", "I3")
        cfg_channels = list(g("channels", ["V1", "V2", "V3"]))
        self.channels: list[PhasorName] = [c for c in cfg_channels if c in valid] or [
            "V1",
            "V2",
            "V3",
        ]

        eps = float(g("epsilon", 0.0))
        nominal = float(g("nominal_hz", 60.0))
        mode = str(g("mode", "neg_to_pos"))
        self.agg_mode: Literal["median", "mean"] = g("agg", "median")

        self.cores: dict[PhasorName, ZCDEstimatorBase] = {
            ch: ZCDEstimatorBase(ZCDConfig(epsilon=eps, nominal_hz=nominal, mode=mode))
            for ch in self.channels
        }

    def reset(self) -> None:
        for core in self.cores.values():
            core.reset()
        super().reset()

    def update(self, measures: PMU_Input) -> PMU_Output:
        # Prefer direct attribute access over getattr for fixed names (ruff B009)
        ts: float = float(measures.timestamp)

        f_list: list[float] = []
        r_list: list[float] = []
        for ch, core in self.cores.items():
            # ch es PhasorName, garantizado en self.channels
            x = float(getattr(measures, ch, 0.0))
            f, r, _crossed, _tc = core.update_scalar(x, ts)
            f_list.append(f)
            r_list.append(r)

        # aggregate across phases (robust by default via median)
        f_hat = _agg(f_list, self.agg_mode)
        r_hat = _agg(r_list, self.agg_mode)

        # Phasors con claves tipadas como PhasorName (cumple PhasorMap)
        phasors: dict[PhasorName, complex] = {
            ch: complex(float(getattr(measures, ch, 0.0)), 0.0) for ch in self.channels
        }
        phasors_typed: PhasorMap = phasors

        return PMU_Output(
            phasors=phasors_typed,
            frequency_hz=float(f_hat),
            rocof_hz_s=float(r_hat),
            timestamp_utc=ts,
        )
