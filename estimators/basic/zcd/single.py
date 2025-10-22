from __future__ import annotations

from typing import Any, Literal, cast

from estimators.base import EstimatorBase
from utils.pmu.pmu_input import PMU_Input
from utils.pmu.pmu_output import PMU_Output

from .core import ZCDConfig, ZCDEstimatorBase


def _cfg_get(cfg: Any, attr: str, key: str, default: Any) -> Any:
    """Fetch config value from object attribute or mapping key (fallback to default)."""
    try:
        return getattr(cfg, attr)
    except AttributeError:
        try:
            return cfg[key]
        except (KeyError, TypeError):
            return default


def _normalize_mode(
    value: Any, default: Literal["neg_to_pos"] = "neg_to_pos"
) -> Literal["neg_to_pos", "pos_to_neg", "either"]:
    """Coerce arbitrary config value to a valid ZCD mode literal."""
    allowed = {"neg_to_pos", "pos_to_neg", "either"}
    s = str(value) if value is not None else ""
    return cast(Literal["neg_to_pos", "pos_to_neg", "either"], s if s in allowed else default)


class ZCDSingle(EstimatorBase):
    def __init__(
        self, config: Any, name: str = "zcd_single", profile: Literal["P", "M"] = "M"
    ) -> None:
        super().__init__(config=config, name=name, profile=profile)

        eps: float = float(_cfg_get(config, "epsilon", "epsilon", 0.0))
        if eps == 0.0:
            eps = float(_cfg_get(config, "eps", "eps", 0.0))

        nominal: float = float(_cfg_get(config, "nominal_hz", "nominal_hz", 60.0))
        mode = _normalize_mode(_cfg_get(config, "mode", "mode", "neg_to_pos"))

        self.zcd = ZCDEstimatorBase(ZCDConfig(epsilon=eps, nominal_hz=nominal, mode=mode))
        self.channel: str = str(_cfg_get(config, "channel", "channel", "V1"))

    def update(self, measures: PMU_Input) -> PMU_Output:
        x = float(getattr(measures, self.channel))
        ts = float(measures.timestamp)  # Ruff B009: direct attribute access
        f, r, _crossed, _t_cross = self.zcd.update_scalar(x, ts)
        return PMU_Output(
            phasors={},
            frequency_hz=float(f),
            rocof_hz_s=float(r),
            timestamp_utc=ts,
        )
