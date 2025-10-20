from estimators.base import EstimatorBase
from utils.pmu.pmu_input import PMU_Input
from utils.pmu.pmu_output import PMU_Output
from .core import ZCDEstimatorBase, ZCDConfig

class ZCDSingle(EstimatorBase):
    def __init__(self, config, name="zcd_single", profile="M"):
        super().__init__(config=config, name=name, profile=profile)
        eps = getattr(config, "epsilon", getattr(config, "eps", 0.0)) if hasattr(config, "__dict__") else config.get("epsilon", 0.0)
        nominal = getattr(config, "nominal_hz", 60.0) if hasattr(config, "__dict__") else config.get("nominal_hz", 60.0)
        mode = getattr(config, "mode", "neg_to_pos") if hasattr(config, "__dict__") else config.get("mode", "neg_to_pos")
        self.zcd = ZCDEstimatorBase(ZCDConfig(epsilon=eps, nominal_hz=nominal, mode=mode))
        self.channel = getattr(config, "channel", "V1") if hasattr(config, "__dict__") else config.get("channel", "V1")

    def update(self, measures: PMU_Input) -> PMU_Output:
        x = float(getattr(measures, self.channel))
        ts = float(getattr(measures, "timestamp"))
        f, r, crossed, t_cross = self.zcd.update_scalar(x, ts)
        return PMU_Output(phasors={}, frequency_hz=f, rocof_hz_s=r, timestamp_utc=ts)
