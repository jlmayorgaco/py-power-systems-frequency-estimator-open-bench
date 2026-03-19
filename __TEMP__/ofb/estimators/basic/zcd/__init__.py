from ofb.core.registry import register

# Simple, human-friendly names; IDs should be stable and unique.

register(
    "estimators",
    id="basic.zcd.single",
    category="single",
    name="ZCD (single-channel)",
    module="ofb.estimators.basic.zcd.single",  # where the implementation lives
    entry="Estimator",                          # optional: class/function name to instantiate
    description="Zero-Crossing Detector (single input).",
)

register(
    "estimators",
    id="basic.zcd.multi",
    category="multi",
    name="ZCD (multi-channel)",
    module="ofb.estimators.basic.zcd.multi",
    entry="Estimator",
    description="Zero-Crossing Detector (multi input).",
)

register(
    "estimators",
    id="basic.zcd.distributed-single",
    category="distributed-single",
    name="ZCD (distributed single)",
    module="ofb.estimators.basic.zcd.distributed_single",
    entry="Estimator",
    description="Zero-Crossing Detector (distributed, single stream per worker).",
)

register(
    "estimators",
    id="basic.zcd.distributed-multi",
    category="distributed-multi",
    name="ZCD (distributed multi)",
    module="ofb.estimators.basic.zcd.distributed_multi",
    entry="Estimator",
    description="Zero-Crossing Detector (distributed, multi-stream).",
)
