# utils/pmu/pmu_output.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import Literal

import numpy as np

__all__ = ["PhasorName", "PhasorMap", "PMUStatus", "PMUClass", "PMU_Output"]

PhasorName = Literal["V1", "V2", "V3", "I1", "I2", "I3"]
PhasorMap = Mapping[PhasorName, complex]


class PMUStatus(IntFlag):
    OK = 0x0000
    DATA_ERROR = 0x0001
    CLOCK_NOT_SYNCED = 0x0002
    PLL_UNLOCKED = 0x0004
    OVER_RANGE = 0x0008


class PMUClass(str, Enum):
    P = "P"  # protection profile
    M = "M"  # measurement profile


@dataclass(slots=True)
class PMU_Output:
    phasors: PhasorMap
    frequency_hz: float
    rocof_hz_s: float
    timestamp_utc: float
    status_word: PMUStatus = PMUStatus.OK

    def to_standard_dict(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {
            "TIMESTAMP_UTC": float(self.timestamp_utc),
            "FREQUENCY_HZ": float(self.frequency_hz),
            "ROCOF_HZ_S": float(self.rocof_hz_s),
            "STATUS_WORD": int(self.status_word),
        }
        for name, p in self.phasors.items():
            out[f"{name}_MAG"] = float(np.abs(p))
            out[f"{name}_ANGLE_RAD"] = float(np.angle(p))
        return out
