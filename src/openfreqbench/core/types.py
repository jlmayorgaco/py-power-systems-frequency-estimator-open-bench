"""Shared type aliases used across the package."""

from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray
ParamsDict = dict[str, Any]
MetricDict = dict[str, Any]
AggDict = dict[str, dict[str, float]]
