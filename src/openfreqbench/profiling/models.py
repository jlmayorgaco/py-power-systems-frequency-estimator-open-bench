"""Profiling result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimingResult:
    exec_time_s: float
    time_per_sample_us: float
    n_samples: int


@dataclass
class MemoryResult:
    peak_rss_kb: float
    peak_tracemalloc_kb: float
