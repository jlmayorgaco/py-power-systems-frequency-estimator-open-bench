from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import time  # <--- Fundamental para medir nanosegundos

@dataclass
class EstimatorInfo:
    name: str
    family: str
    latency_samples: int
    params: Dict[str, Any]
    last_exec_us: float = 0.0

class BaseEstimator:
    """
    Clase base. Centraliza el perfilamiento de CPU y la gestión de latencia.
    """
    NAME: str = "Base"
    FAMILY: str = "Base"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self._params: Dict[str, Any] = params or {}
        self.last_execution_time_us: float = 0.0
        self.reset()

    def reset(self) -> None:
        pass

    def _step(self, v_sample: float) -> float:
        """
        NO REESCRIBIR ESTE MÉTODO en los estimadores.
        Mide el tiempo y llama internamente a _step.
        """
        t_start = time.perf_counter_ns()
        
        # Ejecuta la lógica matemática definida en la subclase
        f_hat = self._step(v_sample)
        
        t_end = time.perf_counter_ns()
        
        # Cálculo de microsegundos (us)
        self.last_execution_time_us = (t_end - t_start) / 1000.0
        return f_hat

    def step(self, v_sample: float) -> float:
        """
        REESCRIBIR ESTE MÉTODO en kalman.py, sogi.py, etc.
        """
        raise NotImplementedError("Debes implementar _step(self, v_sample)")

    @property
    def latency_samples(self) -> int:
        return int(self._params.get("latency_samples", 1))

    def info(self) -> EstimatorInfo:
        return EstimatorInfo(
            name=self.NAME,
            family=self.FAMILY,
            latency_samples=self.latency_samples,
            params=dict(self._params),
            last_exec_us=self.last_execution_time_us
        )

    def run(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros(len(v), dtype=float)
        for i, x in enumerate(v):
            out[i] = self.step(float(x))
        return out