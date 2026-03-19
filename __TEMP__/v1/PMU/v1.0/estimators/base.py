from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import time


@dataclass
class EstimatorInfo:
    name: str
    family: str
    latency_samples: int
    params: Dict[str, Any]
    last_exec_us: float = 0.0


class BaseEstimator:
    """
    Base class:
    - Centraliza perfilamiento CPU (us) en step()
    - Centraliza latencia y params
    Subclases deben implementar _step(v_sample) (solo matemática).
    """

    NAME: str = "Base"
    FAMILY: str = "Base"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self._params: Dict[str, Any] = dict(params or {})
        self.last_execution_time_us: float = 0.0
        self.reset()

    def reset(self) -> None:
        """Opcional: subclases pueden inicializar estados aquí."""
        return

    def _step(self, v_sample: float) -> float:
        """
        Subclases implementan este método.
        Debe retornar f_hat [Hz] (float).
        """
        raise NotImplementedError(
            "Debes implementar _step(self, v_sample) en la subclase"
        )

    def step(self, v_sample: float) -> float:
        """
        NO REESCRIBIR en subclases.
        Wrapper de timing + llamada a la matemática.
        """
        t_start = time.perf_counter_ns()
        f_hat = float(self._step(float(v_sample)))
        t_end = time.perf_counter_ns()
        self.last_execution_time_us = (t_end - t_start) / 1000.0

        # Guardrail anti-NaN/inf (opcional pero MUY recomendable para runs largos)
        if not np.isfinite(f_hat):
            # Si quieres fail-fast:
            # raise FloatingPointError(f"{self.NAME} produced non-finite f_hat={f_hat}")
            # Si quieres soft-fail:
            f_hat = float("nan")
        return f_hat

    @property
    def latency_samples(self) -> int:
        # Yo recomiendo default 0, no 1 (1 introduce sesgo si no hay ventana real)
        return int(self._params.get("latency_samples", 0))

    def info(self) -> EstimatorInfo:
        return EstimatorInfo(
            name=self.NAME,
            family=self.FAMILY,
            latency_samples=self.latency_samples,
            params=dict(self._params),
            last_exec_us=self.last_execution_time_us,
        )

    def run(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        out = np.empty(len(v), dtype=float)
        for i, x in enumerate(v):
            out[i] = self.step(float(x))
        return out
