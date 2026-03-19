"""
pfebench/estimators/base.py

Arquitectura Base para Estimadores (Q1 Research Grade).
Características:
  1. Estandarización: step(), run(), reset().
  2. Profiling: Medición precisa de tiempo de ejecución (last_exec_us).
  3. Tuning System: Definición de rangos y Optimización Automática (GSO).
"""

from __future__ import annotations

import time
import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np

# Importamos métricas básicas para el optimizador (GSO)
try:
    from pfebench.metrics.metrics import (
        rmse,
        mae,
        frequency_error_mhz,
        rocof_error_rmse,
    )
except ImportError:
    # Fallback básico si no se encuentra el paquete completo aun
    def rmse(e, r):
        return np.sqrt(np.mean((e - r) ** 2))

    def mae(e, r):
        return np.mean(np.abs(e - r))

    def frequency_error_mhz(e, r):
        return np.max(np.abs(e - r)) * 1000

    def rocof_error_rmse(e, r, fs):
        return 0.0


# =============================================================================
# 1. DEFINICIÓN DE PARÁMETROS DE TUNING
# =============================================================================


@dataclass
class TuningParam:
    """
    Define un grado de libertad del estimador para el Grid Search.
    """

    name: str
    default: Any
    type: Literal["int", "float", "bool", "categorical"]
    # Para Grid Search:
    values: Optional[List[Any]] = None  # Lista explícita: [10, 20, 50]
    range: Optional[Tuple[float, float, int]] = None  # (min, max, steps)
    # Metadatos para documentación automática
    description: str = ""

    def generate_grid(self) -> List[Any]:
        """Expande la definición a una lista de valores probables."""
        if self.values is not None:
            return self.values

        if self.range is not None:
            min_v, max_v, steps = self.range
            if self.type == "int":
                # np.unique para evitar duplicados por redondeo
                vals = np.linspace(min_v, max_v, steps)
                return np.unique(np.round(vals)).astype(int).tolist()
            elif self.type == "float":
                return np.linspace(min_v, max_v, steps).tolist()

        return [self.default]


@dataclass
class EstimatorMeta:
    name: str
    family: str
    params: Dict[str, Any]
    latency: int


# =============================================================================
# 2. CLASE BASE CON OPTIMIZADOR (GSO)
# =============================================================================


class BaseEstimator(ABC):
    """
    Clase madre para todos los algoritmos.
    Incluye el motor de optimización (GSO).
    """

    NAME: str = "Base"
    FAMILY: str = "Generic"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        # 1. Cargar defaults de la definición de tuning
        defaults = {p.name: p.default for p in self.tuning_ranges()}

        # 2. Aplicar parámetros de usuario
        self._params = defaults.copy()
        if params:
            self._params.update(params)

        # 3. Estado
        self.last_exec_us: float = 0.0
        self._t_start_ns: int = 0

        # 4. Boot
        self.reset()

    # --- INTERFAZ A IMPLEMENTAR POR EL USUARIO ---

    @abstractmethod
    def _step(self, v_sample: float) -> float:
        """Matemática pura del estimador. Retorna f_hat [Hz]."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reinicia buffers y estado interno."""
        pass

    @property
    @abstractmethod
    def latency_samples(self) -> int:
        """Latencia intrínseca (para alineación causal)."""
        return 0

    @classmethod
    def tuning_ranges(cls) -> List[TuningParam]:
        """
        Define el espacio de búsqueda para este estimador.
        Ejemplo: return [TuningParam("window", 10, "int", values=[10, 20, 40])]
        """
        return []

    # --- PORT TUNING & GSO (Grid Search Optimization) ---

    def set_params(self, **kwargs) -> None:
        """
        Cambia parámetros al vuelo y resetea el estimador.
        Usado por Monte Carlo y GSO.
        """
        changed = False
        for k, v in kwargs.items():
            if k in self._params:
                self._params[k] = v
                changed = True

        if changed:
            self.reset()

    def optimize(
        self,
        v: np.ndarray,
        t: np.ndarray,
        f_true: np.ndarray,
        metric: str = "RMSE_HZ",
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], float]:
        """
        GSO: Grid Search Optimization.
        Recibe la señal completa y busca la mejor configuración de parámetros
        para minimizar la métrica indicada.

        Retorna: (best_params, best_score)
        """
        # 1. Generar Grid
        definitions = self.tuning_ranges()
        if not definitions:
            return self._params, float("nan")

        param_names = [p.name for p in definitions]
        param_grids = [p.generate_grid() for p in definitions]

        # Producto cartesiano de todas las combinaciones
        combinations = list(itertools.product(*param_grids))

        best_score = float("inf")
        best_cfg = self._params.copy()

        if verbose:
            print(f"[{self.NAME}] Iniciando GSO. Combinaciones: {len(combinations)}")

        # 2. Fuerza Bruta
        # Nota: Esto asume que v, t, f_true están limpios y alineados (RAW input).
        fs = 1.0 / np.mean(np.diff(t)) if len(t) > 1 else 60.0

        # Pre-alineación simple para métrica rápida (recortar al tamaño mínimo)
        n_eval = min(len(v), len(f_true))
        f_true_eval = f_true[:n_eval]

        for vals in combinations:
            # Crear config dict
            current_cfg = dict(zip(param_names, vals))

            # Inyectar y Correr
            self.set_params(**current_cfg)

            try:
                # Run rápido
                f_est = self.run(v[:n_eval])

                # Calcular Costo (Métrica)
                score = self._calculate_cost(f_est, f_true_eval, metric, fs)

                # Minimizar
                if score < best_score:
                    best_score = score
                    best_cfg = current_cfg.copy()

            except Exception:
                # Si una configuración explota, la ignoramos
                continue

        # 3. Restaurar ganador
        self.set_params(**best_cfg)
        return best_cfg, best_score

    def _calculate_cost(
        self, f_est: np.ndarray, f_ref: np.ndarray, metric: str, fs: float
    ) -> float:
        """Helper interno para calcular el costo durante la optimización."""
        # Recortar warm-up (hardcoded simple para GSO: 10% inicial o 5 muestras)
        warmup = min(10, len(f_est) // 10)
        e = f_est[warmup:]
        r = f_ref[warmup:]

        if len(e) == 0:
            return float("inf")

        if metric == "RMSE_HZ":
            return rmse(e, r)
        elif metric == "MAE_HZ":
            return mae(e, r)
        elif metric == "FE_MAX":
            return frequency_error_mhz(e, r)
        elif "ROCOF" in metric:
            # Requiere derivada, más costoso
            return rocof_error_rmse(e, r, fs)

        # Default
        return rmse(e, r)

    # --- EJECUCIÓN ESTÁNDAR ---

    def step(self, v_sample: float) -> float:
        self._t_start_ns = time.perf_counter_ns()
        try:
            val = float(self._step(float(v_sample)))
        except Exception:
            val = float("nan")

        dt = time.perf_counter_ns() - self._t_start_ns
        self.last_exec_us = dt / 1000.0

        if math.isnan(val) or math.isinf(val):
            return float("nan")
        return val

    def run(self, v_array: np.ndarray) -> np.ndarray:
        v = np.asarray(v_array, dtype=float)
        out = np.empty_like(v)
        self.reset()
        for i, x in enumerate(v):
            out[i] = self.step(x)
        return out

    @property
    def info(self) -> EstimatorMeta:
        return EstimatorMeta(
            name=self.NAME,
            family=self.FAMILY,
            params=self._params.copy(),
            latency=self.latency_samples,
        )
