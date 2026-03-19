# scenarios/registry.py
from __future__ import annotations
from typing import Dict, Tuple, Any

import numpy as np

# Importamos la fábrica maestra que creamos en ibg_events.py
from .ibg_events import get_test_signal, get_all_scenario_names


def get_test_signals(
    fs: float = 2400.0, T: float = 5.0, seed: int = 123
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Registro central de los 16 escenarios para el Journal Q1.

    Argumentos:
        fs: Frecuencia de muestreo (DSP) pasada desde el JSON de entrada.
        T: Duración de cada simulación.
        seed: Semilla base para reproducibilidad de Monte Carlo.

    Returns:
        dict: mapeo 'G_E_Name' -> (t, v, f_true, meta)
    """

    scenarios = {}

    # Obtenemos la lista de los 16 nombres definidos en ibg_events.py
    all_names = get_all_scenario_names()

    for name in all_names:
        # Generamos cada señal usando el fs correcto
        # Esto elimina el error de los 11 Hz al sincronizar señal y estimador
        t, v, f, meta = get_test_signal(name, fs=fs, T=T, seed=seed)
        scenarios[name] = (t, v, f, meta)

    return scenarios
