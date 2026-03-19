import pandas as pd
import os
import json
import numpy as np
from typing import Dict, Any


class WaveformExporter:
    """
    Exportador jerárquico de señales y metadatos.
    Estructura: artifacts/waveforms/scenario/method/estimacion.csv
    Consolida: artifacts/waveforms/scenario/all.csv
    """

    def __init__(self, base_path: str = "artifacts/waveforms"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def export_scenario(
        self,
        scenario_id: str,
        t: np.ndarray,
        real_f: np.ndarray,
        real_v: np.ndarray,
        estimates: Dict[str, np.ndarray],
        methods_info: Dict[str, Any],
        seed: int = 0,
    ):
        """
        Exporta todos los datos estructurados de un escenario completo.
        """
        # 1. Crear carpeta del escenario
        scenario_path = os.path.join(self.base_path, scenario_id)
        os.makedirs(scenario_path, exist_ok=True)

        # --- EXPORTAR all.csv (El "Master" para PGFPlots) ---
        all_data = {"t": t, "real_f": real_f, "real_v": real_v}
        for m_name, f_hat in estimates.items():
            clean_name = m_name.replace("-", "_").replace(" ", "_")
            all_data[clean_name] = f_hat

        pd.DataFrame(all_data).to_csv(
            os.path.join(scenario_path, "all.csv"), index=False, float_format="%.6f"
        )

        # --- EXPORTAR POR MÉTODO (Estructura Jerárquica) ---
        for m_name, f_hat in estimates.items():
            method_path = os.path.join(scenario_path, m_name.replace(" ", "_"))
            os.makedirs(method_path, exist_ok=True)

            # A. frecuencia_estimada.csv
            pd.DataFrame({"t": t, "f_hat": f_hat}).to_csv(
                os.path.join(method_path, "estimated_f.csv"),
                index=False,
                float_format="%.6f",
            )

            # B. meta.json (Explicación técnica de la simulación)
            meta = {
                "scenario": scenario_id,
                "method": m_name,
                "seed": seed,
                "params": methods_info.get(m_name, {}),
                "fs_hz": 1.0 / (t[1] - t[0]) if len(t) > 1 else 0,
            }
            with open(os.path.join(method_path, "meta.json"), "w") as f:
                json.dump(meta, f, indent=4)

        print(f"[OK] Datos exportados en: {scenario_path}")


# Instancia global
exporter = WaveformExporter()
