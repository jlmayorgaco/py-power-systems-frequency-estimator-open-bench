from __future__ import annotations
import os
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List

# Métricas que queremos en el paper (puedes añadir o quitar aquí)
CORE_KEYS: List[str] = [
    "RMSE",
    "Nadir_Mag_Err",
    "TRIP_TIME_0p2",
    "SETTLING_0p1",
    "TIME_PER_SAMPLE_US",
    "LATENCY_SAMPLES",
]


def build_mc_table(mc_json: Dict[str, Any]) -> pd.DataFrame:
    """Extrae datos del JSON y devuelve un DataFrame ordenado."""
    rows: List[Dict[str, Any]] = []
    results = mc_json.get("results") or {}

    for sc, sc_data in results.items():
        methods = (sc_data or {}).get("methods") or {}
        for method, mdata in methods.items():
            mdata = mdata or {}
            row: Dict[str, Any] = {"scenario": sc, "method": method}

            # Lógica de compatibilidad de esquemas (Legacy vs Current)
            agg = mdata.get("test_agg", mdata) if isinstance(mdata, dict) else {}

            for metric in CORE_KEYS:
                stats = agg.get(metric, {})
                if isinstance(stats, dict):
                    # Guardamos la media para la tabla principal
                    row[metric] = stats.get("mean")
                else:
                    row[metric] = None
            rows.append(row)

    return pd.DataFrame(rows)


def generate_latex_summary(
    json_path: str, output_path: str = "artifacts/tables/summary_q1.tex"
):
    """
    Toma el JSON, construye el DataFrame y exporta una tabla LaTeX profesional.
    """
    if not os.path.exists(json_path):
        print(f"Error: {json_path} no existe.")
        return

    with open(json_path, "r") as f:
        mc_json = json.load(f)

    # 1. Obtener DataFrame base usando tu lógica mejorada
    df = build_mc_table(mc_json)

    # 2. Post-procesamiento: Conversión de unidades (samples -> ms)
    fs_dsp = mc_json["metadata"].get("fs_dsp_hz", 10000.0)
    if "LATENCY_SAMPLES" in df.columns:
        df["LATENCY_MS"] = df["LATENCY_SAMPLES"] * (1000.0 / fs_dsp)

    # 3. Generar LaTeX
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("% --- Tabla Autogenerada para Journal Q1 ---\n")
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write(
            "\\caption{Performance comparison: Proposed RA-EKF vs State-of-the-art estimators.}\n"
        )
        f.write("\\label{tab:main_results}\n")
        f.write("\\begin{tabular}{l l c c c c}\n\\hline\n")
        f.write(
            "Scenario & Method & RMSE (Hz) & Nadir Err. (Hz) & Trip (s) & Latency (ms) \\\\ \\hline\n"
        )

        # Agrupamos por escenario para que la tabla sea legible
        for scenario, group in df.groupby("scenario"):
            first_row = True
            sc_clean = scenario.replace("_", " ")

            for _, row in group.iterrows():
                method = str(row["method"]).replace("_", "-")
                # Resaltar tu propuesta en negrita
                is_proposed = "RA-EKF" in method

                # Formateo de valores
                rmse = f"{row['RMSE']:.5f}" if pd.notnull(row["RMSE"]) else "N/A"
                nadir = (
                    f"{row['Nadir_Mag_Err']:.4f}"
                    if pd.notnull(row["Nadir_Mag_Err"])
                    else "N/A"
                )
                trip = (
                    f"{row['TRIP_TIME_0p2']:.3f}"
                    if pd.notnull(row["TRIP_TIME_0p2"])
                    else "--"
                )
                lat = (
                    f"{row['LATENCY_MS']:.1f}"
                    if pd.notnull(row["LATENCY_MS"])
                    else "--"
                )

                sc_col = sc_clean if first_row else ""

                line = f"{sc_col} & {method} & {rmse} & {nadir} & {trip} & {lat} \\\\\n"
                if is_proposed:
                    f.write("\\rowcolor[gray]{0.9} % Resaltar fila propuesta\n")
                    f.write(
                        f"\\textbf{{{sc_col}}} & \\textbf{{{method}}} & \\textbf{{{rmse}}} & \\textbf{{{nadir}}} & \\textbf{{{trip}}} & \\textbf{{{lat}}} \\\\\n"
                    )
                else:
                    f.write(line)

                first_row = False
            f.write("\\hline\n")

        f.write("\\end{tabular}\n\\end{table*}")

    # También guardamos un CSV por si quieres verlo en Excel rápido
    df.to_csv(output_path.replace(".tex", ".csv"), index=False)
    print(f"[OK] Tablas Q1 generadas en artifacts/tables/")


# Alias para mantener compatibilidad con results_manager.py
def build_metrics_table(results_json):
    return build_mc_table(results_json)
