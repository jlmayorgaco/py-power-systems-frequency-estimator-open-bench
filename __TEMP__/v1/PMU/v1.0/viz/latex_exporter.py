import json
import pandas as pd
import os


def generate_latex_table(json_path, output_tex):
    """
    Lee los resultados del benchmark y genera una tabla LaTeX profesional
    enfocada en precisión, estándares IEEE y dinámica de red.
    """
    if not os.path.exists(json_path):
        print(f"Error: No se encontró el archivo de resultados en {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    scenarios = data["results"].keys()

    # Métricas críticas para un artículo Q1
    # RMSE: Precisión general
    # FE_max_mHz: Cumplimiento IEEE C37.118.1
    # Settling_Time: Rapidez de recuperación (crítico en el paper de Chamorro)
    # Nadir_Mag_Err: Precisión en el punto más bajo de la falla
    # TIME_PER_SAMPLE_US: Viabilidad en tiempo real
    target_metrics = [
        "RMSE",
        "FE_max_mHz",
        "Settling_Time",
        "Nadir_Mag_Err",
        "TIME_PER_SAMPLE_US",
    ]

    rows = []
    for sc in scenarios:
        methods = data["results"][sc]["methods"]
        for m_name, m_data in methods.items():
            # Limpiamos el nombre del escenario para que se vea bien en LaTeX
            clean_sc = sc.replace("_", " ")
            row = {"Escenario": clean_sc, "Método": m_name}

            for met in target_metrics:
                if met in m_data:
                    # Extraemos el promedio (mean) del Monte Carlo
                    val = m_data[met].get("mean", 0.0)
                    # Formateo inteligente: 4 decimales para precisión, 2 para tiempo CPU
                    if "TIME" in met:
                        row[met] = f"{val:.2f}"
                    else:
                        row[met] = f"{val:.4f}"
                else:
                    row[met] = "N/A"
            rows.append(row)

    df = pd.DataFrame(rows)

    # Estructura de la tabla LaTeX con formato profesional (booktabs)
    latex_code = df.to_latex(
        index=False,
        caption="Comparativa de Desempeño: Precisión Estática, Estándares IEEE y Respuesta Dinámica",
        label="tab:benchmark_results",
        column_format="lp{3.5cm}ccccc",
        escape=True,
    )

    # Añadimos retoques estéticos típicos de Journal (ajuste de ancho)
    latex_final = (
        "\\begin{table*}[t]\n\\centering\n\\small\n" + latex_code + "\\end{table*}"
    )

    with open(output_tex, "w") as f:
        f.write(latex_final)

    print(f"¡Éxito! Tabla LaTeX generada en: {output_tex}")


if __name__ == "__main__":
    # Ruta relativa desde viz/ hacia project_io/
    json_input = "../project_io/mc_results.json"
    output_file = "tabla_resultados_final.tex"
    generate_latex_table(json_input, output_file)
