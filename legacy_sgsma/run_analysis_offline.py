#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
from scientific_analysis import run_all_scientific_analyses

# ==========================================
# TRADUCTOR PARA TIPOS DE NUMPY -> JSON NATIVO
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    """ Convierte tipos de Numpy (bool_, float64, int64) a tipos estándar de Python para JSON. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):  # <-- Soluciona el error TypeError: Object of type bool is not JSON serializable
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def main():
    print("==========================================================")
    print(" INICIANDO ANÁLISIS CIENTÍFICO OFFLINE (PAPER EXTENSION)  ")
    print("==========================================================")
    print("Cargando datos precalculados del benchmark...")

    # 1. Cargar el JSON principal
    benchmark_file = "benchmark_results.json"
    if not os.path.exists(benchmark_file):
        print(f"❌ Error: No se encuentra {benchmark_file}. Córrelo desde el directorio correcto.")
        return

    with open(benchmark_file, "r", encoding="utf-8") as f:
        full_data = json.load(f)
    
    results_dict = full_data.get("results", {})

    # 2. Reconstruir 'tuned_params' 
    tuned_params = {}
    for sc_name, sc_data in results_dict.items():
        tuned_params[sc_name] = {}
        for method, m_data in sc_data.get("methods", {}).items():
            if "tuned_params" in m_data:
                tuned_params[sc_name][method] = m_data["tuned_params"]

    # 3. Cargar raw_mc.json para los análisis estadísticos avanzados (Levene, Mann-Whitney)
    raw_mc = {}
    raw_mc_path = os.path.join("results_raw", "raw_mc.json")
    
    if os.path.exists(raw_mc_path):
        with open(raw_mc_path, "r", encoding="utf-8") as f:
            raw_mc_serialisable = json.load(f)
        
        # Convertir llaves "Metodo__Escenario" a tuplas ("Metodo", "Escenario")
        for key, val in raw_mc_serialisable.items():
            if "__" in key:
                method, sc_name = key.split("__", 1)
                raw_mc[(method, sc_name)] = val
        print(f"✅ Datos Monte Carlo cargados: {len(raw_mc)} combinaciones.")
    else:
        print(f"⚠️ ADVERTENCIA: No se encontró {raw_mc_path}.")
        print("Los test de varianza y robustez estocástica se saltarán.")

    # 4. Extraer el barrido de fase si existe en el JSON
    phase_jump_data = full_data.get("statistical_analysis", {}).get("phase_jump_sweep", {})

    # 5. Ejecutar los análisis científicos
    print("\nEjecutando suite de análisis científicos y pruebas de hipótesis...")
    try:
        sci_results = run_all_scientific_analyses(
            results_dict=results_dict,
            mc_raw=raw_mc,
            tuned_params=tuned_params,
            phase_jump_data=phase_jump_data
        )
        
        # 6. Guardar los resultados usando el NumpyEncoder para evitar el TypeError
        output_file = "scientific_analysis_extended.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sci_results, f, indent=4, cls=NumpyEncoder)
            
        print(f"\n✅ ¡Análisis completado exitosamente! ")
        print(f"Resultados guardados en: {output_file}")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()