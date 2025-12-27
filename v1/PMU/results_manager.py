import os
import argparse
# Importamos tus funciones de ploteo y tablas existentes
from viz.journal_plots import plot_comparison, plot_nadir_zoom
from viz.tables import generate_latex_summary
# Importamos las funciones del Mega Dashboard
from viz.mega_dashboard import (
    create_dynamic_dashboard, 
    create_global_dashboard, 
    create_physics_dashboard
)

def main():
    parser = argparse.ArgumentParser(description="Q1 Post-Processing Pipeline - Jorge Luis Mayorga")
    parser.add_argument("--scenario", type=str, help="Escenario específico (ej: G3_E12_Composite)")
    parser.add_argument("--table", action="store_true", help="Generar tabla LaTeX y CSV")
    parser.add_argument("--zoom", action="store_true", help="Generar zoom al Nadir en plots individuales")
    parser.add_argument("--mega", action="store_true", help="Generar los 3 Mega Dashboards (Dinámica, Global, Física)")
    args = parser.parse_args()

    base_path = "artifacts/waveforms"
    
    # 1. Procesamiento de Escenarios Individuales
    # Solo se ejecuta si no se pide EXCLUSIVAMENTE el mega dashboard, o si se especifica un escenario
    if args.scenario or (not args.mega and not args.table):
        scenarios = [args.scenario] if args.scenario else os.listdir(base_path)
        for sc in scenarios:
            sc_dir = os.path.join(base_path, sc)
            if not os.path.isdir(sc_dir): continue
            
            print(f"\n>>> Analizando Escenario: {sc}")
            
            # Plot de comparación general (PDF + PNG)
            plot_comparison(sc)
            
            # Zoom al Nadir para auditoría de latencia
            if args.zoom:
                plot_nadir_zoom(sc)

    # 2. Generación de los Mega Dashboards (Las 3 páginas para el paper)
    if args.mega:
        print("\n" + "="*50)
        print(">>> GENERANDO MEGA DASHBOARDS (ESTRATEGIA Q1)")
        print("="*50)
        
        create_dynamic_dashboard()  # Página 1: Respuesta Dinámica Crítica
        print("[1/3] Dashboard de Dinámica: Completado.")
        
        create_global_dashboard()   # Página 2: Paisaje Estadístico y Pareto
        print("[2/3] Dashboard Global: Completado.")
        
        create_physics_dashboard()  # Página 3: Causalidad Física (Voltaje vs Freq)
        print("[3/3] Dashboard de Física: Completado.")

    # 3. Tabla de resultados globales (Desde mc_results.json)
    if args.table:
        print("\n>>> Generando Reportes de Métricas...")
        generate_latex_summary("results_mc/mc_results.json")

if __name__ == "__main__":
    main()