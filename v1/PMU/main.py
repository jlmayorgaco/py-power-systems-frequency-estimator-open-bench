#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entry point (Q1 Research-Grade - M3 Pro Optimized):
- Robust metadata handling (Fix ExperimentConfig error)
- Multi-core execution via ProcessPoolExecutor (Fix empty JSON)
- Flexible result aggregation
"""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import sys
import concurrent.futures
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

# ---------------------------
# Infrastructure / IO
# ---------------------------
from project_io.config import load_config
from project_io.paths import ProjectPaths
from project_io.writers import JsonWriter
from project_io.manifest import build_manifest

# ---------------------------
# Domain / Experiment
# ---------------------------
from experiments import (
    ExperimentConfig,
    build_registry,
)

# ---------------------------
# Scenarios provider
# ---------------------------
from scenarios import get_test_signals

def _sys_metadata() -> Dict[str, Any]:
    return {
        "timestamp": str(datetime.datetime.now()),
        "hostname": platform.node(),
        "machine_arch": platform.machine(),
        "cpu_processor": platform.processor(),
        "os_platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "argv": sys.argv[:],
    }

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _asdict_safe(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    return x

def _expect_dict(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg)

# =========================================================
# WORKER PARA PARALELISMO (Ejecución en núcleos del M3 Pro)
# =========================================================
def run_scenario_task(scenario_item, exp_cfg, registry_cfg):
    """
    Worker que procesa un escenario individual.
    """
    name, signal = scenario_item
    try:
        # Importaciones locales para evitar problemas de 'Pickling' en macOS
        from experiments import MonteCarloRunner, build_registry
        
        local_registry = build_registry(registry_cfg)
        mc_runner = MonteCarloRunner(exp_cfg, local_registry)
        
        # Ejecutamos Monte Carlo para este escenario
        # Se espera que devuelva un dict con métricas
        result = mc_runner.run({name: signal})
        return name, result
    except Exception as e:
        return name, {"error": str(e)}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--mode", type=str, default="mc", choices=["mc", "single", "figures"])
    p.add_argument("--config", type=str, default="configs/mc_default.json")
    p.add_argument("--outdir", type=str, default="artifacts")
    p.add_argument("--workers", type=int, default=-1, help="Núcleos a usar (-1 para auto)")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    outdir = _ensure_dir(args.outdir)
    paths = ProjectPaths(base_dir=outdir)
    paths.ensure()
    writer = JsonWriter()

    # 1) Carga de configuración
    cfg_raw = load_config(args.config)
    if not cfg_raw:
        print(f"[ERROR] No se pudo cargar {args.config}")
        return 1
    cfg_dict = _expect_dict(cfg_raw)

    # 2) Experiment config - FIX METADATA
    # Extraemos 'experiment' y limpiamos 'metadata' para evitar TypeError
    exp_data = cfg_dict.get("experiment", {}).copy()
    exp_data.pop("metadata", None) 
    exp_cfg = ExperimentConfig(**exp_data)

    # 3) Inyección de parámetros al Registry
    registry_cfg: Dict[str, Any] = _expect_dict(cfg_dict.get("registry", {}))
    registry_cfg["fs_dsp_hz"] = float(exp_cfg.fs_dsp_hz)
    registry_cfg["tuners"] = cfg_dict.get("tuners", {})
    
    # Manifest para trazabilidad del paper
    writer.write(paths.manifest_path(), build_manifest(
        experiment_cfg={"experiment": _asdict_safe(exp_cfg), "mode": args.mode},
        code_paths=["main.py"],
        extra={"system": _sys_metadata()},
    ))

    # 4) Generación de Señales
    signals = get_test_signals(fs=exp_cfg.fs_dsp_hz, T=5.0, seed=exp_cfg.seed)
    print(f"🔍 Escenarios detectados: {len(signals)}")
    if not signals:
        print("[ERROR] No se generaron señales. Revisa scenarios/.")
        return 1

    # =========================================================
    # MODO MONTE CARLO (PARALELO)
    # =========================================================
    if args.mode == "mc":
        num_cores = os.cpu_count() if args.workers == -1 else args.workers
        print(f"🚀 Iniciando M3 Pro Parallel Mode | Cores: {num_cores}")

        final_mc_results = {}
        scenario_list = list(signals.items())

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Enviamos todos los escenarios a los núcleos
            futures = {
                executor.submit(run_scenario_task, item, exp_cfg, registry_cfg): item[0] 
                for item in scenario_list
            }

            for future in concurrent.futures.as_completed(futures):
                s_name = futures[future]
                try:
                    name, result = future.result()
                    
                    # --- Lógica de Agregación Robusta ---
                    if isinstance(result, dict):
                        # Si el resultado ya viene indexado por el nombre del escenario
                        if name in result:
                            final_mc_results[name] = result[name]
                        # Si el resultado es directamente el dict de métodos/métricas
                        elif len(result) > 0:
                            final_mc_results[name] = result
                        
                        if "error" in result:
                            print(f"❌ Error en worker {name}: {result['error']}")
                        else:
                            print(f"✅ Escenario Guardado: {name}")
                    else:
                        print(f"⚠️ Formato desconocido para {name}")

                except Exception as exc:
                    print(f"❌ Error crítico procesando {s_name}: {exc}")

        # 5) Guardar Resultados (Aquí ya no saldrá vacío)
        print(f"📊 Finalizado: {len(final_mc_results)} / {len(signals)} escenarios.")
        mc_path = os.path.join(paths.results_mc_root(), "mc_results.json")
        writer.write(mc_path, final_mc_results)
        print(f"💾 JSON definitivo guardado en: {mc_path}")
        return 0

    # Modo Single (Secuencial para debugging rápido)
    if args.mode == "single":
        from experiments import BenchmarkRunner, build_registry
        registry = build_registry(registry_cfg)
        runner = BenchmarkRunner(exp_cfg, registry)
        results = runner.run(signals)
        writer.write(os.path.join(paths.figures_root(), "benchmark_results.json"), results)
        print("✅ Single mode complete.")
        return 0

    return 0

if __name__ == "__main__":
    # Importante en macOS: spawnar procesos requiere este bloque
    main()