# results_manager.py
from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, Callable, List

# ------------------------------------------------------------
# Existing per-scenario plots / tables
# ------------------------------------------------------------
from viz.journal_plots import plot_comparison, plot_nadir_zoom
from viz.tables import generate_latex_summary

# ------------------------------------------------------------
# Mega dashboards: robust imports (do not crash if missing)
# ------------------------------------------------------------
create_physics_dashboard: Optional[Callable[[], None]] = None
create_dynamic_dashboard: Optional[Callable[[], None]] = None
create_global_dashboard: Optional[Callable[[], None]] = None

try:
    from viz.mega_dashboard import create_physics_dashboard as _create_physics_dashboard

    create_physics_dashboard = _create_physics_dashboard
except Exception as e:
    print(
        f"[ERROR] Could not import create_physics_dashboard from viz.mega_dashboard: {e}",
        file=sys.stderr,
    )

try:
    from viz.mega_dashboard import create_dynamic_dashboard as _create_dynamic_dashboard

    create_dynamic_dashboard = _create_dynamic_dashboard
except Exception:
    create_dynamic_dashboard = None  # not available, handled gracefully

try:
    from viz.mega_dashboard import create_global_dashboard as _create_global_dashboard

    create_global_dashboard = _create_global_dashboard
except Exception:
    create_global_dashboard = None  # not available, handled gracefully


# ============================================================
# Helpers
# ============================================================
def _die(msg: str, code: int = 2) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(code)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(msg)


def _list_scenarios(base_path: str) -> List[str]:
    if not os.path.exists(base_path):
        _die(
            f"base_path not found: '{base_path}'. Fix path or run the generator first."
        )
    if not os.path.isdir(base_path):
        _die(f"base_path exists but is not a directory: '{base_path}'.")

    items = sorted(os.listdir(base_path))
    sc = [x for x in items if os.path.isdir(os.path.join(base_path, x))]
    if not sc:
        _die(f"No scenario folders found under: '{base_path}'.")
    return sc


def _assert_scenario_exists(base_path: str, sc: str) -> None:
    sc_dir = os.path.join(base_path, sc)
    if not os.path.isdir(sc_dir):
        _die(
            f"Scenario '{sc}' not found at '{sc_dir}'. Check --scenario name and base_path."
        )


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q1 Post-Processing Pipeline - Jorge Luis Mayorga",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Escenario específico (ej: G3_E12_Composite_Islanding)",
    )
    parser.add_argument(
        "--table", action="store_true", help="Generar tabla LaTeX y CSV"
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Generar zoom al Nadir en plots individuales",
    )
    parser.add_argument(
        "--mega",
        action="store_true",
        help="Generar Mega Dashboards (Dinámica, Global, Física)",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="artifacts/results_mc/waveforms",
        help="Ruta base donde viven los escenarios (cada uno con ground_truth.csv, etc.)",
    )
    args = parser.parse_args()

    base_path = args.base_path

    # --------------------------------------------------------
    # 0) Sanity checks
    # --------------------------------------------------------
    if args.mega and create_physics_dashboard is None:
        _die(
            "Mega dashboards requested (--mega) but create_physics_dashboard could not be imported. "
            "Fix viz/mega_dashboard.py exports."
        )

    # --------------------------------------------------------
    # 1) Per-scenario processing
    # Runs if: a) --scenario is set, OR b) not (mega-only or table-only)
    # --------------------------------------------------------
    run_per_scenario = bool(args.scenario) or (not args.mega and not args.table)

    if run_per_scenario:
        _info("\n" + "=" * 60)
        _info(">>> PER-SCENARIO PROCESSING")
        _info("=" * 60)

        if args.scenario:
            _assert_scenario_exists(base_path, args.scenario)
            scenarios = [args.scenario]
        else:
            scenarios = _list_scenarios(base_path)

        _info(f"[INFO] base_path: {base_path}")
        _info(f"[INFO] scenarios to process: {len(scenarios)}")

        for sc in scenarios:
            sc_dir = os.path.join(base_path, sc)
            if not os.path.isdir(sc_dir):
                continue

            _info(f"\n>>> Analizando Escenario: {sc}")

            # Your existing plotting functions may assume internal paths.
            # If they need base_path, update them later; here we keep your API unchanged.
            plot_comparison(sc)

            if args.zoom:
                plot_nadir_zoom(sc)

    # --------------------------------------------------------
    # 2) Mega dashboards
    # --------------------------------------------------------
    if args.mega:
        _info("\n" + "=" * 60)
        _info(">>> GENERANDO MEGA DASHBOARDS (ESTRATEGIA Q1)")
        _info("=" * 60)

        # Dynamic
        if create_dynamic_dashboard is not None:
            create_dynamic_dashboard()
            _info("[1/3] Dashboard de Dinámica: Completado.")
        else:
            _warn(
                "[1/3] create_dynamic_dashboard() no existe en viz/mega_dashboard.py — skipping."
            )

        # Global
        if create_global_dashboard is not None:
            create_global_dashboard()
            _info("[2/3] Dashboard Global: Completado.")
        else:
            _warn(
                "[2/3] create_global_dashboard() no existe en viz/mega_dashboard.py — skipping."
            )

        # Physics (required)
        create_physics_dashboard()
        _info("[3/3] Dashboard de Física: Completado.")

    # --------------------------------------------------------
    # 3) Table / summary
    # --------------------------------------------------------
    if args.table:
        _info("\n" + "=" * 60)
        _info(">>> Generando Reportes de Métricas...")
        _info("=" * 60)

        # Keep your original call as-is
        generate_latex_summary("results_mc/mc_results.json")
        _info("[OK] Tabla/summary generado.")

    _info("\n✅ Done.")


if __name__ == "__main__":
    main()
