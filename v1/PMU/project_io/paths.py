from __future__ import annotations
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ProjectPaths:
    """
    Canonical project output structure (Q1 reproducibility-friendly).

    base_dir/
      results_raw/<scenario>/[seed_<seed>/] <scenario>__<method>.json
      results_mc/<scenario>__mc_summary.json
      results_mc/mc_results.json
      figures/... (owned by plotting layer, but we keep reference)
      manifest.json
    """
    base_dir: str = "."
    results_raw_dir: str = "results_raw"
    results_mc_dir: str = "results_mc"
    figures_dir: str = "figures_estimatores_benchmark"

    def ensure(self) -> None:
        os.makedirs(self.results_raw_root(), exist_ok=True)
        os.makedirs(self.results_mc_root(), exist_ok=True)
        os.makedirs(self.figures_root(), exist_ok=True)

    def results_raw_root(self) -> str:
        return os.path.join(self.base_dir, self.results_raw_dir)

    def results_mc_root(self) -> str:
        return os.path.join(self.base_dir, self.results_mc_dir)

    def figures_root(self) -> str:
        return os.path.join(self.base_dir, self.figures_dir)

    def manifest_path(self) -> str:
        return os.path.join(self.base_dir, "manifest.json")

    # ---- Raw deterministic runs
    def raw_run_path(self, scenario: str, method: str) -> str:
        return os.path.join(self.results_raw_root(), scenario, f"{scenario}__{method}.json")

    # ---- Raw MC per-seed runs
    def raw_seed_run_path(self, scenario: str, seed: int, method: str) -> str:
        return os.path.join(self.results_raw_root(), scenario, f"seed_{seed}", f"{scenario}__{method}.json")

    # ---- Monte Carlo summaries
    def mc_scenario_summary_path(self, scenario: str) -> str:
        return os.path.join(self.results_mc_root(), f"{scenario}__mc_summary.json")

    def mc_global_path(self) -> str:
        return os.path.join(self.results_mc_root(), "mc_results.json")
