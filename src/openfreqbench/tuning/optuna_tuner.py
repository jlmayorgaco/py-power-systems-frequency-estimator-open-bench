"""
openfreqbench/tuning/optuna_tuner.py

Bayesian Optimization Engine for Estimator Hyperparameters.
Operates on the TuningSpec of an estimator class.
"""

from __future__ import annotations

import logging
from typing import Any, Type

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from openfreqbench.estimators.common.base import BaseEstimator
from openfreqbench.scenarios._base import ScenarioBase
from openfreqbench.runners.scenario_runner import ScenarioRunner
from openfreqbench.config.benchmark_config import BenchmarkConfig

log = logging.getLogger("ofb.tuning")

class OptunaTuner:
    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials
        if not HAS_OPTUNA:
            log.warning("Optuna is not installed. Bayesian tuning will be skipped. Install via pip install optuna")

    def tune_estimator(self, est_class: Type[BaseEstimator], scenarios: list[ScenarioBase], base_config: BenchmarkConfig) -> dict[str, Any]:
        if not HAS_OPTUNA:
            return est_class.default_config()

        spec = est_class.tuning_spec()
        if not spec.params:
            return est_class.default_config()

        def objective(trial: optuna.Trial) -> float:
            # Reconstruct config from trial suggestions
            cfg = est_class.default_config()
            for p in spec.params:
                if p.type == "float":
                    cfg[p.name] = trial.suggest_float(p.name, min(p.values), max(p.values))
                elif p.type == "int":
                    cfg[p.name] = trial.suggest_int(p.name, min(p.values), max(p.values))
                else:
                    cfg[p.name] = trial.suggest_categorical(p.name, p.values)
                    
            # Instantiate with trial config
            est = est_class(cfg)
            runner = ScenarioRunner(store=None)
            
            total_rmse = 0.0
            valid_scenarios = 0
            
            for scen in scenarios:
                res = runner.run(scen, est, base_config)
                # Parse aggregated RMSE_HZ
                if res and res.metrics:
                    rmse = res.metrics.get("RMSE_HZ", {}).get("value", None)
                    if rmse is not None and not math.isnan(rmse):
                        total_rmse += rmse
                        valid_scenarios += 1
                        
            if valid_scenarios == 0:
                raise optuna.TrialPruned("No valid results computed.")
                
            return total_rmse / valid_scenarios

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, catch=(Exception,))
        
        best_cfg = est_class.default_config()
        best_cfg.update(study.best_params)
        log.info(f"Tuning {est_class.SPEC.name} complete. Best Params: {study.best_params}")
        
        return best_cfg
