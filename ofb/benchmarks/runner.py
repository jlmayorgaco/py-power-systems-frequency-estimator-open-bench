from __future__ import annotations
from pathlib import Path
import pandas as pd

def plan_benchmark(cfg) -> dict:
    ns = len(cfg.scenarios); ne = len(cfg.estimators)
    return {"benchmark": cfg.benchmark["name"], "num_scenarios": ns, "num_estimators": ne,
            "num_runs_est": ns * ne}

def run_benchmark(cfg, jobs=1, resume=True, results_dir=None, seed=None, mapping_path=None, show_progress=True):
    data = [{"scenario": s.name, "estimator": e.name, "TVE_mean": 0.0}
            for s in cfg.scenarios for e in cfg.estimators]
    df = pd.DataFrame(data)
    out_dir = Path(results_dir or f"results/{cfg.benchmark['name']}")
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tables" / "summary_wide.csv", index=False)
    return df

def render_reports(cfg, results_dir=None): return None

def calibrate_then_map(cfg, jobs=1, p=0.95, k=1.1, out_path=None):
    out = Path(out_path or f"results/{cfg.benchmark['name']}/mapping.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"p": %.3f, "k": %.3f}' % (p, k), encoding="utf-8")
    return str(out)
