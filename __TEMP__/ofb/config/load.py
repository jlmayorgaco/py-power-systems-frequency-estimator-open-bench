import yaml
from .models import BenchmarkCfg

def load_cfg(path: str) -> BenchmarkCfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return BenchmarkCfg.model_validate(raw)
