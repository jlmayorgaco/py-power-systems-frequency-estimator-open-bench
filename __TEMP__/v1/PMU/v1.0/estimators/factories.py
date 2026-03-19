from __future__ import annotations

from typing import Dict, Type
import pkgutil
import importlib
import inspect

from domain.protocols import EstimatorFactory
from .base import BaseEstimator


def _discover_estimators() -> Dict[str, Type[BaseEstimator]]:
    """
    Finds all subclasses of BaseEstimator inside estimators.* modules.
    Key = class.NAME (string shown to user / CLI).
    """
    out: Dict[str, Type[BaseEstimator]] = {}

    pkg = importlib.import_module(__package__)  # "estimators"
    for m in pkgutil.iter_modules(pkg.__path__, prefix=pkg.__name__ + "."):
        modname = m.name
        try:
            mod = importlib.import_module(modname)
        except Exception:
            # Import may fail if optional deps missing (torch, etc).
            # We just skip those modules.
            continue

        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if obj is BaseEstimator:
                continue
            if issubclass(obj, BaseEstimator):
                name = getattr(obj, "NAME", None)
                if isinstance(name, str) and name.strip():
                    out[name.strip()] = obj

    return out


def build_method_factories() -> Dict[str, EstimatorFactory]:
    classes = _discover_estimators()
    return {
        name: (lambda cls: (lambda p: cls(p)))(cls) for name, cls in classes.items()
    }
