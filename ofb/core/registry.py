from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
import importlib
import pkgutil

# ---------------- Registry storage ----------------

KIND_KEYS = ("estimators", "scenarios", "metrics", "reports")
_REGISTRY: Dict[str, List[Dict[str, Any]]] = {k: [] for k in KIND_KEYS}
_DISCOVERED = False


@dataclass
class Item:
    id: str
    category: Optional[str] = None
    meta: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"id": self.id}
        if self.category is not None:
            d["category"] = self.category
        if self.meta:
            d.update(self.meta)
        return d


# --------------- Core API ----------------

def register(kind: str, id: str, *, category: Optional[str] = None, **meta: Any) -> None:
    """Register an item with kind in {estimators,scenarios,metrics,reports}."""
    if kind not in KIND_KEYS:
        raise ValueError(f"Unknown kind '{kind}'. Expected one of {KIND_KEYS}.")
    # De-dup by id
    bucket = _REGISTRY[kind]
    bucket[:] = [x for x in bucket if x.get("id") != id]
    bucket.append(Item(id=id, category=category, meta=meta).to_dict())


def list_items(kind: str, *, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """List items, optionally filtered by category. Auto-discovers first time."""
    ensure_discovered()
    if kind not in KIND_KEYS:
        raise ValueError(f"Unknown kind '{kind}'. Expected one of {KIND_KEYS}.")
    items = list(_REGISTRY[kind])
    if category:
        items = [i for i in items if i.get("category") == category]
    # sort by id for stable output
    return sorted(items, key=lambda x: x["id"])


def get_meta(kind: str, item_id: str) -> Dict[str, Any]:
    """Fetch metadata for one item. Auto-discovers first time."""
    ensure_discovered()
    if kind not in KIND_KEYS:
        raise ValueError(f"Unknown kind '{kind}'. Expected one of {KIND_KEYS}.")
    for i in _REGISTRY[kind]:
        if i.get("id") == item_id:
            return i
    return {}


# --------------- Decorator helpers (optional) ----------------

def register_estimator(id: str, *, category: Optional[str] = None, **meta: Any) -> Callable:
    def _decorator(obj: Any) -> Any:
        register("estimators", id, category=category, **meta)
        return obj
    return _decorator

def register_scenario(id: str, *, category: Optional[str] = None, **meta: Any) -> Callable:
    def _decorator(obj: Any) -> Any:
        register("scenarios", id, category=category, **meta)
        return obj
    return _decorator

def register_metric(id: str, *, category: Optional[str] = None, **meta: Any) -> Callable:
    def _decorator(obj: Any) -> Any:
        register("metrics", id, category=category, **meta)
        return obj
    return _decorator

def register_report(id: str, *, category: Optional[str] = None, **meta: Any) -> Callable:
    def _decorator(obj: Any) -> Any:
        register("reports", id, category=category, **meta)
        return obj
    return _decorator


# --------------- Auto-discovery ----------------

_BASE_PACKAGES = (
    "ofb.estimators",
    "ofb.scenarios",
    "ofb.metrics",
    "ofb.reports",
)

def ensure_discovered() -> None:
    """Import all submodules under base packages once, to trigger registrations."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    autodiscover(_BASE_PACKAGES)
    _DISCOVERED = True

def autodiscover(packages: tuple[str, ...] = _BASE_PACKAGES) -> None:
    """Walk packages and import every submodule to run their register() side-effects."""
    for pkg_name in packages:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception:
            continue  # package missing is okay (optional components)

        # If it's a namespace pkg without __path__, skip
        pkg_path = getattr(pkg, "__path__", None)
        if not pkg_path:
            continue

        for mod_info in pkgutil.walk_packages(pkg_path, prefix=pkg_name + "."):
            name = mod_info.name
            try:
                importlib.import_module(name)
            except Exception:
                # Keep going; a broken optional module shouldn't stop discovery
                continue
