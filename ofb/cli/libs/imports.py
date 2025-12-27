from __future__ import annotations
from typing import Callable, Tuple
from ofb.cli.libs.errors import echo_err

__all__ = ["lazy_registry", "lazy_bench", "lazy_reports"]

def lazy_registry() -> Tuple[Callable[..., list], Callable[..., dict]]:
    """
    Return (list_items, get_meta) with compatibility for legacy names,
    and trigger auto-discovery on first use.
    """
    try:
        import ofb.core.registry as reg  # type: ignore
        # Trigger discovery so registration side-effects run
        if hasattr(reg, "discover"):
            reg.discover()
        # Prefer new API names
        if hasattr(reg, "list_items") and hasattr(reg, "get_meta"):
            return reg.list_items, reg.get_meta  # type: ignore[return-value]
        # Fallback to legacy names
        if hasattr(reg, "list_") and hasattr(reg, "meta"):
            return reg.list_, reg.meta  # type: ignore[return-value]
        raise RuntimeError("registry functions not found (list_items/get_meta or list_/meta)")
    except Exception as e:
        echo_err(f"core.registry not available: {e}")
        raise  # unreachable: echo_err exits


def lazy_bench() -> Tuple[Callable[..., dict], Callable[..., list], Callable[..., bool]]:
    """
    Resolve benchmark/config functions.

    Returns
    -------
    (load_cfg, plan_matrix, run_matrix)
        - load_cfg(path: Path) -> dict
        - plan_matrix(cfg_obj: dict) -> list[dict]
        - run_matrix(matrix: list[dict], *, outdir, seed, fail_fast) -> bool
    """
    try:
        from ofb.config.load import load_cfg  # type: ignore
        from ofb.benchmarks.runner import plan_matrix, run_matrix  # type: ignore
        return load_cfg, plan_matrix, run_matrix  # type: ignore[return-value]
    except Exception as e:  # pragma: no cover
        echo_err(f"benchmarks/config not available: {e}")
        raise  # unreachable, echo_err exits


def lazy_reports() -> Callable[..., None]:
    """
    Resolve standard report builder.

    Returns
    -------
    build_standard_report(results_dir: Path, outdir: Path) -> None
    """
    try:
        from ofb.reports.standard import build_standard_report  # type: ignore
        return build_standard_report  # type: ignore[return-value]
    except Exception as e:  # pragma: no cover
        echo_err(f"reports.standard not available: {e}")
        raise  # unreachable, echo_err exits
