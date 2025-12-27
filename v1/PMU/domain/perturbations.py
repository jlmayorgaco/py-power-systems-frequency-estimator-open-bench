from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class PerturbationConfig:
    amp_pct: float = 0.05
    sigma_rel: float = 0.0005
    impulsive_prob: float = 0.002
    impulsive_scale: float = 6.0


def apply_perturbations(
    v: np.ndarray,
    rng: np.random.Generator,
    cfg: PerturbationConfig,
) -> np.ndarray:
    """
    Apply amplitude jitter + Gaussian + impulsive noise.
    """
    out = v.astype(float).copy()

    # Amplitude jitter
    g = 1.0 + rng.uniform(-cfg.amp_pct, cfg.amp_pct)
    out *= g

    # Gaussian noise
    sigma = cfg.sigma_rel * np.std(out) if np.std(out) > 0 else cfg.sigma_rel
    out += rng.normal(0.0, sigma, size=out.shape)

    # Impulsive noise
    mask = rng.random(out.shape[0]) < cfg.impulsive_prob
    if np.any(mask):
        out[mask] += rng.normal(
            0.0, cfg.impulsive_scale * sigma, size=int(mask.sum())
        )

    return out
