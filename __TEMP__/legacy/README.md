# Legacy Code Archive

This directory contains **archived** implementations that preceded the `src/openfreqbench/` canonical package.

## Why it exists

The original research system lived in two places:

| Path | Description |
|------|-------------|
| `v1/PMU/pfebench/` | **Scientific source of truth** — real estimators, scenarios, metrics, Monte Carlo pipeline, Q1 artifacts |
| `ofb/` | Modern skeleton (Pydantic + Typer + Registry) with stub runner returning `TVE_mean: 0.0` |
| `estimators/`, `scenarios/`, `evaluation/` | Old legacy with PMU phasor contract |

## Status

`v1/PMU/pfebench/` is **NOT archived** — it remains the reference implementation until numerical parity with `src/openfreqbench/` is verified for every ported estimator and scenario.

Once parity is confirmed per estimator:
1. The estimator is marked ✅ in the parity table (see `docs/reports/`).
2. The corresponding `pfebench/` file is moved here as reference.

## Do NOT import from legacy/ in production code

These files exist for regression testing and archaeology. They may not install cleanly without `pfebench` on `PYTHONPATH`.

## Parity tracking

See `docs/reports/openfreqbench_tickets_and_roadmap.md` ticket P1-01 for the parity verification checklist.
