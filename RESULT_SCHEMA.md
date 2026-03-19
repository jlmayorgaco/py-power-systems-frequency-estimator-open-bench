# Result Schema

## Artifact Layout

```
artifacts/
└── <scenario_id>/
    └── <method_id>/
        └── <method_id>_<YYYYMMDD_HHMMSS>_report.json
```

## report.json Schema

```json
{
  "scenario_id": "G1_E1_Pure_60Hz",
  "method_id": "ZeroCrossing",
  "seed": 0,
  "schema_version": "v1",
  "metrics": {
    "RMSE_HZ": 0.012,
    "MAE_HZ": 0.010,
    ...
  },
  "aggregated": {
    "RMSE_HZ": {"mean": 0.012, "std": 0.001, "max": 0.015, "p95": 0.014, "n": 100}
  }
}
```
