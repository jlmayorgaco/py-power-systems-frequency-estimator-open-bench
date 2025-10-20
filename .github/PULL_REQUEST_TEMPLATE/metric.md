### 📏 Metric Implementation PR

**Metric ID:** `<short-name>`
**Implements:** `MetricBase`
**Related Proposal:** #<issue-number>

---

#### 📋 Summary
Briefly describe:
- What does this metric measure?
- Why is it needed in the benchmark?
- Units and interpretation.

---

#### ⚙️ Implementation
- [ ] Implemented in `src/metrics/<id>.py`
- [ ] Subclassed from `MetricBase`
- [ ] Includes docstring and math definition
- [ ] Handles vectorized and streaming modes
- [ ] Deterministic results with fixed seeds
- [ ] Registered in `src/metrics/__init__.py`

---

#### 🧪 Validation
- [ ] Unit tests (`tests/metrics/test_<id>.py`)
- [ ] Analytical fixtures pass (known input → known output)
- [ ] Compared against baseline (TVE/FE)
- [ ] Supports `--json` export for reports
- [ ] Works in multi-metric benchmark run

---

#### 📊 Example Output
```json
{
  "metric_id": "rfe_windowed",
  "aggregate": { "mean": 0.007, "p95": 0.021 },
  "by_window": [{ "t0": 0.0, "t1": 0.04, "rfe": 0.005 }]
}
