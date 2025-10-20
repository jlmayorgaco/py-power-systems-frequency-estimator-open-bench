### 🧠 Estimator Implementation PR

**Estimator ID:** `<short-name>`
**Implements:** `EstimatorBase`
**Related Proposal:** #<issue-number>

---

#### 📋 Summary
Describe the new estimator briefly:
- What physical/model principle does it use?
- Intended operating range (steady-state, transients, IBRs, etc.)
- Key expected advantages.

---

#### ⚙️ Implementation Details
- [ ] Added in `src/estimators/<id>/<file>.py`
- [ ] Registered via `register_model()` or estimator registry
- [ ] Includes configurable parameters (`params` dict)
- [ ] Includes docstring and usage example
- [ ] Complies with `EstimatorBase` I/O contract (`PMU_Input` / `PMU_Output`)
- [ ] Deterministic random seed behavior (if stochastic)

---

#### 🧪 Testing
- [ ] Unit tests added (`tests/estimators/test_<id>.py`)
- [ ] Tested on built-in scenarios (`synthetic_sine`, `ieee13`)
- [ ] Metrics validated (`TVE`, `FE`, `RFE`)
- [ ] Reproducible results across runs/seeds
- [ ] Benchmarks report consistent performance

---

#### 📊 Performance Targets
| Metric | Target | Achieved | Pass |
|---------|---------|-----------|------|
| TVE (%) | ≤ 1.0 | ... | ✅/❌ |
| FE (Hz) | ≤ 0.005 | ... | ✅/❌ |
| Latency (ms) | ≤ 50 | ... | ✅/❌ |

---

#### 🧩 Documentation
- [ ] Added README section (`docs/estimators/<id>.md`)
- [ ] Example config snippet (`examples/<id>_demo.yml`)
- [ ] Added to API index / registry

---

#### 🏷️ Labels
Apply: `type:feat`, `area:estimator`

---

**Reviewer Checklist**
- [ ] Code style and docstrings
- [ ] Unit tests & reproducibility
- [ ] Metric performance verified
- [ ] Scenario coverage adequate
- [ ] Ready to merge
