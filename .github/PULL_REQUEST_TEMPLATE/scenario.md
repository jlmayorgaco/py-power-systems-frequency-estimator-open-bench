### 🗺️ Scenario Integration PR

**Scenario ID:** `<short-name>`
**Type:** Synthetic / Co-sim / HIL / Real-world
**Related Proposal:** #<issue-number>

---

#### 📋 Summary
Describe what this scenario represents (e.g., “IEEE13 feeder with 70% IBR penetration under ramp+sag events”).

---

#### ⚙️ Files & Structure
- [ ] Added to `data/scenarios/<id>/`
- [ ] Includes `signals.parquet` or equivalent
- [ ] Includes `scenario.json` metadata
- [ ] Includes `events.csv` or annotations
- [ ] Checksums (`sha256`) verified
- [ ] License declared in metadata

---

#### 🧪 Validation
- [ ] Schema validated (columns, units, sampling rate)
- [ ] Metadata consistent (`fs_hz`, `duration_s`, etc.)
- [ ] Truth signals (f_true, dfdt_true) aligned
- [ ] Deterministic generation (same seed → identical data)
- [ ] Smoke test passes in CI

---

#### 🧩 Integration
- [ ] Registered in `src/scenarios/__init__.py`
- [ ] Example config added (`examples/scenarios/<id>_demo.yml`)
- [ ] Added to benchmark suite if applicable

---

#### 📊 Quality Metrics
| Property | Target | Achieved | Pass |
|-----------|---------|-----------|------|
| SNR (dB) | ≥ 20 | ... | ✅/❌ |
| THD (%) | ≤ 8 | ... | ✅/❌ |
| Missing (%) | ≤ 0.1 | ... | ✅/❌ |

---

#### 🏷️ Labels
Apply: `type:feat`, `area:scenario`

---

**Reviewer Checklist**
- [ ] Schema validated
- [ ] Ground-truth consistency
- [ ] Metadata & license verified
- [ ] CI smoke test passing
- [ ] Approved for merge
