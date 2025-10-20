## 🏁 `benchmark.md`
```md
### 🏁 Benchmark Suite PR

**Benchmark ID:** `<short-name>`
**Related Proposal:** #<issue-number>

---

#### 📋 Summary
Explain what this benchmark tests and why it’s valuable.

---

#### ⚙️ Configuration
- [ ] Added in `benchmarks/<id>/config.yml`
- [ ] Defines `estimators`, `scenarios`, `metrics`, `seeds`
- [ ] Includes `runner` block (mode, output_dir)
- [ ] Uses reproducible seeds and deterministic runs
- [ ] Produces reports in `/reports/<id>/`

---

#### 🧪 Validation
- [ ] Runs successfully locally and on CI
- [ ] Outputs match expected schema (`results.json`, `summary.pdf`)
- [ ] Passes all metric and estimator compatibility checks
- [ ] Aggregate score computed correctly

---

#### 📊 Example Snippet
```yaml
benchmark:
  id: bench_v1.3_transient_stability
  estimators: ["trust_weighted_ekf", "zcd", "iddft"]
  scenarios: ["ieee13_ramp_sag_70ibr", "synthetic_step_flicker"]
  metrics: ["tve", "fe", "rfe_windowed"]
  seeds: [42, 1337, 2025]
  repeats: 3

  