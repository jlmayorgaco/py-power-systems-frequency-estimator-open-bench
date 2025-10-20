### 🧩 Dependency Update PR

**Purpose:**  
_Update or add dependencies (Python packages, Actions, Conda envs, etc.)_

---

### 📦 Summary of Changes
- [ ] Dependency bump(s)
- [ ] New dependency added
- [ ] Removed unused dependency
- [ ] CI / environment update

**Details:**
| Package | Old | New | Reason |
|----------|-----|-----|--------|
| example-pkg | 1.2.3 | 1.3.0 | security patch |

---

### 🧪 Verification
- [ ] Local environment builds successfully (`make setup` / `pip install -e .`)
- [ ] All tests pass (`pytest`, benchmarks, lint)
- [ ] CI pipelines green
- [ ] Docs / examples unaffected

Optional check:
```bash
pip list --outdated
pytest -q
```

---

### 🧠 Notes
- Compatibility notes (e.g., Python 3.12)
- Pinned versions if required
- Known regressions or removals

---

**Reviewer Checklist**
- [ ] Changes scoped and minimal
- [ ] Lockfiles / env.yml updated
- [ ] CI tested on all platforms
- [ ] No broken transitive deps
- [ ] Safe to merge ✅
