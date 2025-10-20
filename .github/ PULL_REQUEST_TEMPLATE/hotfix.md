### 🩹 Hotfix PR

**Purpose:**  
_Short description of the urgent fix (production bug, release blocker, security patch)._

**Scope:**  
- Affects: `<component / path>`
- Introduced in: `<commit / tag>`
- User impact: `<crash / wrong results / outage / security>`

---

### 🔧 Fix Description
- Root cause: …
- Change summary: …
- Risk level: Low / Medium / High (why?)

---

### ✅ Verification
- [ ] Reproduced the issue on the target branch
- [ ] Added a minimal regression test
- [ ] Verified fix locally
- [ ] Relevant CI workflow(s) pass
- [ ] Backport plan (if needed): `<branches>`

Repro / test commands:
```bash
pytest tests/ -k "<hotfix_spec>"
python -m examples.<demo>  # if applicable
```

---

### ⏱️ Performance & Safety
- Perf impact: `<none / measured>`
- Side effects / breaking changes: `<none / describe>`

---

### 📜 Release Notes
> One-line, user-facing note describing the fix.

---

### 🧠 Links
Closes #<bug-issue-number> (or) Security: `<CVE/link>`

---

**Reviewer Checklist**
- [ ] Root cause understood and addressed
- [ ] Regression test present and meaningful
- [ ] No unintended side effects
- [ ] Safe to merge & release ✅
