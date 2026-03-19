#!/usr/bin/env bash
set -e

OUT_DIR="artifacts/inspect_estimators"
OUT_TXT="${OUT_DIR}/estimators_dump.txt"

mkdir -p "${OUT_DIR}"
rm -f "${OUT_TXT}"

python - <<'PY'
import os, sys
from pathlib import Path
from datetime import datetime

out_dir = Path("artifacts/inspect_estimators")
out_txt = out_dir / "estimators_dump.txt"
out_dir.mkdir(parents=True, exist_ok=True)

def w(s=""):
    out_txt.open("a", encoding="utf-8").write(s + "\n")

w("========================================")
w("ESTIMATORS TREE + CONTENT DUMP")
w(f"Generated on: {datetime.now().isoformat()}")
w("========================================")
w("")

root = Path("estimators")
if not root.exists():
    raise SystemExit("ERROR: folder 'estimators/' not found. Run from repo root (PMU/).")

# 1) Tree
w("---- FILE TREE (estimators/) ----")
w("")
files = sorted([p for p in root.rglob("*") if p.is_file()])
for p in files:
    w(str(p))
w("")

# 2) Content
w("---- FILE CONTENT ----")
w("")
for p in files:
    w("========================================")
    w(f"FILE: {p}")
    w("========================================")
    w("")
    data = p.read_bytes()
    text = data.decode("utf-8", errors="replace")
    # indent for readability
    for line in text.splitlines():
        w("    " + line)
    w("")

w("========================================")
w("END OF DUMP")
print(f"[OK] Wrote: {out_txt}")
PY
