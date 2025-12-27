#!/usr/bin/env bash

# ============================================================
# Dump ALL estimators/*.py into ONE audit file (READ-ONLY)
# ============================================================

ESTIMATORS_DIR="./estimators"
OUT_DIR="./_audit"
OUT_FILE="$OUT_DIR/estimators_full_dump.txt"

# Safety checks
if [ ! -d "$ESTIMATORS_DIR" ]; then
  echo "[ERROR] Directory not found: $ESTIMATORS_DIR"
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "============================================================" > "$OUT_FILE"
echo " ESTIMATORS FULL DUMP" >> "$OUT_FILE"
echo " Generated on: $(date)" >> "$OUT_FILE"
echo " Root: $(pwd)" >> "$OUT_FILE"
echo "============================================================" >> "$OUT_FILE"
echo >> "$OUT_FILE"

# Iterate deterministically
find "$ESTIMATORS_DIR" -type f -name "*.py" | sort | while read -r file; do
  echo "============================================================" >> "$OUT_FILE"
  echo " FILE: $file" >> "$OUT_FILE"
  echo "============================================================" >> "$OUT_FILE"
  echo >> "$OUT_FILE"

  # With line numbers (crucial for review)
  nl -ba "$file" >> "$OUT_FILE"

  echo >> "$OUT_FILE"
  echo "------------------------ EOF -------------------------------" >> "$OUT_FILE"
  echo >> "$OUT_FILE"
done

echo "============================================================" >> "$OUT_FILE"
echo " END OF ESTIMATORS DUMP" >> "$OUT_FILE"
echo "============================================================" >> "$OUT_FILE"

echo "[OK] Dump completed:"
echo "     $OUT_FILE"
