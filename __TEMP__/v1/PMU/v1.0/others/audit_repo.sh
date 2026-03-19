#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# pack_repo_to_txt.sh
#   Portable version (macOS Bash 3.2 compatible)
# ------------------------------------------------------------

OUTDIR="_audit"
OUTFILE="$OUTDIR/repo_dump.txt"

# ------------------------------------------------------------
# Excluded directories
# ------------------------------------------------------------
EXCLUDES="
./temp_old
./_audit
./.git
./.github
./.venv
./venv
./env
./__pycache__
./.mypy_cache
./.pytest_cache
./.ruff_cache
./dist
./build
./site
./node_modules
./.idea
./.vscode
"

# ------------------------------------------------------------
# Extensions to include
# ------------------------------------------------------------
EXTENSIONS="
py md rst txt
yaml yml json toml ini cfg
sh bash zsh
tex bib
csv tsv
js ts jsx tsx
html css
xml svg
ipynb
"

MAX_BYTES="${MAX_BYTES:-1500000}"

mkdir -p "$OUTDIR"
: > "$OUTFILE"

echo "=== REPO DUMP START ===" >> "$OUTFILE"
echo "Root: $(pwd)" >> "$OUTFILE"
echo "Date: $(date)" >> "$OUTFILE"
echo "Max bytes per file: $MAX_BYTES" >> "$OUTFILE"
echo "" >> "$OUTFILE"

# ------------------------------------------------------------
# Build find prune expression
# ------------------------------------------------------------
PRUNE_ARGS=()
for d in $EXCLUDES; do
  PRUNE_ARGS+=( -path "$d" -o )
done
PRUNE_ARGS+=( -false )

# ------------------------------------------------------------
# Build extension match expression
# ------------------------------------------------------------
EXT_ARGS=()
for e in $EXTENSIONS; do
  EXT_ARGS+=( -iname "*.$e" -o )
done
EXT_ARGS+=( -false )

TOTAL=0
SKIPPED=0

# ------------------------------------------------------------
# Main loop (NO mapfile)
# ------------------------------------------------------------
find . \( "${PRUNE_ARGS[@]}" \) -prune -o \
       -type f \( "${EXT_ARGS[@]}" \) -print0 |
while IFS= read -r -d '' FILE; do
  TOTAL=$((TOTAL + 1))

  REL="${FILE#./}"

  # Size check
  SIZE=$(wc -c < "$FILE" | tr -d ' ')
  if [ "$SIZE" -gt "$MAX_BYTES" ]; then
    SKIPPED=$((SKIPPED + 1))
    {
      echo "============================================================"
      echo "FILE: $REL"
      echo "NOTE: SKIPPED (too large: $SIZE bytes)"
      echo "============================================================"
      echo
    } >> "$OUTFILE"
    continue
  fi

  # Binary check
  if ! LC_ALL=C grep -Iq . "$FILE" 2>/dev/null; then
    SKIPPED=$((SKIPPED + 1))
    {
      echo "============================================================"
      echo "FILE: $REL"
      echo "NOTE: SKIPPED (binary / non-text)"
      echo "============================================================"
      echo
    } >> "$OUTFILE"
    continue
  fi

  {
    echo "============================================================"
    echo "FILE: $REL"
    echo "BYTES: $SIZE"
    echo "============================================================"
    echo
    cat "$FILE"
    echo
    echo
  } >> "$OUTFILE"

done

{
  echo "=== REPO DUMP DONE ==="
  echo "Files considered: $TOTAL"
  echo "Files skipped:    $SKIPPED"
  echo "Output:           $OUTFILE"
} >> "$OUTFILE"

echo "OK — wrote $OUTFILE"
