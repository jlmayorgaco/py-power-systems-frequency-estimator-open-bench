#!/usr/bin/env bash
# Ensure GitHub labels from data/labels.json
# - Works with {"name","color","#RRGGBB","description"} or {"name","color","desc"}
# - DRY_RUN=1 prints actions only
# - UPSERT: update if exists, else create

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
UPSERT="${UPSERT:-1}"
REPO="${REPO:?REPO env required}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/../data"
FILE="$DATA/labels.json"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Need '$1' installed" >&2; exit 1; }; }
need jq

[ -f "$FILE" ] || { echo "Missing $FILE" >&2; exit 1; }

# url-encode for label names containing spaces/colons/etc.
urlencode() {
  local s="$1" i c out=""
  for ((i=0; i<${#s}; i++)); do
    c="${s:$i:1}"
    case "$c" in
      [a-zA-Z0-9._~-]) out+="$c" ;;
      *) printf -v out '%s%%%02X' "$out" "'$c" ;;
    esac
  done
  printf '%s' "$out"
}

# Normalize JSON: strip leading # from color; accept description or desc
jq -e . "$FILE" >/dev/null
stream='.[] | {
  name: .name,
  color: ( ( .color // "" ) | ltrimstr("#") ),
  description: ( .description // .desc // "" )
}'

if [ "$DRY_RUN" -eq 1 ]; then
  jq -r "$stream | \"- ensure label: \\(.name) color=#\\(.color) desc=\\(.description)\"" "$FILE"
  exit 0
fi

need gh

# Process each label
jq -c "$stream" "$FILE" | while IFS= read -r row; do
  name="$(jq -r '.name' <<<"$row")"
  color="$(jq -r '.color' <<<"$row")"
  desc="$(jq -r '.description' <<<"$row")"

  [ -n "$name" ] || { echo "Skip: empty name" >&2; continue; }

  enc_name="$(urlencode "$name")"

  # Does it exist?
  set +e
  gh api -X GET "repos/$REPO/labels/$enc_name" >/dev/null 2>&1
  exists=$?
  set -e

  if [ $exists -eq 0 ]; then
    if [ "$UPSERT" -eq 1 ]; then
      echo "update label: $name"
      gh api -X PATCH "repos/$REPO/labels/$enc_name" \
        -f "new_name=$name" -f "color=$color" -f "description=$desc" >/dev/null
    else
      echo "exists (skip): $name"
    fi
  else
    echo "create label: $name"
    gh api -X POST "repos/$REPO/labels" \
      -f "name=$name" -f "color=$color" -f "description=$desc" >/dev/null
  fi
  # Small delay to be gentle on API limits
  sleep 0.1
done

echo "OK labels ensured"
