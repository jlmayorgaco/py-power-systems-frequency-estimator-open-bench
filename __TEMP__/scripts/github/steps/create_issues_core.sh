#!/usr/bin/env bash
# Create "core" issues from scripts/github/data/issues.json
# DRY_RUN=1 previews; LIVE uses gh. macOS-safe (Bash 3.2).

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
UPSERT="${UPSERT:-0}"                 # 0=create-only, 1=upsert by title
REPO="${REPO:?REPO env required}"
PROJECT_NUMBER="${PROJECT_NUMBER:-}"  # optional

# Optional filters
FILTER_MS="${FILTER_MS:-}"            # e.g., "M0 – Repo bootstrap"
GREP_TITLE="${GREP_TITLE:-}"          # e.g., "pyproject|CI"
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/../data"
FILE="$DATA/issues.json"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Need '$1' installed" >&2; exit 1; }; }
need jq

[ -f "$FILE" ] || { echo "Missing issues JSON. Looked for: $FILE" >&2; exit 1; }
jq -e . "$FILE" >/dev/null

# --- jq pipeline to normalize fields ---
JQ_NORM='{
  title: .title,
  body: (.body // ""),
  milestone: (.milestone // ""),
  labels: (
    if (.labels|type=="array") then (.labels|map(tostring))
    elif (.labels|type=="string") then (.labels
        | split(",")
        | map(gsub("^\\s+|\\s+$";""))
        | map(select(length>0)))
    else [] end
  )
}'

JQ_EXPR="map($JQ_NORM)"
[ -n "$FILTER_MS" ]  && JQ_EXPR="$JQ_EXPR | map(select(.milestone == \$fms))"
[ -n "$GREP_TITLE" ] && JQ_EXPR="$JQ_EXPR | map(select(.title|test(\$grep; \"i\")))"
if [ "$OFFSET" -gt 0 ] || [ "$LIMIT" -gt 0 ]; then
  if [ "$LIMIT" -gt 0 ]; then
    JQ_EXPR="$JQ_EXPR | .[\$off : (\$off + \$lim)]"
  else
    JQ_EXPR="$JQ_EXPR | .[\$off:]"
  fi
fi

JQ_ARGS=()
[ -n "$FILTER_MS" ]  && JQ_ARGS+=(--arg fms "$FILTER_MS")
[ -n "$GREP_TITLE" ] && JQ_ARGS+=(--arg grep "$GREP_TITLE")
[ "$OFFSET" -gt 0 ]  && JQ_ARGS+=(--argjson off "$OFFSET")
[ "$LIMIT"  -gt 0 ]  && JQ_ARGS+=(--argjson lim "$LIMIT")

jq_run() { jq ${JQ_ARGS[@]+"${JQ_ARGS[@]}"} "$@" ; }

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Previewing issues from $FILE"
  jq_run -r "$JQ_EXPR | .[] | \"- issue: \\(.title) [\\(.milestone // \"<none>\")] labels=\\(.labels|join(\", \"))\"" "$FILE"
  exit 0
fi

need gh

# Resolve a milestone name to… itself (older gh wants milestone *name*, not number)
ms_flag_value() {
  # If you later need numbers, swap to gh api + jq here.
  printf "%s" "$1"
}

# Optional upsert by exact title (uses gh api search)
find_issue_number() {
  local title="$1"
  if [ "$UPSERT" -eq 1 ]; then
    gh api -X GET search/issues \
      -f "q=repo:$REPO in:title \"$title\" is:issue" \
      --jq '.items[] | select(.title=="'"$title"'") | .number' 2>/dev/null | head -n1 || true
  else
    echo ""
  fi
}

# URL helpers that don’t require --json support
issue_url_from_number() {
  local num="$1"
  echo "https://github.com/$REPO/issues/$num"
}
issue_url_by_search_title() {
  local title="$1"
  gh api -X GET search/issues \
    -f "q=repo:$REPO in:title \"$title\" is:issue sort:created-desc" \
    --jq '.items[0].html_url' 2>/dev/null || true
}

# Iterate and create/update
jq_run -c "$JQ_EXPR | .[]" "$FILE" | while IFS= read -r row; do
  title="$(jq -r '.title'   <<<"$row")"
  body="$(jq  -r '.body'    <<<"$row")"
  ms_title="$(jq -r '.milestone' <<<"$row")"

  # Build classic args (compatible with older gh)
  args=(-R "$REPO" -t "$title" -b "$body")
  while IFS= read -r lbl; do
    [ -n "$lbl" ] && args+=(-l "$lbl")
  done < <(jq -r '.labels[]?' <<<"$row")
  [ -n "$ms_title" ] && args+=(-m "$(ms_flag_value "$ms_title")")

  existing="$(find_issue_number "$title" || true)"
  if [ -n "$existing" ]; then
    gh issue edit -R "$REPO" "$existing" -t "$title" -b "$body" >/dev/null
    gh issue edit -R "$REPO" "$existing" --remove-label '*' >/dev/null || true
    while IFS= read -r lbl; do
      [ -n "$lbl" ] && gh issue edit -R "$REPO" "$existing" --add-label "$lbl" >/dev/null
    done < <(jq -r '.labels[]?' <<<"$row")
    [ -n "$ms_title" ] && gh issue edit -R "$REPO" "$existing" --milestone "$ms_title" >/dev/null
    issue_url="$(issue_url_from_number "$existing")"
    echo "OK issue (updated): $issue_url — $title"
  else
    # Create without --json; capture nothing, then find URL via search
    gh issue create "${args[@]}" >/dev/null
    issue_url="$(issue_url_by_search_title "$title")"
    [ -z "$issue_url" ] && issue_url="https://github.com/$REPO/issues"
    echo "OK issue (created): $issue_url — $title"
  fi

  if [ -n "${PROJECT_NUMBER:-}" ] && [ -n "${issue_url:-}" ]; then
    # Best-effort: add to org/user project (classic projects use -p title; Projects v2 needs item-add)
    gh project item-add --project "$PROJECT_NUMBER" --owner "${REPO%%/*}" --url "$issue_url" >/dev/null || true
  fi

  sleep 0.12
done

echo "OK core issues ensured"
