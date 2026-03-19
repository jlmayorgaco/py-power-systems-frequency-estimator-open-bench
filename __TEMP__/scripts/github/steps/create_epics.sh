#!/usr/bin/env bash
# Create "epic" issues from scripts/github/data/epics.json
# - macOS Bash 3.2 safe
# - Uses gh api with JSON body via --input - (handles labels array correctly)
# - DRY_RUN=1 previews; UPSERT=1 updates an issue with the same title

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
UPSERT="${UPSERT:-1}"                 # 0=create-only, 1=upsert by exact title
REPO="${REPO:?REPO env required}"
PROJECT_NUMBER="${PROJECT_NUMBER:-}"  # optional

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/../data"
FILE="$DATA/epics.json"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Need '$1' installed" >&2; exit 1; }; }
need jq
need gh

[ -f "$FILE" ] || { echo "Missing $FILE" >&2; exit 1; }
jq -e . "$FILE" >/dev/null

# Normalize: allow labels as CSV string or array
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

STREAM=".[] | $JQ_NORM"

if [ "$DRY_RUN" -eq 1 ]; then
  jq -r "$STREAM | \"- epic: \\(.title) [\\(.milestone // \"<none>\")] labels=\\(.labels|join(\", \"))\"" "$FILE"
  exit 0
fi

# Cache milestones once
MS_TMP="$(mktemp)"; trap 'rm -f "$MS_TMP"' EXIT
gh api -X GET "repos/$REPO/milestones?state=all&per_page=100" >"$MS_TMP" || echo "[]" >"$MS_TMP"

milestone_number() {
  local t="$1"
  [ -n "$t" ] || { echo ""; return; }
  jq -r --arg t "$t" '.[] | select(.title==$t) | .number' "$MS_TMP" | head -n1 || true
}

# Optional upsert: find existing issue by exact title
find_issue_number() {
  local title="$1"
  if [ "$UPSERT" -eq 1 ]; then
    gh api -X GET search/issues \
      -f "q=repo:$REPO in:title \"$title\" is:issue" \
      --jq '.items[] | select(.title=="'"$title"'") | .number' | head -n1 || true
  else
    echo ""
  fi
}

# POST / PATCH with a full JSON body via stdin
create_issue() {
  jq -c '.' | gh api -X POST "repos/$REPO/issues" --input - --jq .html_url
}

update_issue() {
  local number="$1"
  jq -c '.' | gh api -X PATCH "repos/$REPO/issues/$number" --input - >/dev/null
  gh api -X GET "repos/$REPO/issues/$number" --jq .html_url
}

# Iterate rows
jq -c "$STREAM" "$FILE" | while IFS= read -r row; do
  title="$(jq -r '.title' <<<"$row")"
  body="$(jq -r '.body'  <<<"$row")"
  ms_title="$(jq -r '.milestone' <<<"$row")"
  labels_arr="$(jq -c '.labels' <<<"$row")"
  msn="$(milestone_number "$ms_title")"

  # build JSON body
  if [ -n "$msn" ]; then
    body_json="$(jq -n --arg t "$title" --arg b "$body" --argjson L "$labels_arr" --argjson m "$msn" \
      '{title:$t, body:$b, labels:$L, milestone:$m}')"
  else
    body_json="$(jq -n --arg t "$title" --arg b "$body" --argjson L "$labels_arr" \
      '{title:$t, body:$b, labels:$L}')"
  fi

  existing="$(find_issue_number "$title" || true)"
  if [ -n "$existing" ]; then
    url="$(printf '%s' "$body_json" | update_issue "$existing")"
    echo "OK epic (updated): $url — $title"
    issue_url="$url"
  else
    url="$(printf '%s' "$body_json" | create_issue)"
    echo "OK epic (created): $url — $title"
    issue_url="$url"
  fi

  if [ -n "${PROJECT_NUMBER:-}" ] && [ -n "${issue_url:-}" ]; then
    gh project item-add --project "$PROJECT_NUMBER" --owner "${REPO%%/*}" --url "$issue_url" >/dev/null || true
  fi

  sleep 0.15
done

echo "OK epics ensured"
