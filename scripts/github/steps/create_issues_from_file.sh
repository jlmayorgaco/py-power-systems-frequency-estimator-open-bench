#!/usr/bin/env bash
# Create issues from a JSON file (labels can be CSV string or array).
# Usage:
#   DRY_RUN=1 REPO=owner/repo bash create_issues_from_file.sh path/to/issues.json
#   REPO=owner/repo bash create_issues_from_file.sh path/to/issues.json
# Optional:
#   UPSERT=1   # try to update existing issue with same title (slower, uses search API)
#   PROJECT_NUMBER=2  # add created/updated issues to a GitHub Project

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
UPSERT="${UPSERT:-0}"
REPO="${REPO:?REPO env required}"
PROJECT_NUMBER="${PROJECT_NUMBER:-}"
FILE="${1:?usage: create_issues_from_file.sh <json-file>}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Need '$1'" >&2; exit 1; }; }
need jq
jq -e . "$FILE" >/dev/null

# Normalize: labels can be array OR comma-separated string
JQ_NORM='.[] | {
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

if [ "$DRY_RUN" -eq 1 ]; then
  jq -r "${JQ_NORM} | \"- issue: \\(.title) [\\(.milestone // \"<none>\")] labels=\\(.labels|join(\", \"))\"" "$FILE"
  exit 0
fi

need gh

# Cache milestones once to avoid N calls
MS_TMP="$(mktemp)"
gh api -X GET "repos/$REPO/milestones?state=all&per_page=100" >"$MS_TMP" || echo "[]" >"$MS_TMP"

milestone_number() {
  # $1: milestone title
  local t="$1"
  [ -n "$t" ] || { echo ""; return; }
  jq -r --arg t "$t" -f /dev/stdin "$MS_TMP" <<'JQ' | head -n1
.[]
| select(.title==$t)
| .number
JQ
}

# Optional: upsert by title (uses search API; slower for huge sets)
find_issue_number() {
  local title="$1"
  if [ "$UPSERT" -eq 1 ]; then
    gh api -X GET search/issues -f "q=repo:$REPO in:title \"$title\"" \
      --jq '.items[] | select(.title=="'"$title"'") | .number' | head -n1 || true
  else
    echo ""
  fi
}

jq -c "${JQ_NORM}" "$FILE" | while IFS= read -r row; do
  title="$(jq -r '.title'   <<<"$row")"
  body="$(jq  -r '.body'    <<<"$row")"
  ms_title="$(jq -r '.milestone' <<<"$row")"

  # Build args
  args=(-R "$REPO" -t "$title" -b "$body")

  # Labels
  while IFS= read -r lbl; do
    [ -n "$lbl" ] && args+=(-l "$lbl")
  done < <(jq -r '.labels[]?' <<<"$row")

  # Milestone resolve (optional)
  msn="$(milestone_number "$ms_title")"
  [ -n "$msn" ] && args+=(-m "$msn")

  existing="$(find_issue_number "$title" || true)"

  if [ -n "$existing" ]; then
    gh issue edit -R "$REPO" "$existing" -t "$title" -b "$body" >/dev/null
    # Reset labels to exactly the provided set
    gh issue edit -R "$REPO" "$existing" --remove-label '*' >/dev/null || true
    while IFS= read -r lbl; do
      [ -n "$lbl" ] && gh issue edit -R "$REPO" "$existing" --add-label "$lbl" >/dev/null
    done < <(jq -r '.labels[]?' <<<"$row")
    [ -n "$msn" ] && gh issue edit -R "$REPO" "$existing" --milestone "$msn" >/dev/null
    issue_url="$(gh issue view -R "$REPO" "$existing" --json url -q .url)"
    echo "OK issue (updated): $issue_url — $title"
  else
    issue_url="$(gh issue create "${args[@]}" --json url -q .url)"
    echo "OK issue (created): $issue_url — $title"
  fi

  if [ -n "${PROJECT_NUMBER:-}" ] && [ -n "${issue_url:-}" ]; then
    gh project item-add --project "$PROJECT_NUMBER" --owner "${REPO%%/*}" --url "$issue_url" >/dev/null || true
  fi

  # Gentle pacing
  sleep 0.15
done

rm -f "$MS_TMP"
