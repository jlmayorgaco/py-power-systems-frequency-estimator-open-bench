#!/usr/bin/env bash
set -euo pipefail
DRY_RUN="${DRY_RUN:-0}"
REPO="${REPO:?REPO env required}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DATA="$ROOT/github/data"

command -v jq >/dev/null 2>&1 || { echo "Need jq" >&2; exit 1; }
[ -f "$DATA/milestones.json" ] || { echo "Missing $DATA/milestones.json" >&2; exit 1; }

if [ "$DRY_RUN" -eq 1 ]; then
  jq -r '.[] | "- ensure milestone: \(.title) due=\(.due_on // "<none>") state=\(.state // "open")"' "$DATA/milestones.json"
  exit 0
fi

command -v gh >/dev/null 2>&1 || { echo "Need gh" >&2; exit 1; }
jq -c '.[]' "$DATA/milestones.json" | while read -r row; do
  title="$(jq -r '.title' <<<"$row")"
  due_on="$(jq -r '.due_on // empty' <<<"$row")"
  state="$(jq -r '.state // "open"' <<<"$row")"
  num="$(gh api "repos/$REPO/milestones?state=all" -q ".[]|select(.title==\"$title\")|.number" | head -n1 || true)"
  if [ -n "$num" ]; then
    args=(-f title="$title" -f state="$state"); [ -n "$due_on" ] && args+=(-f due_on="$due_on")
    gh api -X PATCH "repos/$REPO/milestones/$num" "${args[@]}" >/dev/null
    echo "OK milestone (updated): $title"
  else
    args=(-f title="$title"); [ -n "$due_on" ] && args+=(-f due_on="$due_on")
    gh api -X POST "repos/$REPO/milestones" "${args[@]}" >/dev/null
    echo "OK milestone (created): $title"
  fi
  sleep 0.1
done
