#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/config.env"
source "$HERE/lib.sh"
need_gh

# Example: estimators list and tasks list
ESTIMATORS_JSON="$HERE/data/generators/estimators.json"            # ["pll.sogi_fll", ...]
EST_TASKS_JSON="$HERE/data/generators/estimator_tasks.json"         # [{"title":"API conformance...", "labels":"...", "milestone":"..."}, ...]

# Optional: read epic number for auto-append (if exists)
EST_EPIC_URL="$(find_issue_by_title "$REPO" "📈 Epic: Estimators v0.1" || true)"
EST_EPIC_NUM="${EST_EPIC_URL##*/}"

for est in $(jq -r '.[]' "$ESTIMATORS_JSON"); do
  jq -c '.[]' "$EST_TASKS_JSON" | while read -r t; do
    t_title="$(jq -r '.title' <<<"$t")"
    labels="$(jq -r '.labels' <<<"$t")"
    ms="$(jq -r '.milestone' <<<"$t")"
    title="Estimator: ${est} — ${t_title}"
    body="Work item for ${est}."
    url="$(create_issue "$REPO" "$title" "$labels" "$ms" "$body")"
    [[ -n "$url" && "$EST_EPIC_NUM" =~ ^[0-9]+$ ]] && append_task_to_epic "$REPO" "$EST_EPIC_NUM" "$url"
    sleep 0.2
  done
done
