# scripts/github/lib.sh
set -euo pipefail

# DRY-RUN aware executor
run() { if (( ${DRY_RUN:-0} )); then echo "DRY: $*"; else eval "$@"; fi; }

need_gh() {
  (( ${DRY_RUN:-0} )) && return 0
  command -v gh >/dev/null 2>&1 || { echo "❌ gh CLI not found"; exit 1; }
  gh auth status >/dev/null || { echo "❌ gh not authenticated. gh auth login"; exit 1; }
}

# Exact label ensure
ensure_label() {
  local repo="$1" name="$2" color="$3" desc="$4"
  if gh label list -R "$repo" --limit 200 --json name \
       --jq ".[]|select(.name==\"$name\")|.name" | grep -qx "$name"; then
    run gh label edit "$name" -R "$repo" --color "${color#\#}" --description "$desc"
  else
    run gh label create "$name" -R "$repo" --color "${color#\#}" --description "$desc"
  fi
}

# Milestone ensure/update
ensure_milestone() {
  local repo="$1" title="$2" desc="$3" due="$4"
  local num; num="$(gh api -X GET "repos/$repo/milestones?state=all" \
    -q ".[]|select(.title==\"$title\")|.number" | head -n1 || true)"
  if [[ -n "$num" ]]; then
    local st; st="$(gh api -X GET "repos/$repo/milestones/$num" -q .state)"
    [[ "$st" = "closed" ]] && run gh api -X PATCH "repos/$repo/milestones/$num" -f state=open >/dev/null
    run gh api -X PATCH "repos/$repo/milestones/$num" \
      -f title="$title" -f description="$desc" -f due_on="${due}T00:00:00Z" >/dev/null
  else
    run gh api -X POST "repos/$repo/milestones" \
      -f title="$title" -f description="$desc" -f due_on="${due}T00:00:00Z" >/dev/null
  fi
}

get_ms_number() { gh api -X GET "repos/$1/milestones?state=all" -q ".[]|select(.title==\"$2\")|.number"; }

find_issue_by_title() { gh issue list -R "$1" --state all --search "in:title \"$2\"" \
  --json url,title --jq ".[]|select(.title==\"$2\")|.url" | head -n1; }

create_issue() {
  local repo="$1" title="$2" labels_csv="$3" ms_title="$4" body="$5"
  local existing; existing="$(find_issue_by_title "$repo" "$title" || true)"
  [[ -n "$existing" ]] && { echo "$existing"; return 0; }

  local flags=(); IFS=',' read -ra L <<<"$labels_csv"
  for l in "${L[@]}"; do l="$(echo "$l" | xargs)"; [[ -n "$l" ]] && flags+=( -l "$l" ); done
  local msn; msn="$(get_ms_number "$repo" "$ms_title" || true)"
  [[ -n "$msn" ]] && flags+=( -m "$msn" )

  if (( ${DRY_RUN:-0} )); then
    echo "https://github.com/$repo/issues/DRY-$RANDOM"
    echo "DRY: gh issue create -R \"$repo\" -t \"$title\" ${flags[*]} -b <body>"
  else
    gh issue create -R "$repo" -t "$title" "${flags[@]}" -b "$body" --json url -q .url
  fi
}

append_task_to_epic() {
  local repo="$1" epic_num="$2" child_url="$3"
  local child_num="${child_url##*/}"
  local body; body="$(gh issue view -R "$repo" "$epic_num" --json body --jq .body)"
  grep -qE "\-\s\[\s\]\s#${child_num}\b" <<<"$body" && return 0
  body+=$'\n'"- [ ] #${child_num}"
  run gh issue edit -R "$repo" "$epic_num" --body "$body" >/dev/null
}

latest_tag() { git describe --tags --abbrev=0 2>/dev/null || true; }
bump_semver() {
  local last="${1#v}" part="$2"; IFS='.' read -r MA MI PA <<<"${last:-0.0.0}"
  case "$part" in major) ((MA++)); MI=0; PA=0;; minor) ((MI++)); PA=0;; patch) ((PA++));; *) echo "bad bump"; exit 1;; esac
  echo "v${MA}.${MI}.${PA}"
}
