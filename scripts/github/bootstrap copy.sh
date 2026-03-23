#!/usr/bin/env bash
# bootstrap.sh — OpenFreqBench project orchestrator (with rich logging & tracing)
# Runs modular setup steps (labels, milestones, epics, issues, releases)
# Idempotent; supports DRY-RUN; detects REPO from git if not provided.
#
# Tested on macOS (Bash 3.2) and Linux (Bash 4+).

set -euo pipefail
shopt -s lastpipe 2>/dev/null || true     # harmless on bash 3.2 (noop)
set -o errtrace         || true
set -o pipefail         || true

# ------------------------- Resolve paths -------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
STEPS="$HERE/steps"
DATA="$ROOT/data"

# ------------------------- Defaults & env ------------------------
DRY_RUN="${DRY_RUN:-0}"
PROJECT_NUMBER="${PROJECT_NUMBER:-}"
REPO="${REPO:-}"
DEBUG="${DEBUG:-0}"       # extra logging
TRACE="${TRACE:-0}"       # bash -x tracing

# Source optional config.env if present
if [[ -f "$HERE/config.env" ]]; then
  # shellcheck disable=SC1090
  source "$HERE/config.env"
fi

# ------------------------- Colors & timestamps -------------------
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLU=$'\033[34m'; MAG=$'\033[35m'; CYA=$'\033[36m'; NC=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GRN=""; YLW=""; BLU=""; MAG=""; CYA=""; NC=""
fi

ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }

log()   { printf "%s %b%s%b %s\n" "$(ts)" "${BLU}" "INFO" "${NC}" "$*"; }
ok()    { printf "%s %b%s%b %s\n" "$(ts)" "${GRN}" "OK  " "${NC}" "$*"; }
warn()  { printf "%s %b%s%b %s\n" "$(ts)" "${YLW}" "WARN" "${NC}" "$*"; }
err()   { printf "%s %b%s%b %s\n" "$(ts)" "${RED}" "ERR " "${NC}" "$*" 1>&2; }
dbg()   { [[ "${DEBUG}" -eq 1 ]] && printf "%s %b%s%b %s\n" "$(ts)" "${MAG}" "DBG " "${NC}" "$*"; }

# optional shell tracing with file:line
enable_trace() {
  [[ "${TRACE}" -eq 1 ]] || return 0
  export PS4='+ ${BASH_SOURCE##*/}:${LINENO}:${FUNCNAME[0]:-main}() '
  set -x
  dbg "Shell tracing enabled"
}

# ------------------------- Error trap ----------------------------
on_err() {
  local exit_code=$?
  local cmd=${BASH_COMMAND:-"<unknown>"}
  err "Exit code: $exit_code"
  err "While running: $cmd"
  err "Call stack (most recent first):"
  local i=0
  # 'caller' exists in bash 3.2+
  while caller $i; do ((i++)); done | awk '{printf "  at %s:%s\n",$2,$1}' 1>&2
  err "Aborting."
  exit "$exit_code"
}
trap on_err ERR

# ------------------------- Helpers ------------------------------
usage() {
  cat <<EOF
${BOLD}OpenFreqBench bootstrap${NC}

${BOLD}Usage${NC}
  ${DIM}$0${NC} [command] [flags]

${BOLD}Commands${NC}
  all                 Run labels → milestones → epics → issues:core → issues:generated
  labels              Ensure labels from ${DIM}data/labels.json${NC}
  milestones          Ensure milestones from ${DIM}data/milestones.json${NC}
  epics               Create/ensure epics from ${DIM}data/epics.json${NC}
  issues:core         Create core issues block (embedded in step)
  issues:generated    Create generated issues from ${DIM}data/generators/*${NC}
  issues:m{0..11}     Create issues from ${DIM}data/issues/M{0..11}.json${NC} (if present)
  releases [args...]  Delegate to ${DIM}steps/create_releases.sh${NC}

${BOLD}Flags${NC}
  --repo owner/name   Override GitHub repo (default: detect from git remote origin)
  --project N         GitHub Project number; when set, new issues are added to it
  --dry-run           Print the commands that would run; do not mutate
  --no-need-gh        Skip 'gh auth' checks (useful for --dry-run without gh login)
  --debug             Verbose internal logging to help diagnose issues
  --trace             Bash -x tracing with file:line in PS4 (very verbose)
  -h | --help         Show this help
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "Required command not found: $1"; exit 1; }
  dbg "Found command: $1 -> $(command -v "$1")"
}

detect_repo() {
  if [[ -n "${REPO:-}" ]]; then
    dbg "REPO provided: $REPO"
    echo "$REPO"
    return
  fi
  local remote
  # Works on macOS git too
  remote="$(git -C "$ROOT" remote -v 2>/dev/null | awk '/origin[[:space:]].*\(push\)/{print $2}' | head -n1 || true)"
  if [[ -z "$remote" ]]; then
    err "Could not detect REPO from git. Pass --repo owner/name or export REPO."
    exit 1
  fi
  # BSD sed supports -E and these substitutions
  REPO="$(printf "%s" "$remote" | sed -E 's#(git@github.com:|https://github.com/)##; s/\.git$//' )"
  dbg "Detected REPO from origin: $REPO"
  echo "$REPO"
}

NEED_GH=1

# ------------------------- Parse args ---------------------------
CMD="${1:-all}"; shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)        REPO="${2:-}"; shift 2;;
    --project)     PROJECT_NUMBER="${2:-}"; shift 2;;
    --dry-run)     DRY_RUN=1; shift;;
    --no-need-gh)  NEED_GH=0; shift;;
    --debug)       DEBUG=1; shift;;
    --trace)       TRACE=1; shift;;
    -h|--help)     usage; exit 0;;
    *)             break;;
  esac
done

export DRY_RUN PROJECT_NUMBER DEBUG TRACE

enable_trace

# ------------------------- Preconditions ------------------------
log "Preflight: checking required tools"
need_cmd git
need_cmd awk
need_cmd sed
need_cmd bash
need_cmd jq

log "Preflight: resolving repo and environment"
detect_repo >/dev/null
export REPO

if [[ "$DRY_RUN" -eq 0 && "$NEED_GH" -eq 1 ]]; then
  need_cmd gh
  if ! gh auth status >/dev/null 2>&1; then
    err "gh CLI not authenticated. Run: gh auth login"
    exit 1
  fi
  dbg "gh auth OK"
else
  dbg "Skipping gh auth check (DRY_RUN=$DRY_RUN NEED_GH=$NEED_GH)"
fi

# ------------------------- Step discovery -----------------------
require_step() {
  local f="$1"
  if [[ -x "$f" ]]; then
    dbg "Step ready: $f"
    return 0
  fi
  if [[ -f "$f" ]]; then
    err "Step exists but not executable: $f"
    ls -l "$f" || true
    err "Fix with: chmod +x \"$f\""
  else
    err "Missing step: $f"
  fi
  exit 1
}

list_steps() {
  log "Discovering step scripts under: $STEPS"
  if [[ -d "$STEPS" ]]; then
    # BSD ls -1 is fine
    ls -1 "$STEPS" | sed 's/^/  • /' || true
  else
    warn "Steps directory not found: $STEPS"
  fi
}

run_step() {
  local name="$1"; shift || true
  local path="$1"; shift || true
  local started ended dur
  started=$(date +%s)
  log "${BOLD}${name}${NC} — starting (script: ${DIM}${path}${NC})"
  DRY_RUN="${DRY_RUN}" PROJECT_NUMBER="${PROJECT_NUMBER}" REPO="${REPO}" bash "$path" "$@" || {
    err "Step failed: $name (script: $path)"
    exit 1
  }
  ended=$(date +%s)
  dur=$((ended - started))
  ok "${name} done (${dur}s)"
}

# ------------------------- Validate known steps -----------------
list_steps
require_step "$STEPS/ensure_labels.sh"
require_step "$STEPS/ensure_milestones.sh"
require_step "$STEPS/create_epics.sh"
require_step "$STEPS/create_issues_core.sh"
require_step "$STEPS/create_issues_generated.sh"  # keep strict; comment out if optional in your repo

# ------------------------- JSON helpers (robust jq) -------------
# Normalize a single issue object into: title, body, milestone, labels[]
# Accepts labels as array OR comma-separated string; trims whitespace; drops null.
jq_norm_issues='.[] |
  {
    title: .title,
    body: (.body // ""),
    milestone: (.milestone // ""),
    labels: (
      if (.labels|type == "array") then
        (.labels | map(tostring))
      elif (.labels|type == "string") then
        (.labels | split(",") | map(gsub("^\\s+|\\s+$";"")) | map(select(length>0)))
      else
        []
      end
    )
  }'

# ------------------------- Issues by milestone ------------------
run_issues_mx() {
  local mx="$1"  # e.g., M0, M5
  local file="$DATA/issues/${mx}.json"
  log "Issues loader — target file: ${file}"
  if [[ ! -f "$file" ]]; then
    err "Issues file not found: $file"
    exit 1
  fi
  jq -e . "$file" >/dev/null || { err "Invalid JSON in $file"; exit 1; }

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY-RUN preview for ${mx}:"
    jq -r "${jq_norm_issues} | \"- \\(.title) [\\(.milestone // \"<no milestone>\")] labels=\\(.labels|join(\",\"))\"" "$file"
    ok "DRY preview completed for ${mx}"
    return 0
  fi

  need_cmd gh

  # If you have a helper lib with create_issue(), prefer it.
  if [[ -f "$HERE/lib.sh" ]]; then
    # shellcheck disable=SC1090
    source "$HERE/lib.sh"
    dbg "Loaded helper lib: $HERE/lib.sh"
    jq -c "${jq_norm_issues}" "$file" | while IFS= read -r row; do
      # shellcheck disable=SC2001
      local title body ms_title url
      title="$(jq -r '.title'   <<<"$row")"
      body="$(jq  -r '.body'    <<<"$row")"
      ms_title="$(jq -r '.milestone' <<<"$row")"
      # labels array to CSV
      local labels_csv
      labels_csv="$(jq -r '.labels | join(",")' <<<"$row")"

      log "Creating issue: ${title}  (${ms_title:-no milestone})"
      url="$(create_issue "$REPO" "$title" "$labels_csv" "$ms_title" "$body")"
      if [[ -n "${PROJECT_NUMBER:-}" && -n "$url" ]]; then
        gh project item-add --project "$PROJECT_NUMBER" --owner "${REPO%%/*}" --url "$url" || warn "Project add failed for $url"
      fi
      sleep 0.2
    done
  else
    jq -c "${jq_norm_issues}" "$file" | while IFS= read -r row; do
      local title body ms_title
      title="$(jq -r '.title'   <<<"$row")"
      body="$(jq  -r '.body'    <<<"$row")"
      ms_title="$(jq -r '.milestone' <<<"$row")"

      # Build gh args
      local -a args
      args=(-R "$REPO" -t "$title" -b "$body")

      # Append labels (array)
      # macOS bash 3.2 doesn't support "readarray", so iterate via jq
      while IFS= read -r lbl; do
        [[ -n "$lbl" ]] && args+=(-l "$lbl")
      done < <(jq -r '.labels[]?' <<<"$row")

      # Resolve milestone number if provided
      if [[ -n "$ms_title" ]]; then
        local msn
        msn="$(gh api -X GET "repos/$REPO/milestones?state=all" -q ".[]|select(.title==\"$ms_title\")|.number" | head -n1 || true)"
        [[ -n "$msn" ]] && args+=(-m "$msn")
      fi

      log "Creating issue: ${title}  (${ms_title:-no milestone})"
      gh issue create "${args[@]}" --json url -q .url || err "Failed to create issue: $title"
      sleep 0.2
    done
  fi
  ok "Issues created for ${mx}"
}

# ------------------------- Execute ------------------------------
cat <<HDR
$(ts) ${BOLD}Bootstrap starting…${NC}
  Repo:    ${BOLD}${REPO}${NC}
  Project: ${BOLD}${PROJECT_NUMBER:-<none>}${NC}
  Mode:    ${BOLD}$([[ "$DRY_RUN" -eq 1 ]] && echo "DRY-RUN" || echo "LIVE")${NC}
  Debug:   ${BOLD}$([[ "$DEBUG" -eq 1 ]] && echo "ON" || echo "OFF")${NC}
  Trace:   ${BOLD}$([[ "$TRACE" -eq 1 ]] && echo "ON" || echo "OFF")${NC}
  Steps:   ${BOLD}${STEPS}${NC}
  Data:    ${BOLD}${DATA}${NC}
HDR

case "$CMD" in
  all)
    run_step "Ensure labels"      "$STEPS/ensure_labels.sh"
    run_step "Ensure milestones"  "$STEPS/ensure_milestones.sh"
    run_step "Create epics"       "$STEPS/create_epics.sh"
    run_step "Core issues"        "$STEPS/create_issues_core.sh"
    run_step "Generated issues"   "$STEPS/create_issues_generated.sh"
    ;;

  labels)
    run_step "Ensure labels" "$STEPS/ensure_labels.sh"
    ;;

  milestones)
    run_step "Ensure milestones" "$STEPS/ensure_milestones.sh"
    ;;

  epics)
    run_step "Create epics" "$STEPS/create_epics.sh"
    ;;

  issues:core)
    run_step "Core issues" "$STEPS/create_issues_core.sh"
    ;;

  issues:generated)
    run_step "Generated issues" "$STEPS/create_issues_generated.sh"
    ;;

  issues:m0|issues:m1|issues:m2|issues:m3|issues:m4|issues:m5|issues:m6|issues:m7|issues:m8|issues:m9|issues:m10|issues:m11)
    mx="$(tr '[:lower:]' '[:upper:]' <<<"${CMD#issues:}")" # M0..M11
    run_issues_mx "$mx"
    ;;

  releases)
    if [[ -x "$STEPS/create_releases.sh" ]]; then
      run_step "Create release(s)" "$STEPS/create_releases.sh" "$@"
    else
      err "Missing or non-executable: $STEPS/create_releases.sh"
      exit 1
    fi
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    err "Unknown command: $CMD"
    usage
    exit 1
    ;;
esac

ok "Bootstrap completed"
