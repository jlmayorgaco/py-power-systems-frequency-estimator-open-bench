#!/usr/bin/env bash
# OpenFreqBench — NO-OP scaffold (macOS-safe)
# Only logs what would happen; performs no network or file mutations.
# Pass --dry-run to preview steps that themselves support DRY_RUN.

set -u

# ----------------- flags / defaults -----------------
DRY_RUN=0
DEBUG=0
TRACE=0
PROJECT_NUMBER=""
# Accept any form here (URL/SSH/owner/name) — we sanitize below.
REPO="https://github.com/jlmayorgaco/py-power-systems-frequency-estimator-open-bench"

CMD="${1:-all}"; shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --repo)        REPO="${2:-}"; shift 2;;
    --project)     PROJECT_NUMBER="${2:-}"; shift 2;;
    --dry-run)     DRY_RUN=1; shift;;
    --debug)       DEBUG=1; shift;;
    --trace)       TRACE=1; shift;;
    -h|--help)     CMD="help"; shift;;
    *) break;;
  esac
done

[ "$TRACE" -eq 1 ] && { export PS4='+ ${BASH_SOURCE##*/}:${LINENO}:${FUNCNAME[0]:-main}() '; set -x; }

# ----------------- colors / logging -----------------
if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLU=$'\033[34m'; MAG=$'\033[35m'; CYA=$'\033[36m'; NC=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GRN=""; YLW=""; BLU=""; MAG=""; CYA=""; NC=""
fi
ts(){ date +"%Y-%m-%dT%H:%M:%S%z"; }
log(){  printf "%s %b%s%b %s\n" "$(ts)" "${BLU}" "INFO" "${NC}" "$*"; }
ok(){   printf "%s %b%s%b %s\n" "$(ts)" "${GRN}" "OK  " "${NC}" "$*"; }
warn(){ printf "%s %b%s%b %s\n" "$(ts)" "${YLW}" "WARN" "${NC}" "$*"; }
err(){  printf "%s %b%s%b %s\n" "$(ts)" "${RED}" "ERR " "${NC}" "$*" 1>&2; }
dbg(){  [ "$DEBUG" -eq 1 ] && printf "%s %b%s%b %s\n" "$(ts)" "${MAG}" "DBG " "${NC}" "$*"; }

usage(){
  cat <<EOF
${BOLD}OpenFreqBench bootstrap — NO-OP scaffold${NC}

${BOLD}Usage${NC}
  ./bootstrap.sh [command] [flags]

${BOLD}Commands${NC}
  all                 Log labels → milestones → epics → issues:core
  labels              Log label step
  milestones          Log milestone step
  epics               Log epic step
  issues:core         Log core issues step
  issues:m{0..11}     Log issues file load for M0..M11
  releases [args...]  Log release step with passthrough args
  help                Show this help

${BOLD}Flags${NC}
  --repo owner/name
  --project N
  --dry-run
  --debug
  --trace
EOF
}

# ----------------- helpers -----------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
STEPS="$HERE/steps"
DATA="$HERE/data"

sanitize_repo() {
  # Normalize to owner/name
  local r="$1"
  r="${r#git@github.com:}"     # ssh prefix
  r="${r#https://github.com/}" # https prefix
  r="${r%.git}"                # .git suffix
  echo "$r"
}

detect_repo() {
  # If REPO set, just echo (assumed sanitized by caller)
  if [ -n "${REPO:-}" ]; then
    echo "$(sanitize_repo "$REPO")"; return
  fi
  # Fallback to git origin → sanitize
  if command -v git >/dev/null 2>&1; then
    local remote
    remote="$(git -C "$ROOT" remote -v 2>/dev/null | awk '/origin[[:space:]].*\(push\)/{print $2}' | head -n1 || true)"
    if [ -n "$remote" ]; then
      echo "$(sanitize_repo "$remote")"; return
    fi
  fi
  echo "<undetected>"
}

# Ensure REPO is normalized now and exported for child steps
REPO="$(detect_repo)"
export REPO

banner(){
  cat <<HDR
$(ts) ${BOLD}Bootstrap (NO-OP) starting…${NC}
  Repo:    ${BOLD}${REPO}${NC}
  Project: ${BOLD}${PROJECT_NUMBER:-<none>}${NC}
  Mode:    ${BOLD}$([ "$DRY_RUN" -eq 1 ] && echo "DRY-RUN" || echo "LIVE (still no-op)")${NC}
  Debug:   ${BOLD}$([ "$DEBUG" -eq 1 ] && echo "ON" || echo "OFF")${NC}
  Trace:   ${BOLD}$([ "$TRACE" -eq 1 ] && echo "ON" || echo "OFF")${NC}
  Steps:   ${BOLD}${STEPS}${NC}
  Data:    ${BOLD}${DATA}${NC}
HDR
}

noop_step(){
  # args: name, (optional) extra
  local name="$1"; shift || true
  local extra="$*"
  log "${BOLD}${name}${NC} — would run"
  [ -n "$extra" ] && dbg "args: ${extra}"
  sleep 0.05
  ok  "${name} logged"
}

noop_issues_mx(){
  local mx="$1"
  local file="$DATA/issues/${mx}.json"
  log "Issues loader — target file: ${file}"
  if [ -f "$file" ]; then
    log "Would parse ${mx} (showing first titles up to 5 if jq exists)"
    if command -v jq >/dev/null 2>&1; then
      jq -r '.[0:5][]?.title // empty' "$file" 2>/dev/null | sed 's/^/  - /' || true
    else
      warn "jq not installed; skipping preview"
    fi
  else
    warn "File not found (ok in NO-OP): $file"
  fi
  ok "Issues ${mx} logged"
}

run_step() {
  local name="$1"; shift || true
  local script="$1"; shift || true
  log "${name} — starting (${script})"
  # Pass normalized REPO and flags to the child step
  DRY_RUN="${DRY_RUN}" PROJECT_NUMBER="${PROJECT_NUMBER}" REPO="${REPO}" \
    bash "$script" "$@" || { err "Step failed: ${name} (${script})"; exit 1; }
  ok  "${name} done"
}

run_issues_mx() {
  local mx="$1"
  local file="$DATA/issues/${mx}.json"
  log "Issues loader — target file: ${file}"
  if [ ! -f "$file" ]; then
    err "Issues file not found: $file"; exit 1
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    if command -v jq >/dev/null 2>&1; then
      jq -r '.[] | "- \(.title) [\(.milestone // "<none>")]"' "$file"
    else
      warn "jq not found; skipping preview"
    fi
    ok "DRY preview completed for ${mx}"
    return
  fi
  # LIVE path delegates to a generic file-driven creator (if you add one later)
  run_step "Issues ${mx}" "$STEPS/create_issues_from_file.sh" "$file"
}

# ----------------- dispatch -----------------
case "$CMD" in
  help|-h|--help)
    usage
    exit 0
    ;;

  all)
    banner
    #run_step "Ensure labels"      "$STEPS/ensure_labels.sh"
    #run_step "Ensure milestones"  "$STEPS/ensure_milestones.sh"
    #run_step "Create epics"       "$STEPS/create_epics.sh"
    run_step "Core issues"        "$STEPS/create_issues_core.sh"
    ;;

  labels)      banner; run_step "Ensure labels"     "$STEPS/ensure_labels.sh" ;;
  milestones)  banner; run_step "Ensure milestones" "$STEPS/ensure_milestones.sh" ;;
  epics)       banner; run_step "Create epics"      "$STEPS/create_epics.sh" ;;
  issues:core) banner; run_step "Core issues"       "$STEPS/create_issues_core.sh" ;;

  issues:m0|issues:m1|issues:m2|issues:m3|issues:m4|issues:m5|issues:m6|issues:m7|issues:m8|issues:m9|issues:m10|issues:m11)
    banner
    mx="$(printf "%s" "${CMD#issues:}" | tr '[:lower:]' '[:upper:]')"
    run_issues_mx "$mx"
    ;;

  releases)
    banner
    noop_step "Create release(s)" "$*"
    ;;

  *)
    banner
    warn "Unknown command: $CMD"
    usage
    exit 1
    ;;
esac

ok "Bootstrap (NO-OP) completed"
