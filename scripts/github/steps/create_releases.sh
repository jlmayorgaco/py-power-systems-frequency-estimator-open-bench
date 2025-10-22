#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/config.env"
source "$HERE/lib.sh"
need_gh

TAG="${1:-v0.1.0}"
TITLE="${2:-OpenFreqBench $TAG}"
NOTES="${3:-Auto-generated release.}"
PRE="${PRE:-0}"

if ! git rev-parse "$TAG" >/dev/null 2>&1; then
  run git tag -a "$TAG" -m "$TITLE"
  run git push origin "$TAG"
fi
if ! gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
  preflag=""; (( PRE )) && preflag="--prerelease"
  run gh release create "$TAG" -R "$REPO" --generate-notes -t "$TITLE" -n "$NOTES" $preflag
fi
