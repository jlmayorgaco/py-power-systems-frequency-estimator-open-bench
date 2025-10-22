#!/usr/bin/env bash
set -Eeuo pipefail
log(){ printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn(){ printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err(){ printf "\033[1;31m[ERR]\033[0m  %s\n" "$*" >&2; }
die(){ err "$*"; exit 1; }
req(){ command -v "$1" >/dev/null || die "Missing required tool: $1"; }
