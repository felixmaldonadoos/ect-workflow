#!/usr/bin/env bash
set -euo pipefail

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

: "${DS_NMHC_ROOT:?Set DS_NMHC_ROOT to the absolute dataset root}"
command -v realpath >/dev/null 2>&1 || die "Required command not found: realpath"

[[ -d "$DS_NMHC_ROOT" ]] || die "DS_NMHC_ROOT directory not found: $DS_NMHC_ROOT"
NMHC_ROOT="$(realpath -- "$DS_NMHC_ROOT")"

SCRIPT_PATH="$(realpath -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
RUNNER_SCRIPT="$SCRIPT_DIR/run_charming_synthsr.sh"
[[ -f "$RUNNER_SCRIPT" ]] || die "Runner script not found: $RUNNER_SCRIPT"
RUNNER_SCRIPT="$(realpath -- "$RUNNER_SCRIPT")"

bash "$RUNNER_SCRIPT" \
  --subjects-file "$NMHC_ROOT/synthsr/.subjects_charm.run.flags" \
  --dataset-dir "$NMHC_ROOT/synthsr/mri/t1" \
  --subjects-dir "$NMHC_ROOT/synthsr/subjectsdir" \
  --output-dir "$NMHC_ROOT/synthsr/m2ms_2mm" \
  --models both \
  --thickness 2
