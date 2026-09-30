#!/usr/bin/env bash
# Submit all four model/step combinations, including all three demeaning modes.
# Examples:
#   bash run_pca_array_submit.sh
#   ARRAY_TASKS=1,3 bash run_pca_array_submit.sh --demean-by parcel_mean
#   ARRAY_TASKS=2-3 bash run_pca_array_submit.sh --existing-only

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARRAY_SCRIPT="$SCRIPT_DIR/run_pca_array.slurm"
LOG_DIR="$SCRIPT_DIR/logs/atlas_pca"
ARRAY_TASKS="${ARRAY_TASKS:-0-3}"

# Accept selections within 0-3 and an optional Slurm concurrency limit.
if [[ ! "$ARRAY_TASKS" =~ ^[0-3](-[0-3])?(,[0-3](-[0-3])?)*(%[1-9][0-9]*)?$ ]]; then
    echo "Invalid ARRAY_TASKS: $ARRAY_TASKS. Examples: 0-3, 1,3, 2-3, 0-3%2." >&2
    exit 2
fi

if (( $# == 0 )); then
    analysis_args=(--demean-by brain_mean parcel_p95_mean parcel_mean)
elif [[ "$1" == "--existing-only" && $# == 1 ]]; then
    analysis_args=()
elif [[ "$1" == "--demean-by" && $# -ge 2 ]]; then
    for reference in "${@:2}"; do
        case "$reference" in
            brain_mean|parcel_p95_mean|parcel_mean) ;;
            *) echo "Invalid demeaning reference: $reference" >&2; exit 2 ;;
        esac
    done
    analysis_args=("$@")
else
    echo "Usage: $0 [--demean-by MODE ... | --existing-only]" >&2
    exit 2
fi

for file in "$ARRAY_SCRIPT" "$SCRIPT_DIR/run_pca_weighted_global_E_job_synthsr.py"; do
    if [[ ! -r "$file" ]]; then
        echo "Required file not found or unreadable: $file" >&2
        exit 1
    fi
done

# Slurm opens log files before the worker starts; create the parent now.
mkdir -p -- "$LOG_DIR"

echo "Array tasks: $ARRAY_TASKS"
echo "Analysis arguments: ${analysis_args[*]:-none (existing analyses only)}"
echo "Log directory: $LOG_DIR"

sbatch \
    --chdir="$SCRIPT_DIR" \
    --array="$ARRAY_TASKS" \
    --output="$LOG_DIR/%A_%a.out" \
    --error="$LOG_DIR/%A_%a.err" \
    "$ARRAY_SCRIPT" "${analysis_args[@]}"
