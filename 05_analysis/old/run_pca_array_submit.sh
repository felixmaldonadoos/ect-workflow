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

demean_args=(--demean-by brain_mean parcel_p95_mean parcel_mean)
extra_args=()
demean_set=0
while (( $# )); do
    case "$1" in
        --existing-only|--demean-by)
            if (( demean_set )); then
                echo "Choose --existing-only or --demean-by once." >&2; exit 2
            fi
            demean_set=1
            option="$1"
            shift
            demean_args=()
            if [[ "$option" == "--demean-by" ]]; then
                demean_args=(--demean-by)
                while (( $# )) && [[ "$1" != --* ]]; do
                    case "$1" in
                        brain_mean|parcel_p95_mean|parcel_mean) demean_args+=("$1") ;;
                        *) echo "Invalid demeaning reference: $1" >&2; exit 2 ;;
                    esac
                    shift
                done
                if (( ${#demean_args[@]} == 1 )); then
                    echo "--demean-by requires at least one mode." >&2; exit 2
                fi
            fi
            ;;
        --subject-scan-map|--cognitive-scores|--stimulus-sheet|--query-path|--placement-alias)
            if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
                echo "$1 requires a value." >&2; exit 2
            fi
            value="$2"
            case "$1" in
                --subject-scan-map|--cognitive-scores|--query-path)
                    if [[ ! -f "$value" || ! -r "$value" ]]; then
                        echo "Required input file not found or unreadable: $value" >&2; exit 2
                    fi
                    value="$(realpath -e -- "$value")"
                    ;;
            esac
            extra_args+=("$1" "$value")
            shift 2
            ;;
        --selection-only) extra_args+=("$1"); shift ;;
        *) echo "Unsupported argument: $1" >&2; exit 2 ;;
    esac
done
analysis_args=("${demean_args[@]}" "${extra_args[@]}")

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
