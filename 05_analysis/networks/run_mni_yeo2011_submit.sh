#!/usr/bin/env bash
# Validate the results root and create logs before Slurm opens output files.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash run_mni_yeo2011_submit.sh [analysis_name] [options]

  --stage all|prep|correlations  Job to submit (default: all).
  --field-name FIELD            NIfTI field name (default: magnE_mean).
  --plot-results 0|1            Plot full-workflow results (default: 1).
  -h, --help                    Show this help.

RESULTS_DIR must be an absolute directory outside the analysis source tree.
analysis_name defaults to mni_yeo2011; each run writes below that analysis.
The correlations stage runs permutation correlations only, using the raw
master unless MASTER_PATH_ROI points to another existing master CSV.
USAGE
}

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_NAME="${ANALYSIS_NAME:-mni_yeo2011}"
FIELD_NAME="${FIELD_NAME:-magnE_mean}"
PLOT_RESULTS="${PLOT_RESULTS:-1}"
STAGE=all

if (( $# )) && [[ "$1" != -* ]]; then
    ANALYSIS_NAME="$1"
    shift
fi

while (( $# )); do
    case "$1" in
        --stage|--field-name|--plot-results)
            if (( $# < 2 )); then
                printf 'ERROR: %s requires a value.\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --stage) STAGE="$2" ;;
                --field-name) FIELD_NAME="$2" ;;
                --plot-results) PLOT_RESULTS="$2" ;;
            esac
            shift 2
            ;;
        --analysis-name|--analysis-name=*)
            printf 'ERROR: Pass analysis_name as the first positional argument.\n' >&2
            exit 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'ERROR: Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$STAGE" in
    all) JOB_SCRIPT="$SCRIPT_ROOT/run_mni_yeo2011_all.sh" ;;
    prep) JOB_SCRIPT="$SCRIPT_ROOT/run_mni_yeo2011_prep.sh" ;;
    correlations) JOB_SCRIPT="$SCRIPT_ROOT/run_mni_yeo2011_correlations.sh" ;;
    *) printf 'ERROR: --stage must be all, prep, or correlations.\n' >&2; exit 2 ;;
esac
if [[ "$PLOT_RESULTS" != "0" && "$PLOT_RESULTS" != "1" ]]; then
    printf 'ERROR: --plot-results / PLOT_RESULTS must be 0 or 1.\n' >&2
    exit 2
fi
if [[ ! -r "$JOB_SCRIPT" || ! -s "$JOB_SCRIPT" ]]; then
    printf 'ERROR: Job script is missing, empty, or unreadable: %s\n' "$JOB_SCRIPT" >&2
    exit 2
fi

export RESULTS_DIR ANALYSIS_NAME FIELD_NAME PLOT_RESULTS
export YEO_SCRIPT_ROOT="$SCRIPT_ROOT"
export PYTHONDONTWRITEBYTECODE=1

ANALYSIS_DIR="$("${YEO_VALIDATE_PYTHON:-python3}" -B - "$SCRIPT_ROOT/.." "$ANALYSIS_NAME" "$FIELD_NAME" <<'PY'
import re
import sys

sys.path.insert(0, sys.argv[1])
from analysis_paths import resolve_analysis_dir, validate_output_path

if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sys.argv[3]):
    raise SystemExit("FIELD_NAME must be a single filename-safe field name.")
try:
    analysis_dir = resolve_analysis_dir(sys.argv[2])
    for suffix in ("data", "plots/raw", f"plots/normed_brain_{sys.argv[3]}_mean", "logs", "logs/matplotlib_cache"):
        validate_output_path(analysis_dir / suffix, analysis_name=sys.argv[2])
except ValueError as error:
    raise SystemExit(str(error)) from error
print(analysis_dir)
PY
)"

LOG_DIR="$ANALYSIS_DIR/logs"
mkdir -p -- "$LOG_DIR"
export MPLCONFIGDIR="$LOG_DIR/matplotlib_cache"

printf 'Analysis name: %s\nStage: %s\nResults directory: %s\n' \
    "$ANALYSIS_NAME" "$STAGE" "$ANALYSIS_DIR"

sbatch \
    --export=ALL \
    --chdir="$SCRIPT_ROOT" \
    --output="$LOG_DIR/%x_%j.out" \
    --error="$LOG_DIR/%x_%j.err" \
    "$JOB_SCRIPT"
