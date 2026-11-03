#!/usr/bin/env bash
# Usage: bash run_mni_nifti_pca_submit.sh [analysis_name] [pipeline options...]
# Example: bash run_mni_nifti_pca_submit.sh mni_nifti_pca --n-permutations 9999
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_ROOT/run_mni_nifti_pca.slurm"
ANALYSIS_NAME=mni_nifti_pca

if (( $# )) && [[ "$1" != -* ]]; then
    ANALYSIS_NAME="$1"
    shift
fi

for argument in "$@"; do
    case "$argument" in
        --analysis-name|--analysis-name=*)
            echo "Pass analysis_name as the first positional argument, not --analysis-name." >&2
            exit 2
            ;;
    esac
done

for required_file in "$SLURM_SCRIPT" "$SCRIPT_ROOT/run_mni_nifti_pca.py"; do
    if [[ ! -r "$required_file" ]]; then
        echo "Required file not found or unreadable: $required_file" >&2
        exit 2
    fi
done

export RESULTS_DIR
export PCA_SCRIPT_ROOT="$SCRIPT_ROOT"
export PYTHONDONTWRITEBYTECODE=1

# The package import is intentionally light: validation needs only stdlib Python.
# Validate before mkdir so invalid names/roots cannot create stray directories.
LOG_DIR="$("${PCA_VALIDATE_PYTHON:-python3}" -B - "$SCRIPT_ROOT" "$ANALYSIS_NAME" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from nifti_pca.paths import resolve_analysis_dir, validate_output_path

try:
    log_dir = validate_output_path(resolve_analysis_dir(sys.argv[2]) / "logs")
    validate_output_path(log_dir / "matplotlib_cache")
except ValueError as error:
    raise SystemExit(str(error)) from error
print(log_dir)
PY
)"

mkdir -p -- "$LOG_DIR"
export MPLCONFIGDIR="$LOG_DIR/matplotlib_cache"

echo "Analysis name: $ANALYSIS_NAME"
echo "Log directory: $LOG_DIR"

sbatch \
    --export=ALL \
    --chdir="$SCRIPT_ROOT" \
    --output="$LOG_DIR/%x/%j.out" \
    --error="$LOG_DIR/%x/%j.err" \
    "$SLURM_SCRIPT" --analysis-name "$ANALYSIS_NAME" "$@"
