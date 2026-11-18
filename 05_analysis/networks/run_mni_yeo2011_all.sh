#!/bin/bash
#SBATCH --job-name=yeo2011_all
#SBATCH --account=p32903
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --chdir=/projects/p32903/Alex2/ect-workflow/05_analysis/networks
# Submit through run_mni_yeo2011_submit.sh to configure per-analysis Slurm logs.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

if (( $# )); then
    printf 'ERROR: Configure this job with run_mni_yeo2011_submit.sh.\n' >&2
    exit 2
fi

# Slurm can spool this script outside the source tree; use the submitter's path.
SCRIPT_ROOT="${YEO_SCRIPT_ROOT:-/projects/p32903/Alex2/ect-workflow/05_analysis/networks}"
ANALYSIS_NAME="${ANALYSIS_NAME:-mni_yeo2011}"
FIELD_NAME="${FIELD_NAME:-magnE_mean}"
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
    for suffix in ("data", "plots/raw", f"plots/normed_brain_{sys.argv[3]}_mean", "logs/matplotlib_cache"):
        validate_output_path(analysis_dir / suffix, analysis_name=sys.argv[2])
except ValueError as error:
    raise SystemExit(str(error)) from error
print(analysis_dir)
PY
)"
DATA_DIR="$ANALYSIS_DIR/data"
export MPLCONFIGDIR="$ANALYSIS_DIR/logs/matplotlib_cache"
NORM_TAG="normed_brain_${FIELD_NAME}_mean"
RAW_MASTER="${DATA_DIR}/yeo2011_${FIELD_NAME}_series_ROI_master.csv"
NORM_MASTER="${DATA_DIR}/yeo2011_${FIELD_NAME}_series_ROI_master_${NORM_TAG}.csv"

# Preserve the current plotting behavior; PLOT_RESULTS=0 disables plotting.
PLOT_RESULTS="${PLOT_RESULTS:-1}"
if [[ "$PLOT_RESULTS" != "0" && "$PLOT_RESULTS" != "1" ]]; then
    printf 'ERROR: PLOT_RESULTS must be 0 or 1.\n' >&2
    exit 2
fi

source "${YEO_CONDA_SH:-$HOME/software/miniconda3/etc/profile.d/conda.sh}"
conda activate "${YEO_CONDA_ENV:-simnibs_env}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg

mkdir -p -- "$DATA_DIR" "$ANALYSIS_DIR/logs"
cd -- "$SCRIPT_ROOT"

# An empty Python script exits successfully without producing any output.
# Check dependencies before starting prep or reusing stale results.
REQUIRED_SCRIPTS=(
    mni_yeo2011_prep_data.py
    mni_yeo2011_normalize_brain.py
    mni_yeo2011_correlations.py
    mni_yeo2011_correlations_perm.py
)
if [[ "$PLOT_RESULTS" == "1" ]]; then
    REQUIRED_SCRIPTS+=(mni_yeo2011_plot_correlations.py)
fi
for SCRIPT in "${REQUIRED_SCRIPTS[@]}"; do
    if [[ ! -r "${SCRIPT_ROOT}/${SCRIPT}" || ! -s "${SCRIPT_ROOT}/${SCRIPT}" ]]; then
        printf 'ERROR: Required script is missing, empty, or unreadable: %s\n' \
            "${SCRIPT_ROOT}/${SCRIPT}" >&2
        exit 1
    fi
done
# Build the raw master, then normalize by each GM/WM-only field's nonzero mean.
srun python -B -u \
    "${SCRIPT_ROOT}/mni_yeo2011_prep_data.py" \
    --analysis-name "$ANALYSIS_NAME" \
    --field-name "$FIELD_NAME" \
    --output-path "$RAW_MASTER"

srun python -B -u \
    "${SCRIPT_ROOT}/mni_yeo2011_normalize_brain.py" \
    --analysis-name "$ANALYSIS_NAME" \
    --master-path "$RAW_MASTER" \
    --field-name "$FIELD_NAME" \
    --output-path "$NORM_MASTER"

if [[ ! -s "$NORM_MASTER" ]]; then
    printf 'ERROR: Normalization did not produce a master CSV: %s\n' "$NORM_MASTER" >&2
    exit 1
fi

run_correlations() {
    local master_path="$1"
    local suffix="$2"
    local plot_output_dir="$3"
    local correlation_results="${DATA_DIR}/spearman_${FIELD_NAME}_ROImean_and_network${suffix}_annotated.csv"
    local permutation_results="${DATA_DIR}/spearman_${FIELD_NAME}_ROImean_and_network${suffix}_permutation_annotated.csv"

    printf 'Running correlations for: %s\n' "$master_path"

    srun python -B -u \
        "${SCRIPT_ROOT}/mni_yeo2011_correlations.py" \
        --analysis-name "$ANALYSIS_NAME" \
        --master-path "$master_path" \
        --field-name "$FIELD_NAME" \
        --output-path "$correlation_results"

    srun python -B -u \
        "${SCRIPT_ROOT}/mni_yeo2011_correlations_perm.py" \
        --analysis-name "$ANALYSIS_NAME" \
        --master-path "$master_path" \
        --field-name "$FIELD_NAME" \
        --output-path "$permutation_results" \
        --n-permutations 9999 \
        --seed 42 \
        --batch-size 256

    if [[ "$PLOT_RESULTS" == "1" ]]; then
        srun python -B -u \
            "${SCRIPT_ROOT}/mni_yeo2011_plot_correlations.py" \
            --analysis-name "$ANALYSIS_NAME" \
            --correlations-path "$permutation_results" \
            --field-name "$FIELD_NAME" \
            --output-dir "$plot_output_dir"
    fi
}

# Run both analysis variants with distinct results and plotting destinations.
run_correlations "$RAW_MASTER" "" \
    "${ANALYSIS_DIR}/plots/raw"

run_correlations "$NORM_MASTER" "_${NORM_TAG}" \
    "${ANALYSIS_DIR}/plots/${NORM_TAG}"
