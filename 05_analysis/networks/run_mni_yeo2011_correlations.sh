#!/bin/bash
#SBATCH --job-name=yeo2011_corrs
#SBATCH --account=p32903
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:05:00
#SBATCH --chdir=/projects/p32903/Alex2/ect-workflow/05_analysis/networks
# Submit through run_mni_yeo2011_submit.sh to configure per-analysis Slurm logs.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

if (( $# )); then
    printf 'ERROR: Configure this job with run_mni_yeo2011_submit.sh.\n' >&2
    exit 2
fi

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
    for suffix in ("data", "logs/matplotlib_cache"):
        validate_output_path(analysis_dir / suffix, analysis_name=sys.argv[2])
except ValueError as error:
    raise SystemExit(str(error)) from error
print(analysis_dir)
PY
)"
export MPLCONFIGDIR="$ANALYSIS_DIR/logs/matplotlib_cache"
MASTER_PATH_ROI="${MASTER_PATH_ROI:-$ANALYSIS_DIR/data/yeo2011_${FIELD_NAME}_series_ROI_master.csv}"

source "${YEO_CONDA_SH:-$HOME/software/miniconda3/etc/profile.d/conda.sh}"
conda activate "${YEO_CONDA_ENV:-simnibs_env}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg

mkdir -p -- "$ANALYSIS_DIR/data" "$ANALYSIS_DIR/logs"
cd -- "$SCRIPT_ROOT"

# Preserve this stage's permutation-only analysis. The Python entrypoint chooses
# a raw or normalized results filename from the master CSV's metadata.
srun python -B -u \
    "$SCRIPT_ROOT/mni_yeo2011_correlations_perm.py" \
    --analysis-name "$ANALYSIS_NAME" \
    --master-path "$MASTER_PATH_ROI" \
    --field-name "$FIELD_NAME" \
    --n-permutations 9999 \
    --seed 42 \
    --batch-size 256
