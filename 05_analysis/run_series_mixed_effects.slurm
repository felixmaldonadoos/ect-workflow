#!/usr/bin/env bash
#SBATCH --job-name=series_lmm
#SBATCH --account=p32903
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:5:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Submit through run_series_mixed_effects_submit.sh. It creates logs before
# submission and supplies stdout/stderr paths under RESULTS_DIR/analysis_name.
set -euo pipefail

SCRIPT_ROOT="${SERIES_SCRIPT_ROOT:-/projects/p32903/Alex2/ect-workflow/05_analysis}"
RUNNER="$SCRIPT_ROOT/run_series_mixed_effects.py"

if [[ -z "${RESULTS_DIR:-}" ]]; then
    echo "RESULTS_DIR must be set. Submit with run_series_mixed_effects_submit.sh." >&2
    exit 2
fi
if [[ ! -r "$RUNNER" ]]; then
    echo "Analysis runner not found or unreadable: $RUNNER" >&2
    exit 2
fi

cd -- "$SCRIPT_ROOT"
if [[ -n "${SERIES_PYTHON:-}" ]]; then
    ANALYSIS_PYTHON="$SERIES_PYTHON"
else
    source "${SERIES_CONDA_SH:-$HOME/software/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${SERIES_CONDA_ENV:-simnibs_env}"
    ANALYSIS_PYTHON=python
fi

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

echo "Job ID: ${SLURM_JOB_ID:-not set}"
echo "Results root: $RESULTS_DIR"
echo "Python: $ANALYSIS_PYTHON"
printf 'Analysis arguments:'
printf ' %q' "$@"
printf '\n'

exec srun "$ANALYSIS_PYTHON" -B -u "$RUNNER" "$@"
