#!/usr/bin/env bash
#SBATCH --job-name=mni_nifti_pca
#SBATCH --account=p32903
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Submit through run_mni_nifti_pca_submit.sh, which creates the results log
# directory before sbatch opens stdout/stderr and overrides /dev/null above.
set -euo pipefail

SCRIPT_ROOT="${PCA_SCRIPT_ROOT:-/projects/p32903/Alex2/ect-workflow/05_analysis}"
RUNNER="$SCRIPT_ROOT/run_mni_nifti_pca.py"

if [[ -z "${RESULTS_DIR:-}" ]]; then
    echo "RESULTS_DIR must be set. Submit with run_mni_nifti_pca_submit.sh." >&2
    exit 2
fi
if [[ ! -r "$RUNNER" ]]; then
    echo "Analysis runner not found or unreadable: $RUNNER" >&2
    exit 2
fi

cd -- "$SCRIPT_ROOT"
source "${PCA_CONDA_SH:-$HOME/software/miniconda3/etc/profile.d/conda.sh}"
conda activate "${PCA_CONDA_ENV:-simnibs_env}"

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg

echo "Job ID: ${SLURM_JOB_ID:-not set}"
echo "Results root: $RESULTS_DIR"
echo "Python: $(command -v python)"
printf 'Analysis arguments:'
printf ' %q' "$@"
printf '\n'

exec srun python -B -u "$RUNNER" "$@"
