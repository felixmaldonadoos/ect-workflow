#!/bin/bash
#SBATCH --account=p32903
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=30G
#SBATCH --job-name=atlas_pca
#SBATCH --array=0-3
#SBATCH --output=logs/atlas_pca/%A_%a.out
#SBATCH --error=logs/atlas_pca/%A_%a.err

set -euo pipefail

# The submit wrapper sets the working directory to this analysis folder.
MODEL_TYPES=(skin_single skin_single skin_double skin_double)
ANALYSIS_MODES=(static adaptive static adaptive)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not defined}"

if [[ ! "$task_id" =~ ^[0-3]$ ]]; then
    echo "Invalid array task ID: $task_id" >&2
    exit 1
fi

model_type="${MODEL_TYPES[$task_id]}"
analysis_mode="${ANALYSIS_MODES[$task_id]}"
runner="$PWD/run_pca_weighted_global_E_job_synthsr.py"

if [[ ! -r "$runner" ]]; then
    echo "Analysis runner not found: $runner. Submit with run_pca_array_submit.sh." >&2
    exit 1
fi

source "${PCA_CONDA_SH:-$HOME/software/miniconda3/etc/profile.d/conda.sh}"
conda activate simnibs_env

# Rename this array task once it starts, for readable sacct/squeue output.
task_job_name="pca_${model_type}_${analysis_mode}"
task_job_id="${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is not defined}_${task_id}"
scontrol update "JobId=$task_job_id" "JobName=$task_job_name"

echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $task_id"
echo "Model type: $model_type"
echo "Analysis mode: $analysis_mode"
echo "Python: $(command -v python)"
echo "Analysis arguments: ${*:-none (existing analyses only)}"

# One call runs the existing analyses plus any requested demeaning modes.
exec srun python -u "$runner" \
    --model-type "$model_type" \
    --analysis-mode "$analysis_mode" "$@"
