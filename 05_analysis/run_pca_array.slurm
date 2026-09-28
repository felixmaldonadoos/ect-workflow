#!/bin/bash
#SBATCH --account=p32903
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=30G
#SBATCH --job-name=atlas_pca
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A/%a.out
#SBATCH --error=logs/%x_%A/%a.err

set -euo pipefail

source "$HOME/software/miniconda3/etc/profile.d/conda.sh"
conda activate simnibs_env

MODEL_TYPES=(
    "skin_single"
    "skin_single"
    "skin_double"
    "skin_double"
)

ANALYSIS_MODES=(
    "static"
    "adaptive"
    "static"
    "adaptive"
)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not defined}"

if (( task_id < 0 || task_id >= ${#MODEL_TYPES[@]} )); then
    echo "Invalid array task ID: $task_id" >&2
    exit 1
fi

model_type="${MODEL_TYPES[$task_id]}"
analysis_mode="${ANALYSIS_MODES[$task_id]}"

# make the job more easily readable using sacct (updates once job starts running, not while pending)
task_job_name="pca_${model_type}_${analysis_mode}"
task_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
scontrol update JobId="$task_job_id" JobName="$task_job_name"

echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $task_id"
echo "Model type: $model_type"
echo "Analysis mode: $analysis_mode"
echo "Python: $(command -v python)"

# main script runner
srun python run_pca_weighted_global_E_job_synthsr.py \
    --model-type "$model_type" \
    --analysis-mode "$analysis_mode"