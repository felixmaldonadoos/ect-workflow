#!/bin/bash
#SBATCH --job-name=yeo2011_prep
#SBATCH --account=p32903
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --chdir=/projects/p32903/Alex2/ect-workflow/05_analysis/networks
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source "$HOME/software/miniconda3/etc/profile.d/conda.sh"
conda activate simnibs_env

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg

srun python -u \
    /projects/p32903/Alex2/ect-workflow/05_analysis/networks/mni_yeo2011_prep_data.py