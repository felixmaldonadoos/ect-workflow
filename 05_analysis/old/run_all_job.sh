#!/bin/bash
#SBATCH --account=p32903
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --job-name=analysis
#SBATCH --output=logs/run_all_atlases.out
#SBATCH --error=logs/run_all_atlases.err

source "$HOME/software/miniconda3/etc/profile.d/conda.sh"
set -euo pipefail
echo "Using conda install: $CONDA_EXE"
conda activate simnibs_env

# python3 run_all_atlases_job.py
# python3 run_pca_weighted_global_E_job.py

## check parcel counts. helps check what subject-atlas-ROIs do not meet the minimum sample requirement needed to calculate a percentile 
# python3 check_volume_roi_sample_counts.py

#######
# MAIN original + synthsr
#######
python3 run_pca_weighted_global_E_job_synthsr.py

python run_pca_weighted_global_E_job_synthsr.py \
  --model-type skin_single \
  --analysis-mode static