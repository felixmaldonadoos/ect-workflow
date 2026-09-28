#!/bin/bash


# # Only skin_single static
# sbatch --array=0 run_pca_array.slurm

# # Both adaptive analyses
# sbatch --array=1,3 run_pca_array.slurm

# # Both skin_double analyses
sbatch --array=2-3 run_pca_array.slurm

# ALL 
# sbatch run_pca_array.slurm