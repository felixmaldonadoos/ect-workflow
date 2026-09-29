#!/bin/bash
#SBATCH --account=p32903
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --job-name=recon_a2009s_002_003
#SBATCH --output=recon_a2009s_002_003.%j.out
#SBATCH --error=recon_a2009s_002_003.%j.err
#SBATCH --export=ALL

set -euo pipefail

module purge all
module load singularity

SUBJECT_ID="subj-cat-002-003"
SUBJECTS_DIR="/projects/p32903/Alex2/datasets/STU00225089/synthsr/subjectsdir"
CONTAINER_PATH="/software/2025/freesurfer/8.1/freesurfer-8.1-mcr.sif"

if [[ ! -d "$SUBJECTS_DIR/$SUBJECT_ID" ]]; then
    echo "ERROR: FreeSurfer subject does not exist: $SUBJECTS_DIR/$SUBJECT_ID" >&2
    exit 1
fi

echo "SUBJECT_ID=$SUBJECT_ID"
echo "SUBJECTS_DIR=$SUBJECTS_DIR"
echo "CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "== STARTING DESTRIEUX RERUN =="

singularity exec -B /projects:/projects \
    "$CONTAINER_PATH" /bin/bash -c "
        set -e

        export FREESURFER_HOME=/usr/local/freesurfer
        export SUBJECTS_DIR=\"${SUBJECTS_DIR}\"
        export OMP_NUM_THREADS=\"${SLURM_CPUS_PER_TASK}\"
        export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=\"${SLURM_CPUS_PER_TASK}\"

        # Prevent ~/.local/bin/test from shadowing /usr/bin/test.
        export PATH=\"/usr/local/freesurfer/bin:/usr/bin:/bin\"

        source \"\$FREESURFER_HOME/SetUpFreeSurfer.sh\"

        echo \"Using test command: \$(type -P test)\"

        \"\$FREESURFER_HOME/bin/recon-all\" \
            -subjid \"${SUBJECT_ID}\" \
            -hemi lh \
            -cortparc2 \
            -parcstats2 \
            -openmp \"\$OMP_NUM_THREADS\" \
            -sd \"\$SUBJECTS_DIR\"

        \"\$FREESURFER_HOME/bin/recon-all\" \
            -subjid \"${SUBJECT_ID}\" \
            -aparc2aseg \
            -sd \"\$SUBJECTS_DIR\"
    "

STATS_FILE="$SUBJECTS_DIR/$SUBJECT_ID/stats/lh.aparc.a2009s.stats"

echo "== CHECKING OUTPUT =="

if grep -n "G_cingul-Post-ventral" "$STATS_FILE"; then
    echo "SUCCESS: G_cingul-Post-ventral is present."
else
    echo "ERROR: G_cingul-Post-ventral remains absent after rerunning." >&2
    exit 2
fi