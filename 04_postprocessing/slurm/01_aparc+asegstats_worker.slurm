#!/bin/bash

#SBATCH --account=p32903
#SBATCH --partition=normal
#SBATCH --job-name=aparc_segstats
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=logs/aparc_segstats_%A_%a.out
#SBATCH --error=logs/aparc_segstats_%A_%a.err
#SBATCH --export=ALL
set -euo pipefail

module load freesurfer/8.1


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

if (( $# != 3 )); then
    echo "Usage: sbatch 01_aparc+asegstats_worker.slurm VOLUME_LIST PARENT_DIR SUBJECTS_DIR" >&2
    exit 2
fi

VOLUME_LIST="$1"
PARENT_DIR="${2%/}"
SUBJECTS_DIR="${3%/}"

export SUBJECTS_DIR

FIELD_NAME="magnE_mean"


# -----------------------------------------------------------------------------
# Validate Slurm array context
# -----------------------------------------------------------------------------

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Error: this script must be run as a Slurm array job." >&2
    exit 1
fi


# -----------------------------------------------------------------------------
# Get NIfTI corresponding to this array index
# -----------------------------------------------------------------------------

SOURCE_NIFTI="$(
    sed -n "${SLURM_ARRAY_TASK_ID}{p;q;}" "${VOLUME_LIST}"
)"

if [[ -z "${SOURCE_NIFTI}" ]]; then
    echo "Error: no volume found for array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

SOURCE_NIFTI="${SOURCE_NIFTI%$'\r'}"


# -----------------------------------------------------------------------------
# Make sure volume belongs to PARENT_DIR
# -----------------------------------------------------------------------------

EXPECTED_PREFIX="${PARENT_DIR}/"

if [[ "${SOURCE_NIFTI}" != "${EXPECTED_PREFIX}"* ]]; then
    echo "Error: volume does not belong to PARENT_DIR." >&2
    echo "Parent: ${PARENT_DIR}" >&2
    echo "Volume: ${SOURCE_NIFTI}" >&2
    exit 1
fi


# -----------------------------------------------------------------------------
# Extract subject ID
# -----------------------------------------------------------------------------

RELATIVE_PATH="${SOURCE_NIFTI#"${EXPECTED_PREFIX}"}"
SUBJECT_ID="${RELATIVE_PATH%%/*}"

if [[ -z "${SUBJECT_ID}" ]]; then
    echo "Error: could not determine subject ID from:" >&2
    echo "  ${SOURCE_NIFTI}" >&2
    exit 1
fi


# -----------------------------------------------------------------------------
# Extract step
# -----------------------------------------------------------------------------

if [[ "${SOURCE_NIFTI}" =~ /(step_[0-9]+)(/|$) ]]; then
    STEP_NAME="${BASH_REMATCH[1]}"
else
    echo "Error: could not determine step from:" >&2
    echo "  ${SOURCE_NIFTI}" >&2
    exit 1
fi


# -----------------------------------------------------------------------------
# Input / output files
# -----------------------------------------------------------------------------

OUTPUT_DIR="$(dirname "${SOURCE_NIFTI}")"

SEGMENTATION="${SUBJECTS_DIR}/${SUBJECT_ID}/mri/aparc+aseg.mgz"
OUTPUT_STATS="${OUTPUT_DIR}/aparc+aseg.${FIELD_NAME}.stats"


echo
echo "======================================================================"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Subject:    ${SUBJECT_ID}"
echo "Step:       ${STEP_NAME}"
echo "Field:      ${SOURCE_NIFTI}"
echo "Seg:        ${SEGMENTATION}"
echo "Output:     ${OUTPUT_STATS}"
echo "======================================================================"


# -----------------------------------------------------------------------------
# Validate inputs
# -----------------------------------------------------------------------------

if [[ ! -f "${SOURCE_NIFTI}" ]]; then
    echo "Error: field volume does not exist:" >&2
    echo "  ${SOURCE_NIFTI}" >&2
    exit 1
fi

if [[ ! -r "${SOURCE_NIFTI}" ]]; then
    echo "Error: field volume is not readable:" >&2
    echo "  ${SOURCE_NIFTI}" >&2
    exit 1
fi

if [[ ! -f "${SEGMENTATION}" ]]; then
    echo "Error: segmentation does not exist:" >&2
    echo "  ${SEGMENTATION}" >&2
    exit 1
fi

if [[ ! -r "${SEGMENTATION}" ]]; then
    echo "Error: segmentation is not readable:" >&2
    echo "  ${SEGMENTATION}" >&2
    exit 1
fi


# -----------------------------------------------------------------------------
# Geometry helper
#
# Dimensions + vox2ras define the image grid we care about here.
# -----------------------------------------------------------------------------

geometry_signature() {
    local volume="$1"

    {
        mri_info --dim "${volume}" \
            | awk '{print $1, $2, $3}'

        mri_info --vox2ras "${volume}" \
            | awk '{$1=$1; print}'
    }
}


FIELD_DIMS="$(
    mri_info --dim "${SOURCE_NIFTI}" |
        awk '{print $1, $2, $3}'
)"

SEG_DIMS="$(
    mri_info --dim "${SEGMENTATION}" |
        awk '{print $1, $2, $3}'
)"

FIELD_GEOMETRY="$(geometry_signature "${SOURCE_NIFTI}")"
SEG_GEOMETRY="$(geometry_signature "${SEGMENTATION}")"

echo
echo "Field dimensions: ${FIELD_DIMS}"
echo "Seg dimensions:   ${SEG_DIMS}"


# -----------------------------------------------------------------------------
# Determine volume to use for stats
# -----------------------------------------------------------------------------

FIELD_FOR_STATS="${SOURCE_NIFTI}"


# -----------------------------------------------------------------------------
# Resample when grids do not match
# -----------------------------------------------------------------------------

if [[ "${FIELD_GEOMETRY}" != "${SEG_GEOMETRY}" ]]; then

    echo
    echo "Grid mismatch detected."
    echo "Resampling field -> FreeSurfer segmentation grid."

    # Prefer node-local scratch when available.
    TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"

    TASK_TMP="${TMP_ROOT%/}/segstats_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    mkdir -p "${TASK_TMP}"

    cleanup() {
        rm -rf "${TASK_TMP}"
    }

    trap cleanup EXIT

    RESAMPLED_FIELD="${TASK_TMP}/${SUBJECT_ID}.${STEP_NAME}.${FIELD_NAME}.fsgrid.nii.gz"

    echo "Temporary resampled field:"
    echo "  ${RESAMPLED_FIELD}"

    mri_vol2vol \
        --mov "${SOURCE_NIFTI}" \
        --targ "${SEGMENTATION}" \
        --regheader \
        --interp trilin \
        --o "${RESAMPLED_FIELD}" \
        --no-save-reg

    if [[ ! -s "${RESAMPLED_FIELD}" ]]; then
        echo "Error: mri_vol2vol did not produce the expected file:" >&2
        echo "  ${RESAMPLED_FIELD}" >&2
        exit 1
    fi

    RESAMPLED_DIMS="$(
        mri_info --dim "${RESAMPLED_FIELD}" |
            awk '{print $1, $2, $3}'
    )"

    echo "Resampled dimensions: ${RESAMPLED_DIMS}"

    if [[ "${RESAMPLED_DIMS}" != "${SEG_DIMS}" ]]; then
        echo "Error: resampled field dimensions do not match segmentation." >&2
        echo "Resampled: ${RESAMPLED_DIMS}" >&2
        echo "Seg:       ${SEG_DIMS}" >&2
        exit 1
    fi

    FIELD_FOR_STATS="${RESAMPLED_FIELD}"

else

    echo
    echo "Field and segmentation grids already match."
    echo "No resampling needed."

fi


# -----------------------------------------------------------------------------
# Generate stats
# -----------------------------------------------------------------------------

echo
echo "Field used for stats:"
echo "  ${FIELD_FOR_STATS}"
echo

mri_segstats \
    --i "${FIELD_FOR_STATS}" \
    --seg "${SEGMENTATION}" \
    --ctab "${FREESURFER_HOME}/FreeSurferColorLUT.txt" \
    --sum "${OUTPUT_STATS}"


# -----------------------------------------------------------------------------
# Validate output
# -----------------------------------------------------------------------------

if [[ ! -s "${OUTPUT_STATS}" ]]; then
    echo "Error: mri_segstats did not produce a valid stats file:" >&2
    echo "  ${OUTPUT_STATS}" >&2
    exit 1
fi

echo
echo "Finished:"
echo "  ${OUTPUT_STATS}"