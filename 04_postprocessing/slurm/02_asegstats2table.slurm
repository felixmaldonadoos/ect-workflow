#!/bin/bash

#SBATCH --account=p32903
#SBATCH --partition=normal
#SBATCH --job-name=aseg_tables
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/aseg_tables_%j.out
#SBATCH --error=logs/aseg_tables_%j.err
#SBATCH --export=ALL

set -euo pipefail

module load freesurfer/8.1


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

if (( $# != 2 )); then
    echo "Usage: sbatch 02_asegstats2table.slurm VOLUME_LIST PARENT_DIR" >&2
    exit 2
fi

VOLUME_LIST="$1"
PARENT_DIR="${2%/}"

FIELD_NAME="magnE_mean"

STATS_ROOT="${PARENT_DIR}/stats"

STATS_LIST="${PARENT_DIR}/aparc+aseg.subjects.${FIELD_NAME}.stats.txt"

mkdir -p "${STATS_ROOT}"


# -----------------------------------------------------------------------------
# Validate
# -----------------------------------------------------------------------------

if [[ ! -f "${VOLUME_LIST}" ]]; then
    echo "Error: volume list does not exist:" >&2
    echo "  ${VOLUME_LIST}" >&2
    exit 1
fi


# -----------------------------------------------------------------------------
# Reconstruct stats-file information
# -----------------------------------------------------------------------------

OUTPUT_STATS_FILES=()
OUTPUT_SUBJECT_IDS=()
OUTPUT_STEPS=()

EXPECTED_PREFIX="${PARENT_DIR}/"

MISSING_STATS=0


while IFS= read -r SOURCE_NIFTI || [[ -n "${SOURCE_NIFTI}" ]]; do

    SOURCE_NIFTI="${SOURCE_NIFTI%$'\r'}"

    [[ -n "${SOURCE_NIFTI}" ]] || continue


    if [[ "${SOURCE_NIFTI}" != "${EXPECTED_PREFIX}"* ]]; then
        echo "Error: input volume does not belong to PARENT_DIR:" >&2
        echo "  ${SOURCE_NIFTI}" >&2
        exit 1
    fi


    # -------------------------------------------------------------------------
    # Subject ID
    # -------------------------------------------------------------------------

    RELATIVE_PATH="${SOURCE_NIFTI#"${EXPECTED_PREFIX}"}"
    SUBJECT_ID="${RELATIVE_PATH%%/*}"


    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------

    if [[ "${SOURCE_NIFTI}" =~ /(step_[0-9]+)(/|$) ]]; then
        STEP_NAME="${BASH_REMATCH[1]}"
    else
        echo "Error: could not determine step from:" >&2
        echo "  ${SOURCE_NIFTI}" >&2
        exit 1
    fi


    # -------------------------------------------------------------------------
    # Corresponding stats file
    # -------------------------------------------------------------------------

    OUTPUT_DIR="$(dirname "${SOURCE_NIFTI}")"

    OUTPUT_STATS="${OUTPUT_DIR}/aparc+aseg.${FIELD_NAME}.stats"


    if [[ ! -s "${OUTPUT_STATS}" ]]; then
        echo "Missing stats file:" >&2
        echo "  ${OUTPUT_STATS}" >&2

        (( MISSING_STATS += 1 ))
        continue
    fi


    OUTPUT_STATS_FILES+=("${OUTPUT_STATS}")
    OUTPUT_SUBJECT_IDS+=("${SUBJECT_ID}")
    OUTPUT_STEPS+=("${STEP_NAME}")

done < "${VOLUME_LIST}"


# -----------------------------------------------------------------------------
# Fail if any expected worker output is missing
# -----------------------------------------------------------------------------

if (( MISSING_STATS > 0 )); then
    echo >&2
    echo "Error: ${MISSING_STATS} stats file(s) are missing." >&2
    exit 1
fi

if (( ${#OUTPUT_STATS_FILES[@]} == 0 )); then
    echo "Error: no stats files were found." >&2
    exit 1
fi


# -----------------------------------------------------------------------------
# Save complete list of stats files
# -----------------------------------------------------------------------------

printf '%s\n' "${OUTPUT_STATS_FILES[@]}" > "${STATS_LIST}"

echo
echo "Stats files: ${#OUTPUT_STATS_FILES[@]}"
echo "Stats list:"
echo "  ${STATS_LIST}"


# -----------------------------------------------------------------------------
# Measures
# -----------------------------------------------------------------------------

MEASURES=(
    mean
    std
)


# -----------------------------------------------------------------------------
# Temporary-file cleanup
# -----------------------------------------------------------------------------

TEMP_TABLES=()

cleanup() {
    if (( ${#TEMP_TABLES[@]} > 0 )); then
        rm -f "${TEMP_TABLES[@]}"
    fi
}

trap cleanup EXIT


# -----------------------------------------------------------------------------
# Determine unique step names
# -----------------------------------------------------------------------------

mapfile -t UNIQUE_STEPS < <(
    printf '%s\n' "${OUTPUT_STEPS[@]}" |
        sort -uV
)


# -----------------------------------------------------------------------------
# Generate tables separately for every step
# -----------------------------------------------------------------------------

for STEP_NAME in "${UNIQUE_STEPS[@]}"; do

    STEP_STATS_FILES=()
    STEP_SUBJECT_IDS=()


    # -------------------------------------------------------------------------
    # Collect files belonging to this step
    # -------------------------------------------------------------------------

    for INDEX in "${!OUTPUT_STATS_FILES[@]}"; do

        if [[ "${OUTPUT_STEPS[$INDEX]}" == "${STEP_NAME}" ]]; then

            STEP_STATS_FILES+=(
                "${OUTPUT_STATS_FILES[$INDEX]}"
            )

            STEP_SUBJECT_IDS+=(
                "${OUTPUT_SUBJECT_IDS[$INDEX]}"
            )

        fi

    done


    if (( ${#STEP_STATS_FILES[@]} == 0 )); then
        continue
    fi


    STEP_OUTPUT_DIR="${STATS_ROOT}/${STEP_NAME}"

    mkdir -p "${STEP_OUTPUT_DIR}"


    echo
    echo "======================================================================"
    echo "Step:     ${STEP_NAME}"
    echo "Subjects: ${#STEP_STATS_FILES[@]}"
    echo "======================================================================"


    # -------------------------------------------------------------------------
    # Generate mean and standard-deviation tables
    # -------------------------------------------------------------------------

    for MEASURE in "${MEASURES[@]}"; do

        OUTPUT_TABLE="${STEP_OUTPUT_DIR}/aparc+aseg.${FIELD_NAME}.${MEASURE}.stats.table"

        TEMP_TABLE="${OUTPUT_TABLE}.tmp"

        TEMP_TABLES+=("${TEMP_TABLE}")


        echo
        echo "Generating ${MEASURE} table..."


        asegstats2table \
            --inputs "${STEP_STATS_FILES[@]}" \
            --meas "${MEASURE}" \
            --tablefile "${TEMP_TABLE}"


        # ---------------------------------------------------------------------
        # With --inputs, asegstats2table uses a numeric row identifier.
        #
        # Replace that first column with the actual subject IDs.
        # ---------------------------------------------------------------------

        {
            printf 'subject_id\t'
            head -n 1 "${TEMP_TABLE}" | cut -f2-

            paste \
                <(printf '%s\n' "${STEP_SUBJECT_IDS[@]}") \
                <(tail -n +2 "${TEMP_TABLE}" | cut -f2-)

        } > "${OUTPUT_TABLE}"


        rm -f "${TEMP_TABLE}"

        echo "Saved:"
        echo "  ${OUTPUT_TABLE}"

    done

done


echo
echo "======================================================================"
echo "Table generation complete."
echo "======================================================================"