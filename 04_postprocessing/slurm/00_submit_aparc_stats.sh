#!/bin/bash

set -euo pipefail

# -----------------------------------------------------------------------------
# Scripts
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPT_BUILD_VOLUMES="${SCRIPT_DIR}/build_subjvolumes_list.sh"
SCRIPT_WORKER="${SCRIPT_DIR}/01_aparc+asegstats_worker.slurm"
SCRIPT_TABLES="${SCRIPT_DIR}/02_asegstats2table.slurm"

# -----------------------------------------------------------------------------
# Output directories for lists/logs
# -----------------------------------------------------------------------------

LIST_DIR="${SCRIPT_DIR}/lists"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LIST_DIR}"
mkdir -p "${LOG_DIR}"

# -----------------------------------------------------------------------------
# Fixed field for now
# -----------------------------------------------------------------------------

FIELD_NAME="magnE_mean"

# Maximum simultaneous jobs PER array.
#
# With four arrays submitted, keep this reasonably conservative initially.
# Override from command line if desired:
#
#   MAX_CONCURRENT=50 bash 00_submit_aparc_stats.sh
#
MAX_CONCURRENT="${MAX_CONCURRENT:-25}"


# -----------------------------------------------------------------------------
# Submit one dataset / skin-model combination
# -----------------------------------------------------------------------------

run_stats() {
    local label="$1"
    local parent_dir="$2"
    local subjects_dir="$3"

    local job_tag="${label//./_}"
    local volumes_file="${LIST_DIR}/subjects.${label}.${FIELD_NAME}.txt"

    echo
    echo "======================================================================"
    echo "Dataset:      ${label}"
    echo "Parent dir:   ${parent_dir}"
    echo "Subjects dir: ${subjects_dir}"
    echo "Volume list:  ${volumes_file}"
    echo "======================================================================"

    # -------------------------------------------------------------------------
    # Validate directories
    # -------------------------------------------------------------------------

    if [[ ! -d "${parent_dir}" ]]; then
        echo "Error: parent directory does not exist:" >&2
        echo "  ${parent_dir}" >&2
        exit 1
    fi

    if [[ ! -d "${subjects_dir}" ]]; then
        echo "Error: FreeSurfer subjects directory does not exist:" >&2
        echo "  ${subjects_dir}" >&2
        exit 1
    fi

    # -------------------------------------------------------------------------
    # Build list of magnE_mean NIfTI files
    # -------------------------------------------------------------------------

    bash "${SCRIPT_BUILD_VOLUMES}" \
        --parent-dir "${parent_dir}" \
        --output "${volumes_file}"

    N_VOLUMES="$(wc -l < "${volumes_file}")"
    N_VOLUMES="${N_VOLUMES//[[:space:]]/}"

    if (( N_VOLUMES == 0 )); then
        echo "Error: volume list is empty: ${volumes_file}" >&2
        exit 1
    fi

    echo "Volumes found: ${N_VOLUMES}"

    # -------------------------------------------------------------------------
    # Submit one Slurm array task per NIfTI
    # -------------------------------------------------------------------------

ARRAY_SUBMIT="$(
    sbatch \
        --parsable \
        --job-name="segstats_${job_tag}" \
        --array="1-${N_VOLUMES}%${MAX_CONCURRENT}" \
        --output="${LOG_DIR}/segstats_${job_tag}_%A_%a.out" \
        --error="${LOG_DIR}/segstats_${job_tag}_%A_%a.err" \
        "${SCRIPT_WORKER}" \
        "${volumes_file}" \
        "${parent_dir}" \
        "${subjects_dir}"
)"

    # On some Slurm installations --parsable may return:
    #
    #   JOBID;CLUSTER
    #
    # Keep only JOBID for dependencies.
    ARRAY_JOB_ID="${ARRAY_SUBMIT%%;*}"

    echo "Submitted array job: ${ARRAY_JOB_ID}"

    # -------------------------------------------------------------------------
    # Build tables only after every array task succeeds
    # -------------------------------------------------------------------------

    TABLE_SUBMIT="$(
        sbatch \
            --parsable \
            --job-name="tables_${job_tag}" \
            --dependency="afterok:${ARRAY_JOB_ID}" \
            --output="${LOG_DIR}/tables_${job_tag}_%j.out" \
            --error="${LOG_DIR}/tables_${job_tag}_%j.err" \
            "${SCRIPT_TABLES}" \
            "${volumes_file}" \
            "${parent_dir}"
    )"

    TABLE_JOB_ID="${TABLE_SUBMIT%%;*}"

    echo "Submitted table job: ${TABLE_JOB_ID}"
    echo "  dependency: afterok:${ARRAY_JOB_ID}"
}


###############################################################################
# ORIGINAL DATASET
###############################################################################

ORG_RESULTS_DIR="/projects/b32903/Alex/results/STU00225089"
ORG_SUBJECTS_DIR="/projects/p32903/Alex2/datasets/STU00225089/subjectsdir"

run_stats \
    "org.skin_single" \
    "${ORG_RESULTS_DIR}/skin_single" \
    "${ORG_SUBJECTS_DIR}"

run_stats \
    "org.skin_double" \
    "${ORG_RESULTS_DIR}/skin_double" \
    "${ORG_SUBJECTS_DIR}"


###############################################################################
# SYNTHSR DATASET
###############################################################################

SYNTHSR_RESULTS_DIR="/projects/b32903/Alex/results/STU00225089_synthsr"
SYNTHSR_SUBJECTS_DIR="/projects/p32903/Alex2/datasets/STU00225089/synthsr/subjectsdir"

run_stats \
    "synthsr.skin_single" \
    "${SYNTHSR_RESULTS_DIR}/skin_single" \
    "${SYNTHSR_SUBJECTS_DIR}"

run_stats \
    "synthsr.skin_double" \
    "${SYNTHSR_RESULTS_DIR}/skin_double" \
    "${SYNTHSR_SUBJECTS_DIR}"


echo
echo "======================================================================"
echo "All jobs submitted."
echo "======================================================================"