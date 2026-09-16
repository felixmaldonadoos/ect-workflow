#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: bash build_subjvolumes_list.sh --parent-dir PARENT_DIR --output OUTPUT_FILE"
}

PARENT_DIR=""
OUTPUT_FILE=""

# Fixed for now.
FIELD_NAME="magnE_mean"
NIFTI_FILENAME="tdcs_uq_gpc_${FIELD_NAME}.nii.gz"


# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------

while (( $# > 0 )); do
    case "$1" in
        --parent-dir)
            if (( $# < 2 )); then
                echo "Error: --parent-dir requires a value." >&2
                usage >&2
                exit 2
            fi

            PARENT_DIR="$2"
            shift 2
            ;;

        --output)
            if (( $# < 2 )); then
                echo "Error: --output requires a value." >&2
                usage >&2
                exit 2
            fi

            OUTPUT_FILE="$2"
            shift 2
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        --*)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;

        *)
            echo "Error: unexpected positional argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done


# -----------------------------------------------------------------------------
# Validate
# -----------------------------------------------------------------------------

if [[ -z "${PARENT_DIR}" ]]; then
    echo "Error: --parent-dir is required." >&2
    usage >&2
    exit 2
fi

if [[ -z "${OUTPUT_FILE}" ]]; then
    echo "Error: --output is required." >&2
    usage >&2
    exit 2
fi

if [[ ! -d "${PARENT_DIR}" ]]; then
    echo "Error: parent directory does not exist:" >&2
    echo "  ${PARENT_DIR}" >&2
    exit 1
fi

PARENT_DIR="${PARENT_DIR%/}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"


# -----------------------------------------------------------------------------
# Discover volumes
# -----------------------------------------------------------------------------

echo "Searching:"
echo "  ${PARENT_DIR}"
echo
echo "For:"
echo "  */step_*/subject_volumes/${NIFTI_FILENAME}"
echo

find "${PARENT_DIR}" \
    -type f \
    -path "*/step_[0-9]*/subject_volumes/${NIFTI_FILENAME}" \
    -print \
    | sort -u \
    > "${OUTPUT_FILE}"


# -----------------------------------------------------------------------------
# Validate output
# -----------------------------------------------------------------------------

if [[ ! -s "${OUTPUT_FILE}" ]]; then
    echo "Error: no matching field volumes were found." >&2
    echo "Parent directory:" >&2
    echo "  ${PARENT_DIR}" >&2
    echo "Expected filename:" >&2
    echo "  ${NIFTI_FILENAME}" >&2
    exit 1
fi

N_VOLUMES="$(wc -l < "${OUTPUT_FILE}")"

echo "Saved ${N_VOLUMES} paths to:"
echo "  ${OUTPUT_FILE}"