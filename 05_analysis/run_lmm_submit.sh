# run_lmm_submit.sh

#!/bin/bash
# Submit one LMM array: 2 models x 2 simulation types x 2 aggregations.
# Examples:
#   bash run_lmm_submit.sh
#   MAX_CONCURRENT=4 bash run_lmm_submit.sh --prepare-only
#   ARRAY_TASKS=0-3 bash run_lmm_submit.sh       # sum only
#   ARRAY_TASKS=4-7 bash run_lmm_submit.sh       # mean only
#   ARRAY_TASKS=2,6 bash run_lmm_submit.sh       # selected combinations

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARRAY_SCRIPT="${SCRIPT_DIR}/run_lmm.slurm"
ANALYSIS_DIR="$(cd -- "${ANALYSIS_DIR:-${SCRIPT_DIR}}" && pwd)"
export ANALYSIS_DIR
ARRAY_TASKS="${ARRAY_TASKS:-0-7}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
LOG_DIR="${LMM_LOG_DIR:-${ANALYSIS_DIR}/logs/lmm}"

if [[ ! "${ARRAY_TASKS}" =~ ^[0-7](-[0-7])?(,[0-7](-[0-7])?)*$ ]]; then
    printf 'ERROR: ARRAY_TASKS must select IDs 0-7, e.g. 0-7, 0-3, or 2,6.\n' >&2
    exit 2
fi
IFS=',' read -r -a TASK_RANGES <<< "${ARRAY_TASKS}"
for TASK_RANGE in "${TASK_RANGES[@]}"; do
    if (( ${TASK_RANGE%-*} > ${TASK_RANGE#*-} )); then
        printf 'ERROR: ARRAY_TASKS contains a reversed range: %s\n' "${TASK_RANGE}" >&2
        exit 2
    fi
done
if [[ ! "${MAX_CONCURRENT}" =~ ^[1-8]$ ]]; then
    printf 'ERROR: MAX_CONCURRENT must be an integer from 1 to 8.\n' >&2
    exit 2
fi

for INPUT_FILE in "${ARRAY_SCRIPT}" "${ANALYSIS_DIR}/run_lmm.py"; do
    if [[ ! -f "${INPUT_FILE}" || ! -r "${INPUT_FILE}" ]]; then
        printf 'ERROR: Required script is missing or unreadable: %s\n' "${INPUT_FILE}" >&2
        exit 1
    fi
done

# Slurm opens logs before the worker starts, so create the directory here.
mkdir -p -- "${LOG_DIR}"
LOG_DIR="$(cd -- "${LOG_DIR}" && pwd)"
printf 'Submitting LMM array %s (maximum %s concurrent tasks).\nLogs: %s\n' "${ARRAY_TASKS}" "${MAX_CONCURRENT}" "${LOG_DIR}"

exec sbatch \
    --chdir="${ANALYSIS_DIR}" \
    --array="${ARRAY_TASKS}%${MAX_CONCURRENT}" \
    --job-name=lmm \
    --output="${LOG_DIR}/%A_%a.out" \
    --error="${LOG_DIR}/%A_%a.err" \
    --export=ALL \
    "${ARRAY_SCRIPT}" "$@"
