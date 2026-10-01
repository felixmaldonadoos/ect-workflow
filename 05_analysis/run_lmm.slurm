#!/bin/bash
#SBATCH --account=p32903
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=lmm
#SBATCH --output=logs/lmm/%x_%j.out
#SBATCH --error=logs/lmm/%x_%j.err
#SBATCH --export=ALL

# Submit from 05_analysis, or use sbatch --chdir=/path/to/05_analysis.
# Create logs/lmm BEFORE submission; Slurm opens logs before this script runs.
# Example:
#   mkdir -p logs/lmm
#   sbatch --job-name=lmm_skin_single_static_sum run_lmm.slurm skin_single static
# Optional run_lmm.py arguments follow the two selectors, e.g. --prepare-only.
# AGGREGATION defaults to sum; use AGGREGATION=mean sbatch ... for means.

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit using run_lmm_submit.sh}"

if [[ ! "${TASK_ID}" =~ ^[0-7]$ ]]; then
    printf 'ERROR: Invalid array task ID: %s.\\n' "${TASK_ID}" >&2
    exit 2
fi

COMBINATIONS=(
    "skin_single static sum"
    "skin_single adaptive sum"
    "skin_double static sum"
    "skin_double adaptive sum"

)

# "skin_single static mean"
# "skin_single adaptive mean"
# "skin_double static mean"
# "skin_double adaptive mean"

read -r MODEL_TYPE SIMULATION_TYPE AGGREGATION <<< "${COMBINATIONS[TASK_ID]}"

# These selectors also determine the output directory; prevent mismatched labels.
for ARGUMENT in "$@"; do
    case "${ARGUMENT%%=*}" in
        --model-type|--simulation-type|--aggregation|--output-dir)
            printf 'ERROR: %s is set by this script; use the positional selectors, AGGREGATION, or OUTPUT_ROOT.\n' "${ARGUMENT}" >&2
            exit 2
            ;;
    esac
done

: "${SLURM_JOB_ID:?Submit this script using sbatch}"
# Slurm executes a spooled script copy: resolve the runner from the working directory.
ANALYSIS_DIR="${ANALYSIS_DIR:-$PWD}"
cd -- "${ANALYSIS_DIR}"
ANALYSIS_DIR="$PWD"
QUERY_FILE="${QUERY_FILE:-${ANALYSIS_DIR}/query_result.csv}"
CLINICAL_FILE="${CLINICAL_FILE:-/projects/p32903/Alex2/datasets/STU00225089/.clincal_notes/cgi_batch_agg_2_clean.xlsx}"
SUBJECT_SCAN_MAP="${SUBJECT_SCAN_MAP:-${ANALYSIS_DIR}/subject_scan_map.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ANALYSIS_DIR}/results/lmm}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_TYPE}/${SIMULATION_TYPE}/clinical_${AGGREGATION}/job_${SLURM_JOB_ID}"
LMM_CONDA_SH="${LMM_CONDA_SH:-$HOME/software/miniconda3/etc/profile.d/conda.sh}"
LMM_CONDA_ENV="${LMM_CONDA_ENV:-simnibs_env}"

for INPUT_FILE in "${ANALYSIS_DIR}/run_lmm.py" "${QUERY_FILE}" "${CLINICAL_FILE}" "${SUBJECT_SCAN_MAP}" "${LMM_CONDA_SH}"; do
    if [[ ! -f "${INPUT_FILE}" || ! -r "${INPUT_FILE}" ]]; then
        printf 'ERROR: Input file is missing or unreadable: %s\n' "${INPUT_FILE}" >&2
        exit 1
    fi
done

source "${LMM_CONDA_SH}"
conda activate "${LMM_CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"

printf 'Job: %s\nModel: %s\nSimulation: %s\nAggregation: %s\n' "${SLURM_JOB_ID}" "${MODEL_TYPE}" "${SIMULATION_TYPE}" "${AGGREGATION}"
printf 'Python: %s\nOutput: %s\n' "$(command -v python)" "${OUTPUT_DIR}"

exec srun --ntasks=1 python -u "${ANALYSIS_DIR}/run_lmm.py" \
    --sessions-xlsx "${CLINICAL_FILE}" \
    --session-sheet stimulus --outcome-sheet Sheet1 \
    --query-path "${QUERY_FILE}" \
    --model-type "${MODEL_TYPE}" --simulation-type "${SIMULATION_TYPE}" \
    --subject-scan-map "${SUBJECT_SCAN_MAP}" \
    --dose-column ef_sf --aggregation "${AGGREGATION}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
