#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "${SCRIPT_DIR}/00_submit_aparc_stats.sh"
    "${SCRIPT_DIR}/build_subjvolumes_list.sh"
    "${SCRIPT_DIR}/01_aparc+asegstats_worker.slurm"
    "${SCRIPT_DIR}/02_asegstats2table.slurm"
)

echo "Checking scripts..."

for script in "${SCRIPTS[@]}"; do
    if [[ ! -f "${script}" ]]; then
        echo "Error: missing script:" >&2
        echo "  ${script}" >&2
        exit 1
    fi

    echo "Syntax check: $(basename "${script}")"

    bash -n "${script}"
done

echo
echo "Setting executable permissions..."

chmod +x "${SCRIPTS[@]}"

echo
echo "Creating working directories..."

mkdir -p \
    "${SCRIPT_DIR}/logs" \
    "${SCRIPT_DIR}/lists"

echo
echo "Build successful."
echo
echo "Run pipeline with:"
echo "  bash ${SCRIPT_DIR}/00_submit_aparc_stats.sh"
echo
echo "Or with higher array concurrency:"
echo "  MAX_CONCURRENT=50 bash ${SCRIPT_DIR}/00_submit_aparc_stats.sh"