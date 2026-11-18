#!/bin/bash
set -euo pipefail

SUBJECT_SCAN_MAP="${PROJECTS_DIR}/ect-workflow/05_analysis/subject_scan_map.json"

cd -- "${PROJECTS_DIR}/ect-workflow/05_analysis"

ARRAY_TASKS=0-3 bash run_pca_array_submit.sh \
    --subject-scan-map "$SUBJECT_SCAN_MAP" \
    --demean-by brain_mean parcel_p95_mean parcel_mean