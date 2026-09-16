#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(realpath -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"

CHARM_SETTINGS_SINGLE_DEFAULT="${CHARM_SETTINGS_SINGLE:-/projects/p32903/Alex2/analysis/03_meshing/settings_skin_single.ini}"
CHARM_SETTINGS_DOUBLE_DEFAULT="${CHARM_SETTINGS_DOUBLE:-/projects/p32903/Alex2/analysis/03_meshing/settings_skin_double.ini}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

usage() {
  cat <<EOF
Usage:
  /absolute/path/${SCRIPT_NAME} [options]

Required:
  --subjects-file PATH      CSV with header: subjid,additional_flags
  --dataset-dir PATH        Dataset root containing SUBJECT_ID/T1.nii.gz.
  --subjects-dir PATH       FreeSurfer SUBJECTS_DIR root.
  --output-dir PATH         Common m2m output root.

Options:
  --models MODE             single, double, or both (default: both).
  --t1-name NAME            T1 filename inside each subject directory (default: T1.nii.gz).
  --skin-label INT          Added thin-skin label for double models (default: 13).
  --thickness FLOAT         Thin-skin shell thickness in mm (default: 1.0).
  --settings-single PATH    CHARM settings for single-skin jobs.
  --settings-double PATH    CHARM settings for double-skin jobs.
  --worker-script PATH      One-subject Slurm worker
                            (default: ${SCRIPT_DIR}/charming_synthsr.sh).
  --log-dir PATH            Slurm log directory
                            (default: OUTPUT_DIR/logs/charm_synthsr).
  --dry-run                 Validate everything and print commands without creating
                            output directories or submitting jobs.
  -h, --help                Show this help.

Example:
  /absolute/path/${SCRIPT_NAME} \\
    --subjects-file /absolute/path/subjects.csv \\
    --dataset-dir /absolute/path/synthsr/mri/t1 \\
    --subjects-dir /absolute/path/synthsr/subjectsdir \\
    --output-dir /absolute/path/m2ms \\
    --models both
EOF
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "$value" ]] || die "$option requires a value"
}

parse_positive_float() {
  awk -v value="$1" 'BEGIN { exit !(value ~ /^([0-9]+([.][0-9]+)?|[.][0-9]+)$/ && value > 0) }'
}

canonical_file() {
  local path="$1"
  [[ -f "$path" ]] || die "File not found: $path"
  realpath -- "$path"
}

canonical_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Directory not found: $path"
  realpath -- "$path"
}

absolute_path_allow_missing() {
  realpath -m -- "$1"
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

SUBJECTS_FILE=""
DATASET_DIR=""
SUBJECTS_DIR=""
OUTPUT_DIR=""
MODELS="both"
T1_NAME="T1.nii.gz"
SKIN_LABEL=13
THICKNESS_MM=1.0
SKIN_LABEL_SET=false
THICKNESS_SET=false
SETTINGS_SINGLE="$CHARM_SETTINGS_SINGLE_DEFAULT"
SETTINGS_DOUBLE="$CHARM_SETTINGS_DOUBLE_DEFAULT"
WORKER_SCRIPT="$SCRIPT_DIR/charming_synthsr.sh"
LOG_DIR=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subjects-file)
      require_value "$1" "${2:-}"
      SUBJECTS_FILE="$2"
      shift 2
      ;;
    --subjects-file=*)
      SUBJECTS_FILE="${1#*=}"
      require_value "--subjects-file" "$SUBJECTS_FILE"
      shift
      ;;
    --dataset-dir)
      require_value "$1" "${2:-}"
      DATASET_DIR="$2"
      shift 2
      ;;
    --dataset-dir=*)
      DATASET_DIR="${1#*=}"
      require_value "--dataset-dir" "$DATASET_DIR"
      shift
      ;;
    --subjects-dir)
      require_value "$1" "${2:-}"
      SUBJECTS_DIR="$2"
      shift 2
      ;;
    --subjects-dir=*)
      SUBJECTS_DIR="${1#*=}"
      require_value "--subjects-dir" "$SUBJECTS_DIR"
      shift
      ;;
    --output-dir)
      require_value "$1" "${2:-}"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      require_value "--output-dir" "$OUTPUT_DIR"
      shift
      ;;
    --models)
      require_value "$1" "${2:-}"
      MODELS="$2"
      shift 2
      ;;
    --models=*)
      MODELS="${1#*=}"
      require_value "--models" "$MODELS"
      shift
      ;;
    --t1-name)
      require_value "$1" "${2:-}"
      T1_NAME="$2"
      shift 2
      ;;
    --t1-name=*)
      T1_NAME="${1#*=}"
      require_value "--t1-name" "$T1_NAME"
      shift
      ;;
    --skin-label)
      require_value "$1" "${2:-}"
      SKIN_LABEL="$2"
      SKIN_LABEL_SET=true
      shift 2
      ;;
    --skin-label=*)
      SKIN_LABEL="${1#*=}"
      require_value "--skin-label" "$SKIN_LABEL"
      SKIN_LABEL_SET=true
      shift
      ;;
    --thickness)
      require_value "$1" "${2:-}"
      THICKNESS_MM="$2"
      THICKNESS_SET=true
      shift 2
      ;;
    --thickness=*)
      THICKNESS_MM="${1#*=}"
      require_value "--thickness" "$THICKNESS_MM"
      THICKNESS_SET=true
      shift
      ;;
    --settings-single)
      require_value "$1" "${2:-}"
      SETTINGS_SINGLE="$2"
      shift 2
      ;;
    --settings-single=*)
      SETTINGS_SINGLE="${1#*=}"
      require_value "--settings-single" "$SETTINGS_SINGLE"
      shift
      ;;
    --settings-double)
      require_value "$1" "${2:-}"
      SETTINGS_DOUBLE="$2"
      shift 2
      ;;
    --settings-double=*)
      SETTINGS_DOUBLE="${1#*=}"
      require_value "--settings-double" "$SETTINGS_DOUBLE"
      shift
      ;;
    --worker-script)
      require_value "$1" "${2:-}"
      WORKER_SCRIPT="$2"
      shift 2
      ;;
    --worker-script=*)
      WORKER_SCRIPT="${1#*=}"
      require_value "--worker-script" "$WORKER_SCRIPT"
      shift
      ;;
    --log-dir)
      require_value "$1" "${2:-}"
      LOG_DIR="$2"
      shift 2
      ;;
    --log-dir=*)
      LOG_DIR="${1#*=}"
      require_value "--log-dir" "$LOG_DIR"
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1 (use --help)"
      ;;
  esac
done

[[ -n "$SUBJECTS_FILE" ]] || die "Missing required argument: --subjects-file"
[[ -n "$DATASET_DIR" ]] || die "Missing required argument: --dataset-dir"
[[ -n "$SUBJECTS_DIR" ]] || die "Missing required argument: --subjects-dir"
[[ -n "$OUTPUT_DIR" ]] || die "Missing required argument: --output-dir"

case "$MODELS" in
  single|skin_single)
    MODELS="single"
    ;;
  double|skin_double)
    MODELS="double"
    ;;
  both)
    ;;
  *)
    die "--models must be single, double, or both (got: $MODELS)"
    ;;
esac

if [[ "$MODELS" == "single" ]]; then
  [[ "$SKIN_LABEL_SET" == false ]] || die "--skin-label is only valid when submitting double-skin jobs"
  [[ "$THICKNESS_SET" == false ]] || die "--thickness is only valid when submitting double-skin jobs"
fi

[[ "$T1_NAME" != */* && "$T1_NAME" != '.' && "$T1_NAME" != '..' ]] || die "--t1-name must be a filename, not a path (got: $T1_NAME)"
[[ "$SKIN_LABEL" =~ ^[1-9][0-9]*$ ]] || die "--skin-label must be a positive integer (got: $SKIN_LABEL)"
parse_positive_float "$THICKNESS_MM" || die "--thickness must be greater than zero (got: $THICKNESS_MM)"

command -v realpath >/dev/null 2>&1 || die "Required command not found: realpath"
if [[ "$DRY_RUN" == false ]]; then
  command -v sbatch >/dev/null 2>&1 || die "Required command not found: sbatch"
fi

# All paths become absolute during preflight. Jobs never depend on the submitter's cwd.
SUBJECTS_FILE="$(canonical_file "$SUBJECTS_FILE")"
DATASET_DIR="$(canonical_dir "$DATASET_DIR")"
SUBJECTS_DIR="$(canonical_dir "$SUBJECTS_DIR")"
WORKER_SCRIPT="$(canonical_file "$WORKER_SCRIPT")"
OUTPUT_DIR="$(absolute_path_allow_missing "$OUTPUT_DIR")"
[[ -n "$LOG_DIR" ]] || LOG_DIR="$OUTPUT_DIR/logs/charm_synthsr"
LOG_DIR="$(absolute_path_allow_missing "$LOG_DIR")"

if [[ "$MODELS" == "single" || "$MODELS" == "both" ]]; then
  SETTINGS_SINGLE="$(canonical_file "$SETTINGS_SINGLE")"
fi
if [[ "$MODELS" == "double" || "$MODELS" == "both" ]]; then
  SETTINGS_DOUBLE="$(canonical_file "$SETTINGS_DOUBLE")"
fi

declare -a SUBJECT_IDS=()
declare -a T1_FILES=()
declare -a EXTRA_FLAGS=()
declare -A SEEN_SUBJECT_IDS=()

IFS= read -r header < "$SUBJECTS_FILE" || die "Could not read subjects file: $SUBJECTS_FILE"
header="${header%$'\r'}"
[[ "$header" == 'subjid,additional_flags' ]] || die "Unexpected CSV header in $SUBJECTS_FILE: expected 'subjid,additional_flags', got '$header'"

line_number=1
while IFS= read -r line || [[ -n "$line" ]]; do
  ((line_number += 1))
  line="${line%$'\r'}"
  [[ -n "$line" ]] || die "Blank row at $SUBJECTS_FILE:$line_number"
  [[ "$line" == *,* ]] || die "Malformed row at $SUBJECTS_FILE:$line_number: expected two comma-separated columns"

  subjid="${line%%,*}"
  additional_flags="${line#*,}"
  [[ "$additional_flags" != *,* ]] || die "Malformed row at $SUBJECTS_FILE:$line_number: more than two columns"
  [[ "$subjid" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "Invalid subject ID at $SUBJECTS_FILE:$line_number: '$subjid'"
  [[ -z "${SEEN_SUBJECT_IDS[$subjid]+x}" ]] || die "Duplicate subject ID at $SUBJECTS_FILE:$line_number: $subjid"
  SEEN_SUBJECT_IDS["$subjid"]=1

  t1_file="$(canonical_file "$DATASET_DIR/$subjid/$T1_NAME")"
  canonical_dir "$SUBJECTS_DIR/$subjid" >/dev/null

  SUBJECT_IDS+=("$subjid")
  T1_FILES+=("$t1_file")
  EXTRA_FLAGS+=("$additional_flags")
done < <(tail -n +2 -- "$SUBJECTS_FILE")

(( ${#SUBJECT_IDS[@]} > 0 )) || die "Subjects file contains no data rows: $SUBJECTS_FILE"

if [[ "$DRY_RUN" == false ]]; then
  mkdir -p -- "$OUTPUT_DIR" "$LOG_DIR"
  OUTPUT_DIR="$(canonical_dir "$OUTPUT_DIR")"
  LOG_DIR="$(canonical_dir "$LOG_DIR")"
fi

log "Preflight passed for ${#SUBJECT_IDS[@]} subjects"
log "Models       : $MODELS"
log "Output root  : $OUTPUT_DIR"
log "Log directory: $LOG_DIR"
[[ "$DRY_RUN" == false ]] || log "Dry run: no directories created and no jobs submitted"

submitted_count=0
declare -a JOB_IDS=()

submit_job() {
  local index="$1"
  local model="$2"
  local subjid="${SUBJECT_IDS[$index]}"
  local t1_file="${T1_FILES[$index]}"
  local additional_flags="${EXTRA_FLAGS[$index]}"
  local job_name settings

  case "$model" in
    skin_single)
      job_name="charm1-${subjid}"
      settings="$SETTINGS_SINGLE"
      ;;
    skin_double)
      job_name="charm2-${subjid}"
      settings="$SETTINGS_DOUBLE"
      ;;
    *)
      die "Internal error: unsupported model $model"
      ;;
  esac

  local -a command=(
    sbatch
    --parsable
    --job-name "$job_name"
    --output "$LOG_DIR/%x_%j.out"
    --error "$LOG_DIR/%x_%j.err"
    "$WORKER_SCRIPT"
    --subject-id "$subjid"
    --subject-t1-nifti "$t1_file"
    --subjects-dir "$SUBJECTS_DIR"
    --output-dir "$OUTPUT_DIR"
    --model "$model"
    --settings "$settings"
  )

  if [[ "$model" == "skin_double" ]]; then
    command+=(--skin-label "$SKIN_LABEL" --thickness "$THICKNESS_MM")
  fi
  if [[ -n "$additional_flags" ]]; then
    command+=(--charm-flags "$additional_flags")
  fi

  print_command "${command[@]}"
  if [[ "$DRY_RUN" == false ]]; then
    local job_id
    job_id="$("${command[@]}")"
    JOB_IDS+=("$job_id")
    log "Submitted $job_name as $job_id"
  fi
  ((submitted_count += 1))
}

for index in "${!SUBJECT_IDS[@]}"; do
  if [[ "$MODELS" == "double" || "$MODELS" == "both" ]]; then
    submit_job "$index" skin_double
  fi
  if [[ "$MODELS" == "single" || "$MODELS" == "both" ]]; then
    submit_job "$index" skin_single
  fi
done

if [[ "$DRY_RUN" == true ]]; then
  log "Dry run complete: validated and printed $submitted_count jobs"
else
  log "Submission complete: submitted $submitted_count jobs"
  printf 'Job IDs:'
  printf ' %s' "${JOB_IDS[@]}"
  printf '\n'
fi
