#!/usr/bin/env bash
#SBATCH --account=p32903
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --job-name=charm-synthsr
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# These environment variables can override cluster-specific defaults.
CHARM_SETTINGS_SINGLE_DEFAULT="${CHARM_SETTINGS_SINGLE:-/projects/p32903/Alex2/analysis/03_meshing/settings_skin_single.ini}"
CHARM_SETTINGS_DOUBLE_DEFAULT="${CHARM_SETTINGS_DOUBLE:-/projects/p32903/Alex2/analysis/03_meshing/settings_skin_double.ini}"
FREESURFER_MODULE="${FREESURFER_MODULE:-freesurfer/8.1}"
FSL_MODULE="${FSL_MODULE:-fsl/6.0.7.8}"
SIMNIBS_CONDA_ENV="${SIMNIBS_CONDA_ENV:-simnibs_env}"
CONDA_INIT_SCRIPT="${CONDA_INIT_SCRIPT:-${HOME}/software/miniconda3/etc/profile.d/conda.sh}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

usage() {
  cat <<EOF
Usage:
  sbatch [SBATCH_OPTIONS] /absolute/path/${SCRIPT_NAME} [options]

Required:
  --subject-id TEXT         Subject ID, e.g. subj-cat-030-001.
  --subject-t1-nifti PATH   Subject T1 NIfTI.
  --subjects-dir PATH       FreeSurfer SUBJECTS_DIR root.
  --output-dir PATH         Common output root. Results are written beneath
                            PATH/skin_single or PATH/skin_double.
  --model MODEL             skin_single or skin_double (single/double aliases accepted).

Options:
  --settings PATH           CHARM settings INI. If omitted, the model-specific
                            CHARM_SETTINGS_SINGLE or CHARM_SETTINGS_DOUBLE value is used.
  --skin-label INT          Added thin-skin label for skin_double (default: 13).
  --thickness FLOAT         Thin-skin shell thickness in mm for skin_double (default: 1.0).
  --charm-flags "FLAGS"     Whitespace-separated extra CHARM flags. No shell evaluation occurs.
  -h, --help                Show this help.

Output layout:
  OUTPUT_DIR/MODEL/m2m_SUBJECT_ID

Example:
  mkdir -p /absolute/path/logs
  sbatch \\
    --output=/absolute/path/logs/%x_%j.out \\
    --error=/absolute/path/logs/%x_%j.err \\
    /absolute/path/${SCRIPT_NAME} \\
    --subject-id subj-cat-030-001 \\
    --subject-t1-nifti /absolute/path/subj-cat-030-001/T1.nii.gz \\
    --subjects-dir /absolute/path/subjectsdir \\
    --output-dir /absolute/path/m2ms \\
    --model skin_double \\
    --skin-label 13 \\
    --thickness 1
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

create_canonical_dir() {
  local path="$1"
  mkdir -p -- "$path" || die "Could not create directory: $path"
  realpath -- "$path"
}

SUBJECT_ID=""
SUBJECT_T1_NIFTI=""
SUBJECTS_DIR=""
OUTPUT_DIR=""
MODEL=""
CHARM_SETTINGS_PATH=""
SKIN_LABEL=""
THICKNESS_MM=""
CHARM_EXTRA_FLAGS_STR=""
declare -a CHARM_EXTRA_FLAGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject-id)
      require_value "$1" "${2:-}"
      SUBJECT_ID="$2"
      shift 2
      ;;
    --subject-id=*)
      SUBJECT_ID="${1#*=}"
      require_value "--subject-id" "$SUBJECT_ID"
      shift
      ;;
    --subject-t1-nifti)
      require_value "$1" "${2:-}"
      SUBJECT_T1_NIFTI="$2"
      shift 2
      ;;
    --subject-t1-nifti=*)
      SUBJECT_T1_NIFTI="${1#*=}"
      require_value "--subject-t1-nifti" "$SUBJECT_T1_NIFTI"
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
    --model)
      require_value "$1" "${2:-}"
      MODEL="$2"
      shift 2
      ;;
    --model=*)
      MODEL="${1#*=}"
      require_value "--model" "$MODEL"
      shift
      ;;
    --settings)
      require_value "$1" "${2:-}"
      CHARM_SETTINGS_PATH="$2"
      shift 2
      ;;
    --settings=*)
      CHARM_SETTINGS_PATH="${1#*=}"
      require_value "--settings" "$CHARM_SETTINGS_PATH"
      shift
      ;;
    --skin-label)
      require_value "$1" "${2:-}"
      SKIN_LABEL="$2"
      shift 2
      ;;
    --skin-label=*)
      SKIN_LABEL="${1#*=}"
      require_value "--skin-label" "$SKIN_LABEL"
      shift
      ;;
    --thickness)
      require_value "$1" "${2:-}"
      THICKNESS_MM="$2"
      shift 2
      ;;
    --thickness=*)
      THICKNESS_MM="${1#*=}"
      require_value "--thickness" "$THICKNESS_MM"
      shift
      ;;
    --charm-flags)
      require_value "$1" "${2:-}"
      CHARM_EXTRA_FLAGS_STR="$2"
      shift 2
      ;;
    --charm-flags=*)
      CHARM_EXTRA_FLAGS_STR="${1#*=}"
      require_value "--charm-flags" "$CHARM_EXTRA_FLAGS_STR"
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

[[ -n "$SUBJECT_ID" ]] || die "Missing required argument: --subject-id"
[[ "$SUBJECT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "Invalid subject ID: $SUBJECT_ID"
[[ -n "$SUBJECT_T1_NIFTI" ]] || die "Missing required argument: --subject-t1-nifti"
[[ -n "$SUBJECTS_DIR" ]] || die "Missing required argument: --subjects-dir"
[[ -n "$OUTPUT_DIR" ]] || die "Missing required argument: --output-dir"
[[ -n "$MODEL" ]] || die "Missing required argument: --model"

case "$MODEL" in
  single|skin_single)
    MODEL="skin_single"
    ;;
  double|skin_double)
    MODEL="skin_double"
    ;;
  *)
    die "--model must be skin_single or skin_double (got: $MODEL)"
    ;;
esac

if [[ "$MODEL" == "skin_single" ]]; then
  [[ -z "$SKIN_LABEL" ]] || die "--skin-label is only valid with --model skin_double"
  [[ -z "$THICKNESS_MM" ]] || die "--thickness is only valid with --model skin_double"
  [[ -n "$CHARM_SETTINGS_PATH" ]] || CHARM_SETTINGS_PATH="$CHARM_SETTINGS_SINGLE_DEFAULT"
else
  SKIN_LABEL="${SKIN_LABEL:-13}"
  THICKNESS_MM="${THICKNESS_MM:-1.0}"
  [[ "$SKIN_LABEL" =~ ^[1-9][0-9]*$ ]] || die "--skin-label must be a positive integer (got: $SKIN_LABEL)"
  parse_positive_float "$THICKNESS_MM" || die "--thickness must be greater than zero (got: $THICKNESS_MM)"
  [[ -n "$CHARM_SETTINGS_PATH" ]] || CHARM_SETTINGS_PATH="$CHARM_SETTINGS_DOUBLE_DEFAULT"
fi

command -v realpath >/dev/null 2>&1 || die "Required command not found: realpath"

# Resolve every path before changing the working directory.
SUBJECT_T1_NIFTI="$(canonical_file "$SUBJECT_T1_NIFTI")"
SUBJECTS_DIR="$(canonical_dir "$SUBJECTS_DIR")"
CHARM_SETTINGS_PATH="$(canonical_file "$CHARM_SETTINGS_PATH")"
SUBJECT_FS_DIR="$(canonical_dir "$SUBJECTS_DIR/$SUBJECT_ID")"
OUTPUT_DIR="$(create_canonical_dir "$OUTPUT_DIR")"
OUT_MODEL_DIR="$(create_canonical_dir "$OUTPUT_DIR/$MODEL")"
FINAL_M2M_DIR="$OUT_MODEL_DIR/m2m_${SUBJECT_ID}"

if [[ -n "$CHARM_EXTRA_FLAGS_STR" ]]; then
  read -r -a CHARM_EXTRA_FLAGS <<< "$CHARM_EXTRA_FLAGS_STR"
fi

# Hold the lock until the worker exits so two jobs cannot modify the same m2m directory.
command -v flock >/dev/null 2>&1 || die "Required command not found: flock"
LOCK_FILE="$OUT_MODEL_DIR/.m2m_${SUBJECT_ID}.lock"
exec {LOCK_FD}> "$LOCK_FILE"
flock -n "$LOCK_FD" || die "Another job is already writing to: $FINAL_M2M_DIR"

# An old marker must not make a failed rerun appear complete.
rm -f -- "$FINAL_M2M_DIR/.done"

type module >/dev/null 2>&1 || die "Environment Modules is not available"
module load "$FREESURFER_MODULE"

CONDA_INIT_SCRIPT="$(canonical_file "$CONDA_INIT_SCRIPT")"
set +u
# shellcheck source=/dev/null
source "$CONDA_INIT_SCRIPT"
conda activate "$SIMNIBS_CONDA_ENV"
set -u

module load "$FSL_MODULE"
[[ -n "${FSLDIR:-}" ]] || die "FSLDIR is not set after loading $FSL_MODULE"
FSL_CONFIG="$(canonical_file "$FSLDIR/etc/fslconf/fsl.sh")"
set +u
# shellcheck source=/dev/null
source "$FSL_CONFIG"
set -u
export FSLOUTTYPE=NIFTI_GZ

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-1}"

required_commands=(charm)
if [[ "$MODEL" == "skin_double" ]]; then
  required_commands+=(fslinfo fslmaths fslcpgeom add_tissues_to_upsampled)
fi
for command_name in "${required_commands[@]}"; do
  command -v "$command_name" >/dev/null 2>&1 || die "Required command not found after environment setup: $command_name"
done

create_thin_skin_mask() {
  local m2m_subject_dir="$1"
  [[ -d "$m2m_subject_dir" ]] || die "Directory not found: $m2m_subject_dir"

  local label_prep_dir="$m2m_subject_dir/label_prep"
  local tissue_labels="$label_prep_dir/tissue_labeling_upsampled.nii.gz"
  local tissue_labels_copy="$label_prep_dir/tissue_labeling_upsampled_cp.nii.gz"
  local skin_mask="$label_prep_dir/tissue_labeling_upsampled_skin_mask.nii.gz"
  local masking_dir="$label_prep_dir/masking"

  [[ -f "$tissue_labels" ]] || die "Required tissue-label file not found: $tissue_labels"
  mkdir -p -- "$masking_dir"
  cp -f -- "$tissue_labels" "$tissue_labels_copy"

  local voxel_x voxel_y voxel_z
  voxel_x="$(fslinfo "$tissue_labels" | awk '/pixdim1/{print $2}')"
  voxel_y="$(fslinfo "$tissue_labels" | awk '/pixdim2/{print $2}')"
  voxel_z="$(fslinfo "$tissue_labels" | awk '/pixdim3/{print $2}')"
  [[ -n "$voxel_x" && -n "$voxel_y" && -n "$voxel_z" ]] || die "Could not read voxel sizes from: $tissue_labels"

  local tolerance=0.0001
  if ! awk -v x="$voxel_x" -v y="$voxel_y" -v z="$voxel_z" -v tol="$tolerance" '
    function abs(value) { return value < 0 ? -value : value }
    BEGIN { exit !(abs(x-y) <= tol && abs(x-z) <= tol && abs(y-z) <= tol) }
  '; then
    die "Voxel sizes are anisotropic: pixdim1=$voxel_x, pixdim2=$voxel_y, pixdim3=$voxel_z"
  fi

  local scalp_label=5
  local scalp_bin="$masking_dir/scalp_bin.nii.gz"
  local scalp_dil="$masking_dir/scalp_dil.nii.gz"
  local shell_bin="$masking_dir/shell_bin.nii.gz"
  local shell_label="$masking_dir/shell_label${SKIN_LABEL}.nii.gz"
  local zeros_mask="$masking_dir/zeros_mask.nii.gz"
  local shell_placeable="$masking_dir/shell_placeable.nii.gz"

  log "Creating thin-skin mask: thickness=${THICKNESS_MM} mm, scalp_label=$scalp_label, new_label=$SKIN_LABEL"
  fslmaths "$tissue_labels_copy" -thr "$scalp_label" -uthr "$scalp_label" -bin "$scalp_bin"
  fslmaths "$scalp_bin" -kernel sphere "$THICKNESS_MM" -dilM "$scalp_dil"
  fslmaths "$scalp_dil" -sub "$scalp_bin" -thr 0 -bin "$shell_bin"
  fslmaths "$shell_bin" -mul "$SKIN_LABEL" "$shell_label"
  fslmaths "$tissue_labels_copy" -bin -binv "$zeros_mask"
  fslmaths "$shell_label" -mas "$zeros_mask" "$shell_placeable" -odt int
  cp -f -- "$shell_placeable" "$skin_mask"
  fslcpgeom "$tissue_labels_copy" "$skin_mask" -d
  [[ -f "$skin_mask" ]] || die "Thin-skin mask was not created: $skin_mask"

  add_tissues_to_upsampled -i "$skin_mask" -t "$tissue_labels" -o "$tissue_labels" --offset 0
  log "Thin-skin mask created and inserted"
}

log "Subject ID       : $SUBJECT_ID"
log "Input T1        : $SUBJECT_T1_NIFTI"
log "FreeSurfer dir  : $SUBJECT_FS_DIR"
log "Model           : $MODEL"
log "Settings        : $CHARM_SETTINGS_PATH"
log "Final output    : $FINAL_M2M_DIR"
log "Conda env       : $SIMNIBS_CONDA_ENV"
if [[ "$MODEL" == "skin_double" ]]; then
  log "Thin-skin label : $SKIN_LABEL"
  log "Thickness       : $THICKNESS_MM mm"
fi
if (( ${#CHARM_EXTRA_FLAGS[@]} )); then
  log "Extra flags     : ${CHARM_EXTRA_FLAGS[*]}"
else
  log "Extra flags     : <none>"
fi

pushd "$OUT_MODEL_DIR" >/dev/null

CHARM_CMD=(
  charm
  "$SUBJECT_ID"
  "$SUBJECT_T1_NIFTI"
  --fs-dir "$SUBJECT_FS_DIR"
  --usesettings "$CHARM_SETTINGS_PATH"
  --forcerun
)

if (( ${#CHARM_EXTRA_FLAGS[@]} )); then
  CHARM_CMD+=("${CHARM_EXTRA_FLAGS[@]}")
fi

if [[ "$MODEL" == "skin_double" ]]; then
  CHARM_CMD+=(--initatlas --segment --surfaces)
  log "Running initial skin_double segmentation and surface stages"
else
  log "Running standard skin_single CHARM pipeline"
fi

printf '[%s] Running:' "$(date '+%F %T')" >&2
printf ' %q' "${CHARM_CMD[@]}" >&2
printf '\n' >&2
"${CHARM_CMD[@]}"

[[ -d "$FINAL_M2M_DIR" ]] || die "Expected CHARM output directory not found: $FINAL_M2M_DIR"

if [[ "$MODEL" == "skin_double" ]]; then
  create_thin_skin_mask "$FINAL_M2M_DIR"
  [[ -f "$FINAL_M2M_DIR/settings.ini" ]] || die "Generated settings file not found: $FINAL_M2M_DIR/settings.ini"
  log "Re-running the CHARM mesh stage with the modified tissue labels"
  charm "$SUBJECT_ID" --mesh --usesettings "$FINAL_M2M_DIR/settings.ini"
fi

popd >/dev/null
touch -- "$FINAL_M2M_DIR/.done"
log "Completed: $FINAL_M2M_DIR"
