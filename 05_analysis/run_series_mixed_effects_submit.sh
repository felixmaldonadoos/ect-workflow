#!/usr/bin/env bash
# Usage: bash run_series_mixed_effects_submit.sh analysis_name [--dry-run] [runner options...]
set -euo pipefail

DRY_RUN=false
INPUT_ARGUMENTS=()
for argument in "$@"; do
    if [[ "$argument" == --dry-run ]]; then
        DRY_RUN=true
    else
        INPUT_ARGUMENTS+=("$argument")
    fi
done
set -- "${INPUT_ARGUMENTS[@]}"

if (( $# )) && [[ "$1" == --help || "$1" == -h ]]; then
    echo "Usage: bash run_series_mixed_effects_submit.sh analysis_name [--dry-run] [runner options...]"
    exit 0
fi
if (( $# == 0 )) || [[ "$1" == -* ]]; then
    echo "Usage: bash run_series_mixed_effects_submit.sh analysis_name [--dry-run] [runner options...]" >&2
    exit 2
fi

ANALYSIS_NAME="$1"
shift
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_ROOT/run_series_mixed_effects.slurm"
VALIDATE_PYTHON="${SERIES_VALIDATE_PYTHON:-python3}"
RUNNER_ARGUMENTS=()

for argument in "$@"; do
    case "$argument" in
        --analysis-name|--analysis-name=*)
            echo "Pass analysis_name as the first positional argument, not --analysis-name." >&2
            exit 2
            ;;
        *)
            RUNNER_ARGUMENTS+=("$argument")
            ;;
    esac
done

for argument in "${RUNNER_ARGUMENTS[@]}"; do
    if [[ "$argument" == --help || "$argument" == -h ]]; then
        exec "$VALIDATE_PYTHON" -B "$SCRIPT_ROOT/run_series_mixed_effects.py" --help
    fi
done

for required_file in "$SLURM_SCRIPT" "$SCRIPT_ROOT/run_series_mixed_effects.py"; do
    if [[ ! -r "$required_file" ]]; then
        echo "Required file not found or unreadable: $required_file" >&2
        exit 2
    fi
done

export RESULTS_DIR
export SERIES_SCRIPT_ROOT="$SCRIPT_ROOT"
export PYTHONDONTWRITEBYTECODE=1

# Parse the same options as the runner before creating logs or submitting. This
# import uses only the standard library. Resolve the master against the caller's
# directory, since sbatch subsequently changes the job's working directory.
VALIDATION_JSON="$("$VALIDATE_PYTHON" -B - "$SCRIPT_ROOT" --analysis-name "$ANALYSIS_NAME" "${RUNNER_ARGUMENTS[@]}" <<'PY'
import json
import sys

sys.path.insert(0, sys.argv[1])
from analysis_paths import resolve_analysis_dir, validate_output_path
from run_series_mixed_effects import parse_args

args = parse_args(sys.argv[2:])
try:
    master = args.master_path.expanduser().resolve(strict=True)
    if not master.is_file():
        raise ValueError(f"Master path is not a file: {master}")
    with master.open("rb"):
        pass
    directory = validate_output_path(
        resolve_analysis_dir(args.analysis_name) / "logs", args.analysis_name
    )
    validate_output_path(directory / "%j.out", args.analysis_name)
    validate_output_path(directory / "%j.err", args.analysis_name)
except (OSError, ValueError) as error:
    raise SystemExit(str(error)) from error
print(json.dumps([str(master), str(directory)]))
PY
)"

# JSON followed by NUL-delimited strings preserves spaces and embedded newlines
# without evaluating shell text or splitting paths on whitespace.
mapfile -d '' -t VALIDATED_PATHS < <(
    "$VALIDATE_PYTHON" -B -c 'import json, sys; sys.stdout.write("\0".join(json.loads(sys.argv[1])) + "\0")' "$VALIDATION_JSON"
)
if (( ${#VALIDATED_PATHS[@]} != 2 )); then
    echo "Could not decode validated submission paths." >&2
    exit 2
fi
MASTER_PATH="${VALIDATED_PATHS[0]}"
LOG_DIR="${VALIDATED_PATHS[1]}"

# Retain CLI argument order and both argparse spellings while replacing master
# values with the validated absolute path. parse_args has already checked that
# each option taking a separate value actually has one.
FORWARDED_ARGUMENTS=()
index=0
while (( index < ${#RUNNER_ARGUMENTS[@]} )); do
    argument="${RUNNER_ARGUMENTS[index]}"
    case "$argument" in
        --master-path)
            FORWARDED_ARGUMENTS+=(--master-path "$MASTER_PATH")
            index=$((index + 2))
            ;;
        --master-path=*)
            FORWARDED_ARGUMENTS+=("--master-path=$MASTER_PATH")
            index=$((index + 1))
            ;;
        *)
            FORWARDED_ARGUMENTS+=("$argument")
            index=$((index + 1))
            ;;
    esac
done

echo "Analysis name: $ANALYSIS_NAME"
echo "Log directory: $LOG_DIR"
echo "Source master: $MASTER_PATH"

SBATCH_COMMAND=(
    sbatch
    --export=ALL
    --chdir="$SCRIPT_ROOT"
    --job-name="$ANALYSIS_NAME"
    --output="$LOG_DIR/%x_%j.out"
    --error="$LOG_DIR/%x_%j.err"
    "$SLURM_SCRIPT" --analysis-name "$ANALYSIS_NAME" "${FORWARDED_ARGUMENTS[@]}"
)

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run; no job submitted."
    printf '%q ' "${SBATCH_COMMAND[@]}"
    printf '\n'
    exit 0
fi

mkdir -p -- "$LOG_DIR"
exec "${SBATCH_COMMAND[@]}"
