#!/usr/bin/env bash
# Submit the shared age/sex null, then ROI/PCA x raw/brain-normalized masters.
# Usage: bash run_series_mixed_effects_all_submit.sh analysis_prefix [shared runner options...]
set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: bash run_series_mixed_effects_all_submit.sh analysis_prefix [--dry-run] [shared runner options...]" >&2
    exit 2
fi
if [[ "$1" == --help || "$1" == -h ]]; then
    echo "Usage: bash run_series_mixed_effects_all_submit.sh analysis_prefix [--dry-run] [shared runner options...]"
    echo "Submits null, roi_raw, roi_brain_mean, pca_raw, and pca_brain_mean jobs."
    echo "Outputs: RESULTS_DIR/lmm/<analysis_prefix>/{null,roi_raw,roi_brain_mean,pca_raw,pca_brain_mean}/"
    echo "Shared options: --sex-effect random|fixed, --prepare-only, --field-name, --reml, --optimizer, --maxiter, --no-fit-plots, --plot-dpi."
    exit 0
fi

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1

exec "${SERIES_VALIDATE_PYTHON:-python3}" -B - "$SCRIPT_ROOT" "$@" <<'PY'
import os
from pathlib import Path
import shutil
import subprocess
import sys

script_root = Path(sys.argv[1])
analysis_prefix = sys.argv[2]
forwarded = sys.argv[3:]
sys.path.insert(0, str(script_root))

from analysis_paths import lmm_analysis_name, lmm_run_dir, results_root
from run_series_mixed_effects import parse_args
from series_models.run_layout import ensure_run_manifest, run_configuration

managed_options = {"--analysis-name", "--output-component", "--feature-type", "--master-path", "--features", "--pca-scope", "--null-model"}
for argument in forwarded:
    if argument.split("=", 1)[0] in managed_options:
        raise SystemExit(
            f"{argument.split('=', 1)[0]} is managed by the all-master launcher; "
            "use run_series_mixed_effects_submit.sh for an individual master or feature subset."
        )
dry_run = "--dry-run" in forwarded
runner_options = [argument for argument in forwarded if argument != "--dry-run"]

try:
    analysis_prefix = lmm_analysis_name(analysis_prefix)
    if len(analysis_prefix.split("/")) != 2:
        raise ValueError("Pass a run name, without a component, to the all-master launcher")
    run_root = lmm_run_dir(analysis_prefix)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"Run directory already exists; choose a new analysis prefix: {run_root}")
    output_root = results_root()
except (ValueError, OSError) as error:
    raise SystemExit(str(error)) from error

wrapper = script_root / "run_series_mixed_effects_submit.sh"
if not wrapper.is_file():
    raise SystemExit(f"Submission wrapper not found: {wrapper}")
cases = [
    ("roi_raw", "roi", "SERIES_ROI_RAW_MASTER",
     script_root / "networks/data/yeo2011_magnE_mean_series_ROI_master.csv"),
    ("roi_brain_mean", "roi", "SERIES_ROI_BRAIN_MASTER",
     script_root / "networks/data/yeo2011_magnE_mean_series_ROI_master_normed_brain_magnE_mean_mean.csv"),
    ("pca_raw", "pca", "SERIES_PCA_RAW_MASTER",
     output_root / "pca/data/pca_magnE_mean_raw_series_PC_master.csv"),
    ("pca_brain_mean", "pca", "SERIES_PCA_BRAIN_MASTER",
     output_root / "pca/data/pca_magnE_mean_normed_brain_magnE_mean_mean_series_PC_master.csv"),
]
commands = []
masters = []
null_source = None
parsed = None
# Check every input and every CLI configuration before the first submission.
for suffix, kind, environment_name, default_master in cases:
    master = Path(os.environ.get(environment_name, str(default_master))).expanduser().resolve()
    if not master.is_file() or not os.access(master, os.R_OK):
        raise SystemExit(f"Master is missing or unreadable ({environment_name}): {master}")
    name = analysis_prefix
    arguments = ["--output-component", suffix, "--master-path", str(master), "--feature-type", kind, *runner_options]
    parsed = parse_args(["--analysis-name", name, *arguments])
    commands.append(["bash", str(wrapper), name, *arguments, *(["--dry-run"] if dry_run else [])])
    masters.append(str(master))
    if suffix == "roi_raw":
        null_source = master

# The null contains no field features or configuration effects. Use one validated
# source for its saved clinical covariates; AIC comparison checks exact cohort
# agreement before reusing this baseline for other masters/configurations.
null_name = analysis_prefix
null_arguments = ["--null-model", "--output-component", "null", "--master-path", str(null_source), "--feature-type", "roi", *runner_options]
parse_args(["--analysis-name", null_name, *null_arguments])
commands.insert(0, ["bash", str(wrapper), null_name, *null_arguments, *(["--dry-run"] if dry_run else [])])

if not dry_run and shutil.which("sbatch") is None:
    raise SystemExit("sbatch is unavailable; run on the cluster or use --dry-run.")
configuration = run_configuration(parsed.sex_effect, {"reml": parsed.reml, "optimizer": parsed.optimizer,
                                  "maxiter": parsed.maxiter}, parsed.field_name, prepare_only=parsed.prepare_only)
ensure_run_manifest(analysis_prefix, configuration, inputs=masters, create=not dry_run)
for command in commands:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
print("Validated five Slurm commands; no jobs submitted." if dry_run else "Submitted one null and four feature-model jobs.")
PY
