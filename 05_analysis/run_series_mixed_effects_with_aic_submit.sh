#!/usr/bin/env bash
# Separate workflow runner: shared null + four masters, then dependent AIC.
# The existing run_series_mixed_effects_all_submit.sh is left intact.
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1

exec "${SERIES_VALIDATE_PYTHON:-python3}" -B - "$SCRIPT_ROOT" "$@" <<'PY'
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

script_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script_root))

from analysis_paths import lmm_analysis_name, lmm_run_dir, results_root, validate_output_path
from run_series_mixed_effects import parse_args as parse_model_args
from run_series_aic_comparison import parse_args as parse_aic_args
from series_models.run_layout import ensure_run_manifest, run_configuration


def main():
    parser = argparse.ArgumentParser(
        description="Submit the null and four series-model jobs, followed by AIC after all five exit successfully.",
        epilog="Other options are forwarded to all model jobs, e.g. --sex-effect fixed, --optimizer, --maxiter, --no-fit-plots.",
        allow_abbrev=False,
    )
    parser.add_argument("run_name", help="Fresh run name under RESULTS_DIR/lmm/.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print all six submissions without writes or jobs.")
    parser.add_argument("--aic-no-plots", action="store_true", help="Skip AIC figures; model-fit plots are controlled separately.")
    parser.add_argument("--aic-top-n", type=int, default=20)
    parser.add_argument("--aic-dpi", type=int, default=800)
    options, forwarded = parser.parse_known_args(sys.argv[2:])
    if options.aic_top_n < 1 or options.aic_dpi < 1:
        parser.error("--aic-top-n and --aic-dpi must be positive")
    managed = {"--analysis-name", "--output-component", "--feature-type", "--master-path",
               "--features", "--pca-scope", "--null-model"}
    for argument in forwarded:
        name = argument.split("=", 1)[0]
        if name in managed:
            parser.error(f"{name} is managed by this all-model workflow")
        if name in {"--reml", "--prepare-only"}:
            parser.error(f"{name} cannot be used here: AIC requires completed ML fits; use the original runner for that option")

    namespace = lmm_analysis_name(options.run_name)
    if len(namespace.split("/")) != 2:
        parser.error("Pass a run name without a component")
    run_name = namespace.split("/")[1]
    run_root = lmm_run_dir(namespace)
    if run_root.exists():
        raise FileExistsError(f"Run directory already exists; choose a fresh run name: {run_root}")
    output_root = results_root()
    wrapper = script_root / "run_series_mixed_effects_submit.sh"
    aic_runner = script_root / "run_series_aic_comparison.py"
    for path in (wrapper, script_root / "run_series_mixed_effects.slurm", aic_runner):
        if not path.is_file() or not os.access(path, os.R_OK):
            raise ValueError(f"Required runner is missing or unreadable: {path}")

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
    models, masters, parsed = [], [], None
    for component, kind, environment_name, default in cases:
        master = Path(os.environ.get(environment_name, str(default))).expanduser().resolve()
        if not master.is_file():
            raise ValueError(f"Master is missing ({environment_name}): {master}")
        with master.open("rb"):
            pass
        arguments = ["--output-component", component, "--master-path", str(master),
                     "--feature-type", kind, *forwarded]
        parsed = parse_model_args(["--analysis-name", namespace, *arguments])
        models.append((component, ["bash", str(wrapper), namespace, *arguments]))
        masters.append(str(master))
    null_arguments = ["--null-model", "--output-component", "null", "--master-path", masters[0],
                      "--feature-type", "roi", *forwarded]
    parse_model_args(["--analysis-name", namespace, *null_arguments])
    models.insert(0, ("null", ["bash", str(wrapper), namespace, *null_arguments]))

    aic_arguments = ["--analysis-name", namespace, "--null-analysis-dir", str(run_root / "null"),
                     "--feature-analysis-dirs", *(str(run_root / component) for component, *_ in cases),
                     "--top-n", str(options.aic_top_n), "--dpi", str(options.aic_dpi)]
    if options.aic_no_plots:
        aic_arguments.append("--no-plots")
    parse_aic_args(aic_arguments)
    aic_namespace = lmm_analysis_name(namespace, "aic")
    log_dir = validate_output_path(run_root / "aic/logs", aic_namespace)
    receipt_path = validate_output_path(run_root / "submission_with_aic.json", namespace)
    configuration = run_configuration(parsed.sex_effect, {"reml": parsed.reml,
        "optimizer": parsed.optimizer, "maxiter": parsed.maxiter}, parsed.field_name)

    # Reuse the model worker's interpreter/Conda contract. All generated command
    # arguments are quoted with shlex, rather than interpolated as shell code.
    aic_script = r'''#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${SERIES_PYTHON:-}" ]]; then
    ANALYSIS_PYTHON="$SERIES_PYTHON"
else
    source "${SERIES_CONDA_SH:-$HOME/software/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${SERIES_CONDA_ENV:-simnibs_env}"
    ANALYSIS_PYTHON=python
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
'''
    aic_script += 'exec srun "$ANALYSIS_PYTHON" -B -u ' + shlex.join([str(aic_runner), *aic_arguments]) + "\n"

    resources = {}
    for name, default in (("ACCOUNT", "p32903"), ("PARTITION", "short"),
                          ("MEMORY", "3G"), ("TIME", "00:05:00")):
        resources[name] = os.environ.get(f"SERIES_AIC_{name}", default).strip()
        if not resources[name]:
            raise ValueError(f"SERIES_AIC_{name} must not be empty")

    def aic_command(dependency):
        return ["sbatch", "--parsable", "--export=ALL", f"--chdir={script_root}",
                f"--account={resources['ACCOUNT']}",
                f"--partition={resources['PARTITION']}",
                "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
                f"--mem={resources['MEMORY']}",
                f"--time={resources['TIME']}",
                f"--dependency={dependency}", "--kill-on-invalid-dep=yes",
                f"--job-name={run_name}_aic", f"--output={log_dir}/%x_%j.out", f"--error={log_dir}/%x_%j.err"]

    if not options.dry_run and shutil.which("sbatch") is None:
        raise ValueError("sbatch is unavailable; use --dry-run or submit on the cluster")

    job_ids, submitted = [], []
    receipt = {"schema_version": 1, "analysis_type": "series_models_with_dependent_aic",
               "run_name": run_name, "run_dir": str(run_root),
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "model_jobs": submitted, "aic_arguments": aic_arguments,
               "aic_batch_script": aic_script,
               "submitter_sha256": hashlib.sha256(
                   (script_root / "run_series_mixed_effects_with_aic_submit.sh").read_bytes()).hexdigest()}

    def save_receipt(status, **details):
        receipt.update(status=status, **details)
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")

    if not options.dry_run:
        # Exclusive run creation prevents two invocations from queuing the same
        # workflow concurrently. No reservation is made during --dry-run.
        run_root.parent.mkdir(parents=True, exist_ok=True)
        run_root.mkdir()
        ensure_run_manifest(namespace, configuration, inputs=masters)

    for model_index, (component, command) in enumerate(models, 1):
        completed = subprocess.run([*command, *(["--dry-run"] if options.dry_run else [])],
                                   capture_output=True, text=True)
        print(completed.stdout, end="", flush=True)
        print(completed.stderr, end="", file=sys.stderr, flush=True)
        ids = re.findall(r"(?m)^Submitted batch job ([0-9]+)[ \t]*$", completed.stdout)
        if not options.dry_run and len(ids) == 1 and ids[0] not in job_ids:
            job_ids.append(ids[0])
            submitted.append({"component": component, "job_id": ids[0], "command": command})
        if completed.returncode != 0:
            if not options.dry_run:
                save_receipt("model_submission_failed", failed_component=component,
                             returncode=completed.returncode)
            print("Model submission stopped; no AIC job submitted. Earlier accepted model jobs remain queued.", file=sys.stderr)
            return completed.returncode if completed.returncode > 0 else 128 - completed.returncode
        if not options.dry_run and (len(ids) != 1 or len(submitted) != model_index):
            save_receipt("model_job_id_unconfirmed", failed_component=component, submission_stdout=completed.stdout)
            raise ValueError("Could not confirm a unique model job ID; no AIC job submitted. Inspect earlier submissions before retrying.")

    if options.dry_run:
        dependency = "afterok:<null-id>:<roi-raw-id>:<roi-brain-id>:<pca-raw-id>:<pca-brain-id>"
        print("AIC submission after all five model jobs (batch script supplied through stdin):")
        print(shlex.join(aic_command(dependency)))
        print(aic_script, end="")
        print("Validated five model submissions and the dependent AIC submission; no files written or jobs submitted.")
        return 0

    if len(job_ids) != 5:
        save_receipt("model_job_id_unconfirmed", confirmed_job_ids=job_ids)
        raise ValueError("Expected exactly five model job IDs; no AIC job submitted")
    dependency = "afterok:" + ":".join(job_ids)
    command = aic_command(dependency)
    receipt.update(dependency=dependency, aic_submission_command=command)
    log_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, input=aic_script, capture_output=True, text=True)
    print(completed.stdout, end="", flush=True)
    print(completed.stderr, end="", file=sys.stderr, flush=True)
    if completed.returncode != 0:
        save_receipt("aic_submission_failed", returncode=completed.returncode,
                     submission_stdout=completed.stdout, submission_stderr=completed.stderr)
        print("The five model jobs remain queued; only the dependent AIC submission failed.", file=sys.stderr)
        return completed.returncode if completed.returncode > 0 else 128 - completed.returncode
    match = re.fullmatch(r"([0-9]+)(?:;[^\s;]+)?", completed.stdout.strip())
    if not match:
        save_receipt("aic_submitted_id_unconfirmed", submission_stdout=completed.stdout)
        raise ValueError("AIC submission succeeded but its job ID is unconfirmed; inspect Slurm before retrying")
    save_receipt("submitted", aic_job_id=match.group(1))
    print(f"AIC job {match.group(1)} waits for model jobs {':'.join(job_ids)}.")
    print(f"Submission receipt: {receipt_path}")
    return 0


try:
    raise SystemExit(main())
except (OSError, ValueError) as error:
    print(f"Series model/AIC submission: {error}", file=sys.stderr)
    raise SystemExit(2)
PY
