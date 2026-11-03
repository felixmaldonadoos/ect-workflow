# Series mixed effects from ROI and PCA masters

`run_series_mixed_effects.py` reads one existing long-format master CSV and
fits one ROI or PCA component at a time:

```r
cgi_change ~ 1 + E + E:ef_sf + (1 | age) + (1 | sex)
```

Every model has three fixed coefficients: the intercept, `E`, and `E:ef_sf`.
Age and sex provide crossed random intercepts, using the saved exact age values
and sex categories. Treatment series remain separate observations, including
multiple series from one person. Models are separated by field, normalization,
`model_type`, `step`, feature name, and PCA basis.

## Master values and normalization

| Master | Feature name | Value used as `E` |
| --- | --- | --- |
| Yeo2011 ROI long master | Complete `roi_org`, or legacy `roi_name` | Saved `mean` |
| PCA long master | `component`, interpreted within its `basis_id` | Saved `score` |

For brain-normalized ROI masters, the saved `mean` is already divided by the
whole-brain mean. In brain-normalized PCA masters, normalization occurred before
PCA fitting. The runner uses these saved values directly. Negative PC scores
retain their signs. Normalization metadata takes precedence over the legacy ROI
filename convention.

The saved `cgi_change` is used unchanged; current masters define it as
`cgi_start - cgi_end`, so a positive value represents a CGI reduction. The
interaction multiplies each row's `E` by that same row's saved `ef_sf`, the legacy
stimulation index from the representative session. No centering or scaling is
applied. The conditional slope for `E` is
`beta_E + beta_interaction * ef_sf`.

## Environment and inputs

Use Python with NumPy, pandas, SciPy, and Statsmodels. The repository's LMM
dependency contract is `statsmodels>=0.14.6,<0.15`. The runner uses CSV data and
does not require imaging or mesh extraction libraries.

Set `RESULTS_DIR` to an absolute directory outside `05_analysis`. All generated
files, including Slurm logs, go under `$RESULTS_DIR/<analysis_name>/`.

```bash
source /home/vum3263/software/miniconda3/etc/profile.d/conda.sh
conda activate simnibs_env
cd /projects/p32903/Alex2/ect-workflow/05_analysis
export RESULTS_DIR=/projects/p32903/Alex2/results
export PYTHONDONTWRITEBYTECODE=1
```

Current inputs are:

```text
networks/data/yeo2011_magnE_mean_series_ROI_master.csv
networks/data/yeo2011_magnE_mean_series_ROI_master_normed_brain_magnE_mean_mean.csv
$RESULTS_DIR/pca/data/pca_magnE_mean_raw_series_PC_master.csv
$RESULTS_DIR/pca/data/pca_magnE_mean_normed_brain_magnE_mean_mean_series_PC_master.csv
```

The runner accepts read-only input paths anywhere. The legacy ROI input
locations above remain usable.

## Prepare and run

Prepare-only mode exports the retained long input, run provenance, and one
design-validation status per planned feature/configuration. It checks the
fixed-design rank and age/sex grouping without fitting the model.

```bash
python -B run_series_mixed_effects.py \
  --master-path networks/data/yeo2011_magnE_mean_series_ROI_master.csv \
  --feature-type roi --field-name magnE_mean \
  --analysis-name series_lmm_roi_raw_prepare --prepare-only

python -B run_series_mixed_effects.py \
  --master-path networks/data/yeo2011_magnE_mean_series_ROI_master_normed_brain_magnE_mean_mean.csv \
  --feature-type roi --field-name magnE_mean \
  --analysis-name series_lmm_roi_brain_mean_prepare --prepare-only

python -B run_series_mixed_effects.py \
  --master-path "$RESULTS_DIR/pca/data/pca_magnE_mean_raw_series_PC_master.csv" \
  --feature-type pca --field-name magnE_mean \
  --analysis-name series_lmm_pca_raw_prepare --prepare-only

python -B run_series_mixed_effects.py \
  --master-path "$RESULTS_DIR/pca/data/pca_magnE_mean_normed_brain_magnE_mean_mean_series_PC_master.csv" \
  --feature-type pca --field-name magnE_mean \
  --analysis-name series_lmm_pca_brain_mean_prepare --prepare-only
```

For actual fits, omit `--prepare-only` and choose a fresh analysis name:

```bash
python -B run_series_mixed_effects.py \
  --master-path "$RESULTS_DIR/pca/data/pca_magnE_mean_raw_series_PC_master.csv" \
  --feature-type pca --field-name magnE_mean \
  --analysis-name series_lmm_pca_raw
```

PCA defaults to components whose saved `selected_for_correlation` flag is true.
Use `--pca-scope all` for all exported components, or `--features PC1 PC2` for
explicit component names. These names are selected separately within each
basis; component numbers from different bases are never pooled. For an ROI
subset, supply complete labels, for example `--features 7Networks_LH_Vis`.

Fits default to ML with Powell and `--maxiter 2000`. Use `--reml` for REML,
`--optimizer powell|lbfgs|bfgs|cg`, or a different positive `--maxiter` to make an
explicit change. These settings are saved in the run manifest. REML AIC/BIC are
unavailable and are exported as missing values.

## Results tables and source identity

Each master receives its own set of files. The filename prefix is
`<safe_source_stem>__<raw|brain_mean>__<hash12>`: the source master's stem
(sanitized and limited to 110 characters), an explicit normalization label, and
the first 12 characters of its SHA-256 hash. Thus a raw master and a
brain-normalized master have separate result files, even when they share an
analysis directory. Existing artifacts for the same master are protected from
overwrite; choose a fresh analysis name for another selection, fit setting, or
prepare/full phase.

```text
$RESULTS_DIR/<analysis_name>/
  analysis_<prefix>.json
  <prefix>_lmm_results.csv
  <prefix>_coefficients.csv
  <prefix>_fit_status.csv
  data/<prefix>_model_input_long.csv
  logs/                                      # Slurm submissions
```

`lmm_results.csv` is the main review table: one row per planned feature and
configuration, including failed fits. It includes the feature identity
(`feature_id`, `roi_name` or `pc_name`, and PCA `basis_id`), `field_name`,
`model_type`, `step`, normalization labels and `is_brain_normalized`,
`master_file` (the source basename), `source_master_path`,
`source_master_sha256`, sample counts, status, diagnostics, and the three
coefficients with their inference fields. These columns state both which ROI/PC
supplied `E` and whether its master was brain-normalized.

`coefficients.csv` holds one row per fixed-effect term for successful fits.
`fit_status.csv` records every planned fit, its sample/group counts,
convergence, warnings, errors, age/sex/residual variances, boundary flags,
log likelihood, and available AIC/BIC. The prepared input retains the master
metadata and adds the chosen feature value `E` and interaction `E_x_ef_sf`.

P-values and 95% confidence intervals use asymptotic normal Wald inference,
without a small-sample or multiplicity correction. There is no new outcome
association step for selecting PCs. Plots, serialized models, predictions,
network averages, joint feature models, and permutation inference are outside
this runner's initial scope.

## Validation and fit failures

Missing/duplicate columns, duplicate observations, conflicting metadata,
ambiguous feature identities, and nonfinite required values cause explicit
validation errors. Observations are never silently dropped or imputed.

Per-feature validation requires a finite full-rank three-column fixed design,
more than three observations, at least two age and sex levels, and some repeated
age values. A constant `ef_sf` makes the interaction collinear with `E`.
All-unique ages confound the requested age variance with residual variance.

The existing masters contain many small age groups and only two sex levels.
Weak variance estimates, boundary estimates, singular Hessians, and
nonconvergence can occur. The runner records boundary estimates separately and
publishes inference only for fits that pass its numerical checks. A failing
feature does not discard successful features. The process returns success
(exit code 0) after exporting every planned fit, including when individual
models do not converge or fail inference checks. Prepare-only status is `prepared` for a
valid design; it does not establish numerical convergence of a future fit.

A Slurm job's success status indicates that processing and exports completed.
Inspect `converged`, `status`, `warnings`, and `error` in the main results table
or source-specific `fit_status.csv` to assess each model. The stdout summary separates optimizer
convergence from fits accepted for Wald inference and counts each failure
reason. A converged fit with invalid coefficient covariance or observed
information remains `failed`; its p-values and confidence intervals are
withheld. A near-zero variance estimate alone is recorded as a boundary and is
not itself a rejection criterion. Increasing Slurm memory or time does not
resolve an uncertainty calculation that fails after completed optimization.
Invalid command-line arguments, unreadable or invalid masters, and output-write
errors still produce a nonzero process exit code. Runs made before this exit
policy change returned code 1 when any feature failed, even after complete exports.

## Slurm submission

Submit all four masters with one command. Each master runs as a separate job,
with separate analysis directories for ROI/PCA and raw/brain-normalized inputs:

```bash
export RESULTS_DIR=/projects/p32903/Alex2/results

# Check all inputs and print the Slurm commands without submitting or writing logs.
bash run_series_mixed_effects_all_submit.sh series_lmm_v1 --dry-run

# Submit the four jobs.
bash run_series_mixed_effects_all_submit.sh series_lmm_v1
```

The analysis names are `series_lmm_v1_roi_raw`,
`series_lmm_v1_roi_brain_mean`, `series_lmm_v1_pca_raw`, and
`series_lmm_v1_pca_brain_mean`. Each job fits every selected feature separately
within each model/step configuration. Shared options such as `--prepare-only`,
`--reml`, `--optimizer lbfgs`, and `--maxiter 2000` are forwarded to every job.
All four input paths and CLI configurations are checked before the first job
is submitted. A scheduler rejection during submission stops the remaining
submissions; earlier accepted jobs remain submitted.

The defaults use the four master locations listed above. Override an input
with `SERIES_ROI_RAW_MASTER`, `SERIES_ROI_BRAIN_MASTER`,
`SERIES_PCA_RAW_MASTER`, or `SERIES_PCA_BRAIN_MASTER`. These paths can point
outside the output root. Use a fresh analysis prefix for subsequent runs to
keep existing results.

To submit one master or select particular ROIs/PCs, use the per-master wrapper:

```bash
bash run_series_mixed_effects_submit.sh series_lmm_pca_raw \
  --master-path "$RESULTS_DIR/pca/data/pca_magnE_mean_raw_series_PC_master.csv" \
  --feature-type pca --field-name magnE_mean
```

Add `--dry-run` to this command to validate and print its submission command.
Relative master paths resolve against the directory where the wrapper is
invoked. The wrapper validates runner arguments, the source master, the results
root, and the analysis name before creating logs or submitting.
It writes them directly to `<analysis_name>/logs/<job_id>.out` and `.err`.
The Slurm worker is `run_series_mixed_effects.slurm`; its defaults are account
`p32903`, partition `short`, one CPU, 5 GB memory, and one hour. It uses the
`simnibs_env` Conda environment. Override
`SERIES_CONDA_ENV` or `SERIES_CONDA_SH`, or set `SERIES_PYTHON` to an absolute
interpreter path. `SERIES_VALIDATE_PYTHON` selects the standard-library Python
used for path validation before submission. `SERIES_SCRIPT_ROOT` is exported by
the wrapper for the job to locate this source tree.

## Verification

The targeted tests include adapter validation, per-feature result accounting,
output-path failures, optimizer/covariance failures, and an actual synthetic
crossed model fitted through both the core and the CSV CLI:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m pytest -q -p no:cacheprovider \
  tests/test_series_models_adapters.py \
  tests/test_series_models_fitting.py \
  tests/test_series_models_workflow.py
```

On 2026-11-02, 36 tests and 54 subtests passed with Statsmodels 0.14.6.
Prepare-only exports were also saved under the four `*_prepare` analysis names
shown above. All designs passed: 204 planned ROI fits per master, 88 selected
PC fits for raw PCA, and 92 selected PC fits for brain-normalized PCA. Every fit
had 35 series, 28 exact age levels, two sex levels, and fixed-design rank three.
These tables have status `prepared` and contain no fitted clinical coefficients.
Those analysis names are now occupied; use fresh names to repeat preparation.
Prepare-only verification does not establish convergence or validity of
clinical-model inference.

Slurm launchers also pass 10 tests and 20 subtests using mocked `sbatch` and
`srun`, including input preflight, paths containing spaces, dry runs, separate
master dispatch, and preservation of worker failure exit codes:

```bash
python -B -m pytest -q -p no:cacheprovider \
  tests/test_series_models_slurm.py tests/test_series_models_batch_submit.py
```
