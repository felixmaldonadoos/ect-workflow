# Series mixed effects from ROI and PCA masters

`run_series_mixed_effects.py` reads one existing long-format master CSV and
fits one ROI or PCA component at a time:

```r
cgi_change ~ 1 + E + E:ef_sf + age + (1 | sex)
```

This is the default `--sex-effect random` specification. It has four fixed
coefficients: the intercept, `E`, `E:ef_sf`, and continuous `age` in years.
Sex provides a random intercept for each category. Choose `--sex-effect fixed`
for the alternative:

```r
cgi_change ~ 1 + E + E:ef_sf + age + C(sex)
```

Fixed sex uses Gaussian OLS because no random effects remain. The first
lexicographically sorted observed sex label is the treatment-reference category;
each other category gets a fixed intercept contrast. `sex_reference`,
`sex_levels`, `sex_effect`, and `sex_effect_choice` record the coding. Age is
neither categorized nor used as a random effect in either mode.
Treatment series remain separate observations, including
multiple series from one person. Models are separated by field, normalization,
`model_type`, `step`, feature name, and PCA basis.

This specification replaces the original age/sex crossed-random-intercept
model on 2026-11-03. The package version is `0.2.0`, the output schema version
is 2, and `model_id` is `continuous_age_sex_random_intercept_v1`. Previous
results remain results of their original model. Fixed-sex IDs are
`continuous_age_sex_fixed_v1` and `continuous_age_sex_fixed_null_v1`.
Use fresh analysis names for a changed specification; the runner protects
existing files. Historical flat result directories remain readable and are
not moved by this change.

## Grouped runs and overwrite protection

Keep `RESULTS_DIR` as the shared results root. The all-master launcher now
creates one LMM run folder:

```text
$RESULTS_DIR/lmm/<run_name>/
  run.json
  null/
  roi_raw/
  roi_brain_mean/
  pca_raw/
  pca_brain_mean/
  aic/
  plots/                  # optional export from saved runs
```

Every master child contains its own result tables, source manifest, `data/`,
`plots/`, and Slurm `logs/`. Automatic fit plots stay in that child's `plots/`.
The parent `run.json` records both formulas, sex mode, model IDs, variance
components, fit backend/settings, input-master plan, and output layout. It is
created atomically and existing scientific settings must match. The all-master
launcher refuses a nonempty existing run directory before submitting anything.
Choose another run name for a repeat, changed model, or prepare/full phase.

For example, these are separate runs:

```bash
bash run_series_mixed_effects_all_submit.sh age_continuous_sex_random_v1 --sex-effect random
bash run_series_mixed_effects_all_submit.sh age_continuous_sex_fixed_v1 --sex-effect fixed
```

These commands are usage examples; they are not executed automatically by
editing the scripts. For an individual master, `--analysis-name RUN` writes
under `$RESULTS_DIR/lmm/RUN/`; add `--output-component roi_raw` (or the matching
other source label) for a child folder. Null models automatically use `null/`.
Explicit `lmm/RUN/COMPONENT` namespaces are also accepted. A parent created by
a single-master run allows only its recorded input; use the all-master
launcher to plan all four inputs under a common run from the outset.

PCA input defaults remain `$RESULTS_DIR/pca/data/`. Output grouping does not
change `RESULTS_DIR` or redirect input lookup into the new LMM run folder.

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

Use Python with NumPy, pandas, SciPy, Statsmodels, and Matplotlib. The repository's LMM
dependency contract is `statsmodels>=0.14.6,<0.15`. The runner uses CSV data and
does not require imaging or mesh extraction libraries.

Set `RESULTS_DIR` to an absolute directory outside `05_analysis`. All generated
files, including Slurm logs, go under `$RESULTS_DIR/lmm/<run_name>/`.

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
fixed-design rank and the selected sex encoding without fitting the model.

```bash
python -B run_series_mixed_effects.py \
  --master-path networks/data/yeo2011_magnE_mean_series_ROI_master.csv \
  --feature-type roi --field-name magnE_mean \
  --analysis-name series_lmm_roi_raw_age_continuous_prepare --prepare-only

python -B run_series_mixed_effects.py \
  --master-path networks/data/yeo2011_magnE_mean_series_ROI_master_normed_brain_magnE_mean_mean.csv \
  --feature-type roi --field-name magnE_mean \
  --analysis-name series_lmm_roi_brain_mean_age_continuous_prepare --prepare-only

python -B run_series_mixed_effects.py \
  --master-path "$RESULTS_DIR/pca/data/pca_magnE_mean_raw_series_PC_master.csv" \
  --feature-type pca --field-name magnE_mean \
  --analysis-name series_lmm_pca_raw_age_continuous_prepare --prepare-only

python -B run_series_mixed_effects.py \
  --master-path "$RESULTS_DIR/pca/data/pca_magnE_mean_normed_brain_magnE_mean_mean_series_PC_master.csv" \
  --feature-type pca --field-name magnE_mean \
  --analysis-name series_lmm_pca_brain_mean_age_continuous_prepare --prepare-only
```

For actual fits, omit `--prepare-only` and choose a fresh analysis name:

```bash
python -B run_series_mixed_effects.py \
  --master-path "$RESULTS_DIR/pca/data/pca_magnE_mean_raw_series_PC_master.csv" \
  --feature-type pca --field-name magnE_mean \
  --analysis-name series_lmm_pca_raw_age_continuous
```

PCA defaults to components whose saved `selected_for_correlation` flag is true.
Use `--pca-scope all` for all exported components, or `--features PC1 PC2` for
explicit component names. These names are selected separately within each
basis; component numbers from different bases are never pooled. For an ROI
subset, supply complete labels, for example `--features 7Networks_LH_Vis`.

Random-sex fits default to ML with Powell and `--maxiter 2000`. Use `--reml` for REML,
`--optimizer powell|lbfgs|bfgs|cg`, or a different positive `--maxiter` to make an
explicit change. These settings are saved in the run manifest. REML AIC/BIC are
unavailable and are exported as missing values.
Fixed-sex fits use an analytic SVD OLS solution and ML residual variance
`RSS/n`; `fit_algorithm=analytic_ols` records the actual algorithm. Iterative
optimizer options are retained as requested settings but do not control OLS.
`--sex-effect fixed --reml` is rejected. Fixed-sex uncertainty follows the
existing normal Wald convention using the ML covariance, rather than OLS
Student-t inference. AIC/BIC explicitly count residual variance as a parameter.

## Shared null and AIC comparison

The null removes both feature terms while preserving continuous age and the
same sex random intercept:

```r
cgi_change ~ 1 + age + (1 | sex)
```

For fixed sex the matching null is `cgi_change ~ 1 + age + C(sex)`, fitted
without random effects. The null and feature models in a run must use the same
sex mode. For two sex categories the nominal parameter counts remain four for
the null and six for the feature model; with additional fixed-sex categories
the counts include their additional contrasts.

It has four nominal parameters (two fixed coefficients, sex variance, residual
variance); each feature model has six. Run it once from a saved master:

```bash
python -B run_series_mixed_effects.py --null-model \
  --master-path networks/data/yeo2011_magnE_mean_series_ROI_master.csv \
  --feature-type roi --analysis-name lmm_age_continuous_null
```

The null exports `analysis.json`, `data/null_model_input.csv`,
`fit_status.csv`, `coefficients.csv`, and `null_results.csv`. The input has only
`person_id`, `series_num`, `cgi_change`, `age`, and `sex`; repeated copies across
features/configurations collapse only after their clinical values agree.
`--features` does not apply to the null. Existing files are protected.

After the feature jobs finish, compare their saved results without refitting:

```bash
python -B run_series_aic_comparison.py \
  --null-analysis-dir "$RESULTS_DIR/lmm/age_continuous_sex_fixed_v1/null" \
  --feature-analysis-dirs \
    "$RESULTS_DIR/lmm/age_continuous_sex_fixed_v1/pca_brain_mean" \
    "$RESULTS_DIR/lmm/age_continuous_sex_fixed_v1/pca_raw" \
    "$RESULTS_DIR/lmm/age_continuous_sex_fixed_v1/roi_brain_mean" \
    "$RESULTS_DIR/lmm/age_continuous_sex_fixed_v1/roi_raw" \
  --analysis-name age_continuous_sex_fixed_v1
```

This comparison writes to the same run's `aic/` child. Existing legacy flat
analysis directories may still be supplied as read-only inputs. Comparisons
that mix fixed-sex and random-sex models are rejected.

The comparator requires identical person/series keys, outcomes, ages and sex
labels; the current fixed/random specifications; and matching ML fit settings.
It checks every planned feature/configuration against its saved input and
status. REML fits, mismatched cohorts/specifications, duplicate identities, or
incomplete exports cannot silently enter a ranking. It reads the saved CSVs
and manifests, without rereading images or recomputing PCA.

`aic_eligible` requires convergence, finite likelihood/AIC, valid recorded
variances, the expected AIC parameter count, and a feature likelihood no lower
than the nested null within numerical tolerance. It is separate from `status`,
which governs coefficient Wald inference. A converged model can therefore be
eligible for this exploratory AIC comparison while its Wald estimates remain
withheld. Boundary estimates and failed Wald checks appear in
`aic_provisional`/`aic_caveat`; an eligible boundary null also makes comparisons
against it provisional. These checks do not establish a global optimum or
remove uncertainty from estimating a sex variance with two categories.

The output directory contains:

```text
analysis.json
aic_comparison.csv       # every candidate plus the shared null
aic_ranked.csv           # eligible rows, increasing AIC, including null
aic_excluded.csv         # exclusions with explicit reasons
aic_summary.csv          # configuration-level counts and best feature
plots/plot_manifest.csv
plots/aic_ranked_features.{png,pdf}
plots/aic_heatmap_*.{png,pdf}
logs/matplotlib_cache/
```

Lower AIC is better. `aic_improvement_over_null = AIC_null - AIC_feature`, so
positive values favor adding the feature terms. `delta_aic_best_including_null`
uses the best supplied candidate **including the null** as its reference;
`delta_aic_best_feature` uses only eligible feature models. Global and
configuration-specific ranks preserve ROI/PC, normalization, model/step, and
PCA-basis identity. Components with the same number in different bases remain
different features.

The figures show improvement over null: a ranking of the top 20 feature models
with the null at zero, plus heatmaps by feature type and normalization.
Provisional fits are marked and exclusions are gray. Use `--top-n`, `--dpi`, or
`--no-plots` to control figure output. Plotting additionally needs Matplotlib.
This is an exploratory comparison of associations in the observed series;
AIC rankings are neither significance tests nor validated prediction scores.

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
$RESULTS_DIR/lmm/<run_name>/<source_component>/
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
`source_master_sha256`, `model_id`, `fixed_formula`, `age_effect`, `sex_effect`,
sample counts, status, diagnostics, and the numeric fixed coefficients plus
any fixed-sex contrasts with their inference fields. Fixed-sex wide columns
use names such as `sex[T.Male]_estimate`; the long coefficients table retains
the exact Patsy term. These columns state both which ROI/PC
supplied `E` and whether its master was brain-normalized.
Age inference is exported as `age_estimate`, `age_std_error`, `age_wald_z`,
`age_p_value`, `age_ci_low`, and `age_ci_high`; `age_estimate` is the estimated
CGI-change difference per additional year, holding the other predictors fixed.

`coefficients.csv` holds one row per fixed-effect term for successful fits.
`fit_status.csv` records every planned fit, its sample/group counts,
convergence, warnings, errors, sex/residual variances, boundary flags,
log likelihood, and available AIC/BIC. The prepared input retains the master
metadata and adds the chosen feature value `E` and interaction `E_x_ef_sf`.
There is no age variance estimate. `n_age_values` and `age_value_counts` are
descriptive summaries of the continuous ages, not model grouping levels.

P-values and 95% confidence intervals use asymptotic normal Wald inference,
without a small-sample or multiplicity correction. There is no new outcome
association step for selecting PCs. Serialized models, network averages, joint
feature models, and permutation inference are outside this runner's scope.

## Automatic fitted-model plots

Full LMM runs now automatically export a PNG per fitted ROI/PC and one PDF
report per source master, `model_type`, and `step`, with a page per feature.
The null produces one shared-cohort report with age on its x-axis.
Prepare-only runs do not generate fitted-model plots. Use `--no-fit-plots` to
disable these exports, or `--plot-dpi 180` to change PNG resolution (default 120).
The existing Slurm wrappers forward these options; plotting is enabled by
default in all five jobs.

Each feature plot shows the saved ROI mean or PC score against CGI change
adjusted to the fit's mean age and median `ef_sf`. Points represent treatment
series and retain the sex-specific intercept offsets. Colored curves show
sex-specific conditional means; the dashed black curve sets the sex random
effect to zero in random-sex mode. For fixed sex, colored curves include the
estimated categorical contrasts and the black curve uses the reference sex;
there is no BLUP shrinkage. Both modes have a common feature slope at fixed
`ef_sf`, so the colored curves are parallel. Curves span
each sex group's observed predictor range. No random slopes are added.

Adjustment subtracts `beta_age * (age - reference_age)` and
`beta_E:ef_sf * E * (ef_sf - reference_ef_sf)` from each observed change. The
adjusted plot values can be fractional or outside the raw outcome range;
the saved original outcome is unchanged. Null plots use unadjusted change
against age. Titles and captions identify the ROI/PC, PCA basis, normalization,
configuration, reference values, sample count, AIC, and fit status.

Point predictions use stable GLS and sex-intercept conditional means at the
saved variance estimates. The reconstructed ML/REML log likelihood must agree
with the fitted value within `1e-6`. This calculation does not repeat model
optimization, imaging extraction, or PCA fitting, and does not use a rejected
Wald covariance matrix. Boundary and failed-inference fits remain marked
provisional. The plots contain point estimates without confidence bands.
Nonconverged or numerically unreconstructable fits receive placeholder pages
and explicit errors in the plot index, allowing the other plots to complete.
Fixed-sex plotting reconstructs the OLS means using the saved ML residual
variance, with no random-effect boundary or few-random-level caveat.

Additional artifacts are:

```text
plots/model_fits/<safe_source_id>/<model_type>/<step>/models.pdf
plots/model_fits/<safe_source_id>/<model_type>/<step>/<feature>__<identity_hash>.png
plots/model_fits/<source_id>_manifest.csv
plots/model_fits/<source_id>_analysis.json
data/fit_plots/<source_id>_observations.csv
data/fit_plots/<source_id>_curves.csv
data/fit_plots/<source_id>_parameters.csv
logs/matplotlib_cache/
```

The manifest gives every fit's PDF, 1-based page, PNG, plot status, and error.
The parameter export labels its values `point_estimate_only`; it does not
replace the inference-qualified coefficients table. All outputs are protected
from overwrite and remain under the selected results analysis directory.

To generate the same plots for existing completed runs, choose a fresh output
analysis name and run:

```bash
python -B run_series_mixed_effects_plots.py \
  --analysis-dirs \
    "$RESULTS_DIR/lmm_age_continuous_null" \
    "$RESULTS_DIR/lmm_age_continuous_pca_brain_mean" \
    "$RESULTS_DIR/lmm_age_continuous_pca_raw" \
    "$RESULTS_DIR/lmm_age_continuous_roi_brain_mean" \
    "$RESULTS_DIR/lmm_age_continuous_roi_raw" \
  --analysis-name lmm_age_continuous_fit_plots
```

This command reads the retained model inputs, statuses, and manifests. It
validates their specifications, coverage, cohorts, and stable hashes without
changing the existing runs. Its output also includes `fit_plot_summary.csv`.
Use `--dpi` to control resolution for this saved-run plotting command. New
saved-run plot exports go under `$RESULTS_DIR/lmm/<analysis-name>/plots/`.

## Validation and fit failures

Missing/duplicate columns, duplicate observations, conflicting metadata,
ambiguous feature identities, and nonfinite required values cause explicit
validation errors. Observations are never silently dropped or imputed.

Per-feature validation requires a finite full-rank four-column fixed design
`[1, E, E*ef_sf, age]`, more than four observations, and at least two sex levels.
A constant `ef_sf` makes the interaction collinear with `E`; constant age is
collinear with the intercept. Every observation may have a distinct age.
Fixed-sex designs additionally include their treatment contrasts, must be
full rank, and require more observations than their fixed-coefficient count.

The existing masters contain only two sex levels for the sex random intercept.
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

The separate `run_series_mixed_effects_with_aic_submit.sh` entry point submits
the same five model jobs and then queues a sixth AIC job with an `afterok`
dependency on all five IDs. The original `run_series_mixed_effects_all_submit.sh`
is retained for comparison until the new runner is tested. The new runner does
not call that original all-model entry point; it uses the existing per-master
submitter and Slurm worker directly.

```bash
# Preview six submissions without creating output directories or jobs.
bash run_series_mixed_effects_with_aic_submit.sh age_continuous_sex_fixed_v2 \
  --sex-effect fixed --dry-run

# Future execution: five model jobs, followed automatically by AIC.
bash run_series_mixed_effects_with_aic_submit.sh age_continuous_sex_fixed_v2 \
  --sex-effect fixed
```

These are examples, not commands executed during the implementation. Use a
fresh run name. The runner requires completed ML fits and rejects `--reml` and
`--prepare-only`; those options remain available in the original model-only
runner. Shared model flags are forwarded unchanged. AIC-only options are
`--aic-no-plots`, `--aic-top-n` (default 20), and `--aic-dpi` (default 150).
`--no-fit-plots` controls model plots independently of AIC figures.

The sixth job writes into the same run's `aic/`, with Slurm output in
`aic/logs/<job_name>_<job_id>.out` and `.err`. It uses the same `SERIES_PYTHON`
or Conda selection as the model worker. AIC defaults are account `p32903`,
partition `short`, one CPU, 3 GB memory, and five minutes; override them with
`SERIES_AIC_ACCOUNT`, `SERIES_AIC_PARTITION`, `SERIES_AIC_MEMORY`, and
`SERIES_AIC_TIME` when needed.

`submission_with_aic.json` records confirmed job IDs, dependency, AIC arguments
and batch script, submitter hash, and submission status. A partial submission
leaves earlier accepted jobs queued and records the failure; it does not
submit the AIC job without five confirmed IDs. An upstream job failure prevents
the dependent comparison from running. Per-fit convergence and inference
remain separate: successful model processing can contain failed fits, and AIC
checks their eligibility after the dependency is satisfied.

The original model-only runner remains available:

Submit the shared null and all four masters with one command. Each runs as a
separate job, with separate analysis directories:

```bash
export RESULTS_DIR=/projects/p32903/Alex2/results

# Check all inputs and print the Slurm commands without submitting or writing logs.
bash run_series_mixed_effects_all_submit.sh series_lmm_age_continuous_v1 --dry-run

# Submit the null and four feature-model jobs.
bash run_series_mixed_effects_all_submit.sh series_lmm_age_continuous_v1
```

The common run is `$RESULTS_DIR/lmm/series_lmm_age_continuous_v1/`, with children
`null`, `roi_raw`, `roi_brain_mean`, `pca_raw`, and `pca_brain_mean`.
Each job fits every selected feature separately
within each model/step configuration, except the null job, which fits once using
the raw ROI master's saved clinical covariates. Shared options such as `--prepare-only`,
`--sex-effect fixed`, `--reml`, `--optimizer lbfgs`, and `--maxiter 2000` are forwarded to every job.
All four input paths and CLI configurations are checked before the first job
is submitted. A scheduler rejection during submission stops the remaining
submissions; earlier accepted jobs remain submitted.
The jobs run independently; the comparison command below runs after their
result files are complete.

The defaults use the four master locations listed above. Override an input
with `SERIES_ROI_RAW_MASTER`, `SERIES_ROI_BRAIN_MASTER`,
`SERIES_PCA_RAW_MASTER`, or `SERIES_PCA_BRAIN_MASTER`. These paths can point
outside the output root. Use a fresh analysis prefix for subsequent runs to
keep existing results.

To submit one master or select particular ROIs/PCs, use the per-master wrapper:

```bash
bash run_series_mixed_effects_submit.sh series_lmm_pca_raw_age_continuous \
  --master-path "$RESULTS_DIR/pca/data/pca_magnE_mean_raw_series_PC_master.csv" \
  --feature-type pca --field-name magnE_mean
```

Add `--dry-run` to this command to validate and print its submission command.
Relative master paths resolve against the directory where the wrapper is
invoked. The wrapper validates runner arguments, the source master, the results
root, and the analysis name before creating logs or submitting.
It writes them directly to
`$RESULTS_DIR/lmm/<run_name>/<component>/logs/<job_name>_<job_id>.out` and `.err`
(omit `<component>` for a single-master run without `--output-component`).
The Slurm worker is `run_series_mixed_effects.slurm`; its defaults are account
`p32903`, partition `short`, one CPU, 3 GB memory, and five minutes. It uses the
`simnibs_env` Conda environment. Override
`SERIES_CONDA_ENV` or `SERIES_CONDA_SH`, or set `SERIES_PYTHON` to an absolute
interpreter path. `SERIES_VALIDATE_PYTHON` selects the standard-library Python
used for path validation before submission. `SERIES_SCRIPT_ROOT` is exported by
the wrapper for the job to locate this source tree.

## Verification

The grouped-run and fixed/random-sex changes were edited and statically reviewed
only, at the user's request. Their new and updated tests have not been run.
The historical counts below apply to earlier revisions, not this change.
New coverage is defined in `tests/test_series_models_run_layout.py` and
`tests/test_series_models_sex_effect_choice.py`, including namespace isolation,
immutable manifests, OLS/category coding, prediction reconstruction, and AIC.

The targeted tests include adapter validation, per-feature result accounting,
output-path failures, optimizer/covariance failures, and an actual synthetic
model with continuous age and a sex random intercept fitted through both the
core and the CSV CLI. Tests verify the exact numeric age column, absence of an
age variance component, acceptance of unique fractional ages, and export of
age coefficients and model provenance:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m pytest -q -p no:cacheprovider \
  tests/test_series_models_adapters.py \
  tests/test_series_models_fitting.py \
  tests/test_series_models_workflow.py
```

The original crossed age/sex model was tested and run on 2026-11-02. Its
`lmm_*` results and earlier `*_prepare` exports have fixed-design rank three
and describe the former specification. They have not been overwritten.
The continuous-age model uses rank four and requires a new run to obtain
revised clinical coefficients. Prepare-only verification does not establish
convergence or validity of clinical-model inference.

On 2026-11-03, the revised implementation passed 50 tests and 74 subtests
with Statsmodels 0.14.6. Read-only validation of all four existing masters
confirmed rank four for all 588 planned models, with 35 series per model.
This check did not refit or overwrite the existing clinical results.

Slurm launchers also pass 10 tests and 20 subtests using mocked `sbatch` and
`srun`, including input preflight, paths containing spaces, dry runs, separate
master dispatch, and preservation of worker failure exit codes:

```bash
python -B -m pytest -q -p no:cacheprovider \
  tests/test_series_models_slurm.py tests/test_series_models_batch_submit.py
```
