# NIfTI E-field PCA and CGI correlations

`run_mni_nifti_pca.py` fits voxelwise PCA separately for each
`model_type × step × normalization` group, preserves treatment-series metadata in
the component-score master tables, and correlates selected component scores with
`cgi_change`. Inputs are NIfTI E-fields. Existing ROI masters can supply the
series manifest; their regional summaries do not enter PCA.

## Results location and environment

Set an absolute results root outside the analysis source tree. Every generated
CSV, fitted model, NIfTI map, figure, and job log belongs under
`$RESULTS_DIR/<analysis_name>/`. The default analysis name is `mni_nifti_pca`.
There is no source-directory fallback when `RESULTS_DIR` is missing.

```bash
cd /projects/p32903/Alex2/ect-workflow/05_analysis
export RESULTS_DIR=/projects/p32903/Alex2/results

source "$HOME/software/miniconda3/etc/profile.d/conda.sh"
conda activate simnibs_env
python -B run_mni_nifti_pca.py --help
```

The environment needs NumPy, pandas, SciPy >= 1.15, nibabel, scikit-learn,
joblib, Matplotlib, nilearn, and the dependencies of the existing clinical/query
loader. The workflow does not
install packages or submit jobs by itself.

## Run or submit

The default run loads the existing clinical/query inputs, selects `step_0` and
`step_9`, and runs both `raw` and `brain_mean` analyses. It fits PCA and computes
Spearman and permutation correlations. Explained-variance figures are saved
after PCA preparation. Component maps and correlation plots require `--plot`.

```bash
python -B run_mni_nifti_pca.py --analysis-name mni_nifti_pca
```

To reuse an existing ROI master as the series manifest:

```bash
python -B run_mni_nifti_pca.py \
    --analysis-name mni_nifti_pca \
    --series-csv /projects/p32903/Alex2/ect-workflow/05_analysis/networks/data/yeo2011_magnE_mean_series_ROI_master.csv \
    --n-permutations 9999 --seed 42 --batch-size 256
```

Repeated ROI rows are collapsed only after their series metadata agrees. The
required manifest columns are `person_id`, `series_num`, `model_type`, `step`,
`field_name`, `full_file_path`, and `cgi_change`. Additional series metadata is
retained. Every field path must identify the original E-field NIfTI; normalization
is recomputed from its voxel data. A supplied `sample_id` must be unique per
series/configuration; otherwise a stable ID is generated.

Submit through the wrapper so log directories exist before Slurm opens files:

```bash
bash run_mni_nifti_pca_submit.sh mni_nifti_pca \
    --n-permutations 9999 --seed 42 --batch-size 256 --plot

# Omitting the first positional analysis name uses mni_nifti_pca.
bash run_mni_nifti_pca_submit.sh --normalizations raw --n-components 5
```

The wrapper accepts the analysis name as its first positional argument. Do not
also pass `--analysis-name`; it is reserved so the job and log paths agree. Use
absolute input paths when submitting. The job starts in this analysis directory,
activates `simnibs_env`, and currently requests one CPU, 5 GB, and 15 minutes on `short`
under account `p32903`. These resource settings are starting values; size them
using the first representative run. `PCA_CONDA_SH` and `PCA_CONDA_ENV` can override
the conda initialization file and environment. `PCA_VALIDATE_PYTHON` can override
the submit wrapper's `python3` used for standard-library path validation.

## Stages and options

Use the same `--analysis-name` when continuing a prepared analysis:

```bash
# Fit PCA and write masters/models/maps and variance figures without correlations.
python -B run_mni_nifti_pca.py --analysis-name pca_trial --stage prepare

# Regenerate just the variance figures from saved variance tables.
# Does not require correlation results, masters, or component NIfTI maps.
python -B run_mni_nifti_pca.py --analysis-name pca_trial --stage variance

# Reuse saved PCA scores for outcome permutations; do not refit PCA.
python -B run_mni_nifti_pca.py --analysis-name pca_trial \
    --stage correlate --n-permutations 9999 --seed 42 --batch-size 256

# Plot the saved analysis after preparing and correlating it.
python -B run_mni_nifti_pca.py --analysis-name pca_trial --stage plot

# Or request all plots at the end of an all-stage run.
python -B run_mni_nifti_pca.py --analysis-name pca_trial --plot
```

Other controls include:

- `--field-name magnE_mean` selects the E-field quantity.
- `--brain-mask /absolute/path/to/mask.nii.gz` replaces the default
  `/projects/p32903/Alex2/ecstim/resources/templates/MNI152_T1_1mm_brain.nii.gz`.
  Positive mask values define the brain, including for a brain-extracted
  intensity image used as a mask.
- `--normalizations raw`, `--normalizations brain_mean`, or
  `--normalizations raw brain_mean` choose the analyses. The default is both.
- `--variance-threshold 0.96` selects the smallest leading set of PCs explaining
  at least 96% of variance for correlation. Use `--n-components K` instead to
  select a fixed count. All nonzero PCs are exported in either case.
- `--n-permutations 0` computes analytic Spearman p-values only.
- `--resample` explicitly permits linear resampling of E-fields to the fixed
  mask grid. The default rejects differing image grids.
- `--query-path`, `--clinical-path`, `--subject-scan-map`, `--steps step_0 step_9`,
  `--all-steps`, and `--exclude-subjects` control the clinical/query input route.
  With `--series-csv`, the manifest supplies the series/configuration rows.
- `--background-path` controls the component-map background; `--dpi 150`
  controls all figures.

## Saved artifacts

For the default field, `<tag>` is `raw` or
`normed_brain_magnE_mean_mean`. Its final `mean` describes the whole-brain mean
normalizer; `magnE_mean` remains the original field name.

```text
$RESULTS_DIR/<analysis_name>/
├── run_manifest.json
├── data/
│   ├── series_inputs.csv
│   ├── basis_manifest.csv
│   ├── pca_magnE_mean_<tag>_series_PC_master.csv
│   ├── pca_magnE_mean_<tag>_series_scores.csv
│   ├── spearman_magnE_mean_<tag>_annotated.csv
│   └── spearman_magnE_mean_<tag>_permutation_annotated.csv
├── models/<tag>/<model_type>/<step>/
│   ├── model.joblib
│   ├── explained_variance.csv
│   ├── brain_mask.nii.gz
│   ├── mean_field.nii.gz
│   ├── analysis_settings.json
│   ├── fitting_samples.csv
│   └── components/
│       ├── PC1.nii.gz
│       ├── PC2.nii.gz
│       └── ...
├── plots/                         # Variance plots are saved during prepare.
│   ├── variance_plot_manifest.csv
│   ├── plot_manifest.csv          # Full plotting stage only.
│   └── <tag>/<model_type>/<step>/
│       ├── explained_variance.png
│       ├── variance_source.csv
│       ├── variance_plot_manifest.csv
│       ├── plot_manifest.csv      # Full plotting stage only.
│       └── ...                    # Other figures when --plot is enabled.
└── logs/                          # Created by the submit wrapper.
    ├── mni_nifti_pca_<job_id>.out
    ├── mni_nifti_pca_<job_id>.err
    └── matplotlib_cache/
```

The long PC master has one row per series and component, with `component` and
`score` replacing ROI identity and ROI means. It retains series metadata and
adds basis, normalization, explained-variance, and component-map provenance.
The wide score CSV has one row per series and one score column per PC.
`basis_manifest.csv` identifies each separately fitted basis. Correlation CSVs
contain only the selected PCs, grouped by their saved basis; masters and spatial
maps retain all nonzero PCs. Plotting uses the saved masters, correlations, and
maps without fitting PCA or recomputing statistics.

Each explained-variance figure shows one bar per nonzero PC (percent variance)
and a cumulative-variance line. Blue bars identify PCs selected for correlation;
gray bars show the remaining PCs. The variance-only stage reads just
`basis_manifest.csv` and its saved variance CSVs, and can run before correlations.

## Interpretation and assumptions

Each treatment series is an independent sample throughout fitting and outcome
permutations. A patient may contribute several series, including repeated
E-field images; those rows contribute separately. Loading an image only once
does not change that weighting. Missing/nonfinite correlations, fewer than
three sample rows, constant scores/outcomes, and conflicting sample metadata
raise errors rather than silently dropping observations.

For `brain_mean`, divide each sample's masked E-field voxels by its whole-brain
mean before PCA, including zero-valued field voxels in that mean. A nonpositive
or nonfinite brain mean is invalid. PCA centers each
voxel across samples, uses full SVD, and does not standardize voxel variance or
whiten scores. Numerically zero components are excluded. Signed component maps
and scores use a deterministic sign: the largest absolute component weight is
positive. A negative PC score is valid.

Each group has its own fitted basis and `basis_id`. `PC1` from two groups is not
automatically the same spatial axis, even with the same component number or
sign convention. Raw and normalized analyses also have separate bases. Selection
uses explained variance, without looking at CGI outcomes.

`p_value` is the analytic two-sided Spearman p-value. `p_permutation` is SciPy's
two-sided outcome-pairing permutation p-value; evaluate it directly against the
chosen significance threshold. All rows are permuted freely. PCA is held fixed,
and the same seeded outcome shuffles are used across components within a group.
SciPy uses exact enumeration when the requested resamples reach `n!`; otherwise
it uses Monte Carlo sampling with the add-one correction. Results preserve the
full null distribution, rank vectors, sample order, raw numeric inputs, RNG
initial state, counts, parameters, and software versions as CSV columns/JSON.
These p-values have no multiple-testing adjustment. With permutations disabled,
`p_permutation` is blank and the saved null is `[]`.

## Local checks

From the analysis directory in the configured environment:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
    python -m unittest discover -s tests -p 'test_nifti_pca*.py'
bash -n run_mni_nifti_pca.slurm
bash -n run_mni_nifti_pca_submit.sh
```

These tests use small synthetic inputs and temporary output directories. They
do not run the full clinical dataset or submit Slurm jobs.
