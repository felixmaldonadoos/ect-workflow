# Yeo2011 ROI and network E-field analysis

The active workflow uses five Python entrypoints: prepare the ROI master,
normalize its regional means, run regular and permutation Spearman correlations,
and plot the results. It reads scalar MNI NIfTI fields and the split-component
Yeo2011 atlas. The notebook files are historical interactive analyses; their
relative output paths do not configure this workflow.

Every generated table, figure, NIfTI map, manifest, and cache is saved under
`$RESULTS_DIR/<analysis_name>/`. Slurm logs go in that analysis's `logs/`
directory. No output directory defaults to `networks/data/` or another source
code directory. Existing results are left where they are.

## Submit the workflow

Set an absolute results root outside `05_analysis/`, then use the submission
wrapper. It creates the log directory **before** Slurm opens the log files.

```bash
export RESULTS_DIR=/projects/p32903/Alex2/results

bash /projects/p32903/Alex2/ect-workflow/05_analysis/networks/run_mni_yeo2011_submit.sh \
    mni_yeo2011 --plot-results 0
```

Omitting the analysis name uses `mni_yeo2011`. Omitting `--plot-results 0`
enables plotting, preserving the full runner's existing behavior. The full
workflow performs these steps in order:

1. Build the raw ROI master from the clinical/query tables and NIfTI fields.
2. Write a separate master normalized by each GM/WM-only field's nonzero mean.
3. Run regular and permutation correlations on the raw master.
4. Run regular and permutation correlations on the normalized master.
5. Plot each variant's permutation results when plotting is enabled.

| Submission option | Default | Effect |
|---|---|---|
| First positional argument | `mni_yeo2011` | Directory name directly below `RESULTS_DIR` |
| `--stage all\|prep\|correlations` | `all` | Select the existing Slurm job script |
| `--field-name FIELD` | `magnE_mean` | Select the NIfTI field and output filename field tag |
| `--plot-results 0\|1` | `1` | Disable or enable plotting in the full workflow |

Analysis and field names must be single filename-safe names containing letters,
digits, underscores, periods, or hyphens, starting with a letter or digit. Pass
an analysis name positionally to the wrapper; Python entrypoints use
`--analysis-name` instead. The wrapper rejects unrecognized options rather than
passing them through to a different command.

```bash
# Prepare only the raw master.
bash networks/run_mni_yeo2011_submit.sh mni_yeo2011 --stage prep

# Run permutation correlations on an existing raw master.
bash networks/run_mni_yeo2011_submit.sh mni_yeo2011 --stage correlations

# Use an existing normalized master with the correlations-only job.
MASTER_PATH_ROI="$RESULTS_DIR/mni_yeo2011/data/yeo2011_magnE_mean_series_ROI_master_normed_brain_magnE_mean_mean.csv" \
    bash networks/run_mni_yeo2011_submit.sh mni_yeo2011 --stage correlations
```

The `correlations` stage preserves the existing standalone job's
**permutation-only** behavior. It does not prepare or normalize data, run a
second master, or plot. The `all` stage handles both masters and both correlation
methods. `--plot-results` applies only to the `all` stage.

The Slurm requests remain unchanged: one node, one task, one CPU, the `short`
partition and `p32903` account; `all` requests 2 GB for 30 minutes, `prep` 1 GB
for 5 minutes, and `correlations` 5 GB for 5 minutes. The jobs activate
`simnibs_env`, use a headless Matplotlib backend, and keep numerical-library
threads within `SLURM_CPUS_PER_TASK`.

The wrapper exports `ANALYSIS_NAME`, `FIELD_NAME`, `PLOT_RESULTS`, the original
`YEO_SCRIPT_ROOT`, and the results/cache configuration to the job. It supplies
absolute Slurm `--output` and `--error` paths. The job files use `/dev/null`
placeholders for those directives; submit through the wrapper to retain logs.
`YEO_VALIDATE_PYTHON` can select the stdlib Python used for path validation.
`YEO_CONDA_SH` and `YEO_CONDA_ENV` can override the Conda initialization file
and environment. Normalization uses the GM/WM-only field itself and does not
require an external MNI brain mask.

## Output layout

For `analysis_name=mni_yeo2011` and `field_name=magnE_mean`:

```text
$RESULTS_DIR/
└── mni_yeo2011/
    ├── data/
    │   ├── yeo2011_magnE_mean_series_ROI_master.csv
    │   ├── yeo2011_magnE_mean_series_ROI_master_normed_brain_magnE_mean_mean.csv
    │   ├── spearman_magnE_mean_ROImean_and_network_annotated.csv
    │   ├── spearman_magnE_mean_ROImean_and_network_permutation_annotated.csv
    │   ├── spearman_magnE_mean_ROImean_and_network_normed_brain_magnE_mean_mean_annotated.csv
    │   └── spearman_magnE_mean_ROImean_and_network_normed_brain_magnE_mean_mean_permutation_annotated.csv
    ├── plots/                         # Created when plotting runs.
    │   ├── raw/
    │   │   ├── overview_maps/         # .nii.gz, .png, and source .csv map sets
    │   │   ├── parent_network_maps/   # .nii.gz, .png, and source .csv map sets
    │   │   └── plot_manifest.csv
    │   └── normed_brain_magnE_mean_mean/
    │       ├── overview_maps/
    │       ├── parent_network_maps/
    │       └── plot_manifest.csv
    └── logs/
        ├── yeo2011_all_<job_id>.out
        ├── yeo2011_all_<job_id>.err
        └── matplotlib_cache/
```

`normed_brain_<field_name>_mean` is the normalization tag. Because the default
field is named `magnE_mean`, the resulting tag ends in `magnE_mean_mean`.
`--no-network-means` on a correlation entrypoint changes its output scope to
`ROImean_only`. Re-running the same analysis overwrites its named outputs;
use another analysis name to keep a separate run. Log names include job IDs.

Explicit output paths must also be inside the selected analysis directory.
Input paths can refer to existing data elsewhere, including legacy CSVs under
`networks/data/`. Reading a legacy input does not redirect outputs there.
Unset, relative, or source-tree `RESULTS_DIR` values are rejected. Resolved
paths and symlinks are checked before output directories are created.

## Master tables and normalization

The raw master is a long table: one row per
`person_id × series_num × model_type × step × field_name × ROI`. It retains
clinical and stimulation metadata, matched `full_file_path`, ROI labels, and
spatial summaries (`mean`, `mode`, `median`, `min`, `max`, `p95`, `p99`, and
`n_voxels`). The full atlas ROI name is `roi_org`; `roi`, `hemi`, and `network`
provide parsed labels.

The default configurations remain `skin_single` and `skin_double`, crossed
with `step_0` and `step_9`. The clinical outcome is
`cgi_change = cgi_start - cgi_end`, and session 2 supplies the representative
stimulation row. Unique field files are extracted once and their statistics
are attached to every matching series. Each treatment series is treated as an
independent sample in correlations, including multiple series from one person.

Normalization expects MNI field NIfTIs already restricted to GM and WM
(SimNIBS tissue labels 1 and 2), with zero encoding excluded tissue/background.
For each unique input file, it validates that every value is finite, selects
`field != 0`, and calculates the arithmetic mean of those selected values:

```text
mean_values_gmwm = mean(field_nifti_values_gmwm[field_nifti_values_gmwm != 0])
normalized_field_values = field_nifti_values_gmwm / mean_values_gmwm
```

The pipeline applies the equivalent regional calculation
`mean = mean_raw / mean_values_gmwm` in a separate master CSV. Dividing every
voxel by the same scalar before taking an ROI mean gives the same result as
dividing that ROI's raw mean by the scalar. No additional normalized NIfTI is
written. The Yeo atlas's voxel selection and its inclusion of zero-valued
field voxels in raw ROI statistics are unchanged.
The raw master must be rebuilt from the same GM/WM-only inputs; the full
runner performs that preparation before normalization.

No external template is loaded, no brain-mask resampling is performed, and
`--brain-mask` is no longer an argument to the normalizer. Foreground selection
is recalculated per file even when two fields share a grid. An all-zero input,
NaN/infinity anywhere in the field, or a nonpositive/nonfinite selected mean
raises an error. A genuine zero-valued GM/WM voxel cannot be distinguished
from background by `field != 0`; inputs must follow the stated zero-background
convention.

The normalized master retains the series/ROI metadata and adds `mean_raw`,
`whole_brain_mean`, `whole_brain_n_voxels`, and normalization provenance.
For compatibility, the existing `whole_brain_*` names and `brain_mean` method
and filename tag are retained: the mean/count now refer to the input's nonzero
GM/WM values. `normalization_domain=gmwm_nonzero`, an empty mask path, and an
explicit `field != 0` rule distinguish these outputs from older template-mask
normalization. Legacy tables retain their recorded mask provenance.
Only `mean` is normalized; the other spatial statistics and `n_voxels` retain
their raw meaning. Correlation and plot CSVs retain the normalization metadata
and the path to their source table. Use a new analysis name, such as
`mni_yeo2011_gmwm`, when running this method alongside historical results.

## Correlations and plots

Each `model_type × step` subset is analyzed separately. The default features
are the 51 regional ROI means and seven combined network means. A combined
network feature is the unweighted mean of its member regional means across
both hemispheres, computed for each series **before** correlation. It is not
an average of correlation coefficients or a voxel-count-weighted mean.

The regular CSV includes Spearman `rho`, the SciPy `p_value`, sample count `n`,
configuration labels, and parsed feature labels. The permutation CSV also
includes `p_permutation`, the observed permutation statistic, the full null
distribution, requested/effective resampling parameters, seed/RNG state,
ranked inputs, sample keys, and software versions. The runner preserves
9,999 requested permutations, seed 42, and batches of 256.

Compare `p_permutation` with a chosen significance threshold, such as 0.05,
to evaluate permutation significance. `p_value` remains the ordinary
Spearman p-value. Neither p-value is adjusted for multiple comparisons.

Plotting paints Spearman rho onto the atlas, with a shared scale from -1 to 1.
It does not filter maps by either p-value. Overview maps show all regional or
combined-network coefficients; parent maps show a selected parent's regional
coefficients separately or its single combined-network coefficient. Each map
set has a NIfTI map, PNG, and source CSV; `plot_manifest.csv` records their paths.

## Run individual Python stages

Run from `05_analysis/` with the same `RESULTS_DIR` environment:

```bash
python -B networks/mni_yeo2011_prep_data.py --analysis-name mni_yeo2011

python -B networks/mni_yeo2011_normalize_brain.py --analysis-name mni_yeo2011

python -B networks/mni_yeo2011_correlations.py --analysis-name mni_yeo2011

python -B networks/mni_yeo2011_correlations_perm.py \
    --analysis-name mni_yeo2011 --n-permutations 9999 --seed 42 --batch-size 256

python -B networks/mni_yeo2011_plot_correlations.py \
    --analysis-name mni_yeo2011 \
    --correlations-path "$RESULTS_DIR/mni_yeo2011/data/spearman_magnE_mean_ROImean_and_network_permutation_annotated.csv"
```

Use `--master-path` to select a normalized or legacy master for correlations.
The default output filename detects raw versus brain-normalized metadata;
plotting also detects the variant and selects `plots/raw/` or
`plots/normed_brain_<field_name>_mean/`. `--output-path` and `--output-dir`
overrides must remain under `$RESULTS_DIR/<analysis_name>/`.

Each entrypoint provides `--help`. Clinical loading, file matching, ROI
extraction, feature aggregation, permutation settings, and plotting behavior
are otherwise unchanged by the results-directory refactor.

## Atlas and feature reference

The split-component atlas contains **51 regional ROIs within seven parent
networks**. The number seven describes the network families, not the number
of separate atlas labels. A regional feature keeps its hemisphere and
component identity; a combined network feature averages its regional means
across both hemispheres before correlation.

| Term | Meaning | Example |
|---|---|---|
| Parent network | One of the seven network families | `Default`, `Cont`, `Vis` |
| Hemisphere | Side containing a regional ROI | `LH` or `RH` |
| Regional ROI | A labeled component within a parent and hemisphere | `7Networks_LH_Cont_Temp` |
| Regional mean feature | Mean field value inside that ROI | `E_mean_7Networks_LH_Cont_Temp` |
| Combined network feature | Mean of all member regional means across LH and RH | `E_mean_Cont_both_hemis` |

| Parent network | LH ROIs | RH ROIs | Total regional ROIs |
|---|---:|---:|---:|
| `Vis` | 1 | 1 | 2 |
| `SomMot` | 1 | 1 | 2 |
| `DorsAttn` | 3 | 3 | 6 |
| `SalVentAttn` | 5 | 6 | 11 |
| `Limbic` | 2 | 2 | 4 |
| `Cont` | 9 | 7 | 16 |
| `Default` | 5 | 5 | 10 |
| **Total** | **26** | **25** | **51** |

For example, `E_mean_7Networks_RH_Default_PFCv` is one RH Default region;
`E_mean_Default_both_hemis` combines all ten Default regions. Equal regional
weights give LH 9/16 of the combined `Cont` feature and 5/10 of `Default`.
Missing regional values propagate through the network mean (`skipna=False`).
Regional suffixes can repeat across parents or hemispheres, so use the full
feature name `x`, or `roi` together with `hemi` and `network`, to identify one.

The default feature set produces 58 correlations per configuration. With
all four default model/step combinations, each regular or permutation CSV
contains 232 rows. Correlations require at least three observations, finite
numeric values, nonconstant columns, and complete, unique ROI records.
Invalid inputs raise errors rather than silently dropping observations.

## Regional extraction reference

The default scalar image is `tdcs_uq_gpc_MNI_magnE_mean.nii.gz`.
Categorical atlas labels are resampled onto each field image's grid with
nearest-neighbor interpolation; the field values are not resampled. The
extractor accepts a 3D image or a fourth dimension of length one.

| Master column | Statistic inside the regional ROI |
|---|---|
| `mean` | Arithmetic mean of voxel values |
| `mode` | Most frequent voxel value |
| `median` | Median |
| `min`, `max` | Minimum and maximum |
| `p95`, `p99` | 95th and 99th percentiles with linear interpolation |
| `n_voxels` | Included voxel count on the field image's grid |

Zero-valued voxels inside an ROI are included in these summaries. Any
non-finite value in the field image raises an error. An ROI absent after
label resampling receives count zero and NaN for its other statistics;
those NaNs subsequently fail the strict correlation checks. The active
extraction uses exact values for `mode` (`mode_decimals=None`). The Python
helper's optional rounding affects only the mode calculation, leaving all
other statistics unchanged.

The regional `mean` is already a spatial average; `n_voxels` records its
support and does not divide it again. Only regional means and their combined
network means enter the default correlations. The remaining summaries stay
in the master table for reference.

## Clinical and simulation matching reference

The course loader reads `Sheet1`, retaining `in_stim == True` and
`batch_num > 0`. Stimulation rows come from the workbook's `stimulus`
sheet. Configured subject exclusions are applied, and stimulation rows are
restricted to persons present in the course table. The representative row
is selected with `session_num == 2`; stimulation settings and fields are
not averaged over sessions within a series.

The selected stimulation and CGI tables must have exactly the same complete,
unique `(person_id, series_num)` keys. Missing keys, duplicate series, or
series present on only one side raise errors with diagnostics. Shared
`age`, `sex`, and `dob` values come from the CGI table.

Field matching uses the selected scan, electrode placement, query filters,
and exact field filename. Exact BT/BL matches are attempted first. If
mapping fails only for BT/BL rows, the unmatched rows are retried once with
BT and BL exchanged; original clinical placement labels are restored after
mapping. Other missing or ambiguous matches raise errors. The subject-scan
map selects modeled scans without aggregating observations or changing
clinical person IDs.

The inventory filters remain `threshold=95`, `asf=asf_5`, and
`step_result=mni_volumes`. The `threshold` filter is simulation-inventory
metadata, separate from the extracted `p95` statistic and any p-value
threshold. Requested steps must have matching field files for each model;
missing steps raise an error listing the available steps.

`ef_sf = pulse_width_ms * frequency_hz * percent_charge` is retained in the
prepared series metadata. This workflow does **not** multiply its E-field
features by `ef_sf` or convert their physical units. Raw summaries retain
the source image's units; normalized means are ratios to the corresponding
whole-brain mean.

## Sources of the definitions

- [Preparation and regional extraction](mni_yeo2011_prep_data.py): clinical
  joins, simulation matching, spatial summaries, and the long master schema.
- [Clinical/query loaders](../simnibs_parcel_analysis/utils/loader.py): input
  sheets, CGI difference, person IDs, and `ef_sf`.
- [Simulation-file mapper](../simnibs_parcel_analysis/utils/prepare_session_file_table.py):
  scan, placement, and field-file matching.
- [Atlas label names](../../atlases/Yeo2011/7Networks_ColorLUT_freeview.txt):
  hemisphere, parent, component names, and counts.
- [Whole-brain normalization](mni_yeo2011_normalize_brain.py),
  [regular correlations](mni_yeo2011_correlations.py),
  [permutation correlations](mni_yeo2011_correlations_perm.py), and
  [plotting](mni_yeo2011_plot_correlations.py): numerical definitions and
  output schemas for the active stages.
