# LMM: session exposures and course-level CGI outcomes

`run_lmm.py` is separate from the PCA runner. It reuses the session-to-HDF5
mapper in `clinical.py`; each session uses its actual electrode placement.
There is one CGI outcome per treatment series. Sessions supply exposure
histories and are never assigned replicated course-level CGI outcomes.

## Install and first run

From `05_analysis`, inside the existing analysis environment:

```bash
python -m pip install -e '.[lmm]'

python run_lmm.py \
    --sessions-xlsx /path/to/clinical.xlsx \
    --session-sheet stimulus --outcome-sheet scores \
    --query-path /path/to/results_query.csv \
    --model-type skin_double --simulation-type adaptive \
    --subject-scan-map /path/to/subject_scan_map.json \
    --dose-column ef_sf --aggregation sum \
    --output-dir /path/to/results/lmm/skin_double/adaptive/clinical_sum \
    --prepare-only
```

The example uses `sum` as a cumulative-exposure hypothesis. `mean` instead
models average session exposure. Neither encodes ordering, spacing, recency,
or recovery between sessions. Choose the aggregation before examining outcome
associations. The option is required so this assumption is never implicit.

Inspect the exported tables first. Remove `--prepare-only` and use a **new
output directory** to fit the model. Existing nonempty output directories are
never overwritten. Model/step/atlas separation is controlled by your output
path and also recorded in the manifest. `static=step_0`, `adaptive=step_2`.

The CLI uses existing exported HDF5 paths without rediscovering files or
opening HDF5 data. One selected scan per patient is chosen by the existing
mapping rules. Multiple candidate scans require the explicit scan map; the
mapper never switches scans based on series/session numbers. Clinical BL/BT
are distinct unless explicitly aliased with `--placement-alias BT=BL`.
Use repeated `--query-filter KEY=VALUE` arguments to resolve dataset, run,
threshold, or filename ambiguity. Full file paths identify simulations.

## Inputs and measurement dates

| Input | Required fields | Interpretation |
|---|---|---|
| Session XLSX | `subjid`, `series_num`, `session_num`, `date`, `electrode_placement`, `pulse_width_ms`, `frequency_hz`, `percent_charge` | Actual session date, placement, and recorded settings; other columns are preserved |
| Outcome CSV/XLSX | `subjid`, `cgi_start`, `cgi_end`, `date_start`, `date_end (acute)` | One pre/post pair per course; these dates define the inclusive exposure window |
| Optional outcome field | `series_num` | If present, must agree with the series found in the date window |
| Optional covariates | Numeric/categorical columns in outcomes, or `--subject-info` CSV/XLSX keyed by `subjid` | Prefer age at course start in the outcome table; no age averaging |
| ROI CSV, when available | `hdf5_fn`, followed by ROI columns | Unscaled, nonnegative finite E magnitudes for one atlas/statistic and selected simulation condition |

`--outcomes` selects a separate file; otherwise outcomes come from the session
workbook. Sheet arguments accept names or zero-based indices. Column headers
are checked before pandas can rename duplicates.

Actual CGI assessment dates, if available as additional columns, remain in
the exported outcome/model tables. The runner **does not infer that treatment
window dates are measurement dates**. Verify the window includes only the
exposure relevant to the post-CGI assessment. Dates are used at calendar-day
resolution; same-day pre/post ordering cannot be inferred from these inputs.
The subject-info interface also preserves an explicitly supplied age-reference
date; it does not calculate age from an unknown date. These are analysis inputs
that must be defined from the clinical source.

All outcome rows must have complete CGI values and match one stimulus series.
Duplicate courses, overlapping windows, ambiguous series, unavailable scans,
missing placements, and missing/ambiguous simulations fail explicitly. There
is no automatic clinical-cohort selection from available simulations; supply
outcome rows for the intended cohort. Sessions without an outcome window also
fail unless `--allow-unmatched-sessions` is supplied; those excluded rows and
their reason are exported. To intentionally exclude maintenance/invalid series
before validation, use `--acute-only`; this invokes the existing loader's
explicit filter, which prints the excluded original rows and count. Neither
option silently discards an incomplete outcome.

All three stimulation inputs must be finite and positive. A placeholder such
as `frequency_hz="age"` is an error when computing `ef_sf`; it is never converted
to the patient's age or an invented frequency. No pulse-width recoding or
missing-value imputation is performed. Percentage settings are used as recorded
(e.g. 50, not 0.50). Do not substitute EEG seizure duration for stimulus duration.

## Session x ROI matrices

The wide ROI interface deliberately separates atlas extraction from modeling:

```csv
hdf5_fn,ctx_lh_precentral,ctx_rh_precentral
/results/.../scan/RUL/.../step_2/tdcs_uq_gpc.hdf5,120.0,95.0
/results/.../scan/BT/.../step_2/tdcs_uq_gpc.hdf5,105.0,110.0
```

Supply a unique row for each HDF5 needed by the session map. Rows can be in any
order, and unused simulations are allowed. HDF5 path matching is exact; a table
keyed only by patient or scan would lose placement identity. A modal-placement
PCA inventory may omit simulations needed for nonmodal sessions. Extract all
required scan/placement fields first, using `selected_hdf5_inventory.csv`.

For an existing `ParcelSummary`, the Python bridge preserves simulation/path
identity and the chosen P95 estimator:

```python
from simnibs_parcel_analysis.lmm import roi_table_from_summary

roi_table = roi_table_from_summary(summary, method="spatial_weighted")
roi_table.to_csv("atlas_roi_fields.csv", index=False)
```

For parcel means, construct the same HDF5-keyed interface from your extractor's
mean values and record `--roi-statistic mean`. This runner does not require
SimNIBS to be importable for modeling or precomputed-matrix preparation; the
existing package's other Python dependencies are still needed.

Add these arguments to the first-run command and use another output directory:

```bash
--roi-table /path/to/atlas_roi_fields.csv \
--atlas-name volume_aparc \
--roi-statistic p95_spatial_weighted \
--roi-model components
```

For patient `p` and session `s`, the matrices are:

```text
raw_E[s, roi]    = ROI field from that session's selected HDF5
ef_sf[s]        = pulse_width_ms[s] * frequency_hz[s] * percent_charge[s]
scaled_E[s, roi] = raw_E[s, roi] * ef_sf[s]
```

Each patient receives `session_matrices/<atlas>/<patient>/raw_E.csv` and
`scaled_E.csv`, each N x M **data values**, plus the `session_id` row labels.
`rows.csv` records the corresponding patient, modeled scan, series, session
number, date, HDF5, and stimulation variables. N counts actual observed session
rows across that patient's included series, not the largest session number.
Order is patient, series, date, session number; session numbers need not restart
or be consecutive. Series-specific matrices can be selected using `rows.csv`.
Global aligned session matrices are also exported. No subject averaging occurs.

`ef_sf` is retained exactly as a legacy composite index. `scaled_E` is an
index formed from E and stimulation settings; it is not physical E amplitude
in V/m or delivered charge. For optional `--dose-column percent_charge`, the
exported `scaled_E` still uses `ef_sf`, while the model's `ef_dose` interaction
uses the selected dose column. This distinction is explicit in the outputs.

Missing, infinite, or negative values in used ROI rows fail. Use
`--exclude-rois ROI_a ROI_b` to remove explicitly chosen parcels; unknown names
raise. No subjects, individual sessions, or missing ROI cells are silently
dropped or imputed. All ROI models in a run use the same course cohort.

## Model definitions

Let `A()` be the selected within-course sum or mean and `D_s` the selected dose
column (`ef_sf` by default). One row is constructed per course:

```text
dose      = A(D_s)
ef_raw    = A(E_s)
ef_dose   = A(E_s * D_s)
ef_scaled = A(E_s * ef_sf_s)
```

Products are formed **within sessions before aggregation**. `A(E_s * D_s)` is
generally different from `A(E_s) * A(D_s)`.

Models, with one separate fit per ROI when a table is supplied:

```r
# Without an ROI table: stimulation-only starting model
cgi_change ~ cgi_start + dose + n_sessions + (1 | subjid_base)

# --roi-model components (default): E + dose + session-level E:dose
cgi_change ~ cgi_start + dose + ef_raw + ef_dose + n_sessions + (1 | subjid_base)

# --roi-model scaled: legacy scaled exposure plus selected dose
cgi_change ~ cgi_start + dose + ef_scaled + n_sessions + (1 | subjid_base)
```

No additional `ef_scaled:dose` term is introduced. That would apply stimulation
weighting twice. ROI values are predictors, never extra observations. The
runner fits each ROI separately rather than putting all M correlated parcels
into one small-sample model. It does not fit PCA or accept signed PC scores as
raw magnitudes. A future PC interface needs a common, prespecified PCA basis.

`cgi_change = cgi_end - cgi_start`. Baseline CGI is always included. Default
`n_sessions` adjustment is explicit; if it is constant or intentionally omitted,
use `--no-session-count`. Session count and dose can depend on prior response:
including them does not solve treatment-selection bias or create causal effects.

Add course covariates with `--numeric-covariates age` and
`--categorical-covariates sex`. Categorical references are the first
lexicographically sorted level and are exported. `--age-sex-interaction`
requires both covariates and adds their fixed interaction. Patient IDs define
the random intercept; age/sex do not define random-effect groups. Patients with
one course remain included alongside those with repeated courses. If no patient
has repeated courses, the LMM fails rather than fitting an unidentifiable
variance decomposition. Fewer than five repeated patients produce a warning,
not an automatic exclusion or a claim that five is a sufficiency threshold.

For numerical stability, continuous course-level predictors are standardized
using population SD after exposure construction; CGI change stays in its
original units. `predictor_scaling.csv` documents the transform. Optional age x
sex terms use the transformed age and unscaled indicator. `--no-standardize`
uses original predictor units. Constant/rank-deficient designs fail with an
explanation; variables are never dropped automatically.

Fitting uses statsmodels MixedLM with a patient random intercept, REML, and
Powell optimization by default. `--ml`, `--optimizer`, and `--maxiter` are
explicit alternatives. Nonconvergence, singular covariance warnings, invalid
Hessians, or nonfinite estimates stop the run and mark the manifest failed.
A valid converged boundary fit (ICC < 1e-6) is exported with a warning and flag;
it is not treated as proof of patient independence. No automatic fallback to
ordinary regression occurs.

## Outputs and inference

- `manifest.json`: input hashes, options, counts, definitions, and run status.
- `session_mapping.csv`, `selected_hdf5_inventory.csv`, `excluded_sessions.csv`.
- `course_exposures.csv`, `patient_course_counts.csv`, and optional course ROI matrices.
- Patient and global session matrices, with row metadata kept separately.
- `model_index.csv`: ROI-to-model-directory lookup.
- `models/model_####/model_data.csv`: exactly one row per fitted course.
- `fixed_effects_design.csv`, `predictor_scaling.csv`, `coefficients.csv`,
  `predictions.csv`, and `diagnostics.json` within each model directory.
- `lmm_coefficients.csv`: combined estimates, 95% Wald intervals, and p-values.

ROI runs additionally report BH q-values across all retained ROIs **separately
for each E-related term** (`ef_raw`, `ef_dose`, or `ef_scaled`). This does not
control multiplicity across atlases, simulation models, aggregation choices,
or all tested terms jointly. Define that broader family in the analysis plan.
No final combined results are written if a model fails partway through; the
failed manifest distinguishes any earlier partial model outputs.

P-values and intervals are asymptotic Gaussian-LMM Wald quantities; no
Satterthwaite, Kenward-Roger, bootstrap, or patient-cluster correction is
implemented here. Few independent patients/repeated series can make these
approximations weak. Inspect residuals and variance estimates, consider the
ordinal/bounded nature of CGI, and prespecify appropriate sensitivity analyses
before confirmatory interpretation. If development/held-out cohorts are used,
split at patient level before running this pipeline; its standardization is
fit on the supplied cohort and is not a held-out prediction workflow.

API reference: [statsmodels MixedLM](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html).

## Verification

```bash
python -m unittest discover -s tests -p 'test_lmm.py' -v
```

Tests use synthetic data, including placement switches, shared HDF5s across
courses, unmatched-session auditing, incomplete/ambiguous inputs, exact row
scaling, session-level product aggregation, singleton patients, and recovery
of a known fixed effect in an actual mixed model. Real atlas fields and clinical
workbooks must be checked in the HPC environment before scientific use.
