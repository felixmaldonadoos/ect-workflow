# Assigned treatment-course regressions

This runner fits exactly two types of model, with one clinical observation per
treatment course. A patient can contribute multiple courses. There is no patient
random effect. The previous patient-intercept LMM remains available separately.

## Models

The clinical null is fitted once:

```r
Tx ~ 1 + age + sex + electrode_placement
```

Each ROI receives a separate mixed-model fit:

```r
Tx ~ 1 + EF_roi + EF_roi:ef_sf + (1 | age) + (1 | sex)
```

`Tx = cgi_end - cgi_start`. Sex and electrode placement are categorical; age is
numeric in the null. Baseline CGI, session count, patient effects, and a separate
`ef_sf` main effect are not added to these assigned formulas.

The random effects follow the notation literally. Each distinct observed age
value and each sex category has a random intercept. Age is not rounded or binned.
Statsmodels implements these as crossed variance components, with no additional
group intercept. Courses sharing an age or sex therefore share a modeled random
effect, even though there is no patient grouping. These terms are not equivalent
to the fixed age/sex adjustment in the null.

Few sex categories can make its random-effect variance poorly estimated. If
every course has a distinct age, age-intercept variance cannot be separated from
residual variance; the runner reports this rather than inventing age bins.

The ROI models use ML and Powell optimization. The null uses OLS. They are
separate models, not a nested null/full pair: model 2 changes the age/sex roles
and omits electrode placement. No nested-model likelihood-ratio or F test is
reported. Predictors remain in their supplied/constructed units.

## Course exposures

For each included session, the existing mapper selects the scan-specific HDF5
for its actual placement and simulation condition. Then:

```python
ef_sf_session = pulse_width_ms * frequency_hz * percent_charge
EF_scaled_session = EF_roi_session * ef_sf_session
```

The model's interaction/exposure column is named **`EF_scaled`** in Python and
the outputs, to make the agreed aggregation of session products explicit.

| `--aggregation` | Course `EF_roi` | Course `EF_scaled` |
|---|---|---|
| `mode` (default) | E for the course's modal placement | That E multiplied by the modal `ef_sf` among sessions with that placement |
| `mean` | Mean of all session E values, independently per ROI | Mean of all session `E * ef_sf` products, independently per ROI |
| `sum` | Sum of all session E values, independently per ROI | Sum of all session `E * ef_sf` products, independently per ROI |

For `mode`, first select the most frequent clinical electrode placement, then
the most frequent factor within those sessions. The selected sessions must use
one HDF5 and identical fields. Tied placement or factor modes produce a clear
error. There is no arbitrary first/lowest-value tie break and no per-ROI modal
selection that could combine different session patterns into a fictitious row.

The null always uses the modal clinical placement, including in mean/sum runs.
Placement ties consequently need resolution for all modes.

For mean/sum, `EF_scaled` is **not** reconstructed as the product of course
`EF_roi` and course `ef_sf`. It is the aggregated session product, as explicitly
chosen for this analysis. Under mode, the interaction is the literal product of
the selected course values.

Example: three RUL sessions have raw E=2 and factors 10, 20, 20; one BT session
has raw E=10 and factor 30. The course predictors are:

| Mode | `EF_roi` | `EF_scaled` | Null-model placement |
|---|---:|---:|---|
| mode | 2 | 40 | RUL |
| mean | 4 | 100 | RUL |
| sum | 16 | 400 | RUL |

Mean/sum include the switched-placement exposure but discard session order.
Mode represents the modal configuration and omits the other configurations from
the course E predictors. Full session fields and placement counts are retained
for inspection. All eligible sessions still require their actual HDF5 mapping.
`ef_sf` remains the legacy index, not measured charge in mC.

## Inputs

The clinical workbook uses `stimulus` and `Sheet1` by default. Existing session
validation, course-window matching, CGI eligibility, and HDF5 mapping functions
are reused. The old LMM design builder and fitter are not used.

- Outcome rows require `subjid`, `cgi_start`, `cgi_end`, `date_start`, and
  `date_end (acute)`, plus `age` and `sex`. Age should refer to the course start.
- Alternatively, `--subject-info` supplies a CSV/XLSX with one row per patient
  and `subjid`, `age`, `sex`. Use course rows if age differs across courses.
- Session rows require the existing session identifiers, date, placement, and
  the three stimulation settings used in `ef_sf`.
- Rows with `series_num == 0` are excluded and exported, as previously requested.
  Missing CGI pairs and associated sessions use the existing eligibility audit.
  Other unmatched sessions require explicit `--allow-unmatched-sessions`.
- Optional subject exclusions are explicit: `--exclude-subjects subj-cat-010`.
  No subject is excluded by default in this new runner.

ROI input is a **wide CSV of unscaled E magnitudes**:

```csv
hdf5_fn,L_V1,R_V1
/results/.../scan/RUL/step_2/result.hdf5,120.0,110.0
/results/.../scan/BT/step_2/result.hdf5,160.0,170.0
```

Use one atlas and one specified field-summary statistic per input table. Supply
the raw parcel magnitudes, before PCA normalization/demeaning and before `ef_sf`
scaling. Every used HDF5 needs a row, including less frequent placements. Missing
or invalid used ROI values are errors; explicit `--exclude-rois` is available.
The existing `roi_table_from_summary()` helper can export a ParcelSummary, but
the extraction must cover all required scan/placement combinations.

To create this input directly from all sessions' actual matched simulations,
run [`prepare_session_table.py`](SESSION_TABLE_README.md). It exports both a
wide `session_table.csv` and `simulation_EF_roi.csv`; pass the latter as this
runner's `--roi-table`. Its standalone export does not require CGI or age/sex.

`--atlas-name` defaults to `HCP_MMP1`. It labels the supplied ROI table and the
outputs; it is not a covariate and does not load/extract an atlas. To use another
atlas, supply its ROI table and change the label.

Actual CGI assessment dates must be verified from the source. Treatment window
dates are not assumed to be the assessment dates.

## Run

Install the repository's existing analysis extra in the analysis environment:

```bash
cd 05_analysis
python -m pip install -e '.[lmm]'

python run_course_regression.py \
    --clinical-file /projects/p32903/Alex2/datasets/STU00225089/.clincal_notes/cgi_batch_agg_2_clean.xlsx \
    --query-path query_result.csv \
    --subject-scan-map subject_scan_map.json \
    --subject-info /path/to/subject_info.csv \
    --roi-table /path/to/HCP_MMP1_unscaled_E.csv \
    --atlas-name HCP_MMP1 \
    --model-type skin_double --simulation-type adaptive \
    --aggregation mode \
    --output-dir results/course_regression/skin_double/adaptive/HCP_MMP1/mode
```

Replace the two placeholder files. Omit `--subject-info` if the outcome sheet
already supplies age and sex. `--aggregation mean` and `--aggregation sum` are
the alternatives; use separate output directories for each run. Add
`--prepare-only` to inspect course predictors before fitting, then use a new
directory for the fit. `static` maps to `step_0`, `adaptive` to `step_2`.

If needed, pass explicit mapping filters such as `--query-filter dataset_root=org`
or a justified placement alias such as `--placement-alias BT=BL`.

## Outputs and failures

- `course_data.csv`: one row per course, outcome/covariates, modal placement,
  factor summary, session counts, and full placement counts. Counts and the
  factor summary are audit columns, not additional fixed effects.
- `course_EF_roi.csv`, `course_EF_scaled.csv`: courses × ROIs.
- `session_table.csv`: one row per eligible session, all session parameters,
  patient/series/session IDs, selected scan/HDF5, atlas/condition labels, and
  `EF_roi__<ROI>` / `EF_scaled__<ROI>` columns. This export happens before course
  aggregation and covariate validation. It remains available if those later
  steps fail; inspect `analysis.json` for overall run status.
- `session_mapping.csv`: all included sessions, HDF5 paths, factors, and
  `used_for_course_EF`; `session_EF_roi.csv` and `session_EF_scaled.csv` retain
  the sessions × ROIs matrices.
- `coefficients.csv`: null coefficients once and ROI coefficients per parcel.
  Null p-values/CIs use OLS t inference; ROI inference uses asymptotic Wald z.
- `fit_status.csv`: model status, fit criteria, age/sex variance estimates,
  warnings, and explicit errors for failed fits.
- `analysis.json`: formulas, arguments, model definition, counts, and status.
- Exclusion tables: any explicitly excluded subjects, series-zero sessions,
  incomplete CGI courses, and excluded unmatched sessions.

A failed ROI does not discard valid fits from the other ROIs. Failures are
listed and the CLI exits nonzero after exporting results. Invalid covariance
or nonconvergence is not presented as valid inference; there is no automatic
fallback to another formula. BH q-values are computed separately for the raw
and scaled EF terms across the requested ROIs, retaining failed fits as p=1 in
the correction family. This does not adjust across atlases or other runs.

## Verification

```bash
python -m unittest discover -s tests -p test_course_regression.py -v
```

Tests check switching placements, modal selection and ties, aggregation after
session scaling, workbook/query preparation, and an actual crossed age/sex fit
with recovery of known EF coefficients. Clinical workbooks, demographics, and
real HCP_MMP1 ROI fields were not supplied for this implementation.

The implementation follows the [Statsmodels crossed variance-component example](https://www.statsmodels.org/stable/examples/notebooks/generated/variance_components.html).
