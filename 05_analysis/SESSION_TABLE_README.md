# Session-level E-field table

`prepare_session_table.py` produces **`session_table.csv`**, with one row per
acute treatment session. It preserves the patient, series, and session IDs;
session date; original clinical/stimulation columns; matched simulation paths;
and one raw and one scaled E-field column for every ROI. It does not read CGI
outcomes, require age/sex, choose modal placements, aggregate sessions, or fit
models. A course without CGI is still included.

## How each row is created

1. Read the `stimulus` sheet (or a session CSV). Validate the patient, series,
   session number, date, actual electrode placement, and stimulation settings.
   Explicit subject exclusions and `series_num == 0` maintenance sessions are
   saved to exclusion CSVs. Other malformed rows are errors.
2. Filter the query by skin model and simulation type: `static` selects
   `step_0`; `adaptive` selects `step_2`. Reuse `map_sessions_to_hdf5()` to
   select one modeled scan for each patient and match **that session's actual
   electrode placement** to exactly one HDF5. Multiple candidate scans require
   `--subject-scan-map`; there is no nearest-date or series-number scan inference.
   Ambiguous runs/datasets require an explicit `--query-filter`.
3. For each distinct matched HDF5, reuse `build_surface_roi_summary()` to
   extract all atlas ROIs once. For HCP_MMP1, the actual field comes from the
   associated `fsavg_overlays/tdcs_uq_gpc_fsavg.msh` mesh, not directly from the
   HDF5 volume. Multiple placements from the same scan are separate simulations.
4. Join the unscaled ROI vector to every session using the **exact full HDF5
   path**. Compute `ef_sf = pulse_width_ms * frequency_hz * percent_charge` and
   multiply every ROI in that row by this session's factor.

For session `s` and ROI `r`:

```text
EF_roi__[r][s]    = ROI summary from simulation matched to session s
EF_scaled__[r][s] = EF_roi__[r][s] * ef_sf[s]
```

Repeated sessions using the same scan/placement share raw fields. Their scaled
fields differ when their stimulation factors differ. A placement switch selects
the corresponding simulation before scaling. There is no sum across sessions
and no accumulated exposure in this table. The legacy factor is an index, not
measured charge in mC or a physical instantaneous field-amplitude calibration.

Illustrative values for one ROI (not patient data):

| Patient | Series | Session | Placement | Matched HDF5 | ef_sf | EF_roi__L_V1 | EF_scaled__L_V1 |
|---|---:|---:|---|---|---:|---:|---:|
| P1 | 1 | 1 | RUL | scan1/RUL/result.hdf5 | 10 | 2 | 20 |
| P1 | 1 | 2 | RUL | scan1/RUL/result.hdf5 | 20 | 2 | 40 |
| P1 | 1 | 3 | BT | scan1/BT/result.hdf5 | 30 | 10 | 300 |

The actual CSV also includes the original stimulation parameters and dates.

## Run directly from simulation outputs

Run in the repository's analysis/SimNIBS environment, with `openpyxl` installed
for Excel input. No Statsmodels fitting dependency is used by the exporter.

```bash
cd /projects/p32903/Alex2/ect-workflow/05_analysis

python prepare_session_table.py \
    --clinical-file /projects/p32903/Alex2/datasets/STU00225089/.clincal_notes/cgi_batch_agg_2_clean.xlsx \
    --session-sheet stimulus \
    --query-path /projects/p32903/Alex2/ect-workflow/05_analysis/query_result.csv \
    --subject-scan-map /path/to/subject_scan_map.json \
    --model-type skin_double --simulation-type adaptive \
    --atlas-name HCP_MMP1 \
    --output-dir results/session_tables/skin_double/adaptive/HCP_MMP1
```

Replace the scan-map placeholder with the existing mapping. Omit the option
only if each patient has exactly one candidate scan after query filtering.
HDF5 paths and their associated meshes must be accessible on the machine
running extraction. Use a new/empty output directory for each run.

The default is the **area-weighted 95th percentile** of `magnE_mean` within
each ROI, matching the existing surface extractor. To choose another summary:

- `--roi-statistic mean`: arithmetic mean over the ROI's surface nodes.
- `--roi-statistic p95_unweighted`: ordinary interpolated 95th percentile.
- `--roi-statistic p95_spatial_weighted`: area-weighted 95th percentile (default).

These choices summarize locations **inside an ROI**, not sessions within a
course. `--field-name` changes the scalar mesh field; the default is
`magnE_mean`. `--min-nodes-per-roi` defaults to 20 and checks every extracted
ROI. `--surface-relative-path` changes the mesh location relative to the HDF5's
parent directory. Supported direct-extraction atlases are `HCP_MMP1`, `DK40`,
and `a2009s`.

Optional query filters and justified placement aliases use the existing syntax:
`--query-filter dataset_root=STU00225089` and `--placement-alias BT=BL`.
The original clinical placement remains in `electrode_placement`; the selected
simulation placement is recorded separately as `simulation_electrode_placement`.
BT and BL are distinct unless an alias is supplied.

## All four simulation conditions

```bash
set -euo pipefail
CLINICAL_FILE="/projects/p32903/Alex2/datasets/STU00225089/.clincal_notes/cgi_batch_agg_2_clean.xlsx"
QUERY_FILE="/projects/p32903/Alex2/ect-workflow/05_analysis/query_result.csv"
SCAN_MAP="/path/to/subject_scan_map.json"

for model in skin_single skin_double; do
    for simulation in static adaptive; do
        python prepare_session_table.py \
            --clinical-file "${CLINICAL_FILE}" --query-path "${QUERY_FILE}" \
            --subject-scan-map "${SCAN_MAP}" \
            --model-type "${model}" --simulation-type "${simulation}" \
            --atlas-name HCP_MMP1 \
            --output-dir "results/session_tables/${model}/${simulation}/HCP_MMP1"
    done
done
```

Each condition gets its own table, so simulations do not create duplicate
clinical observations within one table.

## Use an existing ROI table or another atlas

Add `--roi-table /path/to/unscaled_roi.csv` to bypass mesh extraction. The
CSV must have one row per full `hdf5_fn` and one column per ROI, containing
unscaled, nonnegative E magnitudes. Input row order does not matter. Every used
HDF5 must have fields for every included ROI; missing values, missing paths,
or duplicate paths are errors. No basename/scan-only fallback or imputation
is used. Explicit `--exclude-rois ROI_NAME` is available.

In this mode `--atlas-name` labels the supplied atlas. Optional `--field-name`
and `--roi-statistic` declare the supplied values' provenance; they do not
recompute them. Undeclared field/statistic metadata is recorded as `provided`.
This allows volume/custom atlases whose fields have been extracted separately.

## Outputs and keys

| File | Contents |
|---|---|
| `session_table.csv` | All session metadata plus `EF_roi__<ROI>` and `EF_scaled__<ROI>` columns |
| `session_EF_roi.csv` | Unscaled N sessions × M ROIs, indexed by `session_id` |
| `session_EF_scaled.csv` | Scaled N sessions × M ROIs, indexed by `session_id` |
| `session_mapping.csv` | Session metadata, paths, and factors without the wide ROI columns |
| `simulation_EF_roi.csv` | Unscaled fields once per used HDF5; reusable as course regression's `--roi-table` |
| `simulation_sources.csv` | Unique simulation/scan/HDF5/surface paths (direct extraction only) |
| `roi_qc.csv`, `roi_node_counts.csv` | Extraction quality control and ROI coverage (direct extraction only) |
| `session_table.json` | Arguments, definitions, counts, scope, and success/failure status |
| `excluded_*.csv` | Explicit subject exclusions and series-zero maintenance rows |

`series_id` is `subjid_base::series_<series_num>`. `session_id` additionally
includes `::session_<session_num>`. Thus session numbers restarting in another
series do not collide. Input session order is preserved; use `date` and
`session_num` when sorting within a series. No session-level CGI is invented.

The course-regression runner also writes `session_table.csv` before course
aggregation or age/sex checks. Its table contains only sessions linked to
eligible complete-CGI courses and additionally retains `treatment_course_id`
from outcome-window matching. Use this standalone exporter for all acute
sessions regardless of outcome availability.

## Verification

```bash
python -m unittest discover -s tests -p 'test_session_table.py' -v
```

Synthetic tests cover a patient with multiple courses and scans, explicit scan
selection, a placement switch, repeated simulations with different factors,
unordered input files, case-normalized placement aliases, missing ROI fields,
and extraction once per used simulation for each supported summary statistic.
The extraction tests use synthetic meshes with known fields; real clinical
workbooks and SimNIBS outputs were not supplied.
