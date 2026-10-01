# SimNIBS anatomical ROI and PCA analysis

For the separate session-exposure/course-outcome mixed-model pipeline, see
[LMM_README.md](LMM_README.md) and `run_lmm.py`. It preserves each session's
actual placement and can export per-patient session x ROI matrices scaled by
the legacy `ef_sf` index.

This package analyzes `magnE_mean` stored in the `mesh_roi` group of SimNIBS
gPC HDF5 files. It supports two anatomically distinct workflows:

- **Volume:** subject-specific FreeSurfer `aparc+aseg.mgz` (Desikan–Killiany)
  or `aparc.a2009s+aseg.mgz` (Destrieux) labels sampled at tetrahedron
  barycenters.
- **Surface:** `HCP_MMP1`, `DK40`, or `a2009s` masks applied to E-fields that
  SimNIBS already mapped to an fsaverage surface mesh.

Do not use `mesh.elm.tag1` as an anatomical atlas. Standard SimNIBS element
tags describe tissues or mesh regions, not DK, Destrieux, or HCP parcels.

## Recommended thesis analysis

1. Use volume `aparc+aseg` as the primary analysis. It includes every cortical
   DK parcel plus bilateral thalamus, caudate, putamen, pallidum, hippocampus,
   amygdala, accumbens, ventral diencephalon, and cerebellar cortex.
2. Use volume `a2009s+aseg` as a finer cortical sensitivity analysis.
3. Use HCP-MMP1 only on a surface E-field mesh. HCP-MMP1 does not provide
   hippocampus, amygdala, thalamus, or other subcortical parcels.
4. Treat tetrahedron-volume-weighted P95 as primary and ordinary P95 as a
   sensitivity analysis. Both are exported.

The PCA input follows the requested order exactly:

1. Calculate one P95 value per parcel and modeled scan.
2. Calculate the equal-parcel global mean P95 within each scan.
3. Divide every parcel by that scan's global mean.
4. Demean the relative parcel values within each scan.
5. Fit unscaled PCA to the scans × parcels matrix. Scikit-learn centers each
   parcel across scans during PCA.

The two global predictors are:

- `global_mean_E_unweighted`: equal mean of parcel P95 values.
- `global_mean_E_weighted`: mean of parcel P95 values weighted by parcel
  volume (volume analysis) or surface area (surface analysis).

## Subject matching

Full modeled IDs remain distinct observations. The recon-all directory is
resolved in this order:

1. Explicit `subject_map` entry.
2. Exact full-ID match.
3. The only recon-all directory sharing the same base ID.

Thus `subj-cat-001-002` can use `SUBJECTS_DIR/subj-cat-001-001` when that is
the only reconstruction for `subj-cat-001`. Multiple candidate directories
raise an error rather than being guessed.

## Installation

Use the same environment as SimNIBS 4.6 and FreeSurfer. The Python dependencies
are `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `nibabel`, and
`simnibs`. `FREESURFER_HOME` must be set unless `freesurfer_lut` is passed.

## Usage

See `example_usage.py`. Its `run_all_atlases()` function accepts your HDF5
path list, cognitive DataFrame, `SUBJECTS_DIR`, and output directory.

The HCP surface analysis expects this default sibling path for every HDF5 file:

```text
<step_dir>/fsavg_overlays/tdcs_uq_gpc_fsavg.msh
```

Pass `include_surface_hcp=False` if those surface overlays have not been
generated. Recon-all surfaces alone do not contain the simulated E-field.

## Required spatial validation

The volume sampler assumes mesh barycenters and the atlas image affine refer
to the same subject scanner-RAS space. Review `atlas_assignment_qc.csv` and
visually verify at least several representative scans before interpreting
results. If spaces differ, resample the atlas into the mesh's subject-space
reference or pass an explicit `mesh_to_atlas_ras` affine.

`min_elements_per_roi=20` and `min_nodes_per_roi=20` are deliberate QC
thresholds. The code stops when an ROI has too few samples; lower a threshold
only after reviewing mesh resolution and the affected regions.

## Scans and treatment courses

`subjid_base` is a join key only. It matches modeled scans to FreeSurfer
directories and clinical treatment courses; it is not the clinical observation
identifier. A treatment course is uniquely identified by
`(subjid_base, date_start)`. Different courses for one person may have different
CGI changes. Two clinical rows with the same base ID and `date_start` are a
conflict and stop the analysis.

The pooled runner (`run_pca_weighted_global_E_job_synthsr.py`) reads the
`stimulus` sheet of the same cognitive-score workbook and selects the modal
placement separately for each `(subjid_base, series_num)`:

Before selection, the acute-analysis path explicitly excludes rows whose
`series_num` is below 1 or is not a finite integer. This removes maintenance
sessions (`series_num=0`), negative/fractional values, missing values, and
nonnumeric labels. Numeric strings such as `"1"` and integer-valued floats
such as `1.0` are accepted. The excluded count and original rows are printed;
no course numbers are inferred, filled, or rounded. Filtering occurs before
session/date/placement validation, and an empty acute table is an error.
Direct `load_ect_sessions()` calls remain strict unless `acute_only=True` is
requested; both acute course-mode functions apply this filter automatically.

1. Match a complete CGI course to `series_num` in the outcome sheet when
   supplied. Otherwise, require exactly one stimulus series with sessions in
   the inclusive `date_start`–`date_end (acute)` interval. Missing matches,
   overlapping series, or two clinical courses assigned to one series fail.
2. For matched acute courses, count only sessions inside that date interval;
   maintenance sessions outside it do not determine the acute-course mode.
   Stimulus series without an eligible CGI course use all their sessions and
   remain eligible for ROI/PCA, without an outcome.
3. Count each session once. Normalize placement whitespace and case; ignore
   blank/missing/literal `nan` placements with a warning and exported counts.
   Ties, entirely unknown placements, invalid labels, and duplicate sessions
   fail explicitly. `frequency_hz="age"` is preserved and does not affect this
   selection. This step does not scale E-fields by dose.
4. Match the winning placement to the selected model type and static/adaptive
   step in the query CSV. Nonmodal simulations are not needed. Multiple runs,
   datasets, or files for the selected scan/placement remain errors; restrict
   the query export instead of choosing one arbitrarily.

The earlier session-to-HDF5 mapping rules still apply: one candidate scan is
selected automatically, while multiple scans for a patient require an explicit
`--subject-scan-map` JSON file, e.g. `{"subj-cat-001": "subj-cat-001-001"}`.
This file is distinct from `SUBJECT_MAP`, which maps scans to FreeSurfer
reconstructions. Neither `series_num` nor `session_num` implies a scan suffix.
BL and BT remain distinct unless explicitly requested with
`--placement-alias BT=BL`; aliases apply after counting clinical placements.

Each unique selected HDF5 supplies one PCA row, indexed by
`simulation_id = dataset_root::modeled_subjid::placement`. Thus one scan can
supply RUL for one course and BT for another. Courses reusing the same HDF5
reuse its predictor row; that field is not duplicated in PCA. Canonical
`subjid` and `subjid_base` remain available in the metadata. Outcomes join only
the explicitly assigned field, never every scan/placement for that patient.

The runner exports `selected_hdf5_inventory.csv` and
`course_hdf5_mapping.csv`, including series, course ID, counting window, modal
counts/fraction, clinical and simulated placement, selected scan, and path.
Use `--selection-only` to generate these tables in `selection_preview/`
without fitting PCA or changing the selection tables for an existing run:

```bash
python run_pca_weighted_global_E_job_synthsr.py \
    --model-type skin_double --analysis-mode adaptive --selection-only

# Add the scan map if multiple candidate scans exist for a patient.
bash run_pca_array_submit.sh --subject-scan-map /path/to/subject_scan_map.json
```

The workbook and query paths remain configurable in the runner, or via
`--cognitive-scores` and `--query-path`; `--stimulus-sheet` defaults to
`stimulus`. The submission wrapper forwards these options and preserves its
default of all three demeaning analyses. `--existing-only` disables only the
added demeaning branches, not modal-placement selection.

For direct all-atlas Python calls, pass `simulation_ids` in HDF5 order and
`course_hdf5_map` from `select_modal_course_hdf5`. Older APIs without these
arguments retain the legacy scan × course matching behavior. The current
Pearson correlations still do not account for repeated courses within people;
their p-values and ordinary Fisher intervals remain descriptive. Simulation,
scan, course, and patient counts are reported separately in the new path.
# Optional raw-E demeaning analyses

The existing spatially weighted/unweighted parcel-P95 analyses, global-E
predictors, and global-P95-weighted PCA remain available with their existing
defaults. Demeaning is an additional, opt-in analysis. It does not replace the
choice of percentile estimator.

Let `q[s,p]` be a parcel's P95 E. The added PCA inputs are:

| `demean_by` | Input for subject s, parcel p | Interpretation |
|---|---|---|
| `brain_mean` | `q[s,p] - mean(E[s, brain elements])` | Parcel upper-tail exposure relative to the subject's mean brain field |
| `parcel_p95_mean` | `q[s,p] - mean(q[s,:])` | Parcel exposure relative to the subject's average parcel P95 |
| `parcel_mean` | `q[s,p] - mean(E[s, parcel p])` | Upper-tail field contrast within that parcel |

All new reference means are **arithmetic**, with equal sample weights. Brain
means use volume tetrahedra with `brain_tags=(1, 2)` by default (WM and GM),
including when the PCA parcels come from a surface atlas. Parcel means use
the volume elements or surface nodes used in that parcel's P95 calculation.
These means are not volume/area weighted; a volume/area-weighted P95 remains
a separate choice. Mean parcel P95 is recalculated over the retained atlas
parcels after explicit ROI exclusions; brain mean uses the selected tissue
tags independently of those exclusions.

The new matrices retain the field's original units and may be signed. They
undergo no division by a global mean, no subsequent within-subject centering,
no global-P95 multiplication, and no parcel variance standardization. PCA
still centers each parcel across subjects. Only `parcel_p95_mean` necessarily
has zero within-subject row means. All three retain sensitivity to a
multiplicative change in a subject's field strength. For `parcel_mean`,
subtracting the mean before computing the percentile gives the same result
as subtracting it afterward (for either supported P95 estimator). This is
an upper-tail contrast metric, not a general measure of variance or a claim
about clinical efficacy.

Run the existing analyses **plus all three additions**, from `05_analysis`:

```bash
python run_pca_weighted_global_E_job_synthsr.py \
    --model-type skin_single --analysis-mode adaptive \
    --demean-by brain_mean parcel_p95_mean parcel_mean
```

The same arguments can be passed through the Slurm array script:

```bash
sbatch run_pca_array.slurm --demean-by brain_mean parcel_p95_mean parcel_mean
```

For the Python all-atlas API, add this keyword to the existing call:

```python
demean_references=("brain_mean", "parcel_p95_mean", "parcel_mean"),
```

Omitting it preserves the existing result branches. Added results have keys
such as `volume_aparc_weighted_demean_brain_mean` and
`surface_HCP_MMP1_unweighted_demean_parcel_mean`. The existing `weighted`
and `unweighted` key suffixes continue to identify the parcel-P95 estimator.
Setting `run_unweighted_sensitivity=False` also skips the new branches that
use unweighted parcel P95s.

Each added result lives in its own `demean_<reference>` subdirectory under
the existing `weighted_p95` or `unweighted_p95_sensitivity` directory. It
exports the exact matrix passed to PCA (`parcel_pca_input.csv`), the subtracted
reference matrix (`parcel_demeaning_reference_E.csv`), scores, loadings,
explained variance, global-E predictors, correlations, figures, and explicit
preprocessing metadata. Added runs report 96%, 97%, 98%, and 99% cumulative
variance cutoffs and correlate PCs up to 96% by default.

Direct use with precomputed statistics:

```python
from simnibs_parcel_analysis import fit_demeaned_parcel_pca

result = fit_demeaned_parcel_pca(parcel_p95, demean_by="brain_mean", global_mean_brain_e=brain_mean)
result = fit_demeaned_parcel_pca(parcel_p95, demean_by="parcel_p95_mean")
result = fit_demeaned_parcel_pca(parcel_p95, demean_by="parcel_mean", parcel_mean_e=parcel_means)
```

References must match the input scan/parcel labels exactly; reordered labels
are aligned, while missing, extra, duplicate, nonfinite, or negative reference
values raise errors. Missing means are not inferred from P95 or imputed.

Compatibility: `fit_parcel_pca(parcel_p95, global_p95_e)` keeps its existing
formula. The older one-argument call now works again and fits the original
normalized, within-subject-demeaned matrix.

All preprocessing branches in the pooled runner use the same explicit
course-to-HDF5 map described above. Correlation p-values and confidence intervals still use
the existing Pearson implementation and do not account for dependence between
repeated observations within people. BH correction is
within each atlas/P95/preprocessing run, not across all added analyses.
