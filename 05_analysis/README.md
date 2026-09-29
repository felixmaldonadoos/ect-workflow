# SimNIBS anatomical ROI and PCA analysis

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

PCA remains scans × parcels. For outcome correlations, every modeled scan is
matched to every treatment course sharing its `subjid_base`, producing a unique
`observation_id = modeled scan + course start date`. The current Pearson
correlations treat these scan-course rows as independent. Because scans and
courses repeat within people, those p-values and ordinary Fisher intervals are
descriptive; clustered or mixed-effects inference should replace them in the
final inferential model.

A modeled scan whose `subjid_base` has no complete treatment course is retained
in the ROI summaries and PCA, but excluded from outcome correlations with an
explicit warning listing the affected scans and base IDs.
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

Treatment-course matching is unchanged: scans still expand to all courses
sharing the base ID. Correlation p-values and confidence intervals still use
the existing Pearson implementation and do not account for dependence between
repeated observations within people. BH correction is
within each atlas/P95/preprocessing run, not across all added analyses.
