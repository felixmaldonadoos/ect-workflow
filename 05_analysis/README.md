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
