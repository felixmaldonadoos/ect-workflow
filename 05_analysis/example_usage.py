"""Single-block usage for DK, Destrieux, and HCP-MMP1 analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, Sequence

import pandas as pd

from simnibs_parcel_analysis import (
    build_surface_roi_summary,
    build_volume_roi_summary,
    compute_cgi_change,
    inspect_hdf5_mesh,
    run_pca_outcome_analysis,
    validate_treatment_courses,
)


def run_all_atlases(
    hdf5_files: Sequence[str | Path],
    cognitive: pd.DataFrame,
    subjects_dir: str | Path,
    output_dir: str | Path,
    *,
    subject_map: Mapping[str, str] | None = None,
    freesurfer_lut: str | Path | None = None,
    outcome_col: str = "cgi_change",
    change_direction: Literal["end_minus_start", "start_minus_end"] = "end_minus_start",
    include_surface_hcp: bool = True,
    run_unweighted_sensitivity: bool = True,
) -> dict[str, object]:
    """Run primary volumetric analyses and the optional HCP surface analysis."""
    if outcome_col not in cognitive.columns:
        cognitive = compute_cgi_change(cognitive, outcome_col=outcome_col, direction=change_direction)
    cognitive = validate_treatment_courses(cognitive, outcome_col=outcome_col)
    output_dir = Path(output_dir)
    results = {}

    print("First HDF5 mesh:")
    print(inspect_hdf5_mesh(hdf5_files[0]))

    for atlas_name in ("aparc", "a2009s"):
        summary = build_volume_roi_summary(
            hdf5_files,
            subjects_dir,
            atlas_name=atlas_name,
            subject_map=subject_map,
            freesurfer_lut=freesurfer_lut,
            field_name="magnE_mean",
            percentile=95,
            min_elements_per_roi=20,
            include_cortex=True,
            include_brainstem=False,
        )
        primary = run_pca_outcome_analysis(
            summary,
            cognitive,
            output_dir / f"volume_{atlas_name}" / "weighted_p95",
            p95_method="spatial_weighted",
            outcome_col=outcome_col,
            variance_thresholds=(0.96, 0.99),
            pc_correlation_threshold=0.96,
        )
        results[f"volume_{atlas_name}_weighted"] = primary
        if run_unweighted_sensitivity:
            results[f"volume_{atlas_name}_unweighted"] = run_pca_outcome_analysis(
                summary,
                cognitive,
                output_dir / f"volume_{atlas_name}" / "unweighted_p95_sensitivity",
                p95_method="unweighted",
                outcome_col=outcome_col,
                variance_thresholds=(0.96, 0.99),
                pc_correlation_threshold=0.96,
            )

    if include_surface_hcp:
        hcp_summary = build_surface_roi_summary(
            hdf5_files,
            atlas_name="HCP_MMP1",
            field_name="magnE_mean",
            percentile=95,
            min_nodes_per_roi=20,
        )
        results["surface_HCP_MMP1_weighted"] = run_pca_outcome_analysis(
            hcp_summary,
            cognitive,
            output_dir / "surface_HCP_MMP1" / "weighted_p95",
            p95_method="spatial_weighted",
            outcome_col=outcome_col,
            variance_thresholds=(0.96, 0.99),
            pc_correlation_threshold=0.96,
        )
        if run_unweighted_sensitivity:
            results["surface_HCP_MMP1_unweighted"] = run_pca_outcome_analysis(
                hcp_summary,
                cognitive,
                output_dir / "surface_HCP_MMP1" / "unweighted_p95_sensitivity",
                p95_method="unweighted",
                outcome_col=outcome_col,
                variance_thresholds=(0.96, 0.99),
                pc_correlation_threshold=0.96,
            )
    return results


# Notebook usage:
#
# SUBJECTS_DIR = Path("/path/to/freesurfer/subjectsdir")
# OUTPUT_DIR = Path("/path/to/output/atlas_pca")
#
# # Usually this can remain empty. Add an entry only when a base person has
# # multiple recon-all directories and the automatic match would be ambiguous.
# SUBJECT_MAP = {
#     # "subj-cat-001-002": "subj-cat-001-001",
# }
#
# results = run_all_atlases(
#     hdf5_paths_adaptive,
#     df_cog,
#     SUBJECTS_DIR,
#     OUTPUT_DIR,
#     subject_map=SUBJECT_MAP,
#     include_surface_hcp=True,
#     run_unweighted_sensitivity=True,
# )
#
# dk = results["volume_aparc_weighted"]
# print(dk.components_by_threshold)
# print(dk.correlations.loc[["global_mean_E_unweighted", "global_mean_E_weighted"]])
