"""Run DK, Destrieux, and HCP-MMP1 PCA weighted by global brain P95 E."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence
from simnibs_parcel_analysis.identifiers import infer_subject_id
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simnibs
import re 

SUBJECT_ID_PATTERN = re.compile(r"subj-[A-Za-z0-9]+-\d{3}-\d{3}")

from simnibs_parcel_analysis import (
    build_surface_roi_summary,
    build_volume_roi_summary,
    compute_cgi_change,
    inspect_hdf5_mesh,
    validate_treatment_courses,
)
from simnibs_parcel_analysis.clinical import prepare_scan_course_outcomes
from simnibs_parcel_analysis.pca import (
    ParcelPCA,
    components_for_variance,
    correlate_predictors,
    fit_parcel_pca,
    global_and_pc_predictors,
    plot_correlation_bars,
    plot_global_outcome,
    plot_pca_variance,
    plot_subject_parcel_heatmap,
    plot_top_loadings,
)
from simnibs_parcel_analysis.summary import P95Method, ParcelSummary


@dataclass(frozen=True)
class GlobalP95WeightedAnalysisResult:
    """Outputs from one global-P95-weighted atlas analysis."""

    summary: ParcelSummary
    outcomes: pd.DataFrame
    pca: ParcelPCA
    global_p95_e: pd.Series
    global_metrics: pd.DataFrame
    predictors: pd.DataFrame
    correlations: pd.DataFrame
    components_by_threshold: dict[float, int]
    n_pcs_correlated: int




def build_global_p95_e(
    hdf5_files: Sequence[str | Path],
    *,
    field_name: str = "magnE_mean",
    mesh_key: str = "mesh_roi",
    brain_tags: Sequence[int] = (1, 2),
    percentile: float = 95.0,
) -> pd.Series:
    """Calculate global P95 E across brain tetrahedra for every modeled scan."""
    hdf5_files = [Path(path) for path in hdf5_files]

    if not hdf5_files:
        raise ValueError("hdf5_files is empty")

    if isinstance(percentile, bool) or not isinstance(
        percentile,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("percentile must be numeric")

    if not 0 <= percentile <= 100:
        raise ValueError(
            f"percentile must be between 0 and 100, got {percentile}"
        )

    if not brain_tags:
        raise ValueError("brain_tags cannot be empty")

    brain_tags = tuple(int(tag) for tag in brain_tags)
    subject_ids = [infer_subject_id(path) for path in hdf5_files]
    subject_index = pd.Index(subject_ids, name="subjid")

    if subject_index.has_duplicates:
        duplicates = (
            subject_index[
                subject_index.duplicated()
            ].unique().tolist()
        )
        raise ValueError(
            f"Duplicate modeled-scan IDs in hdf5_files: {duplicates}"
        )

    global_p95_values = {}

    for subject_id, hdf5_file in zip(
        subject_ids,
        hdf5_files,
        strict=True,
    ):
        if not hdf5_file.is_file():
            raise FileNotFoundError(
                f"HDF5 file does not exist: {hdf5_file}"
            )

        mesh = simnibs.Msh.read_hdf5(
            str(hdf5_file),
            mesh_key,
        )

        if field_name not in mesh.field:
            available = sorted(mesh.field.keys())
            raise KeyError(
                f"Field {field_name!r} was not found for {subject_id}; "
                f"available fields: {available}"
            )

        element_field_names = {
            field.field_name
            for field in mesh.elmdata
        }

        if field_name not in element_field_names:
            raise ValueError(
                f"Field {field_name!r} must be ElementData for {subject_id}"
            )

        field = np.asarray(
            mesh.field[field_name].value,
            dtype=float,
        ).squeeze()

        if field.ndim != 1:
            raise ValueError(
                f"Field {field_name!r} must be scalar for {subject_id}, "
                f"got shape {field.shape}"
            )

        if field.size != mesh.elm.nr:
            raise ValueError(
                f"Field {field_name!r} contains {field.size} values, "
                f"but mesh.elm.nr is {mesh.elm.nr} for {subject_id}"
            )

        if not np.isfinite(field).all():
            raise ValueError(
                f"Field {field_name!r} contains non-finite values "
                f"for {subject_id}"
            )

        if (field < 0).any():
            raise ValueError(
                f"Field {field_name!r} contains negative values "
                f"for {subject_id}"
            )

        element_tags = np.asarray(
            mesh.elm.tag1,
            dtype=int,
        ).squeeze()
        element_types = np.asarray(
            mesh.elm.elm_type,
            dtype=int,
        ).squeeze()

        if element_tags.ndim != 1 or element_tags.size != field.size:
            raise ValueError(
                f"Element tags have shape {element_tags.shape}; "
                f"expected ({field.size},) for {subject_id}"
            )

        if element_types.ndim != 1 or element_types.size != field.size:
            raise ValueError(
                f"Element types have shape {element_types.shape}; "
                f"expected ({field.size},) for {subject_id}"
            )

        tetrahedron_mask = element_types == 4
        brain_mask = tetrahedron_mask & np.isin(
            element_tags,
            brain_tags,
        )

        if not brain_mask.any():
            available_tags = np.unique(
                element_tags[tetrahedron_mask]
            ).tolist()

            raise ValueError(
                f"No tetrahedra with brain tags {list(brain_tags)} "
                f"were found for {subject_id}; available tetrahedral "
                f"tags: {available_tags}"
            )

        brain_field = field[brain_mask]

        global_p95_values[subject_id] = float(
            np.percentile(
                brain_field,
                percentile,
            )
        )

        print(
            f"{subject_id}: "
            f"global P{percentile:g} E = "
            f"{global_p95_values[subject_id]:.6g}; "
            f"brain tetrahedra = {brain_field.size}"
        )

    global_p95_e = pd.Series(
        global_p95_values,
        name=f"global_p{percentile:g}_E",
        dtype=float,
    )
    global_p95_e.index.name = "subjid"

    return global_p95_e


def run_global_p95_weighted_pca_outcome_analysis(
    summary: ParcelSummary,
    cognitive: pd.DataFrame,
    global_p95_e: pd.Series,
    output_dir: str | Path,
    *,
    p95_method: P95Method = "spatial_weighted",
    outcome_col: str = "cgi_change",
    variance_thresholds: Sequence[float] = (0.96, 0.99),
    pc_correlation_threshold: float = 0.96,
    n_pcs_for_correlation: int | None = None,
    global_percentile: float = 95.0,
    brain_tags: Sequence[int] = (1, 2),
) -> GlobalP95WeightedAnalysisResult:
    """Run one atlas analysis using global-P95-weighted PCA."""
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parcel_values = summary.p95(p95_method)
    outcomes = prepare_scan_course_outcomes(
        summary.scans,
        cognitive,
        outcome_col=outcome_col,
    )

    missing_global = parcel_values.index.difference(
        global_p95_e.index
    ).tolist()
    extra_global = global_p95_e.index.difference(
        parcel_values.index
    ).tolist()

    if missing_global or extra_global:
        raise ValueError(
            "Modeled-scan mismatch between the parcel summary and "
            f"global P95 E values; missing={missing_global}, "
            f"extra={extra_global}"
        )

    global_p95_e = global_p95_e.reindex(
        parcel_values.index
    )

    pca_result = fit_parcel_pca(
        parcel_values,
        global_p95_e,
    )

    components_by_threshold = {
        float(threshold): components_for_variance(
            pca_result,
            float(threshold),
        )
        for threshold in variance_thresholds
    }

    if n_pcs_for_correlation is None:
        n_pcs_for_correlation = components_for_variance(
            pca_result,
            pc_correlation_threshold,
        )

    if not 0 <= n_pcs_for_correlation <= pca_result.scores.shape[1]:
        raise ValueError(
            f"n_pcs_for_correlation must be between 0 and "
            f"{pca_result.scores.shape[1]}"
        )

    # Preserve the same global predictors used by the original pipeline.
    # Only the PCA input is newly weighted by global brain P95 E.
    global_metrics = summary.global_metrics(
        p95_method
    )

    predictors = global_and_pc_predictors(
        global_metrics,
        pca_result,
        n_pcs=n_pcs_for_correlation,
    )

    correlations = correlate_predictors(
        predictors,
        outcomes,
        outcome_col=outcome_col,
    )

    summary.scans.to_csv(
        output_dir / "modeled_scans.csv",
        index=False,
    )
    summary.qc.to_csv(
        output_dir / "atlas_assignment_qc.csv"
    )
    summary.p95_unweighted.to_csv(
        output_dir / "parcel_p95_unweighted.csv"
    )
    summary.p95_spatial_weighted.to_csv(
        output_dir / "parcel_p95_spatial_weighted.csv"
    )
    summary.parcel_size.to_csv(
        output_dir / f"parcel_size_{summary.size_unit}.csv"
    )
    summary.parcel_count.to_csv(
        output_dir / "parcel_sample_count.csv"
    )
    _parcel_qc_table(summary).to_csv(
        output_dir / "parcel_qc_summary.csv"
    )

    outcomes.to_csv(
        output_dir / "scan_course_outcomes.csv",
        index=False,
    )
    global_metrics.to_csv(
        output_dir / "global_E_metrics.csv"
    )
    pca_result.global_mean_e.to_frame().to_csv(
        output_dir / "pca_normalization_global_mean_E.csv"
    )
    pca_result.global_p95_e.to_frame().to_csv(
        output_dir / "global_p95_E_weights.csv"
    )
    pca_result.relative_parcels.to_csv(
        output_dir / "parcel_relative_E.csv"
    )
    pca_result.subject_demeaned_parcels.to_csv(
        output_dir / "parcel_subject_demeaned_E.csv"
    )

    # This is the actual matrix supplied to PCA.
    pca_result.global_p95_weighted_parcels.to_csv(
        output_dir / "parcel_pca_input.csv"
    )

    pca_result.scores.to_csv(
        output_dir / "pca_scores.csv"
    )
    pca_result.loadings.to_csv(
        output_dir / "pca_loadings.csv"
    )
    pca_result.variance.to_csv(
        output_dir / "pca_explained_variance.csv"
    )
    predictors.to_csv(
        output_dir / "global_E_and_PC_predictors.csv"
    )
    correlations.to_csv(
        output_dir / "global_E_and_PC_correlations.csv"
    )

    settings = {
        "atlas_name": summary.atlas_name,
        "domain": summary.domain,
        "parcel_statistic": (
            f"{p95_method} 95th percentile EF within each parcel"
        ),
        "p95_method_for_pca": p95_method,
        "pca_normalization": (
            "parcel P95 divided by arithmetic mean of parcel P95 "
            "values within each modeled scan"
        ),
        "within_scan_demeaning": True,
        "global_p95_weighting": {
            "enabled": True,
            "percentile": float(global_percentile),
            "field": "magnE_mean",
            "domain": "brain tetrahedra",
            "brain_tags": [
                int(tag)
                for tag in brain_tags
            ],
            "operation": (
                "within-scan demeaned relative parcel P95 multiplied "
                "by global brain P95 E"
            ),
        },
        "pca_input_file": "parcel_pca_input.csv",
        "outcome_column": outcome_col,
        "components_by_threshold": {
            str(key): value
            for key, value in components_by_threshold.items()
        },
        "n_pcs_correlated": n_pcs_for_correlation,
        "clinical_observation": (
            "each unique (subjid_base, date_start) defines one "
            "treatment course; subjects may have multiple courses"
        ),
        "predictor_matching": (
            "each modeled scan is matched to every treatment course "
            "sharing subjid_base"
        ),
        "unmatched_scans": (
            "retained in ROI/PCA outputs and excluded from "
            "outcome correlations"
        ),
        "correlation": (
            "scan-course Pearson; repeated scans and repeated courses "
            "within people are not independent"
        ),
    }

    settings_path = output_dir / "analysis_settings.json"
    settings_path.write_text(
        json.dumps(
            settings,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    figures = [
        (
            plot_pca_variance(
                pca_result,
                variance_thresholds,
            )[0],
            "pca_explained_variance.png",
        ),
        (
            plot_global_outcome(
                global_metrics,
                outcomes,
                outcome_col=outcome_col,
            )[0],
            "global_E_vs_outcome.png",
        ),
        (
            plot_subject_parcel_heatmap(
                pca_result
            )[0],
            "pca_input_heatmap.png",
        ),
        (
            plot_correlation_bars(
                correlations
            )[0],
            "global_E_and_PC_correlations.png",
        ),
    ]

    n_loading_components = min(
        3,
        pca_result.loadings.shape[1],
    )
    loading_components = [
        f"PC{number}"
        for number in range(
            1,
            n_loading_components + 1,
        )
    ]

    figures.append(
        (
            plot_top_loadings(
                pca_result,
                components=loading_components,
            )[0],
            "top_PC_loadings.png",
        )
    )

    for figure, filename in figures:
        figure.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

    print(
        f"Atlas/domain: "
        f"{summary.atlas_name}/{summary.domain}"
    )
    print(
        f"Parcel P95 method: {p95_method}"
    )
    print(
        f"Global PCA weight: brain P{global_percentile:g} E"
    )
    print(
        f"Modeled scans: {len(summary.scans)}"
    )
    print(
        f"Modeled scans with treatment courses: "
        f"{outcomes['subjid'].nunique()}"
    )
    print(
        f"Unique people: "
        f"{summary.scans['subjid_base'].nunique()}"
    )
    print(
        f"Scan-course observations: {len(outcomes)}"
    )
    print(
        f"Treatment courses: "
        f"{outcomes['treatment_course_id'].nunique()}"
    )
    print(
        f"Parcels: {parcel_values.shape[1]}"
    )

    for threshold, count in components_by_threshold.items():
        print(
            f"PCs for {100 * threshold:.0f}% variance: {count}"
        )

    if n_pcs_for_correlation:
        print(
            f"PCs correlated with {outcome_col}: "
            f"PC1-PC{n_pcs_for_correlation}"
        )
    else:
        print(
            f"No PCs correlated with {outcome_col}"
        )

    global_rows = [
        predictor
        for predictor in (
            "global_mean_E_unweighted",
            "global_mean_E_weighted",
        )
        if predictor in correlations.index
    ]

    if global_rows:
        print(
            correlations.loc[global_rows]
        )

    return GlobalP95WeightedAnalysisResult(
        summary=summary,
        outcomes=outcomes,
        pca=pca_result,
        global_p95_e=global_p95_e,
        global_metrics=global_metrics,
        predictors=predictors,
        correlations=correlations,
        components_by_threshold=components_by_threshold,
        n_pcs_correlated=n_pcs_for_correlation,
    )


def run_all_atlases_global_p95_weighted(
    hdf5_files: Sequence[str | Path],
    cognitive: pd.DataFrame,
    subjects_dir: str | Path,
    output_dir: str | Path,
    *,
    subject_map: Mapping[str, str] | None = None,
    freesurfer_lut: str | Path | None = None,
    outcome_col: str = "cgi_change",
    change_direction: Literal[
        "end_minus_start",
        "start_minus_end",
    ] = "end_minus_start",
    include_surface_hcp: bool = True,
    run_unweighted_sensitivity: bool = True,
    global_percentile: float = 95.0,
    brain_tags: Sequence[int] = (1, 2),
    field_name: str = "magnE_mean",
    mesh_key: str = "mesh_roi",
) -> dict[str, GlobalP95WeightedAnalysisResult]:
    """Run all atlas analyses with global-brain-P95-weighted PCA."""
    hdf5_files = [
        Path(path)
        for path in hdf5_files
    ]

    if not hdf5_files:
        raise ValueError("hdf5_files is empty")

    if outcome_col not in cognitive.columns:
        cognitive = compute_cgi_change(
            cognitive,
            outcome_col=outcome_col,
            direction=change_direction,
        )

    cognitive = validate_treatment_courses(
        cognitive,
        outcome_col=outcome_col,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("First HDF5 mesh:")
    print(
        inspect_hdf5_mesh(
            hdf5_files[0]
        )
    )

    # The same scan-level global P95 values are used for every atlas.
    global_p95_e = build_global_p95_e(
        hdf5_files,
        field_name=field_name,
        mesh_key=mesh_key,
        brain_tags=brain_tags,
        percentile=global_percentile,
    )

    results = {}

    for atlas_name in (
        "aparc",
        "a2009s",
    ):
        summary = build_volume_roi_summary(
            hdf5_files,
            subjects_dir,
            atlas_name=atlas_name,
            subject_map=subject_map,
            freesurfer_lut=freesurfer_lut,
            field_name=field_name,
            percentile=95,
            min_elements_per_roi=20,
            include_cortex=True,
            include_brainstem=False,
        )

        results[
            f"volume_{atlas_name}_weighted"
        ] = run_global_p95_weighted_pca_outcome_analysis(
            summary,
            cognitive,
            global_p95_e,
            output_dir
            / f"volume_{atlas_name}"
            / "weighted_p95",
            p95_method="spatial_weighted",
            outcome_col=outcome_col,
            variance_thresholds=(0.96, 0.99),
            pc_correlation_threshold=0.96,
            global_percentile=global_percentile,
            brain_tags=brain_tags,
        )

        if run_unweighted_sensitivity:
            results[
                f"volume_{atlas_name}_unweighted"
            ] = run_global_p95_weighted_pca_outcome_analysis(
                summary,
                cognitive,
                global_p95_e,
                output_dir
                / f"volume_{atlas_name}"
                / "unweighted_p95_sensitivity",
                p95_method="unweighted",
                outcome_col=outcome_col,
                variance_thresholds=(0.96, 0.99),
                pc_correlation_threshold=0.96,
                global_percentile=global_percentile,
                brain_tags=brain_tags,
            )

    if include_surface_hcp:
        hcp_summary = build_surface_roi_summary(
            hdf5_files,
            atlas_name="HCP_MMP1",
            field_name=field_name,
            percentile=95,
            min_nodes_per_roi=20,
        )

        results[
            "surface_HCP_MMP1_weighted"
        ] = run_global_p95_weighted_pca_outcome_analysis(
            hcp_summary,
            cognitive,
            global_p95_e,
            output_dir
            / "surface_HCP_MMP1"
            / "weighted_p95",
            p95_method="spatial_weighted",
            outcome_col=outcome_col,
            variance_thresholds=(0.96, 0.99),
            pc_correlation_threshold=0.96,
            global_percentile=global_percentile,
            brain_tags=brain_tags,
        )

        if run_unweighted_sensitivity:
            results[
                "surface_HCP_MMP1_unweighted"
            ] = run_global_p95_weighted_pca_outcome_analysis(
                hcp_summary,
                cognitive,
                global_p95_e,
                output_dir
                / "surface_HCP_MMP1"
                / "unweighted_p95_sensitivity",
                p95_method="unweighted",
                outcome_col=outcome_col,
                variance_thresholds=(0.96, 0.99),
                pc_correlation_threshold=0.96,
                global_percentile=global_percentile,
                brain_tags=brain_tags,
            )

    return results


def _parcel_qc_table(
    summary: ParcelSummary,
) -> pd.DataFrame:
    """Create parcel-level size and sample-count QC summaries."""
    return pd.DataFrame(
        {
            f"median_size_{summary.size_unit}": (
                summary.parcel_size.median(axis=0)
            ),
            f"minimum_size_{summary.size_unit}": (
                summary.parcel_size.min(axis=0)
            ),
            "median_sample_count": (
                summary.parcel_count.median(axis=0)
            ),
            "minimum_sample_count": (
                summary.parcel_count.min(axis=0)
            ),
        }
    ).rename_axis("parcel")


# Notebook usage:
#
# SUBJECTS_DIR = Path("/path/to/freesurfer/subjectsdir")
# OUTPUT_DIR = Path("/path/to/output/atlas_pca_global_p95_weighted")
#
# SUBJECT_MAP = {
#     # Add entries only when automatic FreeSurfer matching is ambiguous.
#     # "subj-cat-001-002": "subj-cat-001-001",
# }
#
# results = run_all_atlases_global_p95_weighted(
#     hdf5_paths_adaptive,
#     df_cog,
#     SUBJECTS_DIR,
#     OUTPUT_DIR,
#     subject_map=SUBJECT_MAP,
#     include_surface_hcp=True,
#     run_unweighted_sensitivity=True,
#     global_percentile=95,
#     brain_tags=(1, 2),
#     field_name="magnE_mean",
#     mesh_key="mesh_roi",
# )
#
# dk = results["volume_aparc_weighted"]
#
# print(dk.global_p95_e)
# print(dk.pca.global_p95_weighted_parcels)
# print(dk.components_by_threshold)
# print(dk.correlations)