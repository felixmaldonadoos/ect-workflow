"""End-to-end export pipeline for parcel PCA and outcome correlations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .clinical import prepare_scan_course_outcomes
from .pca import (
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
from .summary import P95Method, ParcelSummary


@dataclass(frozen=True)
class AnalysisResult:
    """PCA, aligned outcomes, predictor correlations, and variance cutoffs."""

    summary: ParcelSummary
    outcomes: pd.DataFrame
    pca: ParcelPCA
    global_metrics: pd.DataFrame
    predictors: pd.DataFrame
    correlations: pd.DataFrame
    components_by_threshold: dict[float, int]
    n_pcs_correlated: int


def run_pca_outcome_analysis(
    summary: ParcelSummary,
    cognitive: pd.DataFrame,
    output_dir: str | Path,
    *,
    p95_method: P95Method = "spatial_weighted",
    outcome_col: str = "cgi_change",
    variance_thresholds: Sequence[float] = (0.96, 0.99),
    pc_correlation_threshold: float = 0.96,
    n_pcs_for_correlation: int | None = None,
) -> AnalysisResult:
    """Fit the requested PCA, correlate global/PC predictors, and save outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parcel_values = summary.p95(p95_method)
    outcomes = prepare_scan_course_outcomes(summary.scans, cognitive, outcome_col=outcome_col)
    pca_result = fit_parcel_pca(parcel_values)
    components_by_threshold = {float(value): components_for_variance(pca_result, float(value)) for value in variance_thresholds}

    if n_pcs_for_correlation is None:
        n_pcs_for_correlation = components_for_variance(pca_result, pc_correlation_threshold)
    if not 0 <= n_pcs_for_correlation <= pca_result.scores.shape[1]:
        raise ValueError(f"n_pcs_for_correlation must be between 0 and {pca_result.scores.shape[1]}")

    global_metrics = summary.global_metrics(p95_method)
    predictors = global_and_pc_predictors(global_metrics, pca_result, n_pcs=n_pcs_for_correlation)
    correlations = correlate_predictors(predictors, outcomes, outcome_col=outcome_col)

    summary.scans.to_csv(output_dir / "modeled_scans.csv", index=False)
    summary.qc.to_csv(output_dir / "atlas_assignment_qc.csv")
    summary.p95_unweighted.to_csv(output_dir / "parcel_p95_unweighted.csv")
    summary.p95_spatial_weighted.to_csv(output_dir / "parcel_p95_spatial_weighted.csv")
    summary.parcel_size.to_csv(output_dir / f"parcel_size_{summary.size_unit}.csv")
    summary.parcel_count.to_csv(output_dir / "parcel_sample_count.csv")
    _parcel_qc_table(summary).to_csv(output_dir / "parcel_qc_summary.csv")
    outcomes.to_csv(output_dir / "scan_course_outcomes.csv", index=False)
    global_metrics.to_csv(output_dir / "global_E_metrics.csv")
    pca_result.relative_parcels.to_csv(output_dir / "parcel_relative_E.csv")
    pca_result.subject_demeaned_parcels.to_csv(output_dir / "parcel_pca_input.csv")
    pca_result.scores.to_csv(output_dir / "pca_scores.csv")
    pca_result.loadings.to_csv(output_dir / "pca_loadings.csv")
    pca_result.variance.to_csv(output_dir / "pca_explained_variance.csv")
    predictors.to_csv(output_dir / "global_E_and_PC_predictors.csv")
    correlations.to_csv(output_dir / "global_E_and_PC_correlations.csv")

    settings = {
        "atlas_name": summary.atlas_name,
        "domain": summary.domain,
        "p95_method_for_pca": p95_method,
        "outcome_column": outcome_col,
        "components_by_threshold": {str(key): value for key, value in components_by_threshold.items()},
        "n_pcs_correlated": n_pcs_for_correlation,
        "clinical_observation": "one treatment course per unique (subjid_base, date_start)",
        "predictor_matching": "each modeled scan is matched to every treatment course sharing subjid_base",
        "unmatched_scans": "retained in ROI/PCA outputs and excluded from outcome correlations",
        "correlation": "scan-course Pearson; repeated scans and repeated courses within people are not independent",
    }
    (output_dir / "analysis_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    figures = [
        (plot_pca_variance(pca_result, variance_thresholds)[0], "pca_explained_variance.png"),
        (plot_global_outcome(global_metrics, outcomes, outcome_col=outcome_col)[0], "global_E_vs_outcome.png"),
        (plot_subject_parcel_heatmap(pca_result)[0], "pca_input_heatmap.png"),
        (plot_correlation_bars(correlations)[0], "global_E_and_PC_correlations.png"),
    ]
    n_loading_components = min(3, pca_result.loadings.shape[1])
    loading_components = [f"PC{number}" for number in range(1, n_loading_components + 1)]
    figures.append((plot_top_loadings(pca_result, components=loading_components)[0], "top_PC_loadings.png"))
    for figure, filename in figures:
        figure.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)

    print(f"Atlas/domain: {summary.atlas_name}/{summary.domain}")
    print(f"Modeled scans: {len(summary.scans)}")
    print(f"Modeled scans with treatment courses: {outcomes['subjid'].nunique()}")
    print(f"Unique people: {summary.scans['subjid_base'].nunique()}")
    print(f"Scan-course observations: {len(outcomes)}")
    print(f"Treatment courses: {outcomes['treatment_course_id'].nunique()}")
    print(f"Parcels: {parcel_values.shape[1]}")
    for threshold, count in components_by_threshold.items():
        print(f"PCs for {100 * threshold:.0f}% variance: {count}")
    print(f"PCs correlated with {outcome_col}: PC1-PC{n_pcs_for_correlation}")
    print(correlations.loc[["global_mean_E_unweighted", "global_mean_E_weighted"]])
    return AnalysisResult(
        summary,
        outcomes,
        pca_result,
        global_metrics,
        predictors,
        correlations,
        components_by_threshold,
        n_pcs_for_correlation,
    )


def _parcel_qc_table(summary: ParcelSummary) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"median_size_{summary.size_unit}": summary.parcel_size.median(axis=0),
            f"minimum_size_{summary.size_unit}": summary.parcel_size.min(axis=0),
            "median_sample_count": summary.parcel_count.median(axis=0),
            "minimum_sample_count": summary.parcel_count.min(axis=0),
        }
    ).rename_axis("parcel")
