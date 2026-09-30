"""Parcel P95 summaries, global-P95 weighting, PCA, correlations, and plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm, pearsonr
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class ParcelPCA:
    """Parcel matrices and fitted PCA outputs."""

    parcel_p95: pd.DataFrame
    global_mean_e: pd.Series
    global_p95_e: pd.Series | None
    relative_parcels: pd.DataFrame
    subject_demeaned_parcels: pd.DataFrame
    global_p95_weighted_parcels: pd.DataFrame
    scores: pd.DataFrame
    loadings: pd.DataFrame
    variance: pd.DataFrame
    model: PCA

    @property
    def pca_input(self) -> pd.DataFrame:
        """Matrix supplied to PCA, before its across-scan centering."""
        return self.global_p95_weighted_parcels

    @property
    def input_label(self) -> str:
        if self.global_p95_e is None:
            return "Demeaned relative parcel P95 E (dimensionless)"
        return "Demeaned relative parcel P95 E × global brain P95 E"


DemeanReference = Literal["brain_mean", "parcel_p95_mean", "parcel_mean"]
DEMEAN_REFERENCES = ("brain_mean", "parcel_p95_mean", "parcel_mean")
DEMEAN_DESCRIPTIONS = {
    "brain_mean": "Parcel P95 E minus arithmetic mean E across selected brain tetrahedra",
    "parcel_p95_mean": "Parcel P95 E minus within-scan arithmetic mean of parcel P95 values",
    "parcel_mean": "Parcel P95 E minus arithmetic mean of raw E samples in that parcel",
}


@dataclass(frozen=True)
class DemeanedParcelPCA:
    """PCA of signed parcel P95 deviations, without normalization or P95 scaling."""

    parcel_p95: pd.DataFrame
    reference_e: pd.DataFrame
    demeaned_parcels: pd.DataFrame
    demean_by: DemeanReference
    scores: pd.DataFrame
    loadings: pd.DataFrame
    variance: pd.DataFrame
    model: PCA

    @property
    def pca_input(self) -> pd.DataFrame:
        return self.demeaned_parcels

    @property
    def input_label(self) -> str:
        return DEMEAN_DESCRIPTIONS[self.demean_by]

def fit_demeaned_parcel_pca(
    parcel_p95: pd.DataFrame,
    *,
    demean_by: DemeanReference,
    global_mean_brain_e: pd.Series | None = None,
    parcel_mean_e: pd.DataFrame | None = None,
) -> DemeanedParcelPCA:
    """Subtract one explicit reference from raw parcel P95 values and fit PCA.

    ``brain_mean`` uses arithmetic mean E over selected brain tetrahedra.
    ``parcel_p95_mean`` uses the arithmetic mean across this scan's parcel P95s.
    ``parcel_mean`` uses arithmetic means of raw samples within each parcel.
    The latter is equivalent to centering samples before taking their P95.
    Signed deviations are valid. No division, additional row demeaning,
    spatial standardization, or global-P95 multiplication is performed.
    """
    if demean_by not in DEMEAN_REFERENCES:
        raise ValueError(f"demean_by must be one of {DEMEAN_REFERENCES}, got {demean_by!r}")
    _validate_subject_parcel_frame(parcel_p95)
    parcels = parcel_p95.astype(float).copy()
    if demean_by == "parcel_p95_mean":
        reference = pd.DataFrame(np.repeat(parcels.mean(axis=1).to_numpy()[:, None], parcels.shape[1], axis=1),
                                 index=parcels.index, columns=parcels.columns)
    elif demean_by == "brain_mean":
        if not isinstance(global_mean_brain_e, pd.Series):
            raise TypeError("brain_mean requires global_mean_brain_e as a pandas Series")
        _require_matching_labels(global_mean_brain_e.index, parcels.index, "global_mean_brain_e scans")
        values = pd.to_numeric(global_mean_brain_e.reindex(parcels.index), errors="raise").to_numpy(dtype=float)
        reference = pd.DataFrame(np.repeat(values[:, None], parcels.shape[1], axis=1),
                                 index=parcels.index, columns=parcels.columns)
    else:
        if not isinstance(parcel_mean_e, pd.DataFrame):
            raise TypeError("parcel_mean requires parcel_mean_e as a pandas DataFrame")
        _require_matching_labels(parcel_mean_e.index, parcels.index, "parcel_mean_e scans")
        _require_matching_labels(parcel_mean_e.columns, parcels.columns, "parcel_mean_e parcels")
        reference = parcel_mean_e.reindex(index=parcels.index, columns=parcels.columns).astype(float).copy()
    reference_values = reference.to_numpy(dtype=float)
    if not np.isfinite(reference_values).all() or (reference_values < 0).any():
        raise ValueError("Demeaning reference must contain finite, nonnegative raw E values")
    demeaned = parcels - reference
    values = demeaned.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Demeaned PCA input contains non-finite values")
    total_variance = demeaned.var(axis=0, ddof=1).sum()
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise ValueError("Demeaned parcel profiles have no finite positive between-scan variance")
    model = PCA(svd_solver="full")
    score_values = model.fit_transform(values)
    names = [f"PC{i + 1}" for i in range(score_values.shape[1])]
    scores = pd.DataFrame(score_values, index=parcels.index.copy(), columns=names)
    scores.index.name = parcels.index.name or "subjid"
    loadings = pd.DataFrame(model.components_.T, index=parcels.columns.copy(), columns=names)
    loadings.index.name = parcels.columns.name or "parcel"
    variance = pd.DataFrame({"explained_variance": model.explained_variance_,
                             "explained_variance_ratio": model.explained_variance_ratio_,
                             "cumulative_variance_ratio": np.cumsum(model.explained_variance_ratio_)},
                            index=pd.Index(names, name="component"))
    return DemeanedParcelPCA(parcels, reference, demeaned, demean_by, scores, loadings, variance, model)


def _require_matching_labels(actual: pd.Index, expected: pd.Index, label: str) -> None:
    """Allow reordering but never duplicate, discard, or invent labels."""
    if actual.has_duplicates:
        raise ValueError(f"Duplicate labels in {label}: {actual[actual.duplicated()].tolist()}")
    missing, extra = expected.difference(actual).tolist(), actual.difference(expected).tolist()
    if missing or extra:
        raise ValueError(f"Mismatched {label}: missing={missing}, extra={extra}")


def calculate_parcel_p95(
    field_values: Sequence[float] | np.ndarray,
    parcel_masks: Mapping[str, Sequence[bool] | np.ndarray],
    *,
    subject_id: str,
    percentile: float = 95.0,
) -> pd.Series:
    """Calculate one unweighted EF percentile per parcel for one modeled scan."""
    _validate_percentile(percentile)

    if not isinstance(subject_id, str) or not subject_id.strip():
        raise ValueError("subject_id must be a non-empty string")

    field = _validate_field_values(field_values, subject_id=subject_id)

    if not parcel_masks:
        raise ValueError(f"No parcel masks were provided for {subject_id}")

    parcel_statistics = {}
    coverage = np.zeros(field.size, dtype=np.int32)

    for parcel_name in sorted(parcel_masks):
        if not isinstance(parcel_name, str) or not parcel_name:
            raise ValueError(f"Parcel names must be non-empty strings for {subject_id}")

        mask = np.asarray(parcel_masks[parcel_name], dtype=bool).squeeze()

        if mask.ndim != 1 or mask.size != field.size:
            raise ValueError(
                f"Mask for parcel {parcel_name!r} has shape {mask.shape}; "
                f"expected ({field.size},) for {subject_id}"
            )

        if not mask.any():
            raise ValueError(f"Parcel {parcel_name!r} is empty for {subject_id}")

        coverage += mask
        parcel_statistics[parcel_name] = float(np.percentile(field[mask], percentile))

    overlapping_locations = int(np.count_nonzero(coverage > 1))

    if overlapping_locations:
        raise ValueError(
            f"Parcel masks overlap at {overlapping_locations} spatial locations "
            f"for {subject_id}"
        )

    return pd.Series(parcel_statistics, name=subject_id, dtype=float)


def calculate_global_e_percentile(
    field_values: Sequence[float] | np.ndarray,
    *,
    subject_id: str,
    brain_mask: Sequence[bool] | np.ndarray | None = None,
    percentile: float = 95.0,
) -> float:
    """Calculate a global EF percentile across all selected brain elements."""
    _validate_percentile(percentile)
    field = _validate_field_values(field_values, subject_id=subject_id)

    if brain_mask is not None:
        mask = np.asarray(brain_mask, dtype=bool).squeeze()

        if mask.ndim != 1 or mask.size != field.size:
            raise ValueError(
                f"brain_mask has shape {mask.shape}; expected ({field.size},) "
                f"for {subject_id}"
            )

        if not mask.any():
            raise ValueError(f"brain_mask selects no elements for {subject_id}")

        field = field[mask]

    return float(np.percentile(field, percentile))


def build_parcel_p95_matrix(subject_statistics: Sequence[pd.Series]) -> pd.DataFrame:
    """Combine per-scan parcel P95 summaries into a scans × parcels matrix."""
    if not subject_statistics:
        raise ValueError("subject_statistics is empty")

    first = subject_statistics[0]

    if not isinstance(first, pd.Series):
        raise TypeError("Every entry in subject_statistics must be a pandas Series")

    expected_parcels = pd.Index(first.index.astype(str))

    if expected_parcels.empty:
        raise ValueError("The first modeled scan contains no parcels")

    if expected_parcels.has_duplicates:
        duplicates = expected_parcels[expected_parcels.duplicated()].unique().tolist()
        raise ValueError(f"The first modeled scan contains duplicate parcels: {duplicates}")

    subject_ids = []
    aligned_rows = []

    for position, series in enumerate(subject_statistics):
        if not isinstance(series, pd.Series):
            raise TypeError(
                f"subject_statistics[{position}] must be a pandas Series, "
                f"got {type(series).__name__}"
            )

        subject_id = series.name

        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError(
                f"subject_statistics[{position}] must have a non-empty subject ID as its name"
            )

        series = series.copy()
        series.index = series.index.astype(str)

        if series.index.has_duplicates:
            duplicates = series.index[series.index.duplicated()].unique().tolist()
            raise ValueError(f"Duplicate parcels for {subject_id}: {duplicates}")

        missing = expected_parcels.difference(series.index).tolist()
        extra = series.index.difference(expected_parcels).tolist()

        if missing or extra:
            raise ValueError(
                f"Parcel mismatch for {subject_id}; "
                f"missing parcels={missing}, extra parcels={extra}"
            )

        subject_ids.append(subject_id)
        aligned_rows.append(series.reindex(expected_parcels).rename(subject_id))

    subject_index = pd.Index(subject_ids, name="subjid")

    if subject_index.has_duplicates:
        duplicates = subject_index[subject_index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate modeled-scan IDs: {duplicates}")

    parcel_p95 = pd.DataFrame(aligned_rows, index=subject_index)
    parcel_p95.columns.name = "parcel"
    _validate_subject_parcel_frame(parcel_p95)

    return parcel_p95


def fit_parcel_pca(
    parcel_p95: pd.DataFrame,
    global_p95_e: pd.Series | None = None,
) -> ParcelPCA:
    """Normalize parcel P95 values, apply global-P95 weights, and fit PCA.

    Processing order:

    1. Begin with one P95 EF value per modeled scan and parcel.
    2. Calculate the arithmetic mean across parcel P95 values for each scan.
    3. Divide each parcel P95 by the corresponding scan-level mean.
    4. Demean the normalized parcels within each scan.
    5. Multiply the demeaned profile by the scan's global brain P95 EF.
    6. Fit PCA with modeled scans as rows and parcels as columns.

    Parcels are not variance-standardized. Scikit-learn centers each parcel
    across scans as part of fitting PCA.
    Omitting global_p95_e restores the original normalized/demeaned pipeline;
    supplying it preserves the existing global-P95-weighted calculation.
    """
    _validate_subject_parcel_frame(parcel_p95)
    parcel_p95 = parcel_p95.astype(float).copy()
    if global_p95_e is not None:
        global_p95_e = _validate_global_p95_e(global_p95_e, parcel_p95.index)

    global_mean_e = parcel_p95.mean(axis=1).rename("global_mean_E")

    if (global_mean_e <= 0).any():
        invalid = global_mean_e.index[global_mean_e <= 0].tolist()
        raise ValueError(f"Global mean E must be positive; invalid scans: {invalid}")

    relative_parcels = parcel_p95.div(global_mean_e, axis="index")

    if not np.allclose(relative_parcels.mean(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("Within-scan normalization failed to produce a parcel mean of 1")

    subject_demeaned = relative_parcels.sub(relative_parcels.mean(axis=1), axis="index")

    if not np.allclose(subject_demeaned.mean(axis=1), 0.0, atol=1e-12):
        raise RuntimeError("Within-scan parcel demeaning failed")

    global_p95_weighted = subject_demeaned.copy() if global_p95_e is None else subject_demeaned.mul(global_p95_e, axis="index")

    if not np.isfinite(global_p95_weighted.to_numpy(dtype=float)).all():
        raise ValueError("Global-P95-weighted PCA matrix contains non-finite values")

    total_variance = global_p95_weighted.var(axis=0, ddof=1).sum()

    if not np.isfinite(total_variance):
        raise ValueError("Global-P95-weighted parcel profiles produced non-finite variance")

    if np.isclose(total_variance, 0.0):
        raise ValueError("Global-P95-weighted parcel profiles have no between-scan variance")

    model = PCA(svd_solver="full")
    score_values = model.fit_transform(global_p95_weighted.to_numpy(dtype=float))
    component_names = [f"PC{i}" for i in range(1, score_values.shape[1] + 1)]

    scores = pd.DataFrame(
        score_values,
        index=parcel_p95.index,
        columns=component_names,
    )
    scores.index.name = parcel_p95.index.name or "subjid"

    loadings = pd.DataFrame(
        model.components_.T,
        index=parcel_p95.columns,
        columns=component_names,
    )
    loadings.index.name = parcel_p95.columns.name or "parcel"

    variance = pd.DataFrame(
        {
            "explained_variance": model.explained_variance_,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(model.explained_variance_ratio_),
        },
        index=pd.Index(component_names, name="component"),
    )

    return ParcelPCA(
        parcel_p95=parcel_p95,
        global_mean_e=global_mean_e,
        global_p95_e=global_p95_e,
        relative_parcels=relative_parcels,
        subject_demeaned_parcels=subject_demeaned,
        global_p95_weighted_parcels=global_p95_weighted,
        scores=scores,
        loadings=loadings,
        variance=variance,
        model=model,
    )


def components_for_variance(pca_result: ParcelPCA | DemeanedParcelPCA, threshold: float) -> int:
    """Return the smallest number of PCs reaching a cumulative threshold."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be numeric")

    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    cumulative = pca_result.variance["cumulative_variance_ratio"].to_numpy()

    if cumulative[-1] + 1e-12 < threshold:
        raise ValueError(
            f"PCA reaches only {cumulative[-1]:.6f}, below {threshold:.6f}"
        )

    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def global_and_pc_predictors(
    global_metrics: pd.DataFrame,
    pca_result: ParcelPCA | DemeanedParcelPCA,
    *,
    n_pcs: int,
) -> pd.DataFrame:
    """Combine one or more global E metrics with PC1 through PCn."""
    if isinstance(n_pcs, bool) or not isinstance(n_pcs, int):
        raise TypeError("n_pcs must be an integer")

    if not 0 <= n_pcs <= pca_result.scores.shape[1]:
        raise ValueError(
            f"n_pcs must be between 0 and {pca_result.scores.shape[1]}, got {n_pcs}"
        )

    if global_metrics.empty or global_metrics.shape[1] < 1:
        raise ValueError("global_metrics must contain at least one metric")

    if global_metrics.index.has_duplicates:
        duplicates = global_metrics.index[
            global_metrics.index.duplicated()
        ].unique().tolist()
        raise ValueError(f"global_metrics contains duplicate scan IDs: {duplicates}")

    if not global_metrics.index.equals(pca_result.scores.index):
        raise ValueError(
            "Modeled-scan order differs between global metrics and PCA scores"
        )

    try:
        global_values = global_metrics.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("global_metrics must contain only numeric values") from exc

    if not np.isfinite(global_values).all():
        raise ValueError("global_metrics contains non-finite values")

    selected_scores = pca_result.scores.iloc[:, :n_pcs]
    collisions = sorted(set(global_metrics.columns).intersection(selected_scores.columns))

    if collisions:
        raise ValueError(f"Global metric names conflict with PCA component names: {collisions}")

    return pd.concat([global_metrics, selected_scores], axis=1)


def correlate_predictors(
    predictors: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    subject_col: str = "subjid",
    base_col: str = "subjid_base",
    outcome_col: str = "cgi_change",
) -> pd.DataFrame:
    """Calculate scan-course Pearson correlations, Fisher CIs, and BH-FDR."""
    if predictors.empty or predictors.shape[1] < 1:
        raise ValueError("predictors must contain at least one predictor")

    aligned = expand_predictors_to_courses(
        predictors,
        outcomes,
        subject_col=subject_col,
        outcome_col=outcome_col,
    )

    y = aligned[outcome_col].to_numpy(dtype=float)
    n_people = (
        aligned[base_col].nunique()
        if base_col in aligned.columns
        else aligned[subject_col].nunique()
    )
    n_courses = (
        aligned["treatment_course_id"].nunique()
        if "treatment_course_id" in aligned.columns
        else len(aligned)
    )

    rows = []

    for predictor_name in predictors.columns:
        x = pd.to_numeric(
            aligned[predictor_name],
            errors="raise",
        ).to_numpy(dtype=float)

        if not np.isfinite(x).all():
            raise ValueError(
                f"Predictor {predictor_name!r} contains non-finite values"
            )

        if np.ptp(x) == 0:
            raise ValueError(f"Predictor {predictor_name!r} has zero variance")

        result = pearsonr(x, y)
        ci_low, ci_high = _pearson_fisher_ci(float(result.statistic), len(x))

        rows.append(
            {
                "predictor": predictor_name,
                "n_observations": len(x),
                "n_scans": aligned[subject_col].nunique(),
                "n_treatment_courses": n_courses,
                "n_people": n_people,
                "r": float(result.statistic),
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "p_value": float(result.pvalue),
            }
        )

    correlations = pd.DataFrame(rows).set_index("predictor")
    correlations["p_fdr_bh"] = _benjamini_hochberg(correlations["p_value"])

    return correlations


def expand_predictors_to_courses(
    predictors: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    subject_col: str = "subjid",
    outcome_col: str = "cgi_change",
    observation_col: str = "observation_id",
) -> pd.DataFrame:
    """Attach one scan-level predictor row to every matched treatment course."""
    if predictors.index.has_duplicates:
        duplicates = predictors.index[
            predictors.index.duplicated()
        ].astype(str).unique().tolist()
        raise ValueError(f"Predictors contain duplicate modeled-scan IDs: {duplicates}")

    required = {subject_col, outcome_col, observation_col}
    missing_columns = sorted(required.difference(outcomes.columns))

    if missing_columns:
        raise KeyError(f"outcomes is missing columns: {missing_columns}")

    duplicate_observations = outcomes.loc[
        outcomes[observation_col].duplicated(keep=False),
        observation_col,
    ].tolist()

    if duplicate_observations:
        raise ValueError(
            "outcomes contains duplicate scan-course observations: "
            f"{duplicate_observations}"
        )

    collisions = sorted(set(predictors.columns).intersection(outcomes.columns))

    if collisions:
        raise ValueError(f"Predictor columns already exist in outcomes: {collisions}")

    expected_scans = pd.Index(predictors.index.astype(str))
    observed_scans = pd.Index(outcomes[subject_col].astype(str).unique())
    extra_scans = observed_scans.difference(expected_scans).tolist()

    if extra_scans:
        raise ValueError(f"Outcomes contain scans absent from predictors: {extra_scans}")

    predictor_table = predictors.copy()
    predictor_table.index = predictor_table.index.astype(str)
    predictor_table.insert(0, subject_col, predictor_table.index)
    predictor_table = predictor_table.reset_index(drop=True)

    expanded = outcomes.copy()
    expanded[subject_col] = expanded[subject_col].astype(str)
    expanded = expanded.merge(
        predictor_table,
        on=subject_col,
        how="left",
        validate="many_to_one",
    )

    if len(expanded) != len(outcomes):
        raise RuntimeError(
            f"Predictor expansion changed row count from "
            f"{len(outcomes)} to {len(expanded)}"
        )

    expanded[outcome_col] = pd.to_numeric(
        expanded[outcome_col],
        errors="raise",
    )
    outcome_values = expanded[outcome_col].to_numpy(dtype=float)

    if not np.isfinite(outcome_values).all():
        raise ValueError(f"{outcome_col!r} contains non-finite values")

    if np.ptp(outcome_values) == 0:
        raise ValueError(f"{outcome_col!r} has zero variance")

    return expanded


def plot_pca_variance(
    pca_result: ParcelPCA | DemeanedParcelPCA,
    thresholds: Sequence[float] = (0.96, 0.99),
) -> tuple[Figure, np.ndarray]:
    """Plot component-wise and cumulative explained variance."""
    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError(f"All thresholds must be in (0, 1], got {threshold}")

    variance = pca_result.variance
    x = np.arange(1, len(variance) + 1)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        constrained_layout=True,
    )

    axes[0].bar(
        x,
        100 * variance["explained_variance_ratio"].to_numpy(),
    )
    axes[0].set(
        xlabel="Principal component",
        ylabel="Explained variance (%)",
        title="Scree plot",
    )
    axes[0].set_xlim(0.5, len(x) + 0.5)

    cumulative = 100 * variance["cumulative_variance_ratio"].to_numpy()
    axes[1].plot(x, cumulative, marker="o", markersize=3)

    for threshold in thresholds:
        count = components_for_variance(pca_result, threshold)
        axes[1].axhline(100 * threshold, linestyle="--", linewidth=1)
        axes[1].axvline(count, linestyle=":", linewidth=1)
        axes[1].annotate(
            f"{100 * threshold:.0f}%: {count} PCs",
            (count, 100 * threshold),
            xytext=(5, 5),
            textcoords="offset points",
        )

    axes[1].set(
        xlabel="Number of components",
        ylabel="Cumulative variance (%)",
        title="Cumulative explained variance",
    )
    axes[1].set_xlim(0.5, len(x) + 0.5)
    axes[1].set_ylim(0, 101)

    return fig, axes


def plot_predictor_outcome(
    predictor: pd.Series,
    outcomes: pd.DataFrame,
    *,
    subject_col: str = "subjid",
    outcome_col: str = "cgi_change",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot one predictor against outcome with a least-squares line."""
    predictor_name = predictor.name or "predictor"
    predictor = pd.to_numeric(
        predictor,
        errors="raise",
    ).astype(float).rename(predictor_name)

    aligned = expand_predictors_to_courses(
        predictor.to_frame(),
        outcomes,
        subject_col=subject_col,
        outcome_col=outcome_col,
    )

    x = aligned[predictor_name].astype(float)
    y = aligned[outcome_col].astype(float)

    correlation = correlate_predictors(
        predictor.to_frame(),
        outcomes,
        subject_col=subject_col,
        outcome_col=outcome_col,
    ).iloc[0]

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(5.5, 4.5),
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    ax.scatter(x, y, alpha=0.8)

    slope, intercept = np.polyfit(x.to_numpy(), y.to_numpy(), deg=1)
    line_x = np.linspace(x.min(), x.max(), 100)
    ax.plot(line_x, intercept + slope * line_x)

    ax.set(
        xlabel=predictor_name,
        ylabel=outcome_col,
        title=f"r = {correlation['r']:.2f}, p = {correlation['p_value']:.3g}",
    )

    return fig, ax


def plot_global_outcome(
    global_metrics: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    subject_col: str = "subjid",
    outcome_col: str = "cgi_change",
) -> tuple[Figure, np.ndarray]:
    """Plot one or more global EF metrics against the outcome."""
    if global_metrics.empty or global_metrics.shape[1] < 1:
        raise ValueError("global_metrics must contain at least one metric")

    n_metrics = global_metrics.shape[1]

    fig, axes_2d = plt.subplots(
        1,
        n_metrics,
        figsize=(6 * n_metrics, 4.5),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes_2d[0]

    for ax, predictor_name in zip(axes, global_metrics.columns, strict=True):
        plot_predictor_outcome(
            global_metrics[predictor_name],
            outcomes,
            subject_col=subject_col,
            outcome_col=outcome_col,
            ax=ax,
        )

    return fig, axes


def plot_subject_parcel_heatmap(
    pca_result: ParcelPCA | DemeanedParcelPCA,
) -> tuple[Figure, Axes]:
    """Plot the actual matrix supplied to PCA, with its transformation label."""
    matrix = pca_result.pca_input
    width = min(20, max(8, matrix.shape[1] * 0.08))
    height = min(14, max(4, matrix.shape[0] * 0.18))

    fig, ax = plt.subplots(
        figsize=(width, height),
        constrained_layout=True,
    )

    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        interpolation="nearest",
    )

    ax.set(
        xlabel="Parcel",
        ylabel="Modeled scan",
        title=f"PCA input: {pca_result.input_label}",
    )
    ax.set_yticks(np.arange(len(matrix)), labels=matrix.index)

    if matrix.shape[1] <= 60:
        ax.set_xticks(
            np.arange(matrix.shape[1]),
            labels=matrix.columns,
            rotation=90,
            fontsize=6,
        )
    else:
        ax.set_xticks([])

    fig.colorbar(
        image,
        ax=ax,
        label=pca_result.input_label,
    )

    return fig, ax


def plot_top_loadings(
    pca_result: ParcelPCA | DemeanedParcelPCA,
    *,
    components: Sequence[str] = ("PC1", "PC2", "PC3"),
    n_parcels: int = 15,
) -> tuple[Figure, np.ndarray]:
    """Plot parcels with the largest absolute loadings for selected PCs."""
    if not components:
        raise ValueError("components cannot be empty")

    missing = sorted(set(components).difference(pca_result.loadings.columns))

    if missing:
        raise KeyError(f"PCA components do not exist: {missing}")

    if (
        isinstance(n_parcels, bool)
        or not isinstance(n_parcels, int)
        or n_parcels < 1
    ):
        raise ValueError("n_parcels must be a positive integer")

    fig, axes_2d = plt.subplots(
        len(components),
        1,
        figsize=(9, 4 * len(components)),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes_2d[:, 0]

    for ax, component in zip(axes, components, strict=True):
        component_loadings = pca_result.loadings[component]
        selected = component_loadings.abs().nlargest(
            min(n_parcels, len(component_loadings))
        ).index
        values = component_loadings.loc[selected].sort_values()
        colors = np.where(values >= 0, "#4472C4", "#C55A11")

        ax.barh(values.index, values.to_numpy(), color=colors)
        ax.set(
            xlabel="PCA loading",
            title=f"Largest absolute parcel loadings: {component}",
        )
        ax.axvline(0, color="black", linewidth=0.8)

    return fig, axes


def plot_correlation_bars(
    correlations: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Plot Pearson correlations and nominal 95% Fisher intervals."""
    required = {"r", "ci_95_low", "ci_95_high"}
    missing = sorted(required.difference(correlations.columns))

    if missing:
        raise KeyError(f"correlations is missing columns: {missing}")

    frame = correlations.iloc[::-1]
    y = np.arange(len(frame))
    lower = frame["r"] - frame["ci_95_low"]
    upper = frame["ci_95_high"] - frame["r"]

    fig, ax = plt.subplots(
        figsize=(8, max(4, 0.35 * len(frame))),
        constrained_layout=True,
    )

    ax.errorbar(
        frame["r"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        yticks=y,
        yticklabels=frame.index,
        xlabel="Pearson r",
        title="Global E and PC correlations",
    )
    ax.set_xlim(-1.05, 1.05)

    return fig, ax


def _validate_subject_parcel_frame(frame: pd.DataFrame) -> None:
    """Validate a scans × parcels EF matrix without dropping data."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(frame).__name__}")

    if frame.shape[0] < 3 or frame.shape[1] < 2:
        raise ValueError(
            f"PCA requires at least 3 scans and 2 parcels, got {frame.shape}"
        )

    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate modeled-scan IDs: {duplicates}")

    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate parcels: {duplicates}")

    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Parcel matrix must contain only numeric values") from exc

    if not np.isfinite(values).all():
        bad_rows, bad_columns = np.where(~np.isfinite(values))
        examples = [
            (str(frame.index[row]), str(frame.columns[column]))
            for row, column in zip(
                bad_rows[:5],
                bad_columns[:5],
                strict=True,
            )
        ]
        raise ValueError(
            f"Parcel matrix contains non-finite values; examples: {examples}"
        )

    if (values < 0).any():
        bad_rows, bad_columns = np.where(values < 0)
        examples = [
            (str(frame.index[row]), str(frame.columns[column]))
            for row, column in zip(
                bad_rows[:5],
                bad_columns[:5],
                strict=True,
            )
        ]
        raise ValueError(
            f"Parcel matrix contains negative EF magnitudes; examples: {examples}"
        )


def _validate_global_p95_e(
    global_p95_e: pd.Series,
    expected_index: pd.Index,
) -> pd.Series:
    """Validate and align one global brain P95 EF value per modeled scan."""
    if not isinstance(global_p95_e, pd.Series):
        raise TypeError("global_p95_e must be a pandas Series")

    if global_p95_e.index.has_duplicates:
        duplicates = global_p95_e.index[
            global_p95_e.index.duplicated()
        ].unique().tolist()
        raise ValueError(f"global_p95_e contains duplicate scan IDs: {duplicates}")

    global_p95_e = global_p95_e.copy()
    global_p95_e.index = global_p95_e.index.astype(str)

    expected_index_string = pd.Index(expected_index.astype(str))
    missing = expected_index_string.difference(global_p95_e.index).tolist()
    extra = global_p95_e.index.difference(expected_index_string).tolist()

    if missing or extra:
        raise ValueError(
            "Subject mismatch between parcel_p95 and global_p95_e; "
            f"missing global P95 values={missing}, extra values={extra}"
        )

    global_p95_e = pd.to_numeric(
        global_p95_e.reindex(expected_index_string),
        errors="raise",
    ).astype(float)
    global_p95_e.index = expected_index
    global_p95_e = global_p95_e.rename("global_p95_E")

    values = global_p95_e.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        invalid = global_p95_e.index[~np.isfinite(values)].tolist()
        raise ValueError(
            f"global_p95_e contains non-finite values for scans: {invalid}"
        )

    if (global_p95_e <= 0).any():
        invalid = global_p95_e.index[global_p95_e <= 0].tolist()
        raise ValueError(f"Global P95 E must be positive; invalid scans: {invalid}")

    return global_p95_e


def _validate_field_values(
    field_values: Sequence[float] | np.ndarray,
    *,
    subject_id: str,
) -> np.ndarray:
    """Validate one scalar nonnegative EF field."""
    field = np.asarray(field_values, dtype=float).squeeze()

    if field.ndim != 1:
        raise ValueError(
            f"EF field must be one-dimensional for {subject_id}, got {field.shape}"
        )

    if field.size == 0:
        raise ValueError(f"EF field is empty for {subject_id}")

    if not np.isfinite(field).all():
        raise ValueError(f"EF field contains non-finite values for {subject_id}")

    if (field < 0).any():
        raise ValueError(f"EF magnitude contains negative values for {subject_id}")

    return field


def _validate_percentile(percentile: float) -> None:
    """Validate a percentile argument."""
    if isinstance(percentile, bool) or not isinstance(
        percentile,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("percentile must be numeric")

    if not 0 <= percentile <= 100:
        raise ValueError(
            f"percentile must be between 0 and 100, got {percentile}"
        )


def _pearson_fisher_ci(
    r: float,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate a nominal Fisher-transformed confidence interval."""
    if n <= 3:
        return float("nan"), float("nan")

    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    clipped = np.clip(
        r,
        -1 + np.finfo(float).eps,
        1 - np.finfo(float).eps,
    )
    center = np.arctanh(clipped)
    critical_value = norm.ppf(0.5 + confidence / 2)
    half_width = critical_value / np.sqrt(n - 3)

    return (
        float(np.tanh(center - half_width)),
        float(np.tanh(center + half_width)),
    )


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Apply Benjamini–Hochberg adjustment to one p-value family."""
    values = p_values.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("p-values must be finite")

    if not ((0 <= values) & (values <= 1)).all():
        raise ValueError("p-values must be between 0 and 1")

    order = np.argsort(values)
    ranked = values[order]
    ranks = np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / ranks)[::-1]
    )[::-1]

    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)

    return pd.Series(adjusted, index=p_values.index, name="p_fdr_bh")
