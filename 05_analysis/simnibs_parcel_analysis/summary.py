"""Common parcel summaries for volumetric and surface ROI analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
import pandas as pd


P95Method = Literal["unweighted", "spatial_weighted"]


@dataclass(frozen=True)
class ParcelSummary:
    """Parcel-level field summaries and ROI-assignment quality control."""

    scans: pd.DataFrame
    p95_unweighted: pd.DataFrame
    p95_spatial_weighted: pd.DataFrame
    parcel_size: pd.DataFrame
    parcel_count: pd.DataFrame
    qc: pd.DataFrame
    atlas_name: str
    domain: Literal["volume", "surface"]
    size_unit: Literal["mm3", "mm2"]

    def p95(self, method: P95Method = "spatial_weighted") -> pd.DataFrame:
        """Return the requested parcel-by-scan P95 matrix."""
        if method == "unweighted":
            return self.p95_unweighted
        if method == "spatial_weighted":
            return self.p95_spatial_weighted
        raise ValueError("method must be 'unweighted' or 'spatial_weighted'")

    def global_metrics(self, method: P95Method = "spatial_weighted") -> pd.DataFrame:
        """Return equal-parcel and parcel-size-weighted global mean P95."""
        return global_metrics_from_parcels(self.p95(method), self.parcel_size)


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """Return a weighted empirical percentile using an inverted weighted CDF.

    The result is the smallest field value whose cumulative physical weight
    reaches the requested percentile. For tetrahedra, weights are volumes; for
    surface nodes, weights are represented cortical areas.
    """
    values = np.asarray(values, dtype=float).squeeze()
    weights = np.asarray(weights, dtype=float).squeeze()
    if values.ndim != 1 or weights.ndim != 1 or values.size != weights.size or values.size == 0:
        raise ValueError("values and weights must be nonempty one-dimensional arrays of equal length")
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("values must be finite and weights must be finite and positive")

    order = np.argsort(values, kind="mergesort")
    ordered_values, ordered_weights = values[order], weights[order]
    target = percentile / 100 * ordered_weights.sum()
    index = int(np.searchsorted(np.cumsum(ordered_weights), target, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def summarize_labeled_field(
    values: np.ndarray,
    spatial_weights: np.ndarray,
    labels: np.ndarray,
    label_names: Mapping[int, str],
    *,
    percentile: float = 95,
    min_samples_per_roi: int = 1,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate both P95 definitions, total size, and sample count per ROI."""
    values = np.asarray(values, dtype=float).squeeze()
    spatial_weights = np.asarray(spatial_weights, dtype=float).squeeze()
    labels = np.asarray(labels).squeeze()
    if values.ndim != 1 or spatial_weights.ndim != 1 or labels.ndim != 1:
        raise ValueError("values, spatial_weights, and labels must be one-dimensional")
    if not (values.size == spatial_weights.size == labels.size):
        raise ValueError("values, spatial_weights, and labels must have equal length")
    if isinstance(min_samples_per_roi, bool) or not isinstance(min_samples_per_roi, int) or min_samples_per_roi < 1:
        raise ValueError("min_samples_per_roi must be a positive integer")
    if not label_names:
        raise ValueError("label_names is empty")
    if len(set(label_names.values())) != len(label_names):
        raise ValueError("ROI names must be unique")

    ordinary, weighted, sizes, counts = {}, {}, {}, {}
    for label_id, name in sorted(label_names.items(), key=lambda item: item[1]):
        mask = labels == int(label_id)
        count = int(mask.sum())
        if count < min_samples_per_roi:
            raise ValueError(
                f"ROI {name!r} (label {label_id}) contains {count} samples; "
                f"minimum is {min_samples_per_roi}"
            )
        roi_values, roi_weights = values[mask], spatial_weights[mask]
        ordinary[name] = float(np.percentile(roi_values, percentile, method="linear"))
        weighted[name] = weighted_percentile(roi_values, roi_weights, percentile)
        sizes[name] = float(roi_weights.sum())
        counts[name] = count
    return (
        pd.Series(ordinary, dtype=float),
        pd.Series(weighted, dtype=float),
        pd.Series(sizes, dtype=float),
        pd.Series(counts, dtype=int),
    )


def build_parcel_summary(
    scans: pd.DataFrame,
    ordinary_rows: list[pd.Series],
    weighted_rows: list[pd.Series],
    size_rows: list[pd.Series],
    count_rows: list[pd.Series],
    qc_rows: list[dict[str, object]],
    *,
    atlas_name: str,
    domain: Literal["volume", "surface"],
) -> ParcelSummary:
    """Assemble per-scan rows after enforcing identical parcel columns."""
    if not (len(scans) == len(ordinary_rows) == len(weighted_rows) == len(size_rows) == len(count_rows) == len(qc_rows)):
        raise ValueError("Summary row counts do not match scan count")
    index = pd.Index(scans["subjid"].astype(str), name="subjid")
    frames = [pd.DataFrame(rows, index=index) for rows in (ordinary_rows, weighted_rows, size_rows, count_rows)]
    expected = frames[0].columns
    for name, frame in zip(("weighted P95", "parcel size", "parcel count"), frames[1:], strict=True):
        if not frame.columns.equals(expected):
            raise ValueError(f"Parcel columns differ in {name}")
    for frame in frames[:3]:
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Parcel summary contains missing or non-finite values")
    if (frames[2] <= 0).any().any() or (frames[3] < 1).any().any():
        raise ValueError("Parcel sizes and counts must be positive")

    qc = pd.DataFrame(qc_rows, index=index)
    return ParcelSummary(
        scans=scans.reset_index(drop=True),
        p95_unweighted=frames[0],
        p95_spatial_weighted=frames[1],
        parcel_size=frames[2],
        parcel_count=frames[3].astype(int),
        qc=qc,
        atlas_name=atlas_name,
        domain=domain,
        size_unit="mm3" if domain == "volume" else "mm2",
    )


def global_metrics_from_parcels(parcel_values: pd.DataFrame, parcel_size: pd.DataFrame) -> pd.DataFrame:
    """Calculate the two requested global E summaries from parcel statistics."""
    if not parcel_values.index.equals(parcel_size.index) or not parcel_values.columns.equals(parcel_size.columns):
        raise ValueError("parcel_values and parcel_size must have identical indices and columns")
    values = parcel_values.to_numpy(dtype=float)
    weights = parcel_size.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Parcel values must be finite and parcel sizes must be finite and positive")
    metrics = pd.DataFrame(
        {
            "global_mean_E_unweighted": parcel_values.mean(axis=1),
            "global_mean_E_weighted": (parcel_values * parcel_size).sum(axis=1) / parcel_size.sum(axis=1),
        },
        index=parcel_values.index,
    )
    if (metrics <= 0).any().any():
        bad = metrics.index[(metrics <= 0).any(axis=1)].tolist()
        raise ValueError(f"Global E must be positive for every scan: {bad}")
    return metrics
