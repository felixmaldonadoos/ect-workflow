"""Parcel statistics from SimNIBS fsaverage scalar surface meshes.

The input meshes must use SimNIBS' standard fsaverage vertex ordering:
left hemisphere followed by right hemisphere. Atlas/node counts are checked;
an arbitrary surface with the same number of vertices is not sufficient.

e_mean is a surface-area-weighted spatial mean. e_percentile is numpy's
ordinary, linearly interpolated spatial percentile across parcel vertices
(default: 95). With percentile_weighting='area', it is the inverse
area-weighted empirical CDF: the smallest field value whose cumulative area
fraction reaches percentile / 100.

For a UQ field such as magnE_mean, these are spatial summaries of the
pointwise UQ mean, not percentiles across conductivity realizations.
"""

from collections.abc import Iterable, Sequence
from numbers import Real
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


def global_mean_e(meshes: Iterable, *, field_name: str = "magnE_mean") -> np.ndarray:
    """Return area-weighted cortical mean magnitudes, shape (N, 1).

    Input is an iterable of loaded scalar surface meshes, including a
    generator. Output follows input order. This summarizes all surface
    vertices, including any vertices outside labeled atlas parcels.
    Pass [mesh] for one mesh; an empty iterable returns shape (0, 1).
    """
    means: list[float] = []
    for i, mesh in enumerate(meshes):
        if not np.all(np.asarray(mesh.elm.elm_type) == 2):
            raise ValueError(f"Mesh {i}: expected a triangle-only cortical surface.")
        fields = [f for f in mesh.nodedata if f.field_name == field_name]
        if len(fields) != 1 or any(f.field_name == field_name for f in mesh.elmdata):
            raise ValueError(f"Mesh {i}: {field_name!r} must be a unique nodal magnitude field.")
        values = np.asarray(fields[0].value, dtype=float)
        if values.shape == (int(mesh.nodes.nr), 1):
            values = values[:, 0]
        areas = np.asarray(mesh.nodes_areas().value, dtype=float)
        if values.shape != (int(mesh.nodes.nr),) or areas.shape != values.shape or not values.size:
            raise ValueError(f"Mesh {i}: expected one scalar magnitude and surface area per node.")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Mesh {i}: magnitudes must be finite and nonnegative.")
        if not np.isfinite(areas).all() or (areas <= 0).any():
            raise ValueError(f"Mesh {i}: surface areas must be finite and positive.")
        means.append(float(np.average(values, weights=areas)))
    return np.asarray(means, dtype=float).reshape(-1, 1)


def _mesh_paths(frame: pd.DataFrame, column: str) -> pd.Series:
    if not frame.columns.is_unique:
        raise ValueError("DataFrame column names must be unique.")
    if column not in frame:
        raise KeyError(f"Missing mesh path column: {column!r}")
    if frame.empty:
        raise ValueError("The input table is empty.")
    paths = frame[column]
    valid = paths.map(lambda p: isinstance(p, str) and bool(p.strip()) and p == p.strip())
    if not valid.all():
        rows = np.flatnonzero(~valid.to_numpy()).tolist()
        raise ValueError(f"{column!r} must contain nonblank path strings; row positions: {rows[:12]}")
    return paths


def _area_percentile(values: np.ndarray, areas: np.ndarray, percentile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(areas[order])
    position = np.searchsorted(cumulative, (percentile / 100) * cumulative[-1], side="left")
    return float(values[order[position]])


def _wide_roi_metrics(metrics: pd.DataFrame, path_column: str) -> pd.DataFrame:
    """Pivot a mesh/parcel table, retaining the mesh key and shared metadata."""
    _mesh_paths(metrics, path_column)
    if "roi" not in metrics:
        raise KeyError("Missing metric column: 'roi'")
    statistics = [column for column in ["e_mean", "e_percentile"] if column in metrics]
    if not statistics:
        raise KeyError("Metrics must contain e_mean, e_percentile, or both.")
    valid_roi = metrics["roi"].map(lambda roi: isinstance(roi, str) and bool(roi.strip()))
    if not valid_roi.all() or metrics.duplicated([path_column, "roi"]).any():
        raise ValueError("Metrics must have one nonmissing ROI row per unique mesh/ROI pair.")
    roi_sets = metrics.groupby(path_column, sort=False)["roi"].agg(frozenset)
    if any(rois != roi_sets.iloc[0] for rois in roi_sets):
        raise ValueError("All meshes must have metrics for the same set of ROIs.")
    roi_order = metrics["roi"].drop_duplicates().tolist()
    wide = metrics.pivot(index=path_column, columns="roi", values=statistics)
    wide.columns = [f"{roi}_{statistic}" for statistic, roi in wide.columns]
    columns = [f"{roi}_{statistic}" for roi in roi_order for statistic in statistics]
    wide = wide.loc[:, columns]
    metadata_columns = [column for column in
                        ["atlas", "field_name", "percentile", "mean_weighting", "percentile_weighting"]
                        if column in metrics]
    metadata = metrics[[path_column, *metadata_columns]].drop_duplicates()
    if metadata.duplicated(path_column).any():
        raise ValueError("Atlas, field, percentile, and weighting metadata must be constant within each mesh.")
    return metadata.set_index(path_column).join(wide, validate="one_to_one").reset_index()


def extract_fsaverage_roi_metrics(
    sessions: pd.DataFrame, *, field_name: str,
    atlas_name: str = "HCP_MMP1", path_column: str = "full_file_path",
    min_nodes_per_roi: int = 20, exclude_rois: Sequence[str] = (),
    percentile: float = 95.0, percentile_weighting: Literal["node", "area"] = "node",
    output_format: Literal["wide", "long"] = "wide",
) -> pd.DataFrame:
    """Read each unique mesh path once; return one row per mesh by default.

    No sessions or invalid field values are dropped. Missing files, non-scalar
    or non-nodal fields, NaNs/infinities, negative magnitudes, invalid areas,
    atlas size mismatches, and undersized parcels raise explicit errors.
    No charge scaling, cross-subject averaging, or normalization is applied.

    Parameters
    ----------
    field_name : str
        Exact scalar magnitude field stored in the mesh. Inspect mesh.field
        before selecting it; E_normal is signed and is not a magnitude.
    exclude_rois : sequence of str
        Exact atlas region names to exclude explicitly, e.g. unknown labels.
    percentile : float
        Finite numeric percentile in [0, 100]; defaults to 95. Both endpoints
        are supported: 0 returns the minimum and 100 returns the maximum.
    percentile_weighting : {'node', 'area'}
        Equal-vertex weighting (default) or the area-weighted empirical CDF.
    output_format : {'wide', 'long'}
        'wide' (default): one row per mesh, with two metric columns per ROI.
        'long': one row per mesh/ROI, including parcel node counts and areas.

    Returns
    -------
    DataFrame
        Wide output retains path_column, atlas, field_name, percentile,
        mean_weighting, and percentile_weighting, and adds '<roi>_e_mean'
        and '<roi>_e_percentile' columns. Long output additionally retains
        roi, n_nodes, area_mm2, e_mean, and e_percentile as separate columns.
        Long-output areas are template areas, not subject-native parcel areas.
        Field units are inherited unchanged from the chosen mesh field.
    """
    import simnibs

    paths = _mesh_paths(sessions, path_column)
    if not isinstance(field_name, str) or not field_name.strip():
        raise ValueError("field_name must be an explicit nonblank magnitude field name.")
    if (isinstance(percentile, (bool, np.bool_)) or not isinstance(percentile, Real)
            or not np.isfinite(percentile) or not 0 <= percentile <= 100):
        raise ValueError("percentile must be a finite number in [0, 100].")
    percentile = float(percentile)
    if percentile_weighting not in {"node", "area"}:
        raise ValueError("percentile_weighting must be 'node' or 'area'.")
    if output_format not in {"wide", "long"}:
        raise ValueError("output_format must be 'wide' or 'long'.")
    if isinstance(min_nodes_per_roi, bool) or not isinstance(min_nodes_per_roi, int) or min_nodes_per_roi < 1:
        raise ValueError("min_nodes_per_roi must be a positive integer.")
    if isinstance(exclude_rois, str):
        raise TypeError("exclude_rois must be a sequence of region names, not one string.")

    unique_paths = paths.unique()
    for filename in unique_paths:
        path = Path(filename)
        if path.suffix.lower() != ".msh":
            raise ValueError(f"Expected a .msh file: {filename}")
        if not path.is_file():
            raise FileNotFoundError(f"Mesh file does not exist: {filename}")

    atlas = simnibs.get_atlas(atlas_name, hemi="both")
    excluded = set(exclude_rois)
    unknown = excluded - set(atlas)
    if unknown:
        raise KeyError(f"Excluded ROI names are not in {atlas_name}: {sorted(unknown)}")
    parcels = {name: np.asarray(mask) for name, mask in atlas.items() if name not in excluded}
    if not parcels:
        raise ValueError("No atlas parcels remain.")

    records: list[dict] = []
    for filename in unique_paths:
        mesh = simnibs.read_msh(filename)
        n_nodes = int(mesh.nodes.nr)
        if not np.all(np.asarray(mesh.elm.elm_type) == 2):
            raise ValueError(f"Expected a triangle-only fsaverage surface mesh: {filename}")
        node_fields = [f for f in mesh.nodedata if f.field_name == field_name]
        element_fields = [f for f in mesh.elmdata if f.field_name == field_name]
        if len(node_fields) != 1 or element_fields:
            raise ValueError(
                f"{filename}: {field_name!r} must occur exactly once as a nodal field "
                f"and never as an element field. Available fields: {sorted(mesh.field)}"
            )
        values = np.asarray(node_fields[0].value, dtype=float)
        if values.shape == (n_nodes, 1):
            values = values[:, 0]
        if values.shape != (n_nodes,):
            raise ValueError(f"{filename}: expected scalar field shape {(n_nodes,)}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{filename}: field {field_name!r} contains NaN or infinity.")
        if (values < 0).any():
            raise ValueError(f"{filename}: magnitude field {field_name!r} contains negative values.")
        areas = np.asarray(mesh.nodes_areas().value, dtype=float)
        if areas.shape != (n_nodes,) or not np.isfinite(areas).all() or (areas <= 0).any():
            raise ValueError(f"{filename}: expected one finite, positive surface area per node.")

        for roi, mask in parcels.items():
            if mask.dtype != np.bool_ or mask.shape != (n_nodes,):
                raise ValueError(
                    f"{filename}: ROI {roi!r} mask must be boolean with shape {(n_nodes,)}; "
                    f"got {mask.dtype}, {mask.shape}. Check fsaverage resolution and hemisphere order."
                )
            n_roi = int(mask.sum())
            if n_roi < min_nodes_per_roi:
                raise ValueError(
                    f"{filename}: ROI {roi!r} has {n_roi} nodes; minimum is {min_nodes_per_roi}. "
                    "Exclude this ROI explicitly if it is outside your analysis."
                )
            roi_values, roi_areas = values[mask], areas[mask]
            e_percentile = (float(np.percentile(roi_values, percentile)) if percentile_weighting == "node"
                            else _area_percentile(roi_values, roi_areas, percentile))
            records.append({
                path_column: filename, "atlas": atlas_name, "field_name": field_name,
                "roi": roi, "n_nodes": n_roi, "area_mm2": float(roi_areas.sum()),
                "e_mean": float(np.average(roi_values, weights=roi_areas)),
                "e_percentile": e_percentile, "percentile": percentile,
                "mean_weighting": "area", "percentile_weighting": percentile_weighting,
            })
    metrics = pd.DataFrame.from_records(records)
    return _wide_roi_metrics(metrics, path_column) if output_format == "wide" else metrics


def join_session_roi_metrics(
    sessions: pd.DataFrame, metrics: pd.DataFrame, *, path_column: str = "full_file_path",
) -> pd.DataFrame:
    """Attach selected ROI metric columns; return one row per original session.

    Accept wide mesh tables containing mean columns, percentile columns,
    or both, plus the required path_column. Also accept long mesh/ROI tables
    for compatibility. Preserve every session column, row order, index, and
    repeated session. Attach any retained mesh metadata once per session.
    Duplicate mesh keys, missing paths, nonfinite/non-numeric ROI metrics,
    and conflicting output column names raise explicit errors.
    """
    paths = _mesh_paths(sessions, path_column)
    _mesh_paths(metrics, path_column)
    wide = _wide_roi_metrics(metrics, path_column) if "roi" in metrics else metrics
    if wide.duplicated(path_column).any():
        raise ValueError("Wide metrics must contain one row per unique mesh path.")
    metric_columns = [column for column in wide.columns
                      if isinstance(column, str) and column != path_column
                      and column.endswith(("_e_mean", "_e_percentile"))]
    if not metric_columns:
        raise ValueError("No ROI metric columns found; retain columns ending in _e_mean or _e_percentile.")
    for column in metric_columns:
        if not pd.api.types.is_numeric_dtype(wide[column]) or pd.api.types.is_bool_dtype(wide[column]):
            raise ValueError(f"ROI metric column {column!r} must be numeric.")
    if not np.isfinite(wide[metric_columns].to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError("ROI metric columns contain NaN or infinity.")
    missing = set(paths) - set(wide[path_column])
    if missing:
        raise ValueError(f"Sessions have no ROI metrics for mesh paths: {sorted(missing)[:12]}")
    overlap = (set(sessions.columns) & set(wide.columns)) - {path_column}
    if overlap:
        raise ValueError(f"Session and ROI output columns overlap: {sorted(overlap)}")
    result = sessions.merge(
        wide, on=path_column, how="left", sort=False, validate="many_to_one",
    )
    result.index = sessions.index.copy()
    return result
