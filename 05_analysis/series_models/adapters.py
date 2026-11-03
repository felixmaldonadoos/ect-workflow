"""Validate saved ROI/PCA series masters without loading spatial artifacts.

Only the existing feature values and covariates are used.  In particular, this
module neither recomputes CGI change nor normalizes or rescales any feature.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from networks._results import normalization_metadata


SHARED_COLUMNS = {
    "person_id", "series_num", "model_type", "step", "field_name",
    "cgi_change", "age", "sex", "ef_sf",
}
SERIES_KEYS = ["person_id", "series_num", "model_type", "step", "field_name"]
ROI_FEATURE_COLUMNS = {
    "roi_org", "roi_name", "roi", "hemi", "hemisphere", "network", "n_voxels",
    "mean", "mode", "median", "min", "max", "p95", "p99", "mean_raw",
}
PCA_FEATURE_COLUMNS = {
    "component", "score", "component_index", "explained_variance",
    "explained_variance_ratio", "cumulative_variance_ratio",
    "selected_for_correlation", "component_map_path",
}
ADDED_FEATURE_COLUMNS = {"feature_type", "feature_id", "E", "E_x_ef_sf", "value_column"}
TEXT_COLUMNS = {
    "person_id", "model_type", "step", "field_name", "sex", "roi_org", "roi_name",
    "sample_id", "basis_id", "component",
}


@dataclass
class PreparedMaster:
    """Long feature rows and the adapter's explicit provenance/selection."""

    frame: pd.DataFrame
    metadata: dict


def _read_csv(path: Path) -> pd.DataFrame:
    # Pandas mangles duplicate headers, so inspect the original header first.
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), None)
    if not header or any(not name.strip() for name in header):
        raise ValueError("Master CSV has missing or empty column names.")
    if len(set(header)) != len(header):
        raise ValueError("Master CSV has duplicate column names.")
    if any(name != name.strip() for name in header):
        raise ValueError("Master CSV column names have surrounding whitespace.")
    frame = pd.read_csv(
        path, encoding="utf-8-sig", dtype={name: str for name in TEXT_COLUMNS & set(header)},
        keep_default_na=False, na_values=[""],
    )
    if frame.empty:
        raise ValueError("Master CSV has no observations.")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Master CSV is missing required columns: {sorted(missing)}")


def _require_labels(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        values = frame[column]
        strings = values.astype(str)
        invalid = values.isna() | strings.str.strip().eq("") | strings.ne(strings.str.strip())
        if invalid.any():
            raise ValueError(f"Missing, empty, or padded category/identity in {column!r}.")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{column!r} must contain finite numeric values; no rows were dropped.")
    return values


def _assign_checked(frame: pd.DataFrame, column: str, values) -> None:
    expected = pd.Series(values, index=frame.index)
    if column in frame:
        existing = frame[column]
        if pd.api.types.is_numeric_dtype(expected):
            equal = pd.to_numeric(existing, errors="coerce").eq(expected)
        else:
            equal = existing.eq(expected)
        if not equal.fillna(False).all():
            raise ValueError(f"Conflicting reserved modeling column {column!r}.")
    frame[column] = expected


def _consistent_metadata(frame: pd.DataFrame, keys: list[str], columns: Iterable[str], context: str) -> None:
    columns = [column for column in columns if column not in keys]
    if not columns:
        return
    counts = frame.groupby(keys, dropna=False, sort=False)[columns].nunique(dropna=False)
    bad_columns = counts.columns[counts.gt(1).any()].tolist()
    if bad_columns:
        raise ValueError(f"Conflicting {context} metadata across repeated rows: {bad_columns}")


def _boolean(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False},
    )
    if parsed.isna().any():
        raise ValueError("Invalid Boolean values in 'selected_for_correlation'.")
    return parsed.astype(bool)


def _validate_roi(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    identity = "roi_org" if "roi_org" in frame else "roi_name"
    _require_columns(frame, [identity, "mean"])
    _require_labels(frame, [identity])
    if "roi_org" in frame and "roi_name" in frame:
        _require_labels(frame, ["roi_name"])
        if not frame["roi_org"].eq(frame["roi_name"]).all():
            raise ValueError("Ambiguous ROI identity: roi_org and roi_name disagree.")
    labels = [column for column in ("roi", "hemi", "hemisphere", "network") if column in frame]
    _consistent_metadata(frame, [identity], labels, "ROI identity")
    return frame[identity], "mean"


def _validate_pca(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    _require_columns(frame, ["basis_id", "sample_id", "component", "score"])
    _require_labels(frame, ["basis_id", "sample_id", "component"])
    if not frame["component"].str.fullmatch(r"PC[1-9][0-9]*").all():
        raise ValueError("PCA component identities must be unambiguous labels such as PC1.")
    if "selected_for_correlation" in frame:
        frame["selected_for_correlation"] = _boolean(frame["selected_for_correlation"])
    if "component_index" in frame:
        indices = _numeric(frame, "component_index")
        expected = frame["component"].str.slice(2).astype(int)
        if not indices.eq(expected).all():
            raise ValueError("PCA component_index disagrees with component identity.")
        frame["component_index"] = indices.astype(int)
    _consistent_metadata(
        frame, ["basis_id"],
        ["field_name", "model_type", "step", "normalization", "normalization_tag"],
        "PCA basis/configuration",
    )
    _consistent_metadata(frame, ["basis_id", "sample_id"], SERIES_KEYS, "PCA sample/series identity")
    _consistent_metadata(frame, ["basis_id", *SERIES_KEYS], ["sample_id"], "PCA series/sample identity")
    if frame.duplicated(["basis_id", "sample_id", "component"]).any():
        raise ValueError("Duplicate PCA basis/sample/component observations.")
    component_metadata = PCA_FEATURE_COLUMNS - {"score", "component"}
    _consistent_metadata(
        frame, ["basis_id", "component"], component_metadata & set(frame), "PCA component",
    )
    basis_metadata = {
        "pca_model_path", "pca_variance_path", "pca_settings_path", "brain_mask_path",
        "n_brain_voxels", "n_fit_samples", "n_unique_fields", "pca_fit_unit",
        "samples_assumed_independent",
    }
    _consistent_metadata(frame, ["basis_id"], basis_metadata & set(frame), "PCA basis")
    return frame["component"], "score"


def read_and_prepare(
    master_path, feature_type: str, field_name: str, *, features=None, pca_scope: str = "selected",
) -> PreparedMaster:
    """Read one master and preserve its series rows in a common representation.

    Explicit feature names override the PCA saved-selection default.  ``all``
    and explicit selection also accept a generic PC master without saved flags.
    A run accepts one normalization; separate invocations retain source identity.
    Validation precedes feature filtering so inconsistent repeated metadata cannot
    be hidden by choosing one feature.
    """
    if feature_type not in {"roi", "pca"}:
        raise ValueError("feature_type must be 'roi' or 'pca'.")
    if pca_scope not in {"selected", "all"}:
        raise ValueError("pca_scope must be 'selected' or 'all'.")
    if not isinstance(field_name, str) or not field_name.strip() or field_name != field_name.strip():
        raise ValueError("field_name must be a nonempty, unpadded string.")
    path = Path(master_path).expanduser().resolve()
    frame = _read_csv(path)
    _require_columns(frame, SHARED_COLUMNS)
    _require_labels(frame, ["field_name"])
    frame = frame.loc[frame["field_name"].eq(field_name)].copy()
    if frame.empty:
        raise ValueError(f"No rows found for field_name={field_name!r}.")
    n_master_rows = len(frame)
    _require_labels(frame, ["person_id", "model_type", "step", "sex"])
    if frame["sex"].str.casefold().isin({"nan", "none", "null", "unknown", "n/a"}).any():
        raise ValueError("Missing or unknown sex categories are not valid modeling values.")
    series = _numeric(frame, "series_num")
    if not (series.gt(0) & series.eq(np.floor(series))).all():
        raise ValueError("series_num must contain positive integers.")
    frame["series_num"] = series.astype(int)
    for column in ("cgi_change", "ef_sf", "age"):
        frame[column] = _numeric(frame, column)
    normalization = normalization_metadata(frame, field_name, path)
    # Canonicalize documented aliases (e.g. normalization='none') and fill legacy
    # absent metadata.  The helper rejects conflicting explicit metadata first.
    frame["normalization"] = normalization["normalization"]
    frame["normalization_tag"] = normalization["normalization_tag"]
    if feature_type == "roi":
        feature_ids, value_column = _validate_roi(frame)
        if "basis_id" not in frame:
            frame["basis_id"] = ""
        else:
            frame["basis_id"] = frame["basis_id"].fillna("")
        feature_columns = ROI_FEATURE_COLUMNS
    else:
        feature_ids, value_column = _validate_pca(frame)
        feature_columns = PCA_FEATURE_COLUMNS
    _assign_checked(frame, "feature_type", feature_type)
    _assign_checked(frame, "feature_id", feature_ids)
    _assign_checked(frame, "E", _numeric(frame, value_column))
    with np.errstate(over="ignore", invalid="ignore"):
        interaction = frame["E"] * frame["ef_sf"]
    if not np.isfinite(interaction.to_numpy(dtype=float)).all():
        raise ValueError("E * ef_sf must be finite; supplied values overflowed.")
    _assign_checked(frame, "E_x_ef_sf", interaction)
    _assign_checked(frame, "value_column", value_column)
    configuration_keys = [*SERIES_KEYS, "normalization", "normalization_tag", "basis_id"]
    if frame.duplicated([*configuration_keys, "feature_id"]).any():
        raise ValueError("Duplicate series/feature/configuration observations.")
    metadata_columns = set(frame) - feature_columns - ADDED_FEATURE_COLUMNS
    _consistent_metadata(frame, configuration_keys, metadata_columns, "series clinical/stimulation")

    if features is not None:
        if isinstance(features, str):
            raise ValueError("features must be a sequence of full feature names, not a string.")
        requested = list(features)
        if not requested or any(not isinstance(value, str) or not value for value in requested):
            raise ValueError("Explicit feature selection must contain nonempty feature names.")
        if len(set(requested)) != len(requested):
            raise ValueError("Explicit feature selection contains duplicate names.")
        missing = set(requested) - set(frame["feature_id"])
        if missing:
            raise ValueError(f"Requested features are absent from master: {sorted(missing)}")
        frame = frame.loc[frame["feature_id"].isin(requested)].copy()
        selection = {"mode": "explicit", "features": requested}
    elif feature_type == "pca" and pca_scope == "selected":
        _require_columns(frame, ["selected_for_correlation"])
        selected_by_basis = frame.groupby("basis_id")["selected_for_correlation"].any()
        if not selected_by_basis.all():
            raise ValueError(
                "No PCA components have a saved selected_for_correlation flag in bases: "
                f"{selected_by_basis.index[~selected_by_basis].tolist()}"
            )
        frame = frame.loc[frame["selected_for_correlation"]].copy()
        selection = {"mode": "saved_selected_for_correlation"}
    else:
        selection = {"mode": "all"}
    frame = frame.reset_index(drop=True)
    return PreparedMaster(frame, {
        "feature_type": feature_type, "field_name": field_name,
        **normalization, "value_column": value_column, "statistic": value_column,
        "feature_selection": selection,
        "feature_ids": sorted(frame["feature_id"].unique().tolist()),
        "n_master_rows": n_master_rows, "n_input_rows": len(frame),
        "n_features": int(frame["feature_id"].nunique()),
        "n_bases": int(frame["basis_id"].nunique()) if feature_type == "pca" else 0,
        "feature_units": "saved_master_units", "outcome_convention": "saved_cgi_change_unchanged",
        "stimulation_factor": "saved_series_ef_sf_unchanged",
    })
