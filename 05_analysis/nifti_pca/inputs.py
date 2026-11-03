"""Reuse the current NIfTI series matching without duplicating its policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SERIES_KEYS = ["person_id", "series_num", "model_type", "step", "field_name"]

# An existing ROI master can be used as an input manifest, but its spatial
# summaries are not voxel-PCA inputs. Brain means are recomputed on the PCA grid.
ROI_COLUMNS = {
    "roi_org", "roi_name", "roi", "hemi", "network", "n_voxels",
    "mean", "mean_raw", "mode", "median", "min", "max", "p95", "p99",
    "whole_brain_mean", "whole_brain_n_voxels", "normalization",
    "normalization_tag", "normalization_applies_to", "normalization_mask_path",
    "normalization_mask_rule", "unnormalized_spatial_statistics",
}


def validate_series_inputs(frame: pd.DataFrame, field_name: str) -> pd.DataFrame:
    """Keep every series metadata column and collapse only verified ROI repeats."""
    required = {*SERIES_KEYS, "full_file_path", "cgi_change"}
    if frame.columns.has_duplicates:
        raise ValueError("Input table contains duplicate column names.")
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing series-input columns: {sorted(missing)}")
    selected = frame.loc[frame["field_name"].eq(field_name)].copy()
    if selected.empty:
        raise ValueError(f"No input rows for field_name={field_name!r}.")
    if selected[[*SERIES_KEYS, "full_file_path"]].isna().any().any():
        raise ValueError("Missing series/configuration identifiers or field paths.")
    for column in ["person_id", "model_type", "step", "field_name", "full_file_path"]:
        selected[column] = selected[column].astype(str).str.strip()
        if selected[column].eq("").any():
            raise ValueError(f"Empty values in {column!r}.")
    series_number = pd.to_numeric(selected["series_num"], errors="raise")
    if (
        not np.isfinite(series_number.to_numpy(dtype=float)).all()
        or series_number.le(0).any()
        or series_number.mod(1).ne(0).any()
    ):
        raise ValueError("series_num must contain positive integers.")
    selected["series_num"] = series_number.astype(np.int64)
    selected["cgi_change"] = pd.to_numeric(selected["cgi_change"], errors="raise")
    if not np.isfinite(selected["cgi_change"].to_numpy(dtype=float)).all():
        raise ValueError("cgi_change contains missing or non-finite values.")

    selected = selected.drop(columns=list(ROI_COLUMNS.intersection(selected.columns)))
    metadata_columns = [column for column in selected if column not in SERIES_KEYS]
    counts = selected.groupby(SERIES_KEYS, sort=False)[metadata_columns].nunique(
        dropna=False
    )
    conflicts = counts.columns[counts.gt(1).any()].tolist()
    if conflicts:
        raise ValueError(
            "Conflicting metadata within a series/configuration; no rows were "
            f"arbitrarily selected. Conflicting columns: {conflicts}"
        )
    selected = selected.drop_duplicates(SERIES_KEYS).reset_index(drop=True)
    outcome_counts = selected.groupby(["person_id", "series_num"])[
        "cgi_change"
    ].nunique(dropna=False)
    if outcome_counts.ne(1).any():
        raise ValueError("Inconsistent CGI change across configurations for a series.")

    if "sample_id" not in selected:
        identities = selected[SERIES_KEYS].itertuples(index=False, name=None)
        selected["sample_id"] = [
            "series_" + hashlib.sha256(
                json.dumps(list(identity), default=str, separators=(",", ":")).encode()
            ).hexdigest()[:20]
            for identity in identities
        ]
    if selected["sample_id"].isna().any():
        raise ValueError("Missing sample_id values.")
    selected["sample_id"] = selected["sample_id"].astype(str).str.strip()
    if selected["sample_id"].eq("").any() or selected["sample_id"].duplicated().any():
        raise ValueError("sample_id must be nonempty and unique per series/configuration.")
    return selected


def load_series_inputs(
    *,
    field_name: str,
    series_csv: Path | None = None,
    query_path: Path | None = None,
    clinical_path: Path | None = None,
    subject_scan_map: Path | None = None,
    steps: list[str] | None = None,
    exclude_subjects: list[str] | None = None,
) -> pd.DataFrame:
    if series_csv is not None:
        source = Path(series_csv).expanduser().resolve()
        frame = pd.read_csv(source)
        if "full_file_path" in frame:
            def resolve_field(value):
                if pd.isna(value):
                    return value
                path = Path(value).expanduser()
                return str((path if path.is_absolute() else source.parent / path).resolve())
            frame["full_file_path"] = frame["full_file_path"].map(resolve_field)
        return validate_series_inputs(frame, field_name)

    # Import only for the clinical/query route. This reuses the existing session-2
    # matching and strict alignment policy while the larger refactor is deferred.
    from networks.mni_yeo2011_prep_data import load_series_tables

    tables = load_series_tables(
        fn_query=query_path,
        fn_cgi=clinical_path,
        subject_scan_map=subject_scan_map,
        field_name=field_name,
        steps=steps,
        subj_exclude=exclude_subjects,
    )
    if not tables:
        raise ValueError("No model/step series tables were loaded.")
    return validate_series_inputs(pd.concat(tables.values(), ignore_index=True), field_name)
