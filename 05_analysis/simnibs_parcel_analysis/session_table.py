"""One exposure row per session, before any course-level selection/aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .lmm import SessionFields


def build_session_table(sessions: pd.DataFrame, fields: SessionFields, *, model_type: str, simulation_type: str) -> pd.DataFrame:
    """Join session metadata and raw/scaled ROI matrices by session_id.

    Input row order is preserved. The series key does not depend on CGI windows;
    an existing treatment_course_id from outcome linkage is retained separately.
    """
    required = {"session_id", "subjid_base", "series_num", "session_num", "date", "electrode_placement",
                "modeled_subjid", "hdf5_fn", "ef_sf"}
    if missing := required.difference(sessions):
        raise ValueError(f"Missing session table columns: {sorted(missing)}")
    if sessions.empty or sessions.columns.has_duplicates or sessions[list(required)].isna().any(axis=None):
        raise ValueError("Session table requires nonempty rows, unique columns, and complete keys")
    result = sessions.set_index("session_id", verify_integrity=True).copy()
    if model_type not in {"skin_single", "skin_double"} or simulation_type not in {"static", "adaptive"}:
        raise ValueError("Specify skin_single/skin_double and static/adaptive")
    if not isinstance(fields.atlas, str) or not fields.atlas.strip():
        raise ValueError("Specify an atlas name")
    if fields.raw.columns.has_duplicates or not fields.raw.columns.equals(fields.scaled.columns) or not len(fields.raw.columns):
        raise ValueError("Raw and scaled matrices require identical, unique, nonempty ROI columns")
    for matrix in (fields.raw, fields.scaled):
        if matrix.index.has_duplicates or matrix.index.isna().any() or set(matrix.index) != set(result.index):
            raise ValueError("ROI matrix session IDs must match the session table exactly")
        if not np.isfinite(matrix.to_numpy(dtype=float)).all() or (matrix.to_numpy(dtype=float) < 0).any():
            raise ValueError("ROI values must be finite and nonnegative")
    raw, scaled = fields.raw.reindex(result.index), fields.scaled.reindex(result.index)
    factors = pd.to_numeric(result["ef_sf"], errors="raise")
    if not np.isfinite(factors).all() or factors.le(0).any():
        raise ValueError("ef_sf must be finite and positive")
    if not np.allclose(scaled, raw.mul(factors, axis=0), rtol=1e-12, atol=0):
        raise ValueError("Scaled ROI values must equal raw E times the same session's ef_sf")
    metadata = {"series_id": result["subjid_base"] + "::series_" + result["series_num"].astype(str),
                "model_type": model_type, "simulation_type": simulation_type, "atlas": fields.atlas}
    raw, scaled = raw.add_prefix("EF_roi__"), scaled.add_prefix("EF_scaled__")
    if collisions := set(result.columns).intersection([*metadata, *raw.columns, *scaled.columns]):
        raise ValueError(f"Session input already contains generated columns: {sorted(collisions)}")
    result = pd.concat([result.assign(**metadata), raw, scaled], axis=1).reset_index()
    leading = ["session_id", "series_id", "treatment_course_id", "subjid", "subjid_base", "series_num", "session_num", "date",
               "electrode_placement", "pulse_width_ms", "frequency_hz", "percent_charge", "ef_sf", "modeled_subjid",
               "hdf5_fn", "surface_file", "model_type", "simulation_type", "atlas", "roi_field", "roi_statistic"]
    leading = [column for column in leading if column in result]
    return result[[*leading, *[column for column in result if column not in leading]]]
