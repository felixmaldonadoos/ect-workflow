"""Load clinical/query tables and map prefiltered stimulation sessions to files.

After selecting the desired rows in each table, prepare the mapped sessions::

    df_stim_mapped = prepare_session_data(
        df_query=df_query,
        df_cgi=df_cgi,
        df_stim=df_stim,
        filename="tdcs_uq_gpc_fsavg.msh",
        subject_scan_map="subject_scan_map.json",
    )
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .prepare_session_file_table import map_session_files


def _add_person_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["person_id"] = df["subjid"].str.extract(
        r"^(subj-cat-\d{3})(?:-\d{3})?$", expand=False
    )

    invalid = df["person_id"].isna()
    if invalid.any():
        raise ValueError(
            f"Invalid/missing subjid at rows {df.index[invalid].tolist()}"
        )
    return df


def load_stimulation_data(fn: str | Path) -> pd.DataFrame:
    """Read the stimulus sheet and add validated person IDs."""
    df = pd.read_excel(fn, sheet_name="stimulus")
    return _add_person_id(df)


def load_course_data(fn: str | Path) -> pd.DataFrame:
    """Read eligible course rows and add CGI change and validated person IDs."""
    df = pd.read_excel(fn, sheet_name="Sheet1")
    df = df.loc[
        df["in_stim"].eq(True) & df["batch_num"].gt(0)
    ].copy()

    df["cgi_change"] = df["cgi_start"] - df["cgi_end"]
    return _add_person_id(df)


def load_query_results(fn: str | Path) -> pd.DataFrame:
    """Read all query rows and attach their FreeSurfer subjects directory."""
    fs_subjects_dirs = {
        "STU00225089": "/projects/p32903/Alex2/datasets/STU00225089/subjectsdir",
        "STU00225089_synthsr": "/projects/p32903/Alex2/datasets/STU00225089/synthsr/subjectsdir"
    }
    df_query = pd.read_csv(fn)
    df_query["fs_subjects_dir"] = df_query["dataset_root"].map(fs_subjects_dirs)

    if df_query["fs_subjects_dir"].isna().any():
        missing = df_query.loc[df_query["fs_subjects_dir"].isna(), "dataset_root"].unique()
        raise ValueError(f"Missing FreeSurfer SUBJECTS_DIR mapping for: {missing}")
    return df_query

def collapse_session(df:pd.DataFrame, col_name:str='session_num', col_value=2):
    # check if col_name is inside df
    # only keep values that match that col_name pair. 
    df_filtered = df[df[col_name]==col_value].copy()
    return df_filtered


def _compute_ef_sf(df: pd.DataFrame) -> pd.DataFrame:
    """Add the stimulation scaling factor without modifying the input table."""
    df = df.copy()
    df["ef_sf"] = df["pulse_width_ms"] * df["frequency_hz"] * df["percent_charge"]
    return df


def prepare_session_data(
    df_query: pd.DataFrame,
    df_cgi: pd.DataFrame,
    df_stim: pd.DataFrame,
    *,
    filename: str | None = None,
    subject_scan_map: Mapping[str, str] | str | Path | None = None,
) -> pd.DataFrame:
    """Compute scaling factors and map prefiltered sessions to simulation files.

    Person IDs are derived from each clinical table's ``subjid`` column. Only
    sessions whose person ID occurs in ``df_cgi`` are retained, preserving their
    original order, index, and columns. No subject exclusions, query metadata
    filters, or session-number selections are applied. CGI rows are used for
    cohort membership only; course outcomes are not merged into the result.

    ``filename`` optionally selects an exact basename; omit it when ``df_query``
    already contains only the desired files. ``subject_scan_map`` explicitly
    selects modeled scans, using a mapping or a JSON path when needed.

    Inputs are not modified. Invalid data and missing or ambiguous mappings
    raise errors from validation or the mapper without being suppressed.
    """
    courses = _add_person_id(df_cgi)
    sessions = _compute_ef_sf(_add_person_id(df_stim))
    sessions = sessions.loc[sessions["person_id"].isin(courses["person_id"])].copy()

    return map_session_files(
        sessions=sessions,
        query=df_query,
        filename=filename,
        subject_scan_map=subject_scan_map,
    )
