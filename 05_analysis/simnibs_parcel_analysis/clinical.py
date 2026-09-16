"""Clinical, demographic, and stimulation-data preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence
import warnings

import numpy as np
import pandas as pd

from .identifiers import add_subject_id_columns, build_scan_table

def collapse_person_rows(
    df: pd.DataFrame,
    *,
    id_col: str = "subjid",
    value_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Collapse repeated scan-labeled records after verifying person-level agreement."""
    result = add_subject_id_columns(df, source_col=id_col, require_scan=False)
    if value_columns is None:
        excluded = {id_col, "subjid_base", "scan_id", "subjid_org"}
        value_columns = [column for column in result.columns if column not in excluded]
    _require_columns(result, value_columns, "Person-level table")

    conflicts = {}
    for column in value_columns:
        counts = result.groupby("subjid_base")[column].nunique(dropna=False)
        bad = counts[counts > 1].index.tolist()
        if bad:
            conflicts[column] = bad
    if conflicts:
        raise ValueError(f"Conflicting person-level values across scan-labeled rows: {conflicts}")
    return result[["subjid_base", *value_columns]].drop_duplicates("subjid_base").reset_index(drop=True)


def compute_cgi_change(
    df: pd.DataFrame,
    *,
    start_col: str = "cgi_start",
    end_col: str = "cgi_end",
    outcome_col: str = "cgi_change",
    direction: Literal["end_minus_start", "start_minus_end"] = "end_minus_start",
) -> pd.DataFrame:
    """Calculate CGI change without discarding any input columns or rows."""
    _require_columns(df, [start_col, end_col], "CGI table")
    if direction not in {"end_minus_start", "start_minus_end"}:
        raise ValueError("direction must be 'end_minus_start' or 'start_minus_end'")

    result = df.copy()
    result[[start_col, end_col]] = result[[start_col, end_col]].apply(pd.to_numeric, errors="raise")
    _require_finite(result, [start_col, end_col], "CGI table")
    if direction == "end_minus_start":
        result[outcome_col] = result[end_col] - result[start_col]
    else:
        result[outcome_col] = result[start_col] - result[end_col]
    return result


def load_cognitive_scores(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    id_col: str = "subjid",
    change_direction: Literal["end_minus_start", "start_minus_end"] = "end_minus_start",
    drop_columns: Sequence[str] = ("mrn",),
) -> pd.DataFrame:
    """Load complete treatment courses and calculate CGI change."""
    path = _require_file(path, "Cognitive-score file")
    df = pd.read_excel(path, sheet_name=sheet_name)
    cleaned_columns = df.columns.astype(str).str.strip()
    if cleaned_columns.duplicated().any():
        duplicates = cleaned_columns[cleaned_columns.duplicated()].tolist()
        raise ValueError(f"Cleaning whitespace creates duplicate columns: {duplicates}")
    df.columns = cleaned_columns
    required = [id_col, "cgi_start", "cgi_end", "date_start", "date_end (acute)"]
    _require_columns(df, required, "Cognitive-score table")

    df["clinical_row_index"] = df.index
    df[["cgi_start", "cgi_end"]] = df[["cgi_start", "cgi_end"]].replace(r"^\s*$", pd.NA, regex=True)
    complete = df[["cgi_start", "cgi_end"]].notna().all(axis=1)
    excluded = df.loc[~complete, ["clinical_row_index", id_col, "cgi_start", "cgi_end"]]
    df = df.loc[complete].copy()
    print(f"Clinical rows loaded: {len(complete)}")
    print(f"Complete treatment courses retained: {len(df)}")
    print(f"Incomplete treatment courses excluded: {len(excluded)}")
    if not excluded.empty:
        print(excluded.to_string(index=False))
    if df.empty:
        raise ValueError("No treatment courses have both CGI start and CGI end values")

    df["subjid_org"] = df[id_col].copy()
    existing_drop_columns = [column for column in drop_columns if column in df.columns]
    if existing_drop_columns:
        df = df.drop(columns=existing_drop_columns)
    df = add_subject_id_columns(df, source_col=id_col, require_scan=False)
    df = compute_cgi_change(df, direction=change_direction)
    return validate_treatment_courses(df, id_col=id_col)


def validate_treatment_courses(
    cognitive: pd.DataFrame,
    *,
    id_col: str = "subjid",
    start_date_col: str = "date_start",
    end_date_col: str = "date_end (acute)",
    outcome_col: str = "cgi_change",
) -> pd.DataFrame:
    """Require one clinical row per base subject and treatment-course start date."""
    required = [id_col, start_date_col, outcome_col]
    _require_columns(cognitive, required, "Cognitive-score table")
    result = add_subject_id_columns(cognitive, source_col=id_col, require_scan=False)
    result[outcome_col] = pd.to_numeric(result[outcome_col], errors="raise")
    _require_finite(result, [outcome_col], "Cognitive-score table")

    date_columns = [start_date_col]
    if end_date_col in result.columns:
        date_columns.append(end_date_col)
    for column in date_columns:
        if result[column].isna().any():
            rows = result.index[result[column].isna()].tolist()
            raise ValueError(f"Missing {column!r} values at rows: {rows}")
        result[column] = pd.to_datetime(result[column], format="mixed", errors="raise").dt.normalize()

    conflict_mask = result.duplicated(["subjid_base", start_date_col], keep=False)
    if conflict_mask.any():
        columns = [id_col, "subjid_base", start_date_col, outcome_col]
        if end_date_col in result.columns:
            columns.insert(3, end_date_col)
        conflicts = result.loc[conflict_mask, columns].sort_values(["subjid_base", start_date_col])
        raise ValueError(
            "Multiple clinical rows have the same subject and treatment-course start date:\n"
            + conflicts.to_string(index=False)
        )

    result["treatment_course_id"] = result["subjid_base"] + "_" + result[start_date_col].dt.strftime("%Y%m%d")
    if result["treatment_course_id"].duplicated().any():
        raise RuntimeError("Treatment-course ID construction produced duplicates")
    return result.reset_index(drop=True)


def prepare_scan_course_outcomes(
    scan_ids: Sequence[str] | pd.DataFrame,
    cognitive: pd.DataFrame,
    *,
    scan_id_col: str = "subjid",
    cognitive_id_col: str = "subjid",
    outcome_col: str = "cgi_change",
    start_date_col: str = "date_start",
) -> pd.DataFrame:
    """Expand modeled scans to all treatment courses matched by ``subjid_base``."""
    if isinstance(scan_ids, pd.DataFrame):
        _require_columns(scan_ids, [scan_id_col], "Scan table")
        scans = scan_ids.rename(columns={scan_id_col: "subjid"}) if scan_id_col != "subjid" else scan_ids.copy()
        scans = add_subject_id_columns(scans, source_col="subjid", require_scan=True)
        duplicate_mask = scans["subjid"].duplicated(keep=False)
        if duplicate_mask.any():
            duplicates = scans.loc[duplicate_mask, "subjid"].tolist()
            raise ValueError(f"Modeled-scan IDs must be unique: {duplicates}")
    else:
        scans = build_scan_table(scan_ids)

    _require_columns(cognitive, [cognitive_id_col, start_date_col, outcome_col], "Cognitive-score table")
    cog = cognitive.copy()
    if cognitive_id_col != "subjid":
        if "subjid" in cog.columns:
            raise ValueError("Cannot rename cognitive ID column because 'subjid' already exists")
        cog = cog.rename(columns={cognitive_id_col: "subjid"})
    cog = validate_treatment_courses(cog, id_col="subjid", start_date_col=start_date_col, outcome_col=outcome_col)
    cog = cog.drop(columns=["scan_id"]).rename(columns={"subjid": "clinical_subjid"})

    modeled_bases = set(scans["subjid_base"])
    clinical_bases = set(cog["subjid_base"])
    missing_bases = sorted(modeled_bases.difference(clinical_bases))
    if missing_bases:
        excluded_scans = scans.loc[scans["subjid_base"].isin(missing_bases), "subjid"].tolist()
        warnings.warn(
            "Excluding modeled scans with no eligible treatment course from outcome analyses: "
            f"{excluded_scans} (base subjects: {missing_bases}). These scans remain in the ROI/PCA analysis.",
            UserWarning,
            stacklevel=2,
        )

    matched_scans = scans.loc[scans["subjid_base"].isin(clinical_bases)].copy()
    if matched_scans.empty:
        raise ValueError("No modeled scans have an eligible treatment course; outcome analysis cannot be run")

    result = matched_scans.merge(cog, on="subjid_base", how="inner", validate="many_to_many", suffixes=("", "_clinical"))
    missing = result.loc[result[outcome_col].isna(), "subjid"].tolist()
    if missing:
        raise ValueError(f"Missing {outcome_col!r} after scan-to-course matching: {missing}")

    scan_counts = matched_scans.groupby("subjid_base").size()
    course_counts = cog.groupby("subjid_base").size()
    expected_rows = int(sum(scan_counts[base] * course_counts[base] for base in scan_counts.index))
    if len(result) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} scan-course rows, got {len(result)}")

    result["observation_id"] = result["subjid"] + "__" + result[start_date_col].dt.strftime("%Y%m%d")
    if result["observation_id"].duplicated().any():
        duplicates = result.loc[result["observation_id"].duplicated(keep=False), "observation_id"].tolist()
        raise RuntimeError(f"Duplicate scan-course observation IDs: {duplicates}")
    return result.sort_values(["subjid", start_date_col]).reset_index(drop=True)


def prepare_scan_outcomes(
    scan_ids: Sequence[str] | pd.DataFrame,
    cognitive: pd.DataFrame,
    *,
    scan_id_col: str = "subjid",
    cognitive_id_col: str = "subjid",
    outcome_col: str = "cgi_change",
    start_date_col: str = "date_start",
) -> pd.DataFrame:
    """Backward-compatible alias for treatment-course-aware outcome matching."""
    return prepare_scan_course_outcomes(
        scan_ids,
        cognitive,
        scan_id_col=scan_id_col,
        cognitive_id_col=cognitive_id_col,
        outcome_col=outcome_col,
        start_date_col=start_date_col,
    )


def compute_ef_scaling_factor(df: pd.DataFrame, *, output_col: str = "ef_sf") -> pd.DataFrame:
    """Calculate pulse width × frequency × percent charge."""
    columns = ["pulse_width_ms", "frequency_hz", "percent_charge"]
    _require_columns(df, columns, "Stimulus table")
    result = df.copy()
    result[columns] = result[columns].apply(pd.to_numeric, errors="raise")
    _require_finite(result, columns, "Stimulus table")
    result[output_col] = result["pulse_width_ms"] * result["frequency_hz"] * result["percent_charge"]
    return result


def load_stimulus(
    path: str | Path,
    *,
    sheet_name: str | int = "stimulus",
    pulse_width_map: dict[float, float] | None = None,
) -> pd.DataFrame:
    """Load stimulus data without silently removing incomplete sessions."""
    path = _require_file(path, "Stimulus file")
    df = pd.read_excel(path, sheet_name=sheet_name)
    required = ["subjid", "date", "pulse_width_ms", "frequency_hz", "percent_charge"]
    _require_columns(df, required, "Stimulus table")
    if df[required].isna().any().any():
        rows = df.index[df[required].isna().any(axis=1)].tolist()
        raise ValueError(f"Missing required stimulus values at rows: {rows}")

    df = add_subject_id_columns(df, require_scan=False)
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="raise").dt.normalize()
    if pulse_width_map is None:
        pulse_width_map = {5: 0.5, 25: 0.25, 50: 0.50, 75: 0.75}
    df["pulse_width_ms"] = df["pulse_width_ms"].replace(pulse_width_map)
    return compute_ef_scaling_factor(df)


def load_subject_info(path: str | Path) -> pd.DataFrame:
    """Load age and sex information from an explicitly supplied file."""
    path = _require_file(path, "Subject-info file")
    df = pd.read_csv(path)
    _require_columns(df, ["subjid", "age", "sex"], "Subject-info table")
    df = add_subject_id_columns(df, require_scan=False)
    df["age"] = pd.to_numeric(df["age"], errors="raise")
    _require_finite(df, ["age"], "Subject-info table")
    if df["sex"].isna().any():
        rows = df.index[df["sex"].isna()].tolist()
        raise ValueError(f"Missing sex values at rows: {rows}")
    return df


def attach_subject_info(
    scans: pd.DataFrame,
    subject_info: pd.DataFrame,
    *,
    age_aggregation: Literal["min", "max", "mean", "first"] | None = None,
) -> pd.DataFrame:
    """Attach person-level age and sex to every modeled-scan row."""
    _require_columns(scans, ["subjid"], "Scan table")
    scans = add_subject_id_columns(scans, require_scan=True)
    _require_columns(subject_info, ["subjid", "age", "sex"], "Subject-info table")
    info = add_subject_id_columns(subject_info, require_scan=False)
    info["age"] = pd.to_numeric(info["age"], errors="raise")
    _require_finite(info, ["age"], "Subject-info table")

    sex_counts = info.groupby("subjid_base")["sex"].nunique(dropna=False)
    sex_conflicts = sex_counts[sex_counts > 1].index.tolist()
    if sex_conflicts:
        raise ValueError(f"Conflicting sex values for base subjects: {sex_conflicts}")
    age_counts = info.groupby("subjid_base")["age"].nunique(dropna=False)
    age_conflicts = age_counts[age_counts > 1].index.tolist()
    if age_conflicts and age_aggregation is None:
        raise ValueError(f"Conflicting ages require an explicit age_aggregation: {age_conflicts}")
    if age_aggregation not in {None, "min", "max", "mean", "first"}:
        raise ValueError("age_aggregation must be None, 'min', 'max', 'mean', or 'first'")

    age_method = age_aggregation or "first"
    person_info = info.groupby("subjid_base", as_index=False).agg(age=("age", age_method), sex=("sex", "first"))
    result = scans.merge(person_info, on="subjid_base", how="left", validate="many_to_one")
    missing = result.loc[result[["age", "sex"]].isna().any(axis=1), "subjid"].tolist()
    if missing:
        raise ValueError(f"Missing subject information for modeled scans: {missing}")
    return result


def combine_stimulus_cognitive(stimulus: pd.DataFrame, cognitive: pd.DataFrame) -> pd.DataFrame:
    """Attach one acute CGI record to each in-window stimulation session."""
    stim = add_subject_id_columns(stimulus, require_scan=False)
    _require_columns(stim, ["date"], "Stimulus table")
    _require_columns(cognitive, ["date_start", "date_end (acute)", "cgi_start", "cgi_end", "cgi_change"], "Cognitive-score table")
    cog = validate_treatment_courses(cognitive)

    stim["stimulus_row_index"] = stim.index
    stim["date"] = pd.to_datetime(stim["date"], format="mixed", errors="raise").dt.normalize()
    result = stim.merge(cog, on="subjid_base", how="left", suffixes=("_stim", "_cog"), validate="many_to_many")
    in_window = result["date"].between(result["date_start"], result["date_end (acute)"], inclusive="both")
    result = result.loc[in_window].copy()
    if result.empty:
        raise ValueError("No stimulation sessions fall within the acute-treatment windows")

    match_counts = result.groupby("stimulus_row_index").size()
    overlapping = match_counts[match_counts > 1].index.tolist()
    if overlapping:
        conflicts = result.loc[result["stimulus_row_index"].isin(overlapping), ["stimulus_row_index", "subjid_base", "date", "date_start", "date_end (acute)"]]
        raise ValueError("Stimulation sessions match overlapping treatment courses:\n" + conflicts.to_string(index=False))

    unmatched = sorted(set(stim["stimulus_row_index"]).difference(result["stimulus_row_index"]))
    if unmatched:
        print(f"Excluded {len(unmatched)} stimulation rows outside every acute-treatment window: {unmatched}")
    result["cgi"] = np.nan
    result.loc[result["date"].eq(result["date_start"]), "cgi"] = result["cgi_start"]
    result.loc[result["date"].eq(result["date_end (acute)"]), "cgi"] = result["cgi_end"]
    return result.sort_values(["subjid_base", "date"]).reset_index(drop=True)


def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def _require_finite(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    values = df[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        rows = df.index[~np.isfinite(values).all(axis=1)].tolist()
        raise ValueError(f"{name} contains missing or non-finite values in {list(columns)} at rows: {rows}")


def _require_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path
