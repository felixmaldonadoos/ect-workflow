"""Clinical, demographic, and stimulation-data preparation."""

from __future__ import annotations

import json
from numbers import Real
from pathlib import Path
from typing import Literal, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from .identifiers import add_subject_id_columns, base_subject_id, build_scan_table


def load_ect_sessions(
    fn: str | Path, sheet_name: str | int = "stimulus", *,
    allow_missing_placement: bool = False, acute_only: bool = False,
) -> pd.DataFrame:
    """Load one row per ECT session, preserving all clinical columns.

    Validate only the session identity, date, and placement needed for matching.
    Other treatment variables remain untouched, including a temporary ``age``
    placeholder in frequency_hz. Session numbers need not restart at each series.
    No CGI eligibility filter or electric-field scaling is applied.
    allow_missing_placement is for course-mode summaries only; it retains
    missing/blank/literal 'nan' placements for explicit counting and warnings.
    acute_only excludes rows without a finite positive integer series_num
    before validating session IDs, dates, or placements. Series 0 is maintenance.
    """
    path = _require_file(fn, "ECT-session file")
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"ECT-session file must be .xlsx: {path}")
    if isinstance(sheet_name, bool) or not isinstance(sheet_name, (str, int)):
        raise TypeError("sheet_name must identify one sheet by name or integer index")
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, keep_default_na=False, na_values=[""])
    return _validate_ect_sessions(_ect_table_from_header(raw, "ECT-session sheet"),
                                  allow_missing_placement=allow_missing_placement, acute_only=acute_only)


def load_hdf5_query(
    fn: str | Path | pd.DataFrame,
    *,
    model_type: Literal["skin_single", "skin_double"],
    simulation_type: Literal["static", "adaptive", "step_0", "step_2"],
    query_filters: Mapping[str, str | int] | None = None,
) -> pd.DataFrame:
    """Select HDF5 records from a query CSV or an already loaded query table.

    model_type filters model_type_dir; static means step_0 and adaptive means
    step_2. Optional query_filters select exact metadata values, for example
    dataset_root, threshold, sim_result, or filename (derived when absent).
    This uses exported paths directly; it never scans directories or opens HDF5s.
    Multiple runs or datasets are retained for the mapper to check for ambiguity.
    """
    if model_type not in {"skin_single", "skin_double"}:
        raise ValueError("model_type must be 'skin_single' or 'skin_double'")
    steps = {"static": "step_0", "adaptive": "step_2", "step_0": "step_0", "step_2": "step_2"}
    if simulation_type not in steps:
        raise ValueError("simulation_type must be 'static', 'adaptive', 'step_0', or 'step_2'")
    step = steps[simulation_type]
    if isinstance(fn, pd.DataFrame):
        query = _ect_clean_columns(fn, "Results-query table")
    else:
        path = _require_file(fn, "Results-query file")
        raw = pd.read_csv(path, header=None, dtype="string", keep_default_na=False)
        query = _ect_table_from_header(raw, "Results-query file")
    required = ["model_type_dir", "step", "subject_dir", "electrode_configuration", "full_file_path"]
    _require_columns(query, required, "Results-query table")
    query["model_type_dir"] = query["model_type_dir"].astype("string").str.strip()
    query["step"] = query["step"].astype("string").str.strip()
    query = query.loc[query["model_type_dir"].eq(model_type) & query["step"].eq(step)].copy()
    if "filename" not in query.columns:
        query["filename"] = query["full_file_path"].map(lambda value: Path(str(value).strip()).name)
    for column, value in (query_filters or {}).items():
        if column in {"model_type_dir", "step"}:
            raise ValueError(f"Select {column!r} through model_type/simulation_type, not query_filters")
        _require_columns(query, [column], "Results-query table")
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError(f"query_filters[{column!r}] must be a nonempty string or integer")
        query = query.loc[query[column].astype("string").str.strip().eq(str(value).strip())].copy()
    query["full_file_path"] = _ect_required_text(query, "full_file_path", "Results-query table")
    is_hdf5 = query["full_file_path"].map(lambda value: Path(value).suffix.lower() in {".hdf5", ".h5"})
    query = query.loc[is_hdf5].copy()
    if query.empty:
        raise ValueError(f"No HDF5 records match model_type={model_type!r}, step={step!r}, {query_filters=}")
    ids = add_subject_id_columns(query[["subject_dir"]].rename(columns={"subject_dir": "subjid"}), require_scan=True)
    query["modeled_subjid"] = ids["subjid"]
    query["subjid_base"] = ids["subjid_base"]
    placement = _ect_required_text(query, "electrode_configuration", "Results-query table")
    query["electrode_placement"] = placement.str.upper()
    _ect_check_placements(query["electrode_placement"], "Results-query table")
    query["hdf5_fn"] = query["full_file_path"]
    return query.reset_index(drop=True)


def load_subject_scan_map(subject_scan_map: Mapping[str, str] | str | Path | None = None) -> dict[str, str]:
    """Read and validate {base_subject_id: preferred_modeled_scan_id}.

    Accept an in-memory dictionary or a JSON object stored in its own file.
    Entries are optional for patients with one candidate scan. A map may include
    other patients, so it can be reused across sheets and simulation conditions.
    Duplicate JSON keys, conflicting normalized keys, and cross-patient choices
    are errors. No scan is inferred from series_num or session_num.
    """
    if subject_scan_map is None:
        return {}
    if isinstance(subject_scan_map, (str, Path)):
        path = _require_file(subject_scan_map, "Subject-scan map")
        with path.open(encoding="utf-8") as handle:
            subject_scan_map = json.load(handle, object_pairs_hook=_ect_unique_json_object)
    if not isinstance(subject_scan_map, Mapping):
        raise TypeError("subject_scan_map must be a dictionary, a JSON-file path, or None")
    result = {}
    for subject, scan in subject_scan_map.items():
        if not isinstance(subject, str) or not isinstance(scan, str):
            raise TypeError("subject_scan_map keys and values must be subject-ID strings")
        subject, scan = subject.strip(), scan.strip()
        if subject != base_subject_id(subject):
            raise ValueError(f"subject_scan_map key must be a base subject ID: {subject!r}")
        if base_subject_id(scan, require_scan=True) != subject:
            raise ValueError(f"Preferred scan {scan!r} belongs to a different patient than {subject!r}")
        if subject in result:
            raise ValueError(f"Duplicate subject_scan_map key after stripping whitespace: {subject!r}")
        result[subject] = scan
    return result


def map_sessions_to_hdf5(
    sessions: pd.DataFrame,
    query_results: str | Path | pd.DataFrame,
    *,
    model_type: Literal["skin_single", "skin_double"],
    simulation_type: Literal["static", "adaptive", "step_0", "step_2"],
    subject_scan_map: Mapping[str, str] | str | Path | None = None,
    query_filters: Mapping[str, str | int] | None = None,
    electrode_placement_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Attach exactly one hdf5_fn and modeled_subjid to every ECT session.

    Match by base patient ID and the placement used in that individual session.
    Use subject_scan_map whenever the selected results contain multiple scans
    for that patient, even if different scans supply different placements.
    Clinical scan suffixes do not override the explicitly preferred scan.
    Multiple remaining files or missing matches raise with diagnostic tables.
    Repeated sessions can share an HDF5; row order, index, and count are retained.

    BL and BT remain distinct unless electrode_placement_map explicitly maps
    them, e.g. {"BT": "BL"}. Such an alias is the caller's modeling decision.
    """
    result = _validate_ect_sessions(sessions)
    reserved = {"hdf5_fn", "modeled_subjid"}.intersection(result.columns)
    if reserved:
        raise ValueError(f"Session table already contains output columns: {sorted(reserved)}")
    query = load_hdf5_query(query_results, model_type=model_type, simulation_type=simulation_type,
                           query_filters=query_filters)
    preferred = load_subject_scan_map(subject_scan_map)
    placements = _ect_placement_aliases(electrode_placement_map)
    subjects = result["subjid_base"].unique()
    query = query.loc[query["subjid_base"].isin(subjects)].copy()
    available = query.groupby("subjid_base")["modeled_subjid"].unique().to_dict()
    chosen, ambiguous = {}, {}
    for subject in subjects:
        scans = sorted(available.get(subject, []))
        if subject in preferred:
            if preferred[subject] not in scans:
                raise ValueError(f"Preferred scan {preferred[subject]!r} is unavailable after filtering; {scans=}")
            chosen[subject] = preferred[subject]
        elif len(scans) > 1:
            ambiguous[subject] = scans
        elif scans:
            chosen[subject] = scans[0]
    if ambiguous:
        raise ValueError("Multiple modeled scans require subject_scan_map={base_subjid: preferred_scan}: "
                         + str(ambiguous))
    query = query.loc[query["modeled_subjid"].eq(query["subjid_base"].map(chosen))].copy()

    keys = ["subjid_base", "electrode_placement"]
    matching = result[keys].copy()
    matching["electrode_placement"] = matching["electrode_placement"].replace(placements)
    wanted_keys = pd.MultiIndex.from_frame(matching)
    query = query.loc[pd.MultiIndex.from_frame(query[keys]).isin(wanted_keys)].copy()
    # Exact repeated records for the same file are one candidate, not extra sessions.
    candidates = query[[*keys, "modeled_subjid", "hdf5_fn"]].drop_duplicates()
    repeated_path = candidates["hdf5_fn"].duplicated(keep=False)
    if repeated_path.any():
        details = candidates.loc[repeated_path].to_string(index=False)
        raise ValueError("One HDF5 path has conflicting subject/placement metadata:\n" + details)
    multiple = candidates.duplicated(keys, keep=False)
    if multiple.any():
        diagnostic_columns = [c for c in [*keys, "dataset_root", "threshold", "sim_result", "hdf5_fn"] if c in query]
        conflicting_keys = pd.MultiIndex.from_frame(candidates.loc[multiple, keys])
        details = query.loc[pd.MultiIndex.from_frame(query[keys]).isin(conflicting_keys), diagnostic_columns]
        raise ValueError("Multiple HDF5 files match a session; restrict query_filters or the query export:\n"
                         + details.drop_duplicates().to_string(index=False))
    joined = matching.merge(candidates, on=keys, how="left", sort=False, validate="many_to_one")
    if len(joined) != len(result):
        raise RuntimeError(f"HDF5 matching changed the session count: {len(result)} -> {len(joined)}")
    missing = joined["hdf5_fn"].isna().to_numpy()
    if missing.any():
        details = result.loc[missing, ["subjid", "series_num", "session_num", "date", "electrode_placement"]]
        raise ValueError("No matching HDF5 for these sessions; no sessions were removed and no fallback was used:\n"
                         + details.to_string())
    result["modeled_subjid"] = joined["modeled_subjid"].to_numpy()
    result["hdf5_fn"] = joined["hdf5_fn"].to_numpy()
    return result


def load_ect_sessions_with_hdf5(
    fn: str | Path,
    sheet_name: str | int,
    query_results: str | Path | pd.DataFrame,
    *,
    model_type: Literal["skin_single", "skin_double"],
    simulation_type: Literal["static", "adaptive", "step_0", "step_2"],
    subject_scan_map: Mapping[str, str] | str | Path | None = None,
    query_filters: Mapping[str, str | int] | None = None,
    electrode_placement_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load an ECT-session sheet and attach its session-specific HDF5 paths."""
    sessions = load_ect_sessions(fn, sheet_name)
    return map_sessions_to_hdf5(sessions, query_results, model_type=model_type, simulation_type=simulation_type,
                               subject_scan_map=subject_scan_map, query_filters=query_filters,
                               electrode_placement_map=electrode_placement_map)


def _ect_clean_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    result = df.copy()
    names = pd.Index(result.columns.astype(str).str.strip())
    if names.duplicated().any():
        raise ValueError(f"{label} has duplicate column names: {names[names.duplicated()].tolist()}")
    result.columns = names
    return result


def _ect_table_from_header(raw: pd.DataFrame, label: str) -> pd.DataFrame:
    if raw.empty:
        raise ValueError(f"{label} is empty")
    header = raw.iloc[0].astype("string").str.strip()
    if header.isna().any() or header.eq("").any():
        raise ValueError(f"{label} contains blank column names")
    table = raw.iloc[1:].reset_index(drop=True).copy()
    table.columns = header.tolist()
    return _ect_clean_columns(table, label)


def _ect_required_text(df: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = df[column].astype("string").str.strip()
    missing = values.isna() | values.eq("")
    if missing.any():
        raise ValueError(f"{label} has missing {column!r} at rows: {df.index[missing].tolist()}")
    return values


def _ect_check_placements(placements: pd.Series, label: str) -> None:
    invalid = ~placements.isin({"RUL", "BL", "BT", "BF"})
    if invalid.any():
        raise ValueError(f"{label} has invalid electrode placements: {placements.loc[invalid].to_dict()}")


def _filter_acute_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Explicitly exclude maintenance and invalid series labels before acute analysis."""
    result = _ect_clean_columns(df, "ECT-session table")
    _require_columns(result, ["series_num"], "ECT-session table")
    # Coercion is used only to identify the intentionally excluded rows. No
    # missing course number is imputed, rounded, or carried forward.
    values = pd.to_numeric(result["series_num"], errors="coerce")
    boolean = result["series_num"].map(lambda value: isinstance(value, (bool, np.bool_)))
    keep = (values.notna() & np.isfinite(values) & values.ge(1) & values.mod(1).eq(0) & ~boolean).fillna(False)
    if (~keep).any():
        columns = [name for name in ["subjid", "series_num", "session_num", "date", "electrode_placement"] if name in result]
        excluded = result.loc[~keep, columns]
        print(f"Excluding {len(excluded)} stimulus rows from acute analysis: series_num is below 1 or not a finite integer.")
        print(excluded.to_string(index=True))
    result = result.loc[keep].copy()
    if result.empty:
        raise ValueError("No acute ECT sessions remain after filtering series_num")
    return result


def _validate_ect_sessions(
    df: pd.DataFrame, *, allow_missing_placement: bool = False, acute_only: bool = False,
) -> pd.DataFrame:
    result = _filter_acute_sessions(df) if acute_only else _ect_clean_columns(df, "ECT-session table")
    required = ["subjid", "series_num", "session_num", "date", "electrode_placement"]
    _require_columns(result, required, "ECT-session table")
    if result.empty:
        raise ValueError("ECT-session table contains no sessions")
    result = add_subject_id_columns(result, require_scan=False)
    for column in ["series_num", "session_num"]:
        if result[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"{column} must contain positive integers, not booleans")
        values = pd.to_numeric(result[column], errors="raise")
        valid = values.notna() & np.isfinite(values) & values.gt(0) & values.lt(2**63) & values.mod(1).eq(0)
        if not valid.all():
            raise ValueError(f"{column} must contain positive integers at rows: {result.index[~valid].tolist()}")
        result[column] = values.astype("int64")
    dates = _ect_required_text(result, "date", "ECT-session table")
    if result["date"].map(lambda value: isinstance(value, (Real, np.bool_))).any():
        raise ValueError("ECT-session dates must be dates or date strings, not unformatted numeric Excel serials")
    result["date"] = pd.to_datetime(dates, format="mixed", errors="raise").dt.normalize()
    if result["date"].isna().any():
        raise ValueError(f"Invalid ECT-session dates at rows: {result.index[result['date'].isna()].tolist()}")
    placement = result["electrode_placement"].astype("string").str.strip().str.upper()
    if allow_missing_placement:
        placement = placement.mask(placement.isin(["", "NAN"]))
        _ect_check_placements(placement.dropna(), "ECT-session table")
    else:
        _ect_check_placements(placement, "ECT-session table")
    result["electrode_placement"] = placement
    duplicate = result.duplicated(["subjid_base", "series_num", "session_num"], keep=False)
    if duplicate.any():
        details = result.loc[duplicate, ["subjid", "series_num", "session_num", "date"]].to_string()
        raise ValueError("Duplicate patient/series/session records:\n" + details)
    return result


def _ect_unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in subject_scan_map: {key!r}")
        result[key] = value
    return result


def _ect_placement_aliases(mapping: Mapping[str, str] | None) -> dict[str, str]:
    aliases = {}
    for source, target in (mapping or {}).items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise TypeError("electrode_placement_map keys and values must be placement strings")
        source, target = source.strip().upper(), target.strip().upper()
        _ect_check_placements(pd.Series([source, target]), "electrode_placement_map")
        if source in aliases:
            raise ValueError(f"Duplicate electrode_placement_map key after normalization: {source!r}")
        aliases[source] = target
    return aliases


def modal_placements_by_course(sessions: pd.DataFrame) -> pd.DataFrame:
    """Count acute sessions per patient/series; require a unique modal placement.

    Exclude rows whose series_num is below 1 or not a finite integer.
    Missing/blank/literal 'nan' placements are omitted from the count, with a
    warning and exported counts. Invalid nonmissing placements, duplicate
    sessions, ties, and series with no known placement are errors. BL and BT
    remain distinct clinical labels. Other stimulus variables are not used.
    """
    sessions = _validate_ect_sessions(sessions, allow_missing_placement=True, acute_only=True)
    rows = []
    for (base, series), group in sessions.groupby(["subjid_base", "series_num"], sort=True):
        counts = group["electrode_placement"].value_counts()
        if counts.empty:
            raise ValueError(f"No known electrode placement for {base}, series_num={series}")
        modes = sorted(counts.index[counts.eq(counts.max())].tolist())
        if len(modes) != 1:
            raise ValueError(f"Tied modal electrode placements for {base}, series_num={series}: {counts.to_dict()}")
        missing = int(group["electrode_placement"].isna().sum())
        if missing:
            warnings.warn(f"Ignoring {missing} missing placements for {base}, series_num={series}",
                          UserWarning, stacklevel=2)
        rows.append({"subjid_base": base, "series_num": int(series), "electrode_placement": modes[0],
                     "n_sessions": len(group), "n_known_placements": int(counts.sum()),
                     "n_missing_placements": missing, "n_modal_sessions": int(counts.max()),
                     "modal_fraction": float(counts.max() / counts.sum()),
                     "placement_counts": json.dumps({key: int(counts[key]) for key in sorted(counts.index)}),
                     "first_session_date": group["date"].min(), "last_session_date": group["date"].max()})
    return pd.DataFrame(rows)


def select_modal_course_hdf5(
    sessions: pd.DataFrame,
    inventory: pd.DataFrame,
    cognitive: pd.DataFrame,
    *,
    model_type: Literal["skin_single", "skin_double"],
    simulation_type: Literal["static", "adaptive", "step_0", "step_2"],
    subject_scan_map: Mapping[str, str] | str | Path | None = None,
    electrode_placement_map: Mapping[str, str] | None = None,
    outcome_col: str = "cgi_change",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one scan/placement per stimulus course, independent of CGI eligibility.

    For complete clinical courses, match series_num explicitly when present;
    otherwise require exactly one series with sessions in the inclusive acute
    date interval. Count only sessions inside that interval. Other stimulus
    series use all their sessions and remain eligible for ROI/PCA, without an
    outcome. A series cannot silently represent two clinical courses.

    The existing session mapper resolves scans (explicit map if ambiguous),
    aliases, and HDF5 candidates. Nonmodal placements need no simulation.
    Unique selected HDF5s enter PCA once even if reused by multiple courses.
    """
    query = load_hdf5_query(inventory, model_type=model_type, simulation_type=simulation_type)
    bases = set(query["subjid_base"])
    sessions = _validate_ect_sessions(sessions, allow_missing_placement=True, acute_only=True)
    sessions = sessions.loc[sessions["subjid_base"].isin(bases)].copy()
    missing = sorted(bases.difference(sessions["subjid_base"]))
    if missing:
        raise ValueError(f"No stimulus sessions for modeled patients; cannot select a course placement: {missing}")
    if outcome_col not in cognitive:
        cognitive = compute_cgi_change(cognitive, outcome_col=outcome_col)
    cognitive = validate_treatment_courses(cognitive, outcome_col=outcome_col)
    cognitive = cognitive.loc[cognitive["subjid_base"].isin(bases)].copy()
    _require_columns(cognitive, ["date_end (acute)"], "Cognitive-score table")
    if cognitive["date_end (acute)"].lt(cognitive["date_start"]).any():
        raise ValueError("Clinical acute end date precedes course start date")

    sessions_by_base = {base: group for base, group in sessions.groupby("subjid_base")}
    windows = {}
    for _, course in cognitive.iterrows():
        available = sessions_by_base[course["subjid_base"]]
        during = available.loc[available["date"].between(course["date_start"], course["date_end (acute)"])]
        candidates = sorted(during["series_num"].unique().tolist())
        supplied_series = course.get("series_num", pd.NA)
        if pd.notna(supplied_series) and str(supplied_series).strip():
            number = pd.to_numeric(supplied_series, errors="raise")
            if isinstance(supplied_series, (bool, np.bool_)) or not np.isfinite(number) or number < 1 or number % 1:
                raise ValueError(f"Invalid clinical series_num for {course['treatment_course_id']}: {supplied_series!r}")
            # Other series inside the same acute window are a clinical ambiguity,
            # even if a series number was supplied in the outcome sheet.
            if candidates != [int(number)]:
                raise ValueError(f"Clinical series_num conflicts with stimulus dates for {course['treatment_course_id']}; "
                                 f"series in acute interval: {candidates}")
            series = int(number)
        elif len(candidates) == 1:
            series = candidates[0]
        else:
            raise ValueError(f"Expected exactly one stimulus series in acute interval for {course['treatment_course_id']}; "
                             f"found {candidates}")
        key = (course["subjid_base"], series)
        if key in windows:
            raise ValueError(f"Multiple clinical courses map to the same stimulus patient/series: {key}")
        windows[key] = course

    pieces, links = [], []
    for key, group in sessions.groupby(["subjid_base", "series_num"], sort=True):
        course = windows.get(key)
        if course is not None:
            group = group.loc[group["date"].between(course["date_start"], course["date_end (acute)"])].copy()
        pieces.append(group)
        links.append({"subjid_base": key[0], "series_num": key[1],
                      "treatment_course_id": course["treatment_course_id"] if course is not None else pd.NA,
                      "mode_scope": "acute_date_window" if course is not None else "full_stimulus_series",
                      "date_start": course["date_start"] if course is not None else pd.NaT,
                      "date_end (acute)": course["date_end (acute)"] if course is not None else pd.NaT})
    counted = pd.concat(pieces, ignore_index=True)
    modes = modal_placements_by_course(counted)
    modes = modes.merge(pd.DataFrame(links), on=["subjid_base", "series_num"], validate="one_to_one")
    # Use an actual session having the modal placement to call the existing
    # strict mapper. Its date/session_num are not used to choose an MRI scan.
    representatives = counted.merge(modes[["subjid_base", "series_num", "electrode_placement"]],
                                    on=["subjid_base", "series_num", "electrode_placement"], validate="many_to_one")
    representatives = representatives.sort_values(["subjid_base", "series_num", "date", "session_num"])
    representatives = representatives.drop_duplicates(["subjid_base", "series_num"])
    mapped = map_sessions_to_hdf5(representatives, query, model_type=model_type, simulation_type=simulation_type,
                                  subject_scan_map=subject_scan_map, electrode_placement_map=electrode_placement_map)
    assignments = modes.merge(mapped[["subjid_base", "series_num", "modeled_subjid", "hdf5_fn"]],
                              on=["subjid_base", "series_num"], validate="one_to_one")
    selected = query.loc[query["hdf5_fn"].isin(assignments["hdf5_fn"])].copy()
    if selected["hdf5_fn"].duplicated().any():
        raise ValueError("Selected inventory contains repeated HDF5 paths")
    # Model/step are fixed by this run; dataset + scan + placement distinguish
    # alternative fields without changing canonical patient or scan IDs.
    _require_columns(selected, ["dataset_root"], "HDF5 inventory")
    selected["simulation_id"] = (selected["dataset_root"].astype(str) + "::" + selected["modeled_subjid"]
                                  + "::" + selected["electrode_placement"])
    if selected["simulation_id"].duplicated().any():
        raise ValueError("Multiple selected HDF5s identify the same dataset/scan/placement")
    assignments = assignments.merge(
        selected[["hdf5_fn", "simulation_id", "dataset_root", "electrode_placement"]].rename(
            columns={"electrode_placement": "simulated_placement"}), on="hdf5_fn", validate="many_to_one")
    return selected.reset_index(drop=True), assignments.reset_index(drop=True)


def prepare_mapped_course_outcomes(
    scans: pd.DataFrame,
    cognitive: pd.DataFrame,
    course_hdf5_map: pd.DataFrame,
    *,
    outcome_col: str = "cgi_change",
) -> pd.DataFrame:
    """Join explicit simulation/course assignments without scan × course expansion."""
    _require_columns(scans, ["subjid", "simulation_id", "hdf5_file"], "Scan table")
    scans = add_subject_id_columns(scans, require_scan=True)
    if scans["simulation_id"].isna().any() or scans["simulation_id"].duplicated().any():
        raise ValueError("Scan simulation_id values must be present and unique")
    required = ["subjid_base", "modeled_subjid", "simulation_id", "hdf5_fn", "treatment_course_id"]
    _require_columns(course_hdf5_map, required, "Course HDF5 map")
    mapping = course_hdf5_map.loc[course_hdf5_map["treatment_course_id"].notna()].copy()
    if mapping[required].isna().any(axis=None) or mapping["treatment_course_id"].duplicated().any():
        raise ValueError("Each mapped clinical course must identify exactly one complete simulation assignment")
    checked = mapping.merge(scans[["simulation_id", "subjid", "subjid_base", "hdf5_file"]],
                            on="simulation_id", how="left", validate="many_to_one", suffixes=("", "_scan"))
    paths_match = [Path(str(a)).resolve() == Path(str(b)).resolve()
                   for a, b in zip(checked["hdf5_fn"], checked["hdf5_file"], strict=True)]
    if (checked["subjid"].isna().any() or checked["subjid"].ne(checked["modeled_subjid"]).any()
            or checked["subjid_base"].ne(checked["subjid_base_scan"]).any() or not all(paths_match)):
        raise ValueError("Course HDF5 map conflicts with selected scan/path identities")
    cog = validate_treatment_courses(cognitive, outcome_col=outcome_col)
    eligible = cog.loc[cog["subjid_base"].isin(scans["subjid_base"])]
    if set(mapping["treatment_course_id"]) != set(eligible["treatment_course_id"]):
        raise ValueError("Course HDF5 map does not cover exactly the eligible clinical courses for selected patients")
    cog = cog.drop(columns=["scan_id"]).rename(columns={"subjid": "clinical_subjid"})
    metadata = mapping.drop(columns=[c for c in ["date_start", "date_end (acute)"] if c in mapping])
    result = metadata.merge(cog, on=["subjid_base", "treatment_course_id"], how="inner", validate="one_to_one",
                            suffixes=("", "_clinical"))
    if len(result) != len(mapping):
        raise ValueError("Course mapping has inconsistent patient/course identities")
    result = result.merge(scans, on=["simulation_id", "subjid_base"], validate="many_to_one", suffixes=("", "_scan"))
    if result.empty:
        raise ValueError("No selected simulations have an eligible treatment course")
    unmatched = sorted(set(scans["simulation_id"]).difference(result["simulation_id"]))
    if unmatched:
        warnings.warn(f"Simulations retained in ROI/PCA without eligible outcomes: {unmatched}", UserWarning, stacklevel=2)
    result["observation_id"] = result["simulation_id"] + "__" + result["treatment_course_id"]
    return result.sort_values(["subjid", "date_start"]).reset_index(drop=True)


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
    course_hdf5_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Use explicit course assignments when supplied; otherwise retain legacy matching."""
    if course_hdf5_map is not None:
        if not isinstance(scan_ids, pd.DataFrame):
            raise TypeError("Explicit course matching requires a scan DataFrame with simulation_id and hdf5_file")
        if scan_id_col != "subjid" or cognitive_id_col != "subjid" or start_date_col != "date_start":
            raise ValueError("Explicit course matching requires the standard subject and date column names")
        return prepare_mapped_course_outcomes(scan_ids, cognitive, course_hdf5_map, outcome_col=outcome_col)
    if isinstance(scan_ids, pd.DataFrame) and "simulation_id" in scan_ids:
        raise ValueError("Simulations indexed by simulation_id require explicit course_hdf5_map")
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
