"""Course-outcome LMMs built from explicitly mapped session exposure histories.

Sessions are exposure records, not replicated CGI observations. Atlas tables
contain unscaled field summaries for unique HDF5 paths. ``ef_sf`` is the legacy
stimulation index, not physical charge or an E-field amplitude calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence
import warnings

import numpy as np
import pandas as pd

from .clinical import _validate_ect_sessions, compute_cgi_change, compute_ef_scaling_factor, validate_treatment_courses
from .identifiers import add_subject_id_columns


Aggregation = Literal["sum", "mean"]
DOSE_COLUMNS = ("ef_sf", "percent_charge")
SESSION_KEYS = ["subjid_base", "series_num", "session_num"]


@dataclass(frozen=True)
class CourseLink:
    sessions: pd.DataFrame
    courses: pd.DataFrame
    excluded_sessions: pd.DataFrame
    excluded_courses: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class SessionFields:
    """Identically indexed session x ROI matrices, indexed by session_id."""

    raw: pd.DataFrame
    scaled: pd.DataFrame
    atlas: str


@dataclass(frozen=True)
class LMMResult:
    coefficients: pd.DataFrame
    design: pd.DataFrame
    scaling: pd.DataFrame
    predictions: pd.DataFrame
    diagnostics: dict


def read_lmm_table(path: str | Path, *, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read CSV/XLSX without header mangling or silent row removal."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".xlsx":
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None, keep_default_na=False, na_values=[""])
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, header=None, keep_default_na=False, na_values=[""], skip_blank_lines=False)
    else:
        raise ValueError(f"Expected .csv or .xlsx: {path}")
    if len(raw) < 2:
        raise ValueError(f"No data rows in {path}")
    header = raw.iloc[0].astype("string").str.strip()
    if header.isna().any() or header.eq("").any() or header.duplicated().any():
        raise ValueError(f"Missing or duplicate column names in {path}")
    result = raw.iloc[1:].reset_index(drop=True).copy()
    result.columns = header.tolist()
    return result


def _numeric(frame: pd.DataFrame, columns: Sequence[str], *, nonnegative: bool = False) -> pd.DataFrame:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    result = frame.copy()
    for name in columns:
        if result[name].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise ValueError(f"Boolean values are not valid numeric measurements in {name}")
        result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
        valid = np.isfinite(result[name])
        if nonnegative:
            valid &= result[name].ge(0)
        if not valid.all():
            raise ValueError(f"Missing, nonfinite, or invalid {name} at rows {result.index[~valid].tolist()}")
    return result


def prepare_session_dose(sessions: pd.DataFrame) -> pd.DataFrame:
    """Preserve the legacy product exactly; reject unknown/invalid parameters."""
    columns = ["pulse_width_ms", "frequency_hz", "percent_charge"]
    result = _numeric(sessions, columns, nonnegative=True)
    if result[columns].le(0).any(axis=None):
        raise ValueError("Stimulation parameters must be positive for every included treatment session")
    expected = compute_ef_scaling_factor(result)
    if "ef_sf" in result:
        supplied = _numeric(result, ["ef_sf"])["ef_sf"]
        if not np.allclose(supplied, expected["ef_sf"], rtol=1e-12, atol=0):
            raise ValueError("Existing ef_sf conflicts with pulse_width_ms * frequency_hz * percent_charge")
    if not np.isfinite(expected["ef_sf"]).all():
        raise ValueError("ef_sf overflowed; check stimulation parameters")
    return expected


def _missing_value(values: pd.Series) -> pd.Series:
    """Recognize absent cells, whitespace, and literal nan without coercing other text."""
    return values.isna() | values.astype("string").str.strip().str.lower().isin(["", "nan"])


def _complete_cgi_courses(outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude incomplete pairs while rejecting invalid nonmissing scores."""
    if outcomes.columns.has_duplicates:
        raise ValueError("Outcome table has duplicate columns")
    missing = pd.DataFrame({name: _missing_value(outcomes[name]) for name in ["cgi_start", "cgi_end"]})
    for name in missing:
        _numeric(outcomes.loc[~missing[name]], [name])
    incomplete = missing.any(axis=1)
    excluded = outcomes.loc[incomplete].copy()
    if "exclusion_reason" in outcomes:
        raise ValueError("Outcome table already contains reserved exclusion_reason column")
    excluded["exclusion_reason"] = np.where(missing.loc[incomplete].all(axis=1), "missing_cgi_start_and_end",
                                            np.where(missing.loc[incomplete, "cgi_start"], "missing_cgi_start", "missing_cgi_end"))
    complete = outcomes.loc[~incomplete].copy()
    if complete.empty:
        raise ValueError(f"No courses with both cgi_start and cgi_end remain; excluded {len(excluded)} incomplete courses")
    return complete, excluded


def _incomplete_course_sessions(sessions: pd.DataFrame, courses: pd.DataFrame, excluded: pd.DataFrame) -> pd.Series:
    """Identify sessions belonging to excluded courses without guessing other unmatched links."""
    mask = pd.Series(False, index=sessions.index)
    if excluded.empty:
        return mask
    # Blank unprocessed outcome rows have no usable patient identity.
    identified = add_subject_id_columns(excluded.loc[~_missing_value(excluded["subjid"])])
    included_patients = set(courses["subjid_base"])
    for row_index, course in identified.iterrows():
        patient = sessions["subjid_base"].eq(course["subjid_base"])
        if not patient.any():
            continue
        if course["subjid_base"] not in included_patients:
            mask |= patient
            continue
        supplied = course.get("series_num", pd.NA)
        if not _missing_value(pd.Series([supplied])).iloc[0]:
            series = pd.to_numeric(supplied, errors="raise")
            if isinstance(supplied, (bool, np.bool_)) or not np.isfinite(series) or series < 1 or series % 1 != 0:
                raise ValueError(f"Invalid series_num for incomplete outcome at row {row_index}: {supplied!r}")
        else:
            dates = pd.Series([course["date_start"], course["date_end (acute)"]])
            if _missing_value(dates).any():
                continue  # Insufficient identity: leave unmatched sessions subject to the existing strict check.
            if dates.map(lambda x: isinstance(x, (int, float, np.number, bool))).any():
                raise ValueError(f"Incomplete outcome dates must be dates, not numeric Excel serials, at row {row_index}")
            start, end = pd.to_datetime(dates, format="mixed", errors="raise").dt.normalize()
            if pd.isna(start) or pd.isna(end) or end < start:
                raise ValueError(f"Invalid incomplete outcome window at row {row_index}")
            candidates = sessions.loc[patient & sessions["date"].between(start, end), "series_num"].unique()
            if len(candidates) == 0:
                continue
            if len(candidates) > 1:
                raise ValueError(f"Incomplete outcome window matches multiple series at row {row_index}")
            series = candidates[0]
        mask |= patient & sessions["series_num"].eq(series)
    return mask


def link_sessions_to_courses(
    sessions: pd.DataFrame, outcomes: pd.DataFrame, *, allow_unmatched_sessions: bool = False,
) -> CourseLink:
    """Assign sessions to courses and report the source rows behind conflicts."""
    sessions = _validate_ect_sessions(sessions)
    session_row_indices = sessions.index.to_numpy(copy=True)
    sessions = sessions.reset_index(drop=True)

    reserved = {"session_id", "treatment_course_id", "cgi_start", "cgi_end", "cgi_change"}
    if reserved.intersection(sessions):
        raise ValueError("Session table already contains outcome/output columns: "
                         + str(sorted(reserved.intersection(sessions))))

    required = {"subjid", "cgi_start", "cgi_end", "date_start", "date_end (acute)"}
    if missing := required.difference(outcomes):
        raise KeyError(f"Outcome table is missing {sorted(missing)}")
    if outcomes.empty:
        raise ValueError("Outcome table is empty")

    complete, excluded_courses = _complete_cgi_courses(outcomes)
    courses = _numeric(complete, ["cgi_start", "cgi_end"])

    for date in ["date_start", "date_end (acute)"]:
        if courses[date].map(lambda x: isinstance(x, (int, float, np.number, bool))).any():
            raise ValueError(f"{date} must contain dates, not numeric Excel serials")

    change = compute_cgi_change(courses)
    if "cgi_change" in courses:
        supplied = _numeric(courses, ["cgi_change"])["cgi_change"]
        if not np.allclose(supplied, change["cgi_change"], rtol=0, atol=1e-12):
            raise ValueError("cgi_change must equal cgi_end - cgi_start")

    courses = validate_treatment_courses(change)
    if courses[["date_start", "date_end (acute)"]].isna().any(axis=None):
        raise ValueError("Missing course window dates")
    if courses["date_end (acute)"].lt(courses["date_start"]).any():
        raise ValueError("Course end precedes course start")

    for _, group in courses.groupby("subjid_base"):
        ordered = group.sort_values("date_start")
        if ordered["date_start"].iloc[1:].reset_index(drop=True).le(
                ordered["date_end (acute)"].iloc[:-1].reset_index(drop=True)).any():
            raise ValueError(f"Overlapping inclusive outcome windows for {group.iloc[0]['subjid_base']}")

    assignment = pd.Series(pd.NA, index=sessions.index, dtype="string")
    series_used, series_numbers = set(), []

    for _, course in courses.iterrows():
        mask = sessions["subjid_base"].eq(course["subjid_base"])
        mask &= sessions["date"].between(course["date_start"], course["date_end (acute)"], inclusive="both")
        candidates = sessions.loc[mask, "series_num"].unique().tolist()

        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one session series for {course['treatment_course_id']}; "
                             f"got {candidates}")

        series = int(candidates[0])
        supplied = course.get("series_num", pd.NA)
        if pd.notna(supplied) and str(supplied).strip():
            value = pd.to_numeric(supplied, errors="raise")
            if isinstance(supplied, (bool, np.bool_)) or not np.isfinite(value) or value != series:
                raise ValueError(f"Outcome series_num conflicts with dates for {course['treatment_course_id']}")

        key = (course["subjid_base"], series)
        if key in series_used:
            raise ValueError(f"Multiple outcomes map to patient/series {key}")

        series_used.add(key)
        series_numbers.append(series)
        assignment.loc[mask] = course["treatment_course_id"]

    courses["series_num"] = series_numbers
    incomplete_sessions = _incomplete_course_sessions(sessions, courses, excluded_courses)
    conflict = incomplete_sessions & assignment.notna()

    if conflict.any():
        columns = ["subjid", "series_num", "date_start", "date_end (acute)", "cgi_start", "cgi_end"]

        # Identify the complete outcomes assigned to the conflicting sessions.
        complete_mask = courses["treatment_course_id"].isin(assignment.loc[conflict])
        complete_details = courses.loc[complete_mask, columns].copy()

        # validate_treatment_courses resets indices but preserves row order.
        original_outcome_rows = complete.index.to_numpy()[complete_mask.to_numpy()]
        complete_details.insert(0, "outcome_row_index", original_outcome_rows)

        # Recheck each excluded outcome separately to identify the exact cause.
        incomplete_details = []
        for position in range(len(excluded_courses)):
            row = excluded_courses.iloc[[position]]
            row_conflict = _incomplete_course_sessions(sessions, courses, row) & conflict
            if not row_conflict.any():
                continue

            report_columns = [name for name in [*columns, "exclusion_reason"] if name in row]
            details = row[report_columns].copy()
            details.insert(0, "outcome_row_index", row.index.to_numpy())
            details["matched_series"] = str(sorted(sessions.loc[row_conflict, "series_num"].unique().tolist()))

            supplied = row.iloc[0].get("series_num", pd.NA)
            inferred = _missing_value(pd.Series([supplied])).iloc[0]
            details["match_basis"] = "date window -> series" if inferred else "explicit series_num"
            incomplete_details.append(details)

        affected = sessions.loc[conflict, SESSION_KEYS + ["date"]].copy()
        affected.insert(0, "session_row_index", session_row_indices[conflict.to_numpy()])
        shown = min(len(affected), 20)

        raise ValueError(
            "Complete and incomplete outcome rows identify the same session series; resolve conflicting courses.\n\n"
            "Complete outcome rows:\n"
            + complete_details.to_string(index=False, na_rep="<missing>")
            + "\n\nConflicting incomplete outcome rows:\n"
            + pd.concat(incomplete_details, ignore_index=True).to_string(index=False, na_rep="<missing>")
            + f"\n\nAffected sessions (showing {shown} of {len(affected)}):\n"
            + affected.head(shown).to_string(index=False)
            + "\n\nAn incomplete outcome excludes the entire matched patient/series, "
              "including sessions outside its date window."
            + "\nRow indices refer to the original input tables. With the runner's Excel reader "
              "and a header in row 1, Excel row = row_index + 2."
        )

    unmatched = assignment.isna()
    excluded = sessions.loc[unmatched].copy()
    excluded["exclusion_reason"] = np.where(
        incomplete_sessions.loc[unmatched], "incomplete_cgi_course", "outside_all_outcome_windows"
    )

    unexplained = unmatched & ~incomplete_sessions
    if unexplained.any() and not allow_unmatched_sessions:
        raise ValueError(
            f"{unexplained.sum()} sessions have no outcome window and cannot be linked to an incomplete CGI course; "
            "explicitly allow exclusion or fix input:\n"
            + sessions.loc[unexplained, SESSION_KEYS + ["date"]].to_string(index=False)
        )

    result = sessions.loc[~unmatched].copy()
    result["treatment_course_id"] = assignment.loc[~unmatched]
    result = result.sort_values(["subjid_base", "series_num", "date", "session_num"]).reset_index(drop=True)
    result["session_id"] = (result["subjid_base"] + "::series_" + result["series_num"].astype(str)
                            + "::session_" + result["session_num"].astype(str))

    courses = courses.sort_values(["subjid_base", "date_start"]).reset_index(drop=True)
    return CourseLink(result, courses, excluded.reset_index(drop=True), excluded_courses)


def attach_course_covariates(courses: pd.DataFrame, subject_info: pd.DataFrame) -> pd.DataFrame:
    """Join explicit patient metadata; prefer course-specific age in outcomes."""
    info = add_subject_id_columns(subject_info)
    if info["subjid_base"].duplicated().any():
        raise ValueError("Subject covariates require exactly one row per base patient; use course rows for varying age")
    columns = [c for c in info if c not in {"subjid", "subjid_base", "scan_id", "mrn"}]
    if overlap := set(columns).intersection(courses):
        raise ValueError(f"Covariates already exist in outcomes: {sorted(overlap)}")
    result = courses.merge(info[["subjid_base", *columns]], on="subjid_base", how="left", validate="many_to_one",
                           indicator="_covariate_match")
    if result["_covariate_match"].ne("both").any():
        raise ValueError("Subject covariates do not cover all outcome patients")
    return result.drop(columns="_covariate_match")


def roi_table_from_summary(summary, *, method: str = "spatial_weighted") -> pd.DataFrame:
    """Bridge an existing ParcelSummary to the HDF5-keyed wide atlas interface.

    Extract every required scan/placement, not only the PCA course-mode subset.
    Selection of atlas and percentile estimator belongs to the atlas extractor.
    """
    values = summary.p95(method).copy()
    scans = summary.scans.copy()
    key = "simulation_id" if "simulation_id" in scans else "subjid"
    if scans[key].duplicated().any() or values.index.has_duplicates:
        raise ValueError(f"ParcelSummary must contain unique {key} values")
    if set(values.index) != set(scans[key]):
        raise ValueError("ParcelSummary predictor and scan identities differ")
    paths = scans.set_index(key)["hdf5_file"].reindex(values.index)
    values.insert(0, "hdf5_fn", paths.to_numpy())
    return values.reset_index(drop=True)


def build_session_roi_matrices(
    sessions: pd.DataFrame, roi_table: pd.DataFrame, *, atlas: str, exclude_rois: Sequence[str] = (),
) -> SessionFields:
    """Broadcast each session's HDF5 ROI row and multiply that row by ef_sf.

    roi_table is wide: hdf5_fn, ROI_1, ..., ROI_M. One atlas/statistic/model
    condition per table. Paths match exactly; no scan-only or basename joins.
    Extra simulation rows are allowed; every used row must have every ROI.
    """
    if not isinstance(atlas, str) or not atlas.strip():
        raise ValueError("Provide an explicit atlas name")
    for frame, names in [(sessions, ["session_id", "hdf5_fn", "ef_sf"]), (roi_table, ["hdf5_fn"])]:
        if missing := set(names).difference(frame):
            raise KeyError(f"Missing matrix input columns: {sorted(missing)}")
        if frame.columns.has_duplicates or frame[names].isna().any(axis=None):
            raise ValueError("Matrix inputs have duplicate columns or missing keys")
    if sessions.empty or sessions["session_id"].duplicated().any():
        raise ValueError("Session IDs must be nonempty and unique")
    for paths in [sessions["hdf5_fn"], roi_table["hdf5_fn"]]:
        if not paths.map(lambda x: isinstance(x, str) and bool(x.strip()) and x == x.strip()).all():
            raise ValueError("hdf5_fn must contain exact nonblank path strings without surrounding whitespace")
    if roi_table["hdf5_fn"].duplicated().any():
        raise ValueError("ROI table has duplicate HDF5 paths")
    roi_names = [c for c in roi_table if c != "hdf5_fn"]
    if any(not isinstance(c, str) or not c.strip() or c != c.strip() for c in roi_names):
        raise ValueError("ROI column names must be nonblank strings without surrounding whitespace")
    if unknown := set(exclude_rois).difference(roi_names):
        raise ValueError(f"Requested ROI exclusions not found: {sorted(unknown)}")
    roi_names = [c for c in roi_names if c not in exclude_rois]
    if not roi_names or "session_id" in roi_names:
        raise ValueError("Need at least one ROI; session_id is a reserved name")
    source = roi_table.set_index("hdf5_fn")
    if missing := set(sessions["hdf5_fn"]).difference(source.index):
        raise ValueError(f"No atlas fields for session HDF5s: {sorted(missing)}")
    raw = _numeric(source.loc[sessions["hdf5_fn"], roi_names], roi_names, nonnegative=True)
    raw.index = pd.Index(sessions["session_id"], name="session_id")
    factors = _numeric(sessions, ["ef_sf"], nonnegative=True)["ef_sf"].to_numpy()
    if (factors <= 0).any():
        raise ValueError("ef_sf must be positive")
    scaled = raw.mul(factors, axis=0)
    if not np.isfinite(scaled.to_numpy()).all():
        raise ValueError("Scaled ROI values overflowed")
    return SessionFields(raw, scaled, atlas.strip())


def aggregate_course_exposures(
    sessions: pd.DataFrame, courses: pd.DataFrame, *, aggregation: Aggregation, dose_column: str = "ef_sf",
    fields: SessionFields | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """One clinical row per course plus optional course x ROI exposure matrices.

    The interaction is aggregated AFTER multiplication within each session.
    Both sum and mean are explicit hypotheses about exposure history. Neither
    encodes temporal ordering, spacing, decay, or a session-specific CGI effect.
    """
    if aggregation not in {"sum", "mean"} or dose_column not in DOSE_COLUMNS:
        raise ValueError("Choose aggregation=sum/mean and dose_column=ef_sf/percent_charge")
    needed = {"session_id", "treatment_course_id", "subjid_base", "series_num", "date", dose_column}
    if missing := needed.difference(sessions):
        raise KeyError(f"Missing session columns: {sorted(missing)}")
    if sessions["session_id"].duplicated().any() or sessions[list(needed)].isna().any(axis=None):
        raise ValueError("Missing session keys or duplicated session_id")
    if courses["treatment_course_id"].duplicated().any():
        raise ValueError("Exactly one outcome row is required per treatment_course_id")
    if set(sessions["treatment_course_id"]) != set(courses["treatment_course_id"]):
        raise ValueError("Session and outcome course sets differ")
    identities = sessions[["treatment_course_id", "subjid_base", "series_num"]].drop_duplicates()
    check = identities.merge(courses[["treatment_course_id", "subjid_base", "series_num"]],
                             on="treatment_course_id", validate="one_to_one", suffixes=("", "_outcome"))
    if (check["subjid_base"].ne(check["subjid_base_outcome"]).any()
            or check["series_num"].ne(check["series_num_outcome"]).any()):
        raise ValueError("Session patient/series identities conflict with outcomes")
    prepared = _numeric(sessions, [dose_column], nonnegative=True)
    group = prepared.groupby("treatment_course_id", sort=False)
    aggregate = group.agg(n_sessions=("session_id", "size"), first_session_date=("date", "min"),
                          last_session_date=("date", "max"))
    aggregate["dose"] = group[dose_column].agg(aggregation)
    if overlap := (set(aggregate.columns) | {"ef_raw", "ef_scaled", "ef_dose"}).intersection(courses):
        raise ValueError(f"Outcome table already contains derived columns: {sorted(overlap)}")
    result = courses.merge(aggregate, left_on="treatment_course_id", right_index=True, validate="one_to_one")
    result["course_duration_days"] = (result["last_session_date"] - result["first_session_date"]).dt.days
    matrices = {}
    if fields is not None:
        index = pd.Index(sessions["session_id"], name="session_id")
        if not fields.raw.index.equals(index) or not fields.scaled.index.equals(index):
            raise ValueError("ROI matrix rows must exactly match the ordered session IDs")
        if not fields.raw.columns.equals(fields.scaled.columns):
            raise ValueError("Raw and scaled ROI columns differ")
        raw = _numeric(fields.raw, list(fields.raw), nonnegative=True)
        scaled = _numeric(fields.scaled, list(fields.scaled), nonnegative=True)
        factors = _numeric(sessions, ["ef_sf"], nonnegative=True)["ef_sf"].to_numpy()
        if not np.allclose(scaled, raw.mul(factors, axis=0), rtol=1e-12, atol=0):
            raise ValueError("Scaled matrix does not equal raw E multiplied by session ef_sf")
        sources = {"ef_raw": raw, "ef_scaled": scaled,
                   "ef_dose": raw.mul(prepared[dose_column].to_numpy(), axis=0)}
        for name, frame in sources.items():
            matrix = frame.groupby(sessions["treatment_course_id"].to_numpy(), sort=False).agg(aggregation)
            matrix.index.name = "treatment_course_id"
            matrices[name] = matrix.reindex(result["treatment_course_id"])
            if not np.isfinite(matrices[name].to_numpy()).all():
                raise ValueError(f"Nonfinite aggregated exposure in {name}")
    return result, matrices


def build_fixed_effects(
    courses: pd.DataFrame, *, roi_model: Literal["none", "scaled", "components"] = "none",
    numeric_covariates: Sequence[str] = (), categorical_covariates: Sequence[str] = (),
    include_session_count: bool = True, age_sex_interaction: bool = False, standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an auditable design with no formula parsing or implicit missing-data drops."""
    if roi_model not in {"none", "scaled", "components"}:
        raise ValueError("roi_model must be none, scaled, or components")
    columns = ["cgi_start", "dose"]
    if include_session_count:
        columns.append("n_sessions")
    columns += {"none": [], "scaled": ["ef_scaled"], "components": ["ef_raw", "ef_dose"]}[roi_model]
    columns += list(numeric_covariates)
    all_names = columns + list(categorical_covariates)
    forbidden = {"Intercept", "cgi_change", "cgi_end", "subjid", "subjid_base", "treatment_course_id",
                 "series_num", "session_num", "mrn"}
    if len(set(all_names)) != len(all_names) or forbidden.intersection(all_names):
        raise ValueError("Duplicate, outcome-derived, ID, or reserved covariate names")
    design = _numeric(courses, columns)[columns].copy()
    scaling = []
    for name in columns:
        mean, sd = float(design[name].mean()), float(design[name].std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Constant or invalid predictor {name}; remove it explicitly (e.g. --no-session-count)")
        scaling.append({"term": name, "center": mean if standardize else 0.0,
                        "scale": sd if standardize else 1.0})
        if standardize:
            design[name] = (design[name] - mean) / sd
    for name in categorical_covariates:
        if name not in courses or courses[name].isna().any():
            raise ValueError(f"Missing categorical covariate {name}")
        values = courses[name].astype(str).str.strip()
        if values.eq("").any() or values.nunique() < 2:
            raise ValueError(f"Empty or constant categorical covariate {name}")
        levels = sorted(values.unique())
        for level in levels[1:]:
            term = f"{name}[{level}]"
            if term in design:
                raise ValueError(f"Covariate name collision: {term}")
            design[term] = values.eq(level).astype(float)
            scaling.append({"term": term, "center": 0.0, "scale": 1.0, "reference": levels[0]})
    if age_sex_interaction:
        if "age" not in numeric_covariates or "sex" not in categorical_covariates:
            raise ValueError("age_sex_interaction requires numeric age and categorical sex")
        for term in [c for c in design if c.startswith("sex[")]:
            design[f"age:{term}"] = design["age"] * design[term]
    design.insert(0, "Intercept", 1.0)
    values = design.to_numpy(dtype=float)
    if len(design) <= design.shape[1]:
        raise ValueError(f"Need more course outcomes ({len(design)}) than fixed coefficients ({design.shape[1]})")
    if not np.isfinite(values).all() or np.linalg.matrix_rank(values) != values.shape[1]:
        raise ValueError("Fixed-effect design is rank deficient/nonfinite; inspect collinearity and covariate coding")
    if np.linalg.cond(values) > 1e8:
        warnings.warn("Fixed-effect design is ill-conditioned; inspect correlated exposure summaries", UserWarning)
    design.index = pd.Index(courses["treatment_course_id"], name="treatment_course_id")
    return design, pd.DataFrame(scaling)


def fit_course_lmm(
    courses: pd.DataFrame, design: pd.DataFrame, *, scaling: pd.DataFrame | None = None,
    reml: bool = True, optimizer: str = "powell", maxiter: int = 2000,
) -> LMMResult:
    """Gaussian patient-random-intercept LMM; stop on failed/invalid inference.

    Confidence intervals and p-values are asymptotic Wald quantities, not
    small-sample corrected. No random series/session effect is identifiable
    from one outcome per series. A zero estimated patient variance is reported
    as a boundary fit; invalid Hessians or nonconvergence fail explicitly.
    """
    import statsmodels
    from statsmodels.regression.mixed_linear_model import MixedLM

    if (optimizer not in {"powell", "lbfgs", "bfgs"}
            or isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1):
        raise ValueError("Choose optimizer=powell/lbfgs/bfgs and a positive integer maxiter")
    index = pd.Index(courses["treatment_course_id"], name="treatment_course_id")
    if index.has_duplicates or index.isna().any() or not design.index.equals(index):
        raise ValueError("Model design must align exactly with unique course outcomes")
    if courses["subjid_base"].isna().any():
        raise ValueError("Missing patient grouping ID")
    groups = courses["subjid_base"].astype(str)
    counts = groups.value_counts()
    repeated = int(counts.gt(1).sum())
    if len(counts) < 2 or repeated == 0:
        raise ValueError("A patient random intercept requires repeated courses; "
                         "all-singleton data cannot separate variances")
    if repeated < 5:
        warnings.warn(f"Only {repeated} patients have repeated courses; "
                      "patient variance may be poorly estimated", UserWarning)
    values = design.to_numpy(dtype=float)
    if (len(design) <= design.shape[1] or not np.isfinite(values).all()
            or np.linalg.matrix_rank(values) < values.shape[1]):
        raise ValueError("Model design is nonfinite, rank deficient, or has insufficient residual degrees of freedom")
    observed = _numeric(courses, ["cgi_change"])["cgi_change"].to_numpy()
    if np.std(observed) == 0:
        raise ValueError("CGI change is constant")
    model = MixedLM(pd.Series(observed, index=index), design, groups=groups.to_numpy(), missing="raise")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fitted = model.fit(reml=reml, method=optimizer, maxiter=maxiter, disp=False)
    messages = [str(item.message) for item in captured]
    for message in messages:
        warnings.warn(f"MixedLM: {message}", UserWarning)
    if not fitted.converged:
        raise RuntimeError(f"MixedLM did not converge ({optimizer=}, {maxiter=}); warnings: {messages}")
    if any("not positive definite" in message or "singular" in message.lower() for message in messages):
        raise RuntimeError(f"MixedLM reported singular covariance or an invalid Hessian; warnings: {messages}")
    k = design.shape[1]
    covariance = np.asarray(fitted.cov_params())[:k, :k]
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
        raise RuntimeError("Invalid fixed-effect covariance; Wald inference is unavailable")
    parameters = np.asarray(fitted.fe_params)
    se = np.asarray(fitted.bse_fe)
    ci = np.asarray(fitted.conf_int())[:k]
    p = np.asarray(fitted.pvalues)[:k]
    if not all(np.isfinite(x).all() for x in [parameters, se, ci, p]):
        raise RuntimeError("Nonfinite MixedLM estimates or uncertainty")
    patient_var, residual_var = float(np.asarray(fitted.cov_re)[0, 0]), float(fitted.scale)
    if not np.isfinite(patient_var + residual_var) or patient_var < 0 or residual_var <= 0:
        raise RuntimeError("Invalid variance estimates")
    icc = patient_var / (patient_var + residual_var)
    boundary = bool(icc < 1e-6)
    if boundary:
        warnings.warn("Patient variance is near zero (ICC < 1e-6); "
                      "this is a boundary fit, not proof of independence", UserWarning)
    coefficients = pd.DataFrame({"term": design.columns, "estimate": parameters, "std_error": se,
                                 "z": parameters / se, "p_value": p, "ci_low": ci[:, 0], "ci_high": ci[:, 1]})
    marginal = values @ parameters
    # Closed-form random-intercept BLUP also remains defined at the zero boundary.
    residual = pd.Series(observed - marginal)
    group_key = groups.reset_index(drop=True)
    shifts = residual.groupby(group_key).sum() * patient_var / (residual_var + counts * patient_var)
    conditional = marginal + group_key.map(shifts).to_numpy()
    predictions = pd.DataFrame({"treatment_course_id": index, "subjid_base": groups.to_numpy(),
                                "observed_cgi_change": observed, "marginal_fitted": marginal,
                                "conditional_fitted": conditional, "conditional_residual": observed - conditional})
    diagnostics = {"n_courses": len(courses), "n_patients": len(counts), "n_repeated_patients": repeated,
                   "courses_per_patient": {str(key): int(value) for key, value in counts.sort_index().items()},
                   "fixed_terms": list(design.columns), "patient_variance": patient_var,
                   "residual_variance": residual_var, "icc": icc, "boundary_fit": boundary,
                   "converged": True, "reml": reml, "optimizer": optimizer, "log_likelihood": float(fitted.llf),
                   "condition_number": float(np.linalg.cond(values)), "warnings": messages,
                   "inference": "asymptotic normal Wald; no small-sample correction",
                   "statsmodels_version": statsmodels.__version__}
    return LMMResult(coefficients, design, scaling if scaling is not None else pd.DataFrame(), predictions, diagnostics)
