"""Align saved treatment-series covariates for null and feature comparisons."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


COHORT_KEYS = ["person_id", "series_num"]
COHORT_VALUES = ["cgi_change", "age", "sex"]
COHORT_COLUMNS = [*COHORT_KEYS, *COHORT_VALUES]


def canonical_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse validated feature/configuration repeats, never distinct series.

    A source master repeats the same clinical values for many features and
    configurations. Conflicting repeated values are an error, not an averaging
    rule. The returned rows are exactly the variables used by the shared null.
    """
    missing = set(COHORT_COLUMNS) - set(frame)
    if missing or frame.empty:
        raise ValueError(f"Empty or invalid cohort; missing columns: {sorted(missing)}")
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate cohort column names")
    data = frame[COHORT_COLUMNS].copy()
    for column in ("person_id", "sex"):
        values = data[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any() or not values.eq(values.str.strip()).all():
            raise ValueError(f"Incomplete or padded cohort identity/category: {column}")
        data[column] = values.astype(str)
    for column in ("series_num", "cgi_change", "age"):
        if data[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"Boolean cohort values are invalid in {column}")
        data[column] = pd.to_numeric(data[column], errors="raise").astype(float)
        if not np.isfinite(data[column]).all():
            raise ValueError(f"Nonfinite cohort values in {column}")
    if not (data.series_num.gt(0) & data.series_num.eq(np.floor(data.series_num))).all():
        raise ValueError("Cohort series numbers must be positive integers")
    data["series_num"] = data.series_num.astype(int)
    if data.age.lt(0).any():
        raise ValueError("Cohort age must be nonnegative")
    counts = data.groupby(COHORT_KEYS, dropna=False)[COHORT_VALUES].nunique(dropna=False)
    if conflicts := counts.columns[counts.gt(1).any()].tolist():
        raise ValueError(f"Conflicting repeated series covariates across features/configurations: {conflicts}")
    return data.drop_duplicates(COHORT_KEYS).sort_values(COHORT_KEYS).reset_index(drop=True)


def cohort_sha256(frame: pd.DataFrame) -> str:
    canonical = canonical_cohort(frame)
    payload = json.dumps(canonical.to_dict("records"), sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_same_cohort(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    """Require identical series and model covariates, allowing CSV roundoff."""
    left, right = canonical_cohort(expected), canonical_cohort(actual)
    if not left[COHORT_KEYS].equals(right[COHORT_KEYS]):
        raise ValueError("Cohort mismatch: null and feature model use different treatment-series keys")
    if not left.sex.equals(right.sex):
        raise ValueError("Cohort mismatch: saved sex categories differ")
    for column in ("cgi_change", "age"):
        if not np.allclose(left[column], right[column], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Cohort mismatch: saved {column} differs")
