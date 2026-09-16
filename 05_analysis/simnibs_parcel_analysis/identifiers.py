"""Strict parsing and alignment of person and modeled-scan identifiers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

import re 

SUBJECT_ID_PATTERN = re.compile(r"subj-[A-Za-z0-9]+-\d{3}-\d{3}")

_BASE_ID_RE = re.compile(r"^subj-[A-Za-z0-9]+-\d{3}$")
_SCAN_ID_RE = re.compile(r"^(?P<base>subj-[A-Za-z0-9]+-\d{3})-(?P<scan>\d{3})$")
_SCAN_ID_IN_PATH_RE = re.compile(r"subj-[A-Za-z0-9]+-\d{3}-\d{3}")


def base_subject_id(subject_id: str, *, require_scan: bool = False) -> str:
    """Return the person-level ID, removing only a terminal ``-###`` scan ID."""
    value = str(subject_id).strip()
    match = _SCAN_ID_RE.fullmatch(value)
    if match:
        return match.group("base")
    if not require_scan and _BASE_ID_RE.fullmatch(value):
        return value
    expected = "subj-<group>-###-###" if require_scan else "subj-<group>-###[-###]"
    raise ValueError(f"Invalid subject ID {subject_id!r}; expected {expected}")


def scan_id(subject_id: str) -> str | None:
    """Return the final three-digit scan suffix, or ``None`` for a base ID."""
    value = str(subject_id).strip()
    match = _SCAN_ID_RE.fullmatch(value)
    if match:
        return match.group("scan")
    if _BASE_ID_RE.fullmatch(value):
        return None
    raise ValueError(f"Invalid subject ID {subject_id!r}")


def subject_id_from_path(path: str | Path) -> str:
    """Extract exactly one modeled-scan ID from a filesystem path."""
    matches = sorted(set(_SCAN_ID_IN_PATH_RE.findall(str(path))))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one modeled-scan ID in path, found {matches}: {path}")
    return matches[0]

def infer_subject_id(path: str | Path) -> str:
    """Infer exactly one modeled-scan ID from a file path."""
    matches = sorted(set(SUBJECT_ID_PATTERN.findall(str(path))))

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one modeled-scan ID in path, "
            f"found {matches}: {path}"
        )

    return matches[0]

def add_subject_id_columns(
    df: pd.DataFrame,
    *,
    source_col: str = "subjid",
    base_col: str = "subjid_base",
    scan_col: str = "scan_id",
    require_scan: bool = False,
) -> pd.DataFrame:
    """Validate IDs and add person-level and scan-suffix columns."""
    if source_col not in df.columns:
        raise KeyError(f"Required subject-ID column {source_col!r} is missing")

    result = df.copy()
    if result[source_col].isna().any():
        rows = result.index[result[source_col].isna()].tolist()
        raise ValueError(f"Missing subject IDs in {source_col!r} at rows: {rows}")
    result[source_col] = result[source_col].astype(str).str.strip()
    if result[source_col].eq("").any():
        rows = result.index[result[source_col].eq("")].tolist()
        raise ValueError(f"Empty subject IDs in {source_col!r} at rows: {rows}")

    derived_base = result[source_col].map(lambda value: base_subject_id(value, require_scan=require_scan))
    derived_scan = result[source_col].map(scan_id).astype("string")
    if require_scan and derived_scan.isna().any():
        bad = result.loc[derived_scan.isna(), source_col].tolist()
        raise ValueError(f"Modeled IDs must include a scan suffix: {bad}")

    if base_col in result.columns:
        existing = result[base_col].astype(str).str.strip()
        mismatch = existing.ne(derived_base)
        if mismatch.any():
            rows = result.index[mismatch].tolist()
            raise ValueError(f"Existing {base_col!r} conflicts with IDs at rows: {rows}")
    result[base_col] = derived_base

    if scan_col in result.columns:
        existing = result[scan_col].astype("string").str.zfill(3)
        mismatch = existing.fillna("<NA>").ne(derived_scan.fillna("<NA>"))
        if mismatch.any():
            rows = result.index[mismatch].tolist()
            raise ValueError(f"Existing {scan_col!r} conflicts with IDs at rows: {rows}")
    result[scan_col] = derived_scan
    return result


def build_scan_table(subject_ids: Sequence[str]) -> pd.DataFrame:
    """Build a unique modeled-scan table while retaining the shared base ID."""
    table = pd.DataFrame({"subjid": [str(value).strip() for value in subject_ids]})
    if table.empty:
        raise ValueError("subject_ids is empty")
    table = add_subject_id_columns(table, require_scan=True)
    duplicate_mask = table["subjid"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = table.loc[duplicate_mask, "subjid"].tolist()
        raise ValueError(f"Modeled-scan IDs must be unique: {duplicates}")
    return table


def resolve_freesurfer_subjects(
    scans: pd.DataFrame,
    subjects_dir: str | Path,
    *,
    subject_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Resolve each modeled scan to one recon-all directory.

    Resolution is deliberately strict. An explicit ``subject_map`` entry wins,
    followed by an exact full-ID directory match. Otherwise, the function uses
    the sole FreeSurfer directory with the same base person ID. Multiple
    candidates are considered ambiguous and raise an error.
    """
    if "subjid" not in scans.columns:
        raise KeyError("scans must contain 'subjid'")
    result = add_subject_id_columns(scans, require_scan=True)
    subjects_dir = Path(subjects_dir).expanduser().resolve()
    if not subjects_dir.is_dir():
        raise NotADirectoryError(f"SUBJECTS_DIR does not exist: {subjects_dir}")

    explicit = {} if subject_map is None else {str(key).strip(): str(value).strip() for key, value in subject_map.items()}
    unknown_keys = sorted(set(explicit).difference(result["subjid"]))
    if unknown_keys:
        raise ValueError(f"subject_map contains modeled IDs absent from scans: {unknown_keys}")

    available = {}
    for path in subjects_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            base = base_subject_id(path.name, require_scan=True)
        except ValueError:
            continue
        available.setdefault(base, []).append(path.name)

    resolved = []
    for row in result.itertuples(index=False):
        if row.subjid in explicit:
            fs_subjid = explicit[row.subjid]
        elif (subjects_dir / row.subjid).is_dir():
            fs_subjid = row.subjid
        else:
            candidates = sorted(available.get(row.subjid_base, []))
            if not candidates:
                raise FileNotFoundError(
                    f"No recon-all directory matches {row.subjid!r} or base ID {row.subjid_base!r} in {subjects_dir}"
                )
            if len(candidates) > 1:
                raise ValueError(
                    f"Ambiguous recon-all match for {row.subjid!r}: {candidates}. "
                    "Provide subject_map={modeled_subjid: freesurfer_subjid}."
                )
            fs_subjid = candidates[0]

        fs_path = subjects_dir / fs_subjid
        if not fs_path.is_dir():
            raise FileNotFoundError(f"Mapped recon-all directory does not exist for {row.subjid!r}: {fs_path}")
        fs_base = base_subject_id(fs_subjid, require_scan=True)
        if fs_base != row.subjid_base:
            raise ValueError(
                f"Mapped FreeSurfer subject {fs_subjid!r} has base ID {fs_base!r}, "
                f"which does not match modeled scan {row.subjid!r}"
            )
        resolved.append(fs_subjid)

    result["fs_subjid"] = resolved
    result["fs_subject_dir"] = [str(subjects_dir / value) for value in resolved]
    return result
