#!/usr/bin/env python3
"""Create preferred scan IDs from a results-query CSV using only the standard library.

Examples:
    python build_subject_scan_map.py results_query.csv
    python build_subject_scan_map.py results_query.csv --model-type skin_double --simulation-type adaptive
    python build_subject_scan_map.py results_query.csv --filter dataset_root=STU00225089 --overwrite

Only patients with multiple distinct modeled scans enter the JSON map. The
lowest numeric scan suffix wins. Repeated records across placements, runs,
models, and steps count as the same scan. By default all models/steps are used;
optional filters should match the filters used by the clinical session mapper.
Only .hdf5/.h5 records are considered. No filesystem discovery or clinical
eligibility filtering is performed, and simulation files are never opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


SCAN_ID = re.compile(r"(?P<base>subj-[A-Za-z0-9]+-\d{3})-(?P<scan>\d{3})\Z")
STEPS = {"static": "step_0", "adaptive": "step_2", "step_0": "step_0", "step_2": "step_2"}


def read_subject_scans(
    fn: str | Path,
    *,
    model_type: str = "all",
    simulation_type: str = "all",
    filters: Mapping[str, str] | None = None,
) -> dict[str, set[str]]:
    """Read unique scan IDs per patient from the selected HDF5 query records."""
    if model_type not in {"all", "skin_single", "skin_double"}:
        raise ValueError("model_type must be all, skin_single, or skin_double")
    if simulation_type not in {"all", *STEPS}:
        raise ValueError("simulation_type must be all, static, adaptive, step_0, or step_2")
    selected = dict(filters or {})
    if {"model_type_dir", "step"}.intersection(selected):
        raise ValueError("Use --model-type/--simulation-type to select model_type_dir/step")
    if any(not isinstance(value, str) or not value.strip() for value in selected.values()):
        raise ValueError("Filter values must be nonempty strings")
    selected = {key: value.strip() for key, value in selected.items()}
    if model_type != "all":
        selected["model_type_dir"] = model_type
    if simulation_type != "all":
        selected["step"] = STEPS[simulation_type]

    path = Path(fn).expanduser()
    subjects: dict[str, set[str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Query CSV is empty: {path}")
        header = [name.strip() for name in header]
        if any(not name for name in header):
            raise ValueError("Query CSV has blank column names")
        if len(header) != len(set(header)):
            raise ValueError("Query CSV has duplicate column names after stripping whitespace")
        required = {"subject_dir", "full_file_path", *selected}
        # The legacy export omits filename; derive it from full_file_path when needed.
        if "filename" not in header:
            required.discard("filename")
        missing = sorted(required.difference(header))
        if missing:
            raise ValueError(f"Query CSV is missing required columns: {missing}")
        for values in reader:
            if len(values) != len(header):
                raise ValueError(f"CSV line {reader.line_num}: expected {len(header)} fields, got {len(values)}")
            row = dict(zip(header, (value.strip() for value in values)))
            row.setdefault("filename", Path(row["full_file_path"]).name)
            if any(row[key] != value for key, value in selected.items()):
                continue
            if not row["full_file_path"]:
                raise ValueError(f"CSV line {reader.line_num}: missing full_file_path")
            if Path(row["full_file_path"]).suffix.lower() not in {".hdf5", ".h5"}:
                continue
            subject = row["subject_dir"]
            match = SCAN_ID.fullmatch(subject)
            if match is None:
                raise ValueError(f"CSV line {reader.line_num}: invalid modeled subject_dir {subject!r}")
            subjects.setdefault(match.group("base"), set()).add(subject)
    if not subjects:
        raise ValueError(f"No HDF5 records match the requested filters in {path}")
    return subjects


def build_subject_scan_map(subject_scans: Mapping[str, set[str]]) -> dict[str, str]:
    """For each patient with multiple scans, choose the lowest numeric suffix."""
    return {base: min(scans, key=lambda scan: int(scan.rsplit("-", 1)[1]))
            for base, scans in sorted(subject_scans.items()) if len(scans) > 1}


def write_subject_scan_map(mapping: Mapping[str, str], fn: str | Path, *, overwrite: bool = False) -> Path:
    """Atomically write JSON; preserve any existing map unless overwrite is explicit."""
    path = Path(fn).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; use --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(dict(mapping), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # A same-directory hard link creates the output atomically without clobbering it.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _parse_filters(values: Sequence[str]) -> dict[str, str]:
    filters = {}
    for value in values:
        key, separator, expected = value.partition("=")
        key, expected = key.strip(), expected.strip()
        if not separator or not key or not expected:
            raise ValueError(f"Invalid --filter {value!r}; expected COLUMN=VALUE")
        if key in filters:
            raise ValueError(f"Duplicate --filter column: {key!r}")
        filters[key] = expected
    return filters


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query_csv", type=Path, help="Results-query CSV with subject_dir and full_file_path")
    parser.add_argument("-o", "--output", type=Path, default=Path("subject_scan_map.json"))
    parser.add_argument("--model-type", choices=["all", "skin_single", "skin_double"], default="all")
    parser.add_argument("--simulation-type", choices=["all", *STEPS], default="all")
    parser.add_argument("--filter", action="append", default=[], metavar="COLUMN=VALUE",
                        help="Repeat for extra filters")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output map")
    args = parser.parse_args(argv)
    try:
        source, output = args.query_csv.expanduser(), args.output.expanduser()
        if source.resolve() == output.resolve() or (output.exists() and source.samefile(output)):
            raise ValueError("Output must not overwrite the input query CSV")
        filters = _parse_filters(args.filter)
        scans = read_subject_scans(source, model_type=args.model_type, simulation_type=args.simulation_type,
                                   filters=filters)
        mapping = build_subject_scan_map(scans)
        output = write_subject_scan_map(mapping, output, overwrite=args.overwrite)
    except (OSError, ValueError, csv.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Patients represented: {len(scans)}")
    print(f"Patients with multiple scans: {len(mapping)}")
    for base, scan in mapping.items():
        print(f"  {base} -> {scan} ({len(scans[base])} distinct scans)")
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
