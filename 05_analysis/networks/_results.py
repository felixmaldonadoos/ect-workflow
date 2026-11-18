"""ROI command-line output conventions backed by the shared path policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys


sys.dont_write_bytecode = True
ANALYSIS_SOURCE = Path(__file__).resolve().parents[1]
if str(ANALYSIS_SOURCE) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SOURCE))

from analysis_paths import resolve_analysis_dir, validate_output_path


DEFAULT_ANALYSIS_NAME = "mni_yeo2011"


def add_analysis_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--analysis-name", default=DEFAULT_ANALYSIS_NAME,
        help="Results subdirectory under RESULTS_DIR (default: %(default)s).",
    )


def bootstrap_cli(argv=None) -> Path:
    """Fail early and configure caches before heavy command-line imports.

    Call this only when executing an entrypoint, never merely when importing its
    numerical functions. Full parsers subsequently validate remaining flags.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_analysis_argument(parser)
    args, _ = parser.parse_known_args(argv)
    try:
        root = resolve_analysis_dir(args.analysis_name)
        cache = validate_output_path(root / "logs/matplotlib_cache", args.analysis_name)
    except ValueError as exc:
        parser.error(str(exc))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    os.environ["MPLBACKEND"] = "Agg"
    return root


def data_dir(analysis_name: str) -> Path:
    return validate_output_path(resolve_analysis_dir(analysis_name) / "data", analysis_name)


def default_master_path(analysis_name: str, field_name: str) -> Path:
    return data_dir(analysis_name) / f"yeo2011_{field_name}_series_ROI_master.csv"


def normalization_metadata(frame, field_name: str, source_path=None) -> dict:
    """Identify a single raw/brain-mean analysis without interpreting rho values."""
    selected = frame.loc[frame["field_name"].eq(field_name)] if "field_name" in frame else frame
    if selected.empty:
        raise ValueError(f"No normalization metadata for field_name={field_name!r}.")

    def unique_value(column):
        if column not in selected:
            return None
        values = selected[column].dropna().astype(str).str.strip()
        values = values.loc[values.ne("")].unique().tolist()
        if len(values) > 1:
            raise ValueError(f"Mixed {column} values; select one normalization per run: {values}")
        return values[0] if values else None

    method = unique_value("normalization")
    tag = unique_value("normalization_tag")
    expected_tag = f"normed_brain_{field_name}_mean"
    if method is None and tag is None:
        # Older CSVs have no normalization columns; their filenames use this tag.
        normalized = source_path is not None and expected_tag in Path(source_path).name
        method = "brain_mean" if normalized else "raw"
    elif method is None:
        method = "brain_mean" if tag == expected_tag else "raw"
    if method.lower() in {"none", "raw"}:
        method, expected = "raw", "raw"
    elif method == "brain_mean":
        expected = expected_tag
    else:
        raise ValueError(f"Unsupported normalization metadata: {method!r}")
    if tag is not None and tag != expected:
        raise ValueError(f"Conflicting normalization/tag metadata: {method!r}/{tag!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", expected):
        raise ValueError(f"Invalid normalization tag: {expected!r}")
    result = {"normalization": method, "normalization_tag": expected}
    for column in (
        "normalization_mask_path", "normalization_mask_rule", "normalization_applies_to",
        "normalization_domain",
    ):
        value = unique_value(column)
        if value is not None:
            result[column] = value
    return result


def correlation_output_path(
    analysis_name: str, field_name: str, scope: str, metadata: dict,
    permutation: bool = False,
) -> Path:
    tag = metadata["normalization_tag"]
    suffix = "" if tag == "raw" else f"_{tag}"
    permutation_suffix = "_permutation" if permutation else ""
    return validate_output_path(
        data_dir(analysis_name) / f"spearman_{field_name}_{scope}{suffix}{permutation_suffix}_annotated.csv",
        analysis_name,
    )


def plot_output_dir(analysis_name: str, metadata: dict) -> Path:
    return validate_output_path(
        resolve_analysis_dir(analysis_name) / "plots" / metadata["normalization_tag"],
        analysis_name,
    )
