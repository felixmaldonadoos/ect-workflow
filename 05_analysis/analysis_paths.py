"""Shared results-directory policy for ROI and NIfTI PCA workflows."""

from __future__ import annotations

import os
from pathlib import Path
import re


SOURCE_DIR = Path(__file__).resolve().parent


def results_root() -> Path:
    value = os.environ.get("RESULTS_DIR", "").strip()
    if not value:
        raise ValueError(
            "RESULTS_DIR must be set; no results directory inside the source "
            "tree is used as a fallback."
        )
    configured = Path(os.path.expandvars(value)).expanduser()
    if not configured.is_absolute():
        raise ValueError("RESULTS_DIR must be an absolute path.")
    root = configured.resolve()
    if root == SOURCE_DIR or SOURCE_DIR in root.parents:
        raise ValueError("RESULTS_DIR must be outside the analysis source tree.")
    return root


def resolve_analysis_dir(analysis_name: str) -> Path:
    if (
        not isinstance(analysis_name, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", analysis_name)
    ):
        raise ValueError(
            "analysis_name must be a single directory name containing letters, "
            "digits, underscores, hyphens, or periods."
        )
    path = (results_root() / analysis_name).resolve()
    if path.parent != results_root():
        raise ValueError("Analysis directory must be directly under RESULTS_DIR.")
    return path


def validate_output_path(path: str | Path, analysis_name: str | None = None) -> Path:
    """Validate an artifact/subdirectory below RESULTS_DIR/<analysis_name>."""
    root = results_root()
    target = Path(path).expanduser().resolve()
    if target == SOURCE_DIR or SOURCE_DIR in target.parents:
        raise ValueError("Analysis outputs cannot be written inside source code.")
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Output must be below RESULTS_DIR: {target}") from exc
    if len(relative.parts) < 2:
        raise ValueError(
            "Artifact paths must be below RESULTS_DIR/<analysis_name>/, "
            "rather than directly in RESULTS_DIR."
        )
    # Also reject unexpected analysis-name/path traversal through symlinks.
    resolve_analysis_dir(relative.parts[0])
    if analysis_name is not None:
        selected = resolve_analysis_dir(analysis_name)
        if selected not in target.parents:
            raise ValueError(f"Output must be inside selected analysis directory: {selected}")
    return target
