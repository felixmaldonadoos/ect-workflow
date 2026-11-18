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
    parts = analysis_name.split("/") if isinstance(analysis_name, str) else []
    valid = bool(parts) and all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) for part in parts)
    if not valid or (len(parts) > 1 and (parts[0] != "lmm" or len(parts) not in (2, 3))):
        raise ValueError(
            "analysis_name must be a directory name containing letters, digits, underscores, "
            "hyphens, or periods, or an lmm/<run-name>[/<component>] namespace."
        )
    root = results_root()
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root) or path.relative_to(root) != Path(*parts):
        raise ValueError("Analysis directory must retain its namespace below RESULTS_DIR; symlink redirection is not allowed.")
    return path


def lmm_analysis_name(analysis_name: str, component: str | None = None) -> str:
    """Place new LMM outputs under one analysis run without changing RESULTS_DIR."""
    resolve_analysis_dir(analysis_name)
    parts = analysis_name.split("/")
    if len(parts) == 1:
        parts = ["lmm", parts[0]]
    if component is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", component):
            raise ValueError("Invalid LMM output component")
        if len(parts) == 3 and parts[2] != component:
            raise ValueError("LMM analysis namespace conflicts with --output-component")
        if len(parts) == 2:
            parts.append(component)
    name = "/".join(parts)
    resolve_analysis_dir(name)
    return name


def lmm_run_dir(analysis_name: str) -> Path:
    parts = lmm_analysis_name(analysis_name).split("/")
    return resolve_analysis_dir("/".join(parts[:2]))


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
    if relative.parts[0] == "lmm" and len(relative.parts) >= 3:
        resolve_analysis_dir("/".join(relative.parts[:2]))
    else:
        resolve_analysis_dir(relative.parts[0])
    if analysis_name is not None:
        selected = resolve_analysis_dir(analysis_name)
        if selected not in target.parents:
            raise ValueError(f"Output must be inside selected analysis directory: {selected}")
    return target
