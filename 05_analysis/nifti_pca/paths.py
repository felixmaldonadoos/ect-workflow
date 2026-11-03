"""Compatibility imports for the shared ROI/PCA results-directory policy."""

from analysis_paths import (
    SOURCE_DIR,
    resolve_analysis_dir,
    results_root,
    validate_output_path,
)

__all__ = ["SOURCE_DIR", "results_root", "resolve_analysis_dir", "validate_output_path"]
