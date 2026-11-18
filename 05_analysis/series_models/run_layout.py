"""Immutable, standard-library run manifests shared by LMM child jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from analysis_paths import lmm_analysis_name, lmm_run_dir, validate_output_path
from .model_spec import get_model_spec


def run_configuration(sex_effect: str, fit_settings: dict, field_name: str = "magnE_mean", *, prepare_only: bool = False) -> dict:
    feature = get_model_spec(sex_effect)
    null = get_model_spec(sex_effect, null=True)
    settings = {"reml": False, "optimizer": "powell", "maxiter": 2000,
                "boundary_tolerance": 1e-8, **fit_settings, "sex_effect": sex_effect}
    return {
        "sex_effect_choice": sex_effect, "sex_effect": feature["sex_effect"],
        "feature_model_id": feature["model_id"], "feature_formula": feature["fixed_formula"],
        "null_model_id": null["model_id"], "null_formula": null["fixed_formula"],
        "variance_components": feature["variance_components"], "fit_backend": feature["fit_backend"],
        "fit_settings": settings, "field_name": field_name, "prepare_only": bool(prepare_only),
    }


def ensure_run_manifest(
    analysis_name: str, configuration: dict, *, inputs: list[str] | None = None,
    create: bool = True,
) -> Path:
    """Create atomically or require the run's existing scientific settings.

    Independent Slurm children may reach this concurrently. An immutable JSON
    document is installed with an exclusive hard link after the temporary file
    is complete, so another reader never sees partially written JSON.
    """
    root = lmm_run_dir(analysis_name)
    namespace = "/".join(lmm_analysis_name(analysis_name).split("/")[:2])
    path = validate_output_path(root / "run.json", namespace)
    requested_inputs = sorted({str(Path(value).expanduser().resolve()) for value in (inputs or [])})

    def check():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("schema_version") != 1 or existing.get("analysis_type") != "series_lmm":
            raise ValueError(f"Incompatible run manifest: {path}")
        if existing.get("configuration") != configuration:
            raise ValueError(f"Run settings differ from {path}; choose a new analysis/run name")
        allowed = existing.get("input_masters", [])
        if requested_inputs and allowed and not set(requested_inputs).issubset(allowed):
            raise ValueError(f"Source masters differ from {path}; choose a new analysis/run name")

    if path.exists():
        check()
        return path
    if not create:
        return path
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1, "analysis_type": "series_lmm", "analysis_name": root.name,
        "analysis_dir": str(root), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration, "input_masters": requested_inputs,
        "layout": {"null": "null/", "roi_raw": "roi_raw/", "roi_brain_mean": "roi_brain_mean/",
                   "pca_raw": "pca_raw/", "pca_brain_mean": "pca_brain_mean/",
                   "aic": "aic/", "saved_fit_plots": "plots/"},
        "overwrite_policy": "Existing artifacts are protected; use a new run name for changed settings or a repeated run.",
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=".run_manifest_", suffix=".json", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            check()
    finally:
        temporary.unlink(missing_ok=True)
    return path
