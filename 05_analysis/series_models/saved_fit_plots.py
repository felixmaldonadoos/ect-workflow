"""Load immutable saved series fits for plots without refitting or image access."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aic import (
    IDENTITY_COLUMNS, _artifact, _boolean, _counts, _csv, _identities,
    _require_columns, _same_values, _sha256, _validate_features,
)
from .cohort import canonical_cohort, cohort_sha256
from .fitting import FitSettings
from .model_spec import get_model_spec


def _reconstruct_prediction(frame, diagnostics, *, feature, grid_points):
    # Prediction is separate from reading saved files; defer its import so that
    # input/specification errors are detected without plotting dependencies.
    from .prediction import reconstruct_prediction
    return reconstruct_prediction(frame, diagnostics, feature=feature, grid_points=grid_points)


def _manifest_path(directory: Path) -> tuple[Path, bool]:
    null = directory / "analysis.json"
    features = sorted(path for path in directory.glob("analysis_*.json") if path.is_file())
    candidates = ([null] if null.is_file() else []) + features
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one saved null or feature analysis manifest in {directory}")
    path = candidates[0].resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Saved analysis manifest resolves outside its analysis directory")
    return path, path == null.resolve()


def _validate_manifest(manifest: dict, *, null: bool, sex_levels: list | None = None) -> tuple[dict, dict]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("Saved plots require a schema-2 continuous-age analysis")
    if manifest.get("analysis_type", "feature") != ("null" if null else "feature"):
        raise ValueError("Saved analysis type disagrees with its manifest filename")
    if manifest.get("prepare_only") is not False:
        raise ValueError("Prepare-only analyses do not contain fitted models to plot")
    settings = manifest.get("fit_settings")
    if not isinstance(settings, dict):
        raise ValueError("Missing saved fit settings")
    # Missing choice is the original schema-2 random-intercept specification.
    # This canonical copy does not alter the saved file or its recorded hash.
    settings = {"sex_effect": "random", **settings}
    try:
        FitSettings(**settings)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid saved fit settings: {error}") from error
    choice = settings["sex_effect"]
    if choice == "fixed" and settings["reml"]:
        raise ValueError("Saved fixed-sex models support ML only")
    model = manifest.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("Missing saved model specification")
    levels = sex_levels if sex_levels is not None else model.get("sex_levels")
    specification = get_model_spec(choice, null=null, sex_levels=levels)
    formula, model_id = specification["fixed_formula"], specification["model_id"]
    for key, expected in {
        "model_id": model_id, "fixed_formula": formula,
        "age_effect": "continuous_fixed", "sex_effect": specification["sex_effect"],
    }.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Incompatible saved model specification: {key}")
    spec = {
        "fixed_formula": formula, "fixed_terms": specification["fixed_terms"],
        "variance_components": specification["variance_components"],
        "outer_group": specification.get("outer_group", None if choice == "fixed" else "constant"),
        "group_random_intercept": False, "patient_random_effect": False,
        "observation_unit": "treatment series", "centering": "none", "standardization": "none",
    }
    for key, expected in spec.items():
        # Fixed contrast terms are validated after reading the saved cohort if
        # an older/new producer omitted optional planned sex-level metadata.
        if key == "fixed_terms" and choice == "fixed" and levels is None:
            continue
        if model.get(key) != expected:
            raise ValueError(f"Incompatible saved model specification: {key}")
    if sex_levels is not None and "sex_levels" in model and model["sex_levels"] != sex_levels:
        raise ValueError("Saved planned sex levels differ from the saved cohort")
    if choice == "fixed" and levels is not None and "sex_reference" in model and model["sex_reference"] != specification["sex_reference"]:
        raise ValueError("Saved fixed-sex reference differs from the first sorted sex category")
    for column in ("master_file", "source_master_path", "source_master_sha256"):
        if not isinstance(manifest.get(column), str) or not manifest[column]:
            raise ValueError(f"Missing saved source identity: {column}")
    digest = manifest["source_master_sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Invalid source master SHA256")
    return spec, settings


def _validate_saved_spec(frame: pd.DataFrame, manifest: dict, *, status: bool, sex_levels: list | None = None) -> None:
    for column in ("model_id", "fixed_formula", "age_effect", "sex_effect",
                   "master_file", "source_master_path", "source_master_sha256"):
        _same_values(frame, column, manifest[column], "Saved model table")
    if not status:
        return
    _same_values(frame, "formula", manifest["fixed_formula"], "Saved fit status")
    _require_columns(frame, ["converged", "status", "reml", "fit_method", "n_observations"], "Saved fit status")
    settings = manifest["fit_settings"]
    if not frame.reml.map(_boolean).eq(settings["reml"]).all():
        raise ValueError("Saved rows disagree with the manifest ML/REML setting")
    _same_values(frame, "fit_method", "REML" if settings["reml"] else "ML", "Saved fit status")
    for column, expected in settings.items():
        if column not in frame or column in {"reml", "sex_effect"}:
            continue
        if isinstance(expected, str):
            _same_values(frame, column, expected, "Saved fit setting")
        elif not np.allclose(pd.to_numeric(frame[column], errors="coerce"), expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Saved rows disagree with the manifest fit setting: {column}")
    if "sex_effect_choice" in frame:
        _same_values(frame, "sex_effect_choice", settings.get("sex_effect", "random"), "Saved fit setting")
    expected_design = [("fixed_terms", manifest["model"]["fixed_terms"]),
                       ("random_effects", manifest["model"]["variance_components"])]
    if sex_levels is not None:
        expected_design.append(("sex_levels", sex_levels))
    for column, expected in expected_design:
        if column not in frame:
            continue
        for value, converged in zip(frame[column], frame.converged, strict=True):
            if value == "" and _boolean(converged) is not True:
                continue
            try:
                actual = json.loads(value) if isinstance(value, str) else value
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid saved design metadata: {column}") from error
            if actual != expected:
                raise ValueError(f"Incompatible saved design metadata: {column}")
    fixed_sex = settings.get("sex_effect", "random") == "fixed"
    if fixed_sex and sex_levels is not None and "sex_reference" in frame:
        present = frame.sex_reference.ne("") | frame.converged.map(_boolean).eq(True)
        if not frame.loc[present, "sex_reference"].eq(sex_levels[0]).all():
            raise ValueError("Saved fixed-sex reference differs from the saved cohort")
    for column, expected in (("n_outer_groups", 0 if fixed_sex else 1), ("n_group_random_intercepts", 0)):
        if column in frame:
            present = frame[column].ne("")
            if not pd.to_numeric(frame.loc[present, column], errors="coerce").eq(expected).all():
                raise ValueError(f"Incompatible saved design metadata: {column}")


def load_saved_fit_records(
    analysis_dir: str | Path, *, grid_points: int = 100,
) -> tuple[list[dict], str, dict]:
    """Validate one saved run, retaining an explicit record for every fit.

    Broken run identity, cohort, or coverage is a global error. A failed fit or
    failed numerical reconstruction produces a record with ``prediction=None``
    and its error, allowing all other fits to be plotted. Random-sex ML/REML and
    fixed-sex ML are supported; no fitting, source-master rereading, or output
    writing occurs.
    """
    if isinstance(grid_points, bool) or not isinstance(grid_points, int) or grid_points < 2:
        raise ValueError("grid_points must be an integer of at least two")
    directory = Path(analysis_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Analysis directory does not exist: {directory}")
    files = []

    def read(path, role, reader):
        digest = _sha256(path)
        result = reader(path)
        if _sha256(path) != digest:
            raise ValueError(f"Saved input changed while being read: {path}")
        files.append({"role": role, "path": str(path), "sha256": digest})
        return result

    def verify_stable():
        for item in files:
            if _sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Saved input changed during plot preparation: {item['path']}")

    manifest_path, null = _manifest_path(directory)
    manifest = read(manifest_path, "manifest", lambda path: json.loads(path.read_text(encoding="utf-8")))
    spec, settings = _validate_manifest(manifest, null=null)
    model_input = _artifact(directory, manifest, "model_input", "data/null_model_input.csv" if null else "data/*_model_input_long.csv")
    fit_status = _artifact(directory, manifest, "fit_status", "fit_status.csv" if null else "*_fit_status.csv")
    prepared = read(model_input, "model_input", _csv)
    statuses = read(fit_status, "fit_status", _csv)
    cohort = canonical_cohort(prepared)
    # Derive the treatment-coded terms from the actual immutable model input,
    # rather than trusting contrast names or reference labels in a manifest.
    sex_levels = sorted(cohort.sex.unique().tolist())
    spec, settings = _validate_manifest(manifest, null=null, sex_levels=sex_levels)
    _validate_saved_spec(statuses, manifest, status=True, sex_levels=sex_levels)
    if null:
        if len(prepared) != len(cohort):
            raise ValueError("Duplicate treatment series in saved null input")
        if len(statuses) != 1 or manifest.get("planned_fits") != 1:
            raise ValueError("Saved null must contain exactly one planned fit and status row")
        for column in IDENTITY_COLUMNS:
            _same_values(statuses, column, manifest.get(column), "Saved null identity")
        if manifest.get("cohort_sha256") != cohort_sha256(cohort):
            raise ValueError("Saved null cohort SHA256 differs from its input")
        if "cohort_sha256" in statuses and not statuses.cohort_sha256.eq(cohort_sha256(cohort)).all():
            raise ValueError("Saved null status cohort SHA256 differs from its input")
        _counts(statuses, cohort, "Saved null")
        groups = {tuple(statuses.iloc[0][IDENTITY_COLUMNS]): prepared}
        source_id = f"null__{manifest['source_master_sha256'][:12]}"
    else:
        _validate_saved_spec(prepared, manifest, status=False)
        _validate_features(prepared, statuses, manifest, cohort, str(directory))
        groups = {identity: group for identity, group in prepared.groupby(IDENTITY_COLUMNS, dropna=False, sort=False)}
        suffix = "_fit_status"
        if not fit_status.stem.endswith(suffix):
            raise ValueError("Saved feature status filename must end with _fit_status.csv")
        source_id = fit_status.stem[:-len(suffix)]
    identities = _identities(statuses, "Saved fit status")
    verify_stable()
    records = []
    for identity, row in zip(identities, statuses.to_dict("records"), strict=True):
        prediction, error = None, ""
        if _boolean(row.get("converged")) is not True:
            error = "Saved model did not converge"
            if row.get("error"):
                error += f": {row['error']}"
        else:
            try:
                prediction = _reconstruct_prediction(groups[identity], row, feature=not null, grid_points=grid_points)
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
        records.append({"identity": row, "prediction": prediction, "error": error})
    verify_stable()
    metadata = {
        "analysis_dir": str(directory), "analysis_name": manifest.get("analysis_name"),
        "is_null": null, "prepare_only": False, "source_id": source_id,
        "model_id": manifest["model_id"], "model_spec": spec, "fit_settings": settings,
        "source_master_path": manifest["source_master_path"],
        "source_master_sha256": manifest["source_master_sha256"],
        "input_files": files, "cohort_sha256": cohort_sha256(cohort),
        "n_observations": len(cohort), "n_people": int(cohort.person_id.nunique()),
        "planned_fits": len(records), "grid_points": grid_points,
    }
    return records, source_id, metadata
