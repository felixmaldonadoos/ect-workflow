"""Compare saved ML series models on an identical, validated clinical cohort.

Likelihood comparison and Wald coefficient inference have separate eligibility
rules. A converged constrained ML solution can have useful AIC while its Wald
Hessian is invalid; those comparisons remain explicitly provisional here.
This module only reads saved tables and manifests. It never refits a model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cohort import COHORT_KEYS, canonical_cohort, cohort_sha256, require_same_cohort
from .model_spec import get_model_spec


IDENTITY_COLUMNS = [
    "feature_type", "field_name", "normalization", "normalization_tag",
    "model_type", "step", "basis_id", "feature_id",
]
CONFIGURATION_COLUMNS = [
    "feature_type", "field_name", "normalization", "model_type", "step", "basis_id",
]
_TEXT_COLUMNS = [
    *IDENTITY_COLUMNS, "person_id", "sex", "source_master_path", "source_master_sha256",
    "master_file", "model_id", "fixed_formula", "formula", "age_effect", "sex_effect",
    "roi_name", "pc_name", "value_column", "sex_effect_choice", "sex_levels", "sex_reference",
]
_OBSERVATION_SPEC = {
    "observation_unit": "treatment series",
    "centering": "none",
    "standardization": "none",
}


@dataclass(frozen=True)
class AICComparison:
    table: pd.DataFrame
    metadata: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _boolean(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return None


def _number(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _csv(path: Path) -> pd.DataFrame:
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate CSV column names: {path}")
    frame = pd.read_csv(
        path, keep_default_na=False,
        dtype={column: str for column in _TEXT_COLUMNS if column in header},
    )
    if frame.empty:
        raise ValueError(f"Empty saved model table: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: list[str], context: str) -> None:
    if missing := set(columns) - set(frame):
        raise ValueError(f"{context}: missing columns {sorted(missing)}")


def _same_values(frame: pd.DataFrame, column: str, expected: Any, context: str) -> None:
    _require_columns(frame, [column], context)
    if not frame[column].eq(expected).all():
        raise ValueError(f"{context}: {column} differs from saved model specification/provenance")


def _manifest(directory: Path, *, null: bool) -> tuple[Path, dict, str]:
    paths = [directory / "analysis.json"] if null else sorted(directory.glob("analysis_*.json"))
    paths = [path for path in paths if path.is_file()]
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {'null' if null else 'feature'} analysis manifest in {directory}")
    path = paths[0].resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f"Analysis manifest escapes its analysis directory: {path}")
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError(f"Unsupported analysis manifest schema: {path}")
    if null and manifest.get("analysis_type") != "null":
        raise ValueError(f"Expected a null-model analysis: {path}")
    if not null and manifest.get("analysis_type", "feature") != "feature":
        raise ValueError(f"Expected a feature-model analysis: {path}")
    if manifest.get("prepare_only") is not False:
        raise ValueError(f"Analysis is not a completed fit run: {path}")
    return path, manifest, hashlib.sha256(payload).hexdigest()


def _artifact(directory: Path, manifest: dict, name: str, pattern: str) -> Path:
    declared = manifest.get("artifacts", {}).get(name)
    if declared:
        path = Path(declared)
        path = (path if path.is_absolute() else directory / path).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Invalid saved {name} artifact in {directory}: {declared}")
        return path
    matches = [path.resolve() for path in directory.glob(pattern) if path.is_file()]
    if len(matches) != 1 or not matches[0].is_relative_to(directory):
        raise ValueError(f"Expected exactly one saved {name} artifact in {directory}")
    return matches[0]


def _normalized_settings(manifest: dict) -> dict:
    settings = manifest.get("fit_settings", {})
    if not isinstance(settings, dict):
        raise ValueError("Saved fit settings must be a dictionary")
    choice = settings.get("sex_effect", "random")
    if not isinstance(choice, str) or choice not in {"random", "fixed"}:
        raise ValueError("Saved sex_effect setting must be random or fixed")
    # Schema-2 random-sex runs made before the switch was introduced omit this
    # setting. Their original metadata is preserved alongside its interpretation.
    return {**settings, "sex_effect": choice}


def _validate_manifest(manifest: dict, *, null: bool, context: str,
                       sex_levels: list[str]) -> dict:
    settings = _normalized_settings(manifest)
    expected_spec = get_model_spec(settings["sex_effect"], null=null, sex_levels=sex_levels)
    for name, expected in {
        "model_id": expected_spec["model_id"], "fixed_formula": expected_spec["fixed_formula"],
        "age_effect": "continuous_fixed", "sex_effect": expected_spec["sex_effect"],
    }.items():
        if manifest.get(name) != expected:
            raise ValueError(f"{context}: incompatible {name}; current continuous-age model required")
    spec = manifest.get("model", {})
    if not isinstance(spec, dict):
        raise ValueError(f"{context}: saved model specification must be a dictionary")
    design_keys = ("fixed_formula", "fixed_terms", "variance_components", "outer_group",
                   "group_random_intercept", "patient_random_effect")
    required = {**_OBSERVATION_SPEC, **{key: expected_spec[key] for key in design_keys}}
    for name, expected in required.items():
        if spec.get(name) != expected:
            raise ValueError(f"{context}: incompatible model specification {name}")
    for label, metadata in (("manifest", manifest), ("model", spec)):
        if "sex_levels" in metadata:
            levels = metadata["sex_levels"]
            try:
                levels = json.loads(levels) if isinstance(levels, str) else levels
            except (TypeError, ValueError) as error:
                raise ValueError(f"{context}: invalid {label} sex_levels") from error
            if levels != sex_levels:
                raise ValueError(f"{context}: {label} sex_levels differ from the saved cohort")
        if settings["sex_effect"] == "fixed" and "sex_reference" in metadata:
            if metadata["sex_reference"] != expected_spec["sex_reference"]:
                raise ValueError(f"{context}: {label} sex_reference differs from the first sorted sex category")
        if "sex_effect_choice" in metadata and metadata["sex_effect_choice"] != settings["sex_effect"]:
            raise ValueError(f"{context}: {label} sex_effect_choice differs from saved fit settings")
    if "fit_backend" in spec and spec["fit_backend"] != expected_spec["fit_backend"]:
        raise ValueError(f"{context}: incompatible model specification fit_backend")
    if settings.get("reml") is not False:
        raise ValueError(f"{context}: ML is required; REML cannot compare these fixed-effect designs")
    boundary_tolerance = _number(settings.get("boundary_tolerance"))
    if not np.isfinite(boundary_tolerance) or boundary_tolerance < 0:
        raise ValueError(f"{context}: invalid boundary_tolerance")
    for name in ("master_file", "source_master_path", "source_master_sha256"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"{context}: missing source provenance {name}")
    digest = manifest["source_master_sha256"]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{context}: invalid source master SHA256")
    return {**_OBSERVATION_SPEC, **expected_spec}


def _validate_saved_spec(frame: pd.DataFrame, manifest: dict, *, status: bool, context: str,
                         sex_levels: list[str] | None = None) -> None:
    for column in ("model_id", "fixed_formula", "age_effect", "sex_effect",
                   "master_file", "source_master_path", "source_master_sha256"):
        _same_values(frame, column, manifest[column], context)
    if status:
        _require_columns(frame, ["reml", "fit_method", "converged", "status", "n_observations"], context)
    settings = _normalized_settings(manifest)
    if "sex_effect_choice" in frame:
        _same_values(frame, "sex_effect_choice", settings["sex_effect"], context)
    if sex_levels is not None and "sex_levels" in frame:
        for index, value in frame.sex_levels.items():
            if status and value == "" and _boolean(frame.loc[index, "converged"]) is not True:
                continue
            try:
                levels = json.loads(value) if isinstance(value, str) else value
            except (TypeError, ValueError) as error:
                raise ValueError(f"{context}: invalid saved sex_levels") from error
            if levels != sex_levels:
                raise ValueError(f"{context}: saved sex_levels differ from the saved cohort")
    if sex_levels is not None and settings["sex_effect"] == "fixed" and "sex_reference" in frame:
        present = frame.sex_reference.ne("")
        if status:
            present |= frame.converged.map(_boolean).eq(True)
        else:
            present = pd.Series(True, index=frame.index)
        if not frame.loc[present, "sex_reference"].eq(sex_levels[0]).all():
            raise ValueError(f"{context}: saved sex_reference differs from the first sorted sex category")
    if not status:
        return
    _same_values(frame, "formula", manifest["fixed_formula"], context)
    if not frame.reml.map(_boolean).eq(False).all() or not frame.fit_method.eq("ML").all():
        raise ValueError(f"{context}: saved rows must use ML, not REML")
    for column in ("optimizer", "maxiter", "boundary_tolerance"):
        if column not in frame:
            continue
        expected = settings.get(column)
        if column == "optimizer":
            if not frame[column].eq(expected).all():
                raise ValueError(f"{context}: fit settings differ for {column}")
        elif not np.allclose(pd.to_numeric(frame[column], errors="coerce"), _number(expected), rtol=1e-12, atol=1e-12):
            raise ValueError(f"{context}: fit settings differ for {column}")
    for column, expected in (("random_effects", manifest["model"]["variance_components"]),
                             ("fixed_terms", manifest["model"]["fixed_terms"])):
        if column not in frame:
            continue
        for value, converged in zip(frame[column], frame.converged, strict=True):
            if value == "" and _boolean(converged) is not True:
                continue  # A design failure can precede these diagnostics.
            try:
                actual = json.loads(value) if isinstance(value, str) else value
            except (TypeError, ValueError) as error:
                raise ValueError(f"{context}: invalid saved {column}") from error
            if actual != expected:
                raise ValueError(f"{context}: incompatible saved {column}")
    n_outer_groups = 1 if settings["sex_effect"] == "random" else 0
    for column, expected in (("n_outer_groups", n_outer_groups), ("n_group_random_intercepts", 0)):
        if column in frame:
            present = frame[column].ne("")
            if not pd.to_numeric(frame.loc[present, column], errors="coerce").eq(expected).all():
                raise ValueError(f"{context}: incompatible saved {column}")


def _identities(frame: pd.DataFrame, context: str) -> list[tuple]:
    _require_columns(frame, IDENTITY_COLUMNS, context)
    for column in IDENTITY_COLUMNS:
        values = frame[column]
        if not values.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{context}: invalid feature identity {column}")
        if not values.eq(values.str.strip()).all() or (column != "basis_id" and values.eq("").any()):
            raise ValueError(f"{context}: empty or padded feature identity {column}")
    return list(frame[IDENTITY_COLUMNS].itertuples(index=False, name=None))


def _counts(status: pd.DataFrame, cohort: pd.DataFrame, context: str) -> None:
    counts = {"n_observations": len(cohort), "n_people": cohort.person_id.nunique(),
              "n_age_values": cohort.age.nunique(), "n_sex_levels": cohort.sex.nunique()}
    for column, expected in counts.items():
        if column not in status:
            if column == "n_observations":
                raise ValueError(f"{context}: missing n_observations")
            continue
        if not pd.to_numeric(status[column], errors="coerce").eq(expected).all():
            raise ValueError(f"{context}: {column} differs from the saved model cohort")


def _validate_features(prepared: pd.DataFrame, status: pd.DataFrame, manifest: dict,
                       cohort: pd.DataFrame, context: str) -> None:
    input_keys, status_keys = _identities(prepared, context), _identities(status, context)
    if len(status_keys) != len(set(status_keys)):
        raise ValueError(f"{context}: duplicate feature/configuration fit-status rows")
    if set(input_keys) != set(status_keys):
        raise ValueError(f"{context}: planned input and fit-status identities differ")
    if manifest.get("planned_fits") != len(status_keys):
        raise ValueError(f"{context}: planned_fits does not match saved fit coverage")
    _require_columns(prepared, [*COHORT_KEYS, "E", "ef_sf"], context)
    if prepared.duplicated([*IDENTITY_COLUMNS, *COHORT_KEYS]).any():
        raise ValueError(f"{context}: duplicate treatment series within a feature/configuration")
    selection = manifest.get("feature_selection", {})
    for column in ("feature_type", "field_name", "normalization", "normalization_tag", "value_column"):
        if column not in selection:
            raise ValueError(f"{context}: missing source-adapter metadata {column}")
        _same_values(prepared, column, selection[column], context)
        _same_values(status, column, selection[column], context)
    kind, normalization = selection["feature_type"], selection["normalization"]
    if kind not in {"roi", "pca"} or normalization not in {"raw", "brain_mean"}:
        raise ValueError(f"{context}: unsupported feature type or normalization")
    expected_tag = "raw" if normalization == "raw" else f"normed_brain_{selection['field_name']}_mean"
    if selection["normalization_tag"] != expected_tag:
        raise ValueError(f"{context}: normalization tag does not match normalization")
    if selection.get("feature_ids") != sorted(set(prepared.feature_id)):
        raise ValueError(f"{context}: selected feature IDs do not match the saved model input")
    for column, expected in (("n_input_rows", len(prepared)), ("n_features", prepared.feature_id.nunique())):
        if column in selection and selection[column] != expected:
            raise ValueError(f"{context}: {column} does not match the saved model input")
    if kind == "pca" and prepared.basis_id.eq("").any():
        raise ValueError(f"{context}: PCA features require an explicit basis_id")
    if kind == "roi" and prepared.basis_id.ne("").any():
        raise ValueError(f"{context}: ROI features must not claim a PCA basis")
    if kind == "pca":
        configurations = prepared[["field_name", "normalization", "model_type", "step", "basis_id"]].drop_duplicates()
        if configurations.duplicated(["basis_id"]).any():
            raise ValueError(f"{context}: one PCA basis claims multiple configurations")
        if configurations.duplicated(["field_name", "normalization", "model_type", "step"]).any():
            raise ValueError(f"{context}: multiple PCA bases claim the same configuration")
        if selection.get("n_bases", prepared.basis_id.nunique()) != prepared.basis_id.nunique():
            raise ValueError(f"{context}: PCA basis count differs from saved source metadata")
    for frame in (prepared, status):
        if "is_brain_normalized" in frame and not frame.is_brain_normalized.map(_boolean).eq(normalization == "brain_mean").all():
            raise ValueError(f"{context}: is_brain_normalized disagrees with normalization")
    for column, expected in (("roi_name", kind == "roi"), ("pc_name", kind == "pca")):
        if column in status:
            wanted = status.feature_id if expected else ""
            if not status[column].eq(wanted).all():
                raise ValueError(f"{context}: {column} disagrees with feature_id")
    for _, group in prepared.groupby(IDENTITY_COLUMNS, dropna=False, sort=False):
        require_same_cohort(cohort, group)
    _counts(status, cohort, context)


def _eligibility(row: pd.Series, *, null: bool, null_likelihood: float | None,
                 likelihood_tolerance: float, information_tolerance: float,
                 n_observations: int, spec: dict) -> tuple[list[str], list[str]]:
    reasons, caveats = [], []
    parameters, fixed_parameters = spec["n_parameters"], spec["n_fixed_effects"]
    random_sex = spec["sex_effect_choice"] == "random"
    likelihood, aic = _number(row.get("log_likelihood")), _number(row.get("aic"))
    sex, residual = _number(row.get("sex_variance")), _number(row.get("residual_variance"))
    if _boolean(row.get("converged")) is not True:
        reasons.append("not_converged")
    if not np.isfinite(likelihood):
        reasons.append("nonfinite_log_likelihood")
    if not np.isfinite(aic):
        reasons.append("nonfinite_aic")
    if random_sex and (not np.isfinite(sex) or sex < 0):
        reasons.append("invalid_sex_variance")
    if not np.isfinite(residual) or residual <= 0:
        reasons.append("invalid_residual_variance")
    if "n_parameters" in row and row.n_parameters != "" and _number(row.n_parameters) != parameters:
        reasons.append("parameter_count_mismatch")
    if "n_fixed_effects" in row and row.n_fixed_effects != "" and _number(row.n_fixed_effects) != fixed_parameters:
        reasons.append("fixed_parameter_count_mismatch")
    if np.isfinite(likelihood) and np.isfinite(aic) and not np.isclose(aic, -2 * likelihood + 2 * parameters, rtol=0, atol=information_tolerance):
        reasons.append("aic_parameter_count_mismatch")
    bic_value = row.get("bic", "")
    if bic_value != "":
        bic = _number(bic_value)
        if not np.isfinite(bic) or (np.isfinite(likelihood) and not np.isclose(bic, -2 * likelihood + np.log(n_observations) * parameters, rtol=0, atol=information_tolerance)):
            reasons.append("bic_parameter_count_mismatch")
    if not null and np.isfinite(likelihood) and likelihood + likelihood_tolerance < null_likelihood:
        reasons.append("log_likelihood_below_nested_null")
    boundary_tolerance = _number(row.get("boundary_tolerance"))
    if random_sex and ((np.isfinite(sex) and 0 <= sex <= boundary_tolerance) or any(
        _boolean(row.get(column)) is True for column in ("sex_variance_boundary", "any_variance_boundary")
    )):
        caveats.append("variance_boundary")
    if str(row.get("status", "")) != "ok":
        caveats.append("wald_inference_unavailable")
    if random_sex and _boolean(row.get("random_effects_covariance_singular")) is True:
        caveats.append("random_effects_covariance_singular")
    if random_sex and _number(row.get("n_sex_levels")) < 5:
        caveats.append("few_sex_levels")
    if str(row.get("warnings", "")):
        caveats.append("fit_warnings")
    return reasons, caveats


def compare_analyses(
    null_analysis_dir: str | Path,
    feature_analysis_dirs: list[str | Path],
    *,
    likelihood_tolerance: float = 1e-6,
    information_tolerance: float = 1e-6,
) -> AICComparison:
    """Validate and rank saved feature models, retaining excluded candidates.

    Ranks use the minimum rank for exact AIC ties. Global ranks include the null;
    within-configuration ranks compare that configuration's features and the
    shared null. The null itself has no feature/configuration rank or feature-only
    delta. AIC deltas and ranks are missing for ineligible candidates.
    """
    if any(not np.isfinite(value) or value < 0 for value in (likelihood_tolerance, information_tolerance)):
        raise ValueError("Comparison tolerances must be finite and nonnegative")
    null_directory = Path(null_analysis_dir).expanduser().resolve()
    feature_directories = [Path(path).expanduser().resolve() for path in feature_analysis_dirs]
    if not feature_directories:
        raise ValueError("At least one feature analysis directory is required")
    if len(feature_directories) != len(set(feature_directories)) or null_directory in feature_directories:
        raise ValueError("Duplicate or overlapping null/feature analysis directories")
    input_files, analysis_settings = [], []

    def load(directory: Path, *, null: bool) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
        if not directory.is_dir():
            raise ValueError(f"Analysis directory does not exist: {directory}")
        manifest_path, manifest, manifest_digest = _manifest(directory, null=null)
        prepared_path = _artifact(directory, manifest, "model_input", "data/null_model_input.csv" if null else "data/*_model_input_long.csv")
        status_path = _artifact(directory, manifest, "fit_status", "fit_status.csv" if null else "*_fit_status.csv")
        digests = {"manifest": manifest_digest, "model_input": _sha256(prepared_path), "fit_status": _sha256(status_path)}
        prepared, status = _csv(prepared_path), _csv(status_path)
        saved_cohort = canonical_cohort(prepared)
        spec = _validate_manifest(manifest, null=null, context=str(directory),
                                  sex_levels=sorted(saved_cohort.sex.unique().tolist()))
        sex_levels = sorted(saved_cohort.sex.unique().tolist())
        _validate_saved_spec(status, manifest, status=True, context=str(status_path), sex_levels=sex_levels)
        if not null:
            _validate_saved_spec(prepared, manifest, status=False, context=str(prepared_path), sex_levels=sex_levels)
        for role, path in (("manifest", manifest_path), ("model_input", prepared_path), ("fit_status", status_path)):
            if _sha256(path) != digests[role]:
                raise ValueError(f"Saved input changed while being read: {path}")
            input_files.append({"analysis_dir": str(directory), "role": role, "path": str(path), "sha256": digests[role]})
        analysis_settings.append({"analysis_dir": str(directory), "analysis_name": manifest.get("analysis_name"),
                                  "is_null": null, "fit_settings": manifest["fit_settings"],
                                  "normalized_fit_settings": _normalized_settings(manifest)})
        return manifest, spec, prepared, status

    null_manifest, null_spec, null_input, null_status = load(null_directory, null=True)
    if len(null_status) != 1 or null_manifest.get("planned_fits") != 1:
        raise ValueError("Null analysis must have exactly one planned fit and one fit-status row")
    for column in IDENTITY_COLUMNS:
        _same_values(null_status, column, null_manifest.get(column), str(null_directory))
    if null_manifest.get("feature_type") != "null" or null_manifest.get("feature_id") != "null":
        raise ValueError("Null analysis must identify its feature as null")
    cohort = canonical_cohort(null_input)
    if len(null_input) != len(cohort):
        raise ValueError("Duplicate treatment series in null model input")
    digest = cohort_sha256(cohort)
    if null_manifest.get("cohort_sha256", digest) != digest:
        raise ValueError("Null manifest cohort SHA256 differs from saved model input")
    if "cohort_sha256" in null_status and not null_status.cohort_sha256.eq(digest).all():
        raise ValueError("Null fit-status cohort SHA256 differs from saved model input")
    _counts(null_status, cohort, str(null_directory))
    null_reasons, null_caveats = _eligibility(
        null_status.iloc[0], null=True, null_likelihood=None,
        likelihood_tolerance=likelihood_tolerance, information_tolerance=information_tolerance,
        n_observations=len(cohort), spec=null_spec,
    )
    if null_reasons:
        raise ValueError(f"Null model is ineligible for AIC comparison: {';'.join(null_reasons)}")
    null_likelihood, null_aic = float(null_status.iloc[0].log_likelihood), float(null_status.iloc[0].aic)
    null_status = null_status.copy()
    null_status["analysis_dir"], null_status["is_null"] = str(null_directory), True
    null_status["aic_eligible"], null_status["aic_exclusion_reason"] = True, ""
    null_status["aic_provisional"], null_status["aic_caveat"] = bool(null_caveats), ";".join(null_caveats)
    null_status["n_parameters"] = null_spec["n_parameters"]
    candidates, seen, feature_spec = [], set(), None
    for directory in feature_directories:
        manifest, spec, prepared, status = load(directory, null=False)
        if spec["sex_effect_choice"] != null_spec["sex_effect_choice"]:
            raise ValueError(f"{directory}: sex-effect mode differs from the null analysis")
        if _normalized_settings(manifest) != _normalized_settings(null_manifest):
            raise ValueError(f"{directory}: fit settings differ from the null analysis")
        _validate_features(prepared, status, manifest, cohort, str(directory))
        identities = _identities(status, str(directory))
        if overlap := seen.intersection(identities):
            raise ValueError(f"Duplicate feature/configuration candidates across analyses: {sorted(overlap)[:3]}")
        seen.update(identities)
        feature_spec = spec
        status = status.copy()
        exclusions, caveats = [], []
        for _, row in status.iterrows():
            reasons, warnings = _eligibility(
                row, null=False, null_likelihood=null_likelihood,
                likelihood_tolerance=likelihood_tolerance, information_tolerance=information_tolerance,
                n_observations=len(cohort), spec=spec,
            )
            if null_caveats:
                warnings.append("null_model_provisional")
            exclusions.append(";".join(reasons))
            caveats.append(";".join(warnings))
        status["analysis_dir"], status["is_null"] = str(directory), False
        status["aic_exclusion_reason"], status["aic_caveat"] = exclusions, caveats
        status["aic_eligible"] = status.aic_exclusion_reason.eq("")
        status["aic_provisional"] = status.aic_eligible & status.aic_caveat.ne("")
        if "n_parameters" in status:
            status["reported_n_parameters"] = status.n_parameters
        status["n_parameters"] = spec["n_parameters"]
        candidates.append(status)
    table = pd.concat([null_status, *candidates], ignore_index=True, sort=False)
    for column in ("aic", "bic", "log_likelihood", "sex_variance", "residual_variance"):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    eligible, features = table.aic_eligible, ~table.is_null
    eligible_features = eligible & features
    for column in ("aic_rank_global", "delta_aic_best_including_null", "delta_aic_best_feature",
                   "aic_improvement_over_null", "aic_rank_within_configuration",
                   "delta_aic_within_configuration_including_null"):
        table[column] = np.nan
    table.loc[eligible, "aic_rank_global"] = table.loc[eligible, "aic"].rank(method="min")
    table.loc[eligible, "delta_aic_best_including_null"] = table.loc[eligible, "aic"] - table.loc[eligible, "aic"].min()
    table.loc[eligible, "aic_improvement_over_null"] = null_aic - table.loc[eligible, "aic"]
    if eligible_features.any():
        table.loc[eligible_features, "delta_aic_best_feature"] = table.loc[eligible_features, "aic"] - table.loc[eligible_features, "aic"].min()
    for _, group in table.loc[eligible_features].groupby(CONFIGURATION_COLUMNS, dropna=False, sort=False):
        values = group.aic
        table.loc[group.index, "aic_rank_within_configuration"] = values.rank(method="min") + values.gt(null_aic).astype(int)
        table.loc[group.index, "delta_aic_within_configuration_including_null"] = values - min(null_aic, values.min())
    table = table.sort_values(["aic_rank_global", "is_null", *IDENTITY_COLUMNS], na_position="last", kind="stable").reset_index(drop=True)
    best_rows = table.loc[table.aic_eligible & ~table.is_null]
    best_candidate = None
    if not best_rows.empty:
        best = best_rows.sort_values("aic", kind="stable").iloc[0]
        best_candidate = {column: best[column] for column in [*IDENTITY_COLUMNS, "analysis_dir"]}
        best_candidate.update(aic=float(best.aic), aic_improvement_over_null=float(best.aic_improvement_over_null), aic_provisional=bool(best.aic_provisional))
    metadata = {
        "schema_version": 1,
        "comparison_type": "saved_ml_aic",
        "sex_effect_choice": null_spec["sex_effect_choice"],
        "sex_effect": null_spec["sex_effect"],
        "null_analysis_dir": str(null_directory),
        "feature_analysis_dirs": [str(path) for path in feature_directories],
        "null_aic": null_aic, "null_log_likelihood": null_likelihood,
        "null_model_provisional": bool(null_caveats), "null_model_caveats": null_caveats,
        "n_candidates": int((~table.is_null).sum()),
        "n_eligible": int((table.aic_eligible & ~table.is_null).sum()),
        "n_excluded": int((~table.aic_eligible & ~table.is_null).sum()),
        "n_provisional": int((table.aic_provisional & ~table.is_null).sum()),
        "n_observations": int(len(cohort)), "n_people": int(cohort.person_id.nunique()),
        "cohort_sha256": digest, "cohort_columns": list(cohort.columns),
        "model_spec": feature_spec, "null_model_spec": null_spec,
        "analysis_settings": analysis_settings, "input_files": input_files,
        "comparison_settings": {
            "likelihood_tolerance": likelihood_tolerance, "information_tolerance": information_tolerance,
            "cohort_numeric_rtol": 1e-12, "cohort_numeric_atol": 1e-12,
            "rank_ties": "minimum", "global_rank_includes_null": True,
            "configuration_rank_includes_null": True,
            "configuration_columns": CONFIGURATION_COLUMNS,
            "wald_inference_required_for_aic": False,
            "aic_definition": "-2 * log_likelihood + 2 * n_parameters",
            "aic_improvement_over_null_definition": "null_aic - feature_aic; positive favors feature",
            "delta_aic_best_feature_null_row": "not applicable",
            "parameter_count": ("fixed coefficients + sex variance + residual variance; boundary estimates flagged provisional"
                                if null_spec["sex_effect_choice"] == "random" else
                                "fixed coefficients including K-1 sex contrasts + residual variance; ML residual variance is RSS/n"),
            "boundary_parameter_count": ("nominal fixed + sex variance + residual variance; boundary estimates flagged provisional"
                                         if null_spec["sex_effect_choice"] == "random" else "not applicable: no random-sex variance"),
        },
        "best_candidate": best_candidate,
        "interpretation": "Relative in-sample ML support on the same saved cohort; no predictive validation or coefficient p-values are computed.",
    }
    for item in input_files:
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Saved input changed during AIC comparison: {item['path']}")
    return AICComparison(table=table, metadata=metadata)
