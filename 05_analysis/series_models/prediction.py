"""Reconstruct point predictions from saved Gaussian sex-intercept fits.

This is a GLS/OLS calculation at saved variance estimates, not another model
optimization. Reconstructed likelihoods must match the saved fit before any
predictions are returned. Wald covariance matrices are never used.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

from .cohort import canonical_cohort
from .fitting import validate_feature_design, validate_null_design
from .model_spec import get_model_spec


LIKELIHOOD_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FitPrediction:
    points: pd.DataFrame
    curves: pd.DataFrame
    coefficients: dict[str, float]
    sex_offsets: dict[str, float]
    metadata: dict


def _bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return None


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Invalid saved {name}: expected a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid saved {name}: expected a finite number") from error
    if not np.isfinite(number):
        raise ValueError(f"Invalid saved {name}: expected a finite number")
    return number


def _fit_method(diagnostics: dict) -> bool:
    method = diagnostics.get("fit_method")
    if "reml" in diagnostics:
        reml = _bool(diagnostics["reml"])
        if reml is None:
            raise ValueError("Invalid saved reml flag")
    elif method in {"ML", "REML"}:
        reml = method == "REML"
    else:
        raise ValueError("Saved fit must identify ML or REML")
    if method is not None and method != ("REML" if reml else "ML"):
        raise ValueError("Saved fit_method and reml flag disagree")
    return reml


def _sex_effect(diagnostics: dict) -> str:
    # Schema-2 random-intercept runs predate the explicit choice setting.
    value = diagnostics.get("sex_effect", "categorical_random_intercept")
    if value in {"random", "categorical_random_intercept"}:
        return "random"
    if value in {"fixed", "categorical_fixed"}:
        return "fixed"
    raise ValueError(f"Unknown saved sex-effect specification: {value!r}")


def _validate_spec(diagnostics: dict, *, feature: bool, n_observations: int, sex_effect: str, levels: list) -> dict:
    spec = get_model_spec(sex_effect, null=not feature, sex_levels=levels)
    formula, model_id = spec["fixed_formula"], spec["model_id"]
    for name, expected in (("formula", formula), ("fixed_formula", formula), ("model_id", model_id)):
        if name in diagnostics and diagnostics[name] != expected:
            raise ValueError(f"Incompatible saved {name} for prediction reconstruction")
    if "n_observations" in diagnostics and _number(diagnostics["n_observations"], "n_observations") != n_observations:
        raise ValueError("Saved observation count differs from supplied model input")
    for name, expected in (("random_effects", spec["variance_components"]),
                           ("fixed_terms", spec["fixed_terms"]),
                           ("sex_levels", levels)):
        if name not in diagnostics:
            continue
        actual = diagnostics[name]
        try:
            actual = json.loads(actual) if isinstance(actual, str) else actual
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid saved {name}") from error
        if actual != expected:
            raise ValueError(f"Incompatible saved {name} for prediction reconstruction")
    if sex_effect == "fixed" and "sex_reference" in diagnostics and diagnostics["sex_reference"] != levels[0]:
        raise ValueError("Saved fixed-sex reference differs from the first sorted sex category")
    for name, expected in (("n_outer_groups", 1 if sex_effect == "random" else 0),
                           ("n_group_random_intercepts", 0)):
        if name in diagnostics and _number(diagnostics[name], name) != expected:
            raise ValueError(f"Incompatible saved {name} for prediction reconstruction")
    return spec


def _grid(values: np.ndarray, size: int) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    return np.array([low]) if low == high else np.linspace(low, high, size)


def reconstruct_prediction(
    frame: pd.DataFrame,
    diagnostics: dict,
    *,
    feature: bool = True,
    grid_points: int = 100,
) -> FitPrediction:
    """Return reference-covariate curves and observed-row predictions.

    Feature plots hold age at its sample mean and ef_sf at its sample median.
    Observed outcomes are adjusted to those references while retaining sex
    intercept offsets. Null-model curves use age directly and leave y unchanged.
    Sex curves span only each observed sex group's predictor range. No uncertainty
    bands are computed, including when point predictions are valid but Wald
    inference was rejected.
    """
    if not isinstance(feature, bool):
        raise ValueError("feature must be Boolean")
    if isinstance(grid_points, bool) or not isinstance(grid_points, int) or grid_points < 2:
        raise ValueError("grid_points must be an integer of at least two")
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate columns in prediction model input")
    sex_effect = _sex_effect(diagnostics)
    validator = validate_feature_design if feature else validate_null_design
    design = validator(frame, sex_effect=sex_effect)
    cohort = canonical_cohort(frame)
    if len(cohort) != len(frame):
        raise ValueError("Duplicate treatment series in prediction model input")
    if _bool(diagnostics.get("converged")) is not True:
        raise ValueError("Point predictions require a converged saved fit")
    reml = _fit_method(diagnostics)
    if sex_effect == "fixed" and reml:
        raise ValueError("Fixed-sex prediction reconstruction supports ML only")
    saved_likelihood = _number(diagnostics.get("log_likelihood"), "log_likelihood")
    sex_variance = _number(diagnostics.get("sex_variance"), "sex_variance") if sex_effect == "random" else 0.0
    residual_variance = _number(diagnostics.get("residual_variance"), "residual_variance")
    if sex_variance < 0 or residual_variance <= 0:
        raise ValueError("Prediction variances require sex >= 0 and residual > 0")

    y, age = frame.cgi_change.to_numpy(float), frame.age.to_numpy(float)
    sex = frame.sex.astype(str).to_numpy()
    levels = np.unique(sex)
    spec = _validate_spec(diagnostics, feature=feature, n_observations=len(frame),
                          sex_effect=sex_effect, levels=levels.tolist())
    indicators = (sex[:, None] == levels[None, :]).astype(float)
    terms = spec["fixed_terms"]
    columns = [np.ones(len(frame))]
    if sex_effect == "fixed":
        columns.extend(indicators[:, index] for index in range(1, len(levels)))
    if feature:
        E, sf = frame.E.to_numpy(float), frame.ef_sf.to_numpy(float)
        columns.extend([E, E * sf, age])
    else:
        columns.append(age)
    X = np.column_stack(columns)

    covariance = residual_variance * np.eye(len(frame)) + sex_variance * (indicators @ indicators.T)
    if not np.isfinite(covariance).all():
        raise ValueError("Nonfinite prediction covariance")
    try:
        cholesky = np.linalg.cholesky(covariance)
        whitened_X = solve_triangular(cholesky, X, lower=True)
        whitened_y = solve_triangular(cholesky, y, lower=True)
        # Scaling protects a joint design containing an intercept, age, a PC
        # score and a much larger score*ef_sf interaction. No values are changed
        # in the exported coefficients or predictions.
        scales = np.max(np.abs(whitened_X), axis=0)
        scaled_beta, _, rank, singular_values = np.linalg.lstsq(whitened_X / scales, whitened_y, rcond=None)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError("Unable to reconstruct the saved Gaussian covariance/design") from error
    if rank != len(terms):
        raise ValueError("Rank-deficient whitened prediction design")
    beta = scaled_beta / scales
    if not np.isfinite(beta).all():
        raise ValueError("Nonfinite reconstructed point coefficients")
    fixed_prediction = X @ beta
    fixed_residual = y - fixed_prediction
    whitened_residual = solve_triangular(cholesky, fixed_residual, lower=True)
    logdet_covariance = 2 * np.log(np.diag(cholesky)).sum()
    degrees = len(frame) - (len(terms) if reml else 0)
    likelihood = -0.5 * (degrees * np.log(2 * np.pi) + logdet_covariance + whitened_residual @ whitened_residual)
    if reml:
        # log|X' V^-1 X|, evaluated through singular values rather than normal
        # equations. Together with V this matches statsmodels' REML convention.
        logdet_information = 2 * (np.log(singular_values).sum() + np.log(scales).sum())
        likelihood -= 0.5 * logdet_information
    difference = float(likelihood - saved_likelihood)
    if not np.isfinite(likelihood) or abs(difference) > LIKELIHOOD_TOLERANCE:
        raise ValueError(f"Reconstructed log likelihood differs from saved fit by {difference:.6g} (tolerance {LIKELIHOOD_TOLERANCE:g})")

    coefficients = {term: float(value) for term, value in zip(terms, beta, strict=True)}
    sex_offsets = {}
    for level in levels:
        mask = sex == level
        if sex_effect == "fixed":
            # Treatment-coded fixed contrasts have no shrinkage; the first
            # sorted category defines the common intercept reference.
            sex_offsets[str(level)] = (0.0 if level == levels[0]
                                       else coefficients[f"C(sex)[T.{level}]"])
        else:
            # Equivalent to tau² Z' V^-1 (y-X beta), including tau²=0.
            shrinkage = 0.0 if sex_variance == 0 else 1 / (mask.sum() + residual_variance / sex_variance)
            sex_offsets[str(level)] = float(shrinkage * fixed_residual[mask].sum())
    offsets = np.array([sex_offsets[level] for level in sex])
    # Fixed-category effects are already present in X beta; adding them again
    # would double their contribution. Random intercepts enter via BLUPs.
    conditional_prediction = fixed_prediction if sex_effect == "fixed" else fixed_prediction + offsets
    reference_age = float(age.mean())
    reference_ef_sf = float(np.median(sf)) if feature else None
    x_column = "E" if feature else "age"
    x_values = E if feature else age
    adjusted = y.copy()
    if feature:
        adjusted -= coefficients["age"] * (age - reference_age)
        adjusted -= coefficients["E:ef_sf"] * E * (sf - reference_ef_sf)

    point_columns = ["person_id", "series_num", "sex", *(["E"] if feature else []), "age"]
    points = frame[point_columns].copy().reset_index(drop=True)
    points["ef_sf"] = sf if feature else (frame.ef_sf.to_numpy(copy=True) if "ef_sf" in frame else None)
    points["observed_cgi_change"] = y
    points["fixed_prediction"] = fixed_prediction
    points["conditional_prediction"] = conditional_prediction
    points["residual"] = y - conditional_prediction
    points["adjusted_cgi_change"] = adjusted

    def curve(values: np.ndarray, *, kind: str, level: str = "") -> pd.DataFrame:
        x = _grid(values, grid_points)
        if feature:
            prediction = coefficients["Intercept"] + coefficients["age"] * reference_age
            prediction = prediction + (coefficients["E"] + coefficients["E:ef_sf"] * reference_ef_sf) * x
        else:
            prediction = coefficients["Intercept"] + coefficients["age"] * x
        if kind == "sex":
            prediction = prediction + sex_offsets[level]
        return pd.DataFrame({"curve_kind": kind, "sex": level, "x": x, "predicted_cgi_change": prediction})

    reference_values = x_values[sex == levels[0]] if sex_effect == "fixed" else x_values
    curves = [curve(reference_values, kind="fixed")]
    curves.extend(curve(x_values[sex == level], kind="sex", level=str(level)) for level in levels)
    curves = pd.concat(curves, ignore_index=True)
    if not np.isfinite(points[["fixed_prediction", "conditional_prediction", "residual", "adjusted_cgi_change"]]).all().all() or not np.isfinite(curves.predicted_cgi_change).all():
        raise ValueError("Nonfinite reconstructed point predictions")

    status = str(diagnostics.get("status", "unknown"))
    caveats = []
    if status != "ok":
        caveats.append("wald_inference_unavailable")
    if sex_effect == "random":
        boundary_tolerance = _number(diagnostics.get("boundary_tolerance", 1e-8), "boundary_tolerance")
        if boundary_tolerance < 0:
            raise ValueError("Invalid saved boundary_tolerance")
        if sex_variance <= boundary_tolerance or any(_bool(diagnostics.get(column)) is True for column in ("sex_variance_boundary", "any_variance_boundary")):
            caveats.append("variance_boundary")
        if _bool(diagnostics.get("random_effects_covariance_singular")) is True:
            caveats.append("random_effects_covariance_singular")
        if design["n_sex_levels"] < 5:
            caveats.append("few_sex_levels")
    warnings = diagnostics.get("warnings")
    if warnings is not None and str(warnings) not in {"", "nan"}:
        caveats.append("fit_warnings")
    aic = diagnostics.get("aic")
    try:
        aic = float(aic)
        aic = aic if np.isfinite(aic) else None
    except (TypeError, ValueError):
        aic = None
    metadata = {
        "x_column": x_column, "reference_age": reference_age, "reference_ef_sf": reference_ef_sf,
        "log_likelihood_reconstructed": float(likelihood), "log_likelihood_difference": difference,
        "likelihood_tolerance": LIKELIHOOD_TOLERANCE,
        "point_estimates_available": True, "prediction_status": "ok",
        "provisional": bool(caveats), "caveat": ";".join(caveats),
        "status": status, "converged": True, "n_observations": int(len(frame)),
        "n_people": int(design["n_people"]), "n_sex_levels": int(design["n_sex_levels"]),
        "fit_method": "REML" if reml else "ML", "aic": aic,
        "formula": spec["fixed_formula"], "model_id": spec["model_id"],
        "sex_effect": spec["sex_effect"], "sex_effect_choice": sex_effect,
        "sex_reference": str(levels[0]) if sex_effect == "fixed" else None,
        "sex_levels": levels.tolist(),
        "point_estimate_source": ("OLS at saved ML residual variance; no optimization" if sex_effect == "fixed" else
                                  "GLS and sex BLUPs at saved variance estimates; no optimization"),
        "residual_definition": "observed minus conditional prediction",
        "fixed_curve_definition": (f"fixed category reference: {levels[0]}" if sex_effect == "fixed" else
                                   "random intercept set to zero"),
        "uncertainty_available": False,
    }
    return FitPrediction(points, curves, coefficients, sex_offsets, metadata)
