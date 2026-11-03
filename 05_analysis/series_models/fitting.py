"""One-feature series models with crossed exact-age and sex intercepts.

This module consumes prepared rows only.  It deliberately does not import the
legacy SimNIBS package or reconstruct any clinical or imaging data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import warnings

import numpy as np
import pandas as pd


FORMULA = "cgi_change ~ 1 + E + E:ef_sf"
RANDOM_EFFECTS = {"age": "0 + C(age)", "sex": "0 + C(sex)"}
FIXED_TERMS = ["Intercept", "E", "E:ef_sf"]
COEFFICIENT_COLUMNS = [
    "term", "estimate", "std_error", "wald_z", "p_value", "ci_low", "ci_high",
    "inference_method",
]


@dataclass(frozen=True)
class FitSettings:
    """Numerical settings that callers must include in saved provenance."""

    reml: bool = False
    optimizer: str = "powell"
    maxiter: int = 2000
    boundary_tolerance: float = 1e-8

    def __post_init__(self):
        if not isinstance(self.reml, bool):
            raise ValueError("reml must be Boolean")
        if self.optimizer not in {"powell", "lbfgs", "bfgs", "cg", "nm"}:
            raise ValueError("Unsupported MixedLM optimizer")
        if isinstance(self.maxiter, bool) or not isinstance(self.maxiter, int) or self.maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        if not np.isfinite(self.boundary_tolerance) or self.boundary_tolerance < 0:
            raise ValueError("boundary_tolerance must be finite and nonnegative")


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"Missing model column: {column}")
    if frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(f"Boolean values are invalid in {column}")
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} must contain finite numeric values") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{column} must contain finite numeric values")
    return values


def _group_sizes(values: pd.Series) -> str:
    counts = values.value_counts(dropna=False, sort=False)
    return json.dumps({str(level): int(count) for level, count in counts.items()}, sort_keys=True)


def sample_counts(frame: pd.DataFrame) -> dict:
    """Describe all supplied series, including a design that later fails."""
    diagnostic = {
        "n_observations": int(len(frame)),
        "n_people": int(frame["person_id"].nunique()) if "person_id" in frame else None,
        "n_age_levels": int(frame["age"].nunique()) if "age" in frame else None,
        "n_sex_levels": int(frame["sex"].nunique()) if "sex" in frame else None,
        "age_group_sizes": _group_sizes(frame["age"]) if "age" in frame else "{}",
        "sex_group_sizes": _group_sizes(frame["sex"]) if "sex" in frame else "{}",
    }
    return diagnostic


def validate_feature_design(frame: pd.DataFrame) -> dict:
    """Validate the requested model without omitting or imputing any row.

    Each call represents a single already selected feature/configuration.  The
    source adapter is responsible for observation and feature identity checks.
    """
    values = {column: _numeric(frame, column) for column in ("cgi_change", "E", "ef_sf", "age")}
    if (values["age"] < 0).any():
        raise ValueError("age must be nonnegative")
    if "sex" not in frame:
        raise ValueError("Missing model column: sex")
    sex = frame["sex"].astype("string")
    stripped = sex.str.strip()
    if sex.isna().any() or stripped.str.lower().isin(["", "nan", "none", "null", "<na>"]).any():
        raise ValueError("sex must contain complete nonempty categories")
    if not sex.eq(stripped).all():
        raise ValueError("sex categories must not contain surrounding whitespace")
    with np.errstate(over="ignore", invalid="ignore"):
        interaction = values["E"] * values["ef_sf"]
    design = np.column_stack([np.ones(len(frame)), values["E"], interaction])
    if not np.isfinite(design).all():
        raise ValueError("Nonfinite fixed-effect design [1, E, E*ef_sf]")
    if len(frame) <= 3:
        raise ValueError("More than three observations are required for three fixed coefficients")
    rank = int(np.linalg.matrix_rank(design))
    if rank != 3:
        raise ValueError(f"Rank-deficient fixed-effect design [1, E, E*ef_sf]: rank={rank}")
    diagnostic = sample_counts(frame)
    # Numeric conversion must also govern exact age-level diagnostics.
    ages = pd.Series(values["age"])
    diagnostic.update(n_age_levels=int(ages.nunique()), age_group_sizes=_group_sizes(ages))
    if diagnostic["n_age_levels"] < 2 or diagnostic["n_sex_levels"] < 2:
        raise ValueError("Both age and sex random intercepts require at least two observed levels")
    if diagnostic["n_age_levels"] == len(frame):
        raise ValueError("Every series has a unique age: age-intercept variance is confounded with residual variance")
    diagnostic.update(
        design_rank=rank,
        fixed_terms=json.dumps(FIXED_TERMS),
        n_outer_groups=1,
        n_group_random_intercepts=0,
        random_effects=json.dumps(RANDOM_EFFECTS, sort_keys=True),
    )
    return diagnostic


def _positive_definite(matrix: np.ndarray, label: str) -> float:
    """Check definiteness after diagonal scaling; supplied units stay unchanged."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise RuntimeError(f"Invalid {label}: nonfinite or nonsquare matrix")
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-12):
        raise RuntimeError(f"Invalid {label}: nonsymmetric matrix")
    diagonal = np.diag(matrix)
    if (diagonal <= 0).any():
        raise RuntimeError(f"Invalid {label}: nonpositive diagonal")
    scales = np.sqrt(diagonal)
    normalized = matrix / np.outer(scales, scales)
    normalized = (normalized + normalized.T) / 2
    minimum = float(np.linalg.eigvalsh(normalized).min())
    if minimum <= 0:
        raise RuntimeError(f"Invalid {label}: matrix is not positive definite")
    np.linalg.cholesky(normalized)
    return minimum


def _result_diagnostics(model, result, status: dict, settings: FitSettings) -> None:
    status["converged"] = bool(result.converged)
    for output, attribute in (("log_likelihood", "llf"), ("aic", "aic"), ("bic", "bic")):
        value = float(getattr(result, attribute))
        status[output] = value if np.isfinite(value) else None
    status["residual_variance"] = float(result.scale)
    if len(model.exog_vc.names) != len(result.vcomp):
        raise RuntimeError("Variance-component result length does not match the requested model")
    component_values = dict(zip(model.exog_vc.names, result.vcomp))
    if set(component_values) != set(RANDOM_EFFECTS):
        raise RuntimeError("Fitted model does not have exactly the requested age and sex variance components")
    for name, value in component_values.items():
        variance = float(value)
        status[f"{name}_variance"] = variance
        status[f"{name}_variance_boundary"] = bool(np.isfinite(variance) and 0 <= variance <= settings.boundary_tolerance)
    status["any_variance_boundary"] = bool(status["age_variance_boundary"] or status["sex_variance_boundary"])


def fit_feature(frame: pd.DataFrame, settings: FitSettings | None = None) -> tuple[pd.DataFrame, dict]:
    """Return valid Wald coefficients or an empty table and explicit failure.

    Numerical failures are local to this feature; callers can continue all other
    planned fits and export a complete status report.  Boundary estimates and
    optimizer warnings are retained, while covariance/Hessian validity is
    checked directly before coefficients are published.
    """
    settings = FitSettings() if settings is None else settings
    status = {
        **sample_counts(frame), **asdict(settings), "formula": FORMULA,
        "fit_method": "REML" if settings.reml else "ML",
        "inference_method": "asymptotic_normal_wald", "status": "failed",
        "converged": False, "warnings": "", "error": "", "error_type": "",
        "log_likelihood": None, "aic": None, "bic": None,
        "age_variance": None, "sex_variance": None, "residual_variance": None,
        "age_variance_boundary": None, "sex_variance_boundary": None,
        "any_variance_boundary": None, "random_effects_covariance_singular": None,
        "fixed_covariance_min_scaled_eigenvalue": None,
        "information_min_scaled_eigenvalue": None,
    }
    coefficients = pd.DataFrame(columns=COEFFICIENT_COLUMNS)
    captured = []
    try:
        status.update(validate_feature_design(frame))
        data = frame.copy()
        for column in ("cgi_change", "E", "ef_sf", "age"):
            data[column] = _numeric(frame, column)
        data["sex"] = data["sex"].astype(str)
        import statsmodels.formula.api as smf

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                FORMULA, data=data, groups=np.ones(len(data)), re_formula="0",
                vc_formula=RANDOM_EFFECTS, missing="raise",
            )
            if model.exog_names != FIXED_TERMS or model.k_re != 0 or len(model.group_labels) != 1:
                raise RuntimeError("Constructed model differs from the exact requested fixed/random design")
            result = model.fit(
                reml=settings.reml, method=settings.optimizer, maxiter=settings.maxiter, disp=False,
            )
            _result_diagnostics(model, result, status, settings)
            if hasattr(result, "nobs") and result.nobs != len(data):
                raise RuntimeError("Fitted observation count differs from the complete supplied series")
            if not status["converged"]:
                raise RuntimeError("Mixed model did not converge")
            for name in RANDOM_EFFECTS:
                value = status[f"{name}_variance"]
                if not np.isfinite(value) or value < 0:
                    raise RuntimeError(f"Invalid {name} variance: expected finite nonnegative estimate")
            if not np.isfinite(status["residual_variance"]) or status["residual_variance"] <= 0:
                raise RuntimeError("Invalid residual variance: expected finite positive estimate")
            if status["log_likelihood"] is None:
                raise RuntimeError("Nonfinite fitted log likelihood")
            if not settings.reml and (status["aic"] is None or status["bic"] is None):
                raise RuntimeError("Nonfinite ML AIC/BIC")
            covariance = result.cov_params().loc[FIXED_TERMS, FIXED_TERMS].to_numpy()
            status["fixed_covariance_min_scaled_eigenvalue"] = _positive_definite(covariance, "fixed-effect covariance")
            hessian, singular = model.hessian(result.params_object)
            # MixedLM.hessian's second return value describes the random-effects
            # covariance/pseudoinverse, not the definiteness of the Hessian.
            status["random_effects_covariance_singular"] = bool(singular)
            status["information_min_scaled_eigenvalue"] = _positive_definite(-hessian, "observed-information Hessian")
            intervals = result.conf_int().loc[FIXED_TERMS]
            estimates = result.fe_params.loc[FIXED_TERMS].to_numpy(dtype=float)
            standard_errors = np.sqrt(np.diag(covariance))
            values = {
                "estimate": estimates,
                "std_error": standard_errors,
                "wald_z": estimates / standard_errors,
                "p_value": result.pvalues.loc[FIXED_TERMS].to_numpy(dtype=float),
                "ci_low": intervals.iloc[:, 0].to_numpy(dtype=float),
                "ci_high": intervals.iloc[:, 1].to_numpy(dtype=float),
            }
            if not np.isfinite(np.column_stack(list(values.values()))).all():
                raise RuntimeError("Nonfinite fixed coefficients or Wald inference")
            if ((values["p_value"] < 0) | (values["p_value"] > 1)).any():
                raise RuntimeError("Invalid Wald p-values")
            coefficients = pd.DataFrame({"term": FIXED_TERMS, **values})
            coefficients["inference_method"] = "asymptotic_normal_wald"
            status["status"] = "ok"
    except Exception as error:
        # Import failures and numerical optimizer/covariance errors remain visible
        # per feature; system-exiting exceptions are intentionally not caught.
        status["error"] = str(error)
        status["error_type"] = type(error).__name__
    finally:
        messages = [f"{warning.category.__name__}: {warning.message}" for warning in captured]
        if status.get("n_age_levels", 0) is not None and 0 < status["n_age_levels"] < 5:
            messages.append(f"FewLevelsWarning: age random intercept has only {status['n_age_levels']} levels")
        if status.get("n_sex_levels", 0) is not None and 0 < status["n_sex_levels"] < 5:
            messages.append(f"FewLevelsWarning: sex random intercept has only {status['n_sex_levels']} levels")
        status["warnings"] = " | ".join(messages)
    return coefficients, status
