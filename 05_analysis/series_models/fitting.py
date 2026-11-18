"""One-feature series models with continuous age and random or fixed sex.

This module consumes prepared rows only.  It deliberately does not import the
legacy SimNIBS package or reconstruct any clinical or imaging data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import warnings

import numpy as np
import pandas as pd

from .model_spec import (
    FIXED_TERMS, FORMULA, MODEL_ID, NULL_FIXED_TERMS, NULL_FORMULA,
    NULL_MODEL_ID, RANDOM_EFFECTS, get_model_spec,
)


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
    sex_effect: str = "random"

    def __post_init__(self):
        if not isinstance(self.reml, bool):
            raise ValueError("reml must be Boolean")
        if not isinstance(self.sex_effect, str) or self.sex_effect not in {"random", "fixed"}:
            raise ValueError("sex_effect must be 'random' or 'fixed'")
        if self.sex_effect == "fixed" and self.reml:
            raise ValueError("REML is unsupported for fixed-sex OLS; use ML (omit --reml)")
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
        "n_age_values": int(frame["age"].nunique()) if "age" in frame else None,
        "n_sex_levels": int(frame["sex"].nunique()) if "sex" in frame else None,
        "age_value_counts": _group_sizes(frame["age"]) if "age" in frame else "{}",
        "sex_group_sizes": _group_sizes(frame["sex"]) if "sex" in frame else "{}",
    }
    return diagnostic


def _validate_design(frame: pd.DataFrame, *, feature: bool, sex_effect: str = "random") -> dict:
    """Validate a complete continuous-age design with the requested sex effect."""
    get_model_spec(sex_effect, null=not feature)
    columns = ("cgi_change", "E", "ef_sf", "age") if feature else ("cgi_change", "age")
    values = {column: _numeric(frame, column) for column in columns}
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
    levels = sorted(sex.unique().tolist())
    if len(levels) < 2:
        label = "random intercept" if sex_effect == "random" else "fixed effect"
        raise ValueError(f"The sex {label} requires at least two observed levels")
    spec = get_model_spec(sex_effect, null=not feature, sex_levels=levels)
    fixed_terms = spec["fixed_terms"]
    if feature:
        with np.errstate(over="ignore", invalid="ignore"):
            interaction = values["E"] * values["ef_sf"]
        design = np.column_stack([np.ones(len(frame)), values["E"], interaction, values["age"]])
        design_label = "[1, E, E*ef_sf, age]"
    else:
        design = np.column_stack([np.ones(len(frame)), values["age"]])
        design_label = "[1, age]"
    if sex_effect == "fixed":
        contrasts = [(sex == level).to_numpy(dtype=float) for level in levels[1:]]
        design = np.column_stack([design[:, 0], *contrasts, design[:, 1:]])
        design_label = f"treatment-coded sex ({levels[0]!r} reference) plus {design_label}"
    if not np.isfinite(design).all():
        raise ValueError(f"Nonfinite fixed-effect design {design_label}")
    if len(frame) <= len(fixed_terms):
        number = ("four" if feature else "two") if sex_effect == "random" else str(len(fixed_terms))
        raise ValueError(f"More than {number} observations are required for {number} fixed coefficients")
    rank = int(np.linalg.matrix_rank(design))
    if rank != len(fixed_terms):
        raise ValueError(f"Rank-deficient fixed-effect design {design_label}: rank={rank}")
    diagnostic = sample_counts(frame)
    # These are descriptive counts of a continuous covariate, not random groups.
    ages = pd.Series(values["age"])
    diagnostic.update(n_age_values=int(ages.nunique()), age_value_counts=_group_sizes(ages))
    diagnostic.update(
        design_rank=rank,
        fixed_terms=json.dumps(fixed_terms),
        n_outer_groups=spec["n_outer_groups"],
        n_group_random_intercepts=0,
        random_effects=json.dumps(spec["variance_components"], sort_keys=True),
        sex_levels=json.dumps(levels),
        sex_reference=spec["sex_reference"],
        n_fixed_effects=spec["n_fixed_effects"],
        n_parameters=spec["n_parameters"],
    )
    return diagnostic


def validate_feature_design(frame: pd.DataFrame, *, sex_effect: str = "random") -> dict:
    """Validate one feature/configuration without omitting or imputing rows.

    The source adapter is responsible for observation and feature identity checks.
    """
    return _validate_design(frame, feature=True, sex_effect=sex_effect)


def validate_null_design(frame: pd.DataFrame, *, sex_effect: str = "random") -> dict:
    """Validate the age/sex null model; E and ef_sf are not required or used."""
    return _validate_design(frame, feature=False, sex_effect=sex_effect)


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
        raise RuntimeError("Fitted variance components differ from the requested sex random intercept")
    for name, value in component_values.items():
        variance = float(value)
        status[f"{name}_variance"] = variance
        status[f"{name}_variance_boundary"] = bool(np.isfinite(variance) and 0 <= variance <= settings.boundary_tolerance)
    status["any_variance_boundary"] = any(status[f"{name}_variance_boundary"] for name in RANDOM_EFFECTS)


def _fit_fixed_model(
    frame: pd.DataFrame, settings: FitSettings, *, feature: bool,
) -> tuple[pd.DataFrame, dict]:
    """Fit Gaussian OLS with treatment-coded sex and ML residual variance.

    OLS has no random effects and requires no iterative optimizer. Normal Wald
    inference uses the ML covariance RSS/n * (X'X)^-1, preserving this workflow's
    inference convention. AIC/BIC count residual variance as a parameter.
    """
    if settings.reml:
        raise ValueError("REML is unsupported for fixed-sex OLS; use ML")
    spec = get_model_spec("fixed", null=not feature)
    status = {
        **sample_counts(frame), **asdict(settings),
        "model_id": spec["model_id"], "formula": spec["fixed_formula"],
        "fixed_formula": spec["fixed_formula"],
        "sex_effect": spec["sex_effect"], "sex_effect_choice": "fixed",
        "age_effect": "continuous_fixed", "fit_backend": "ols",
        "fit_algorithm": "analytic_ols", "fit_method": "ML",
        "n_fixed_effects": None, "n_parameters": None,
        "inference_method": "asymptotic_normal_wald", "status": "failed",
        "converged": False, "warnings": "", "error": "", "error_type": "",
        "log_likelihood": None, "aic": None, "bic": None,
        "sex_variance": None, "residual_variance": None,
        "sex_variance_boundary": None, "any_variance_boundary": None,
        "random_effects_covariance_singular": None,
        "fixed_covariance_min_scaled_eigenvalue": None,
        "information_min_scaled_eigenvalue": None,
    }
    coefficients = pd.DataFrame(columns=COEFFICIENT_COLUMNS)
    captured = []
    try:
        validator = validate_feature_design if feature else validate_null_design
        status.update(validator(frame, sex_effect="fixed"))
        levels = json.loads(status["sex_levels"])
        spec = get_model_spec("fixed", null=not feature, sex_levels=levels)
        data = frame.copy()
        numeric_columns = ("cgi_change", "E", "ef_sf", "age") if feature else ("cgi_change", "age")
        for column in numeric_columns:
            data[column] = _numeric(frame, column)
        # Explicit categories guarantee the saved reference independently of
        # source row order or a preexisting pandas Categorical ordering.
        data["sex"] = pd.Categorical(data["sex"].astype(str), categories=levels, ordered=True)
        import statsmodels.formula.api as smf
        from scipy.stats import norm

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = smf.ols(spec["fixed_formula"], data=data, missing="raise")
            fixed_terms = list(model.exog_names)
            if fixed_terms != spec["fixed_terms"] or len(model.endog) != len(data):
                raise RuntimeError("Constructed OLS model differs from the requested fixed-sex design")
            X, y = np.asarray(model.exog, dtype=float), np.asarray(model.endog, dtype=float)
            scales = np.max(np.abs(X), axis=0)
            scaled_X = X / scales
            left, singular_values, right = np.linalg.svd(scaled_X, full_matrices=False)
            tolerance = max(scaled_X.shape) * np.finfo(float).eps * singular_values.max()
            rank = int(np.count_nonzero(singular_values > tolerance))
            if rank != len(fixed_terms):
                raise RuntimeError("Rank-deficient scaled fixed-sex OLS design")
            estimates = (right.T @ ((left.T @ y) / singular_values)) / scales
            if not np.isfinite(estimates).all():
                raise RuntimeError("Nonfinite fixed-sex OLS point coefficients")
            residual = y - X @ estimates
            rss = float(residual @ residual)
            variance = rss / len(data)
            if not np.isfinite(variance) or variance <= 0:
                raise RuntimeError("Invalid OLS residual variance: expected finite positive ML estimate")
            likelihood = float(-0.5 * len(data) * (np.log(2 * np.pi) + np.log(variance) + 1))
            parameters = len(fixed_terms) + 1
            if not np.isfinite(likelihood):
                raise RuntimeError("Nonfinite OLS log likelihood")
            status.update(
                converged=True, n_fixed_effects=len(fixed_terms), n_parameters=parameters,
                design_rank=rank, fixed_terms=json.dumps(fixed_terms),
                log_likelihood=likelihood, aic=-2 * likelihood + 2 * parameters,
                bic=-2 * likelihood + np.log(len(data)) * parameters,
                residual_variance=variance,
            )
            # SVD avoids squaring the design's condition number through X'X.
            scaled_inverse = (right.T / singular_values) @ (right / singular_values[:, None])
            covariance = variance * scaled_inverse / np.outer(scales, scales)
            status["fixed_covariance_min_scaled_eigenvalue"] = _positive_definite(covariance, "fixed-effect covariance")
            standard_errors = np.sqrt(np.diag(covariance))
            z = estimates / standard_errors
            critical = float(norm.ppf(0.975))
            values = {
                "estimate": estimates, "std_error": standard_errors, "wald_z": z,
                "p_value": 2 * norm.sf(np.abs(z)),
                "ci_low": estimates - critical * standard_errors,
                "ci_high": estimates + critical * standard_errors,
            }
            if not np.isfinite(np.column_stack(list(values.values()))).all():
                raise RuntimeError("Nonfinite OLS coefficients or Wald inference")
            coefficients = pd.DataFrame({"term": fixed_terms, **values})
            coefficients["inference_method"] = "asymptotic_normal_wald"
            status["status"] = "ok"
    except Exception as error:
        status["error"] = str(error)
        status["error_type"] = type(error).__name__
    finally:
        status["warnings"] = " | ".join(
            f"{warning.category.__name__}: {warning.message}" for warning in captured
        )
    return coefficients, status


def _fit_model(
    frame: pd.DataFrame, settings: FitSettings | None, *, feature: bool,
) -> tuple[pd.DataFrame, dict]:
    """Fit feature and null models with identical diagnostics and safeguards."""
    settings = FitSettings() if settings is None else settings
    if settings.sex_effect == "fixed":
        return _fit_fixed_model(frame, settings, feature=feature)
    formula = FORMULA if feature else NULL_FORMULA
    fixed_terms = FIXED_TERMS if feature else NULL_FIXED_TERMS
    model_id = MODEL_ID if feature else NULL_MODEL_ID
    status = {
        **sample_counts(frame), **asdict(settings), "formula": formula, "model_id": model_id,
        "sex_effect_choice": "random", "fit_backend": "mixedlm", "fit_algorithm": "mixedlm_optimizer",
        "n_fixed_effects": len(fixed_terms),
        # MixedLM profiles residual scale out of its parameter vector; AIC/BIC
        # count it along with fixed coefficients and sex variance.
        "n_parameters": len(fixed_terms) + len(RANDOM_EFFECTS) + 1,
        "age_effect": "continuous_fixed", "sex_effect": "categorical_random_intercept",
        "fit_method": "REML" if settings.reml else "ML",
        "inference_method": "asymptotic_normal_wald", "status": "failed",
        "converged": False, "warnings": "", "error": "", "error_type": "",
        "log_likelihood": None, "aic": None, "bic": None,
        "sex_variance": None, "residual_variance": None,
        "sex_variance_boundary": None,
        "any_variance_boundary": None, "random_effects_covariance_singular": None,
        "fixed_covariance_min_scaled_eigenvalue": None,
        "information_min_scaled_eigenvalue": None,
    }
    coefficients = pd.DataFrame(columns=COEFFICIENT_COLUMNS)
    captured = []
    try:
        validator = validate_feature_design if feature else validate_null_design
        status.update(validator(frame))
        data = frame.copy()
        numeric_columns = ("cgi_change", "E", "ef_sf", "age") if feature else ("cgi_change", "age")
        for column in numeric_columns:
            data[column] = _numeric(frame, column)
        data["sex"] = data["sex"].astype(str)
        import statsmodels.formula.api as smf

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                formula, data=data, groups=np.ones(len(data)), re_formula="0",
                vc_formula=RANDOM_EFFECTS, missing="raise",
            )
            if model.exog_names != fixed_terms or model.k_re != 0 or len(model.group_labels) != 1:
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
            covariance = result.cov_params().loc[fixed_terms, fixed_terms].to_numpy()
            status["fixed_covariance_min_scaled_eigenvalue"] = _positive_definite(covariance, "fixed-effect covariance")
            hessian, singular = model.hessian(result.params_object)
            # MixedLM.hessian's second return value describes the random-effects
            # covariance/pseudoinverse, not the definiteness of the Hessian.
            status["random_effects_covariance_singular"] = bool(singular)
            status["information_min_scaled_eigenvalue"] = _positive_definite(-hessian, "observed-information Hessian")
            intervals = result.conf_int().loc[fixed_terms]
            estimates = result.fe_params.loc[fixed_terms].to_numpy(dtype=float)
            standard_errors = np.sqrt(np.diag(covariance))
            values = {
                "estimate": estimates,
                "std_error": standard_errors,
                "wald_z": estimates / standard_errors,
                "p_value": result.pvalues.loc[fixed_terms].to_numpy(dtype=float),
                "ci_low": intervals.iloc[:, 0].to_numpy(dtype=float),
                "ci_high": intervals.iloc[:, 1].to_numpy(dtype=float),
            }
            if not np.isfinite(np.column_stack(list(values.values()))).all():
                raise RuntimeError("Nonfinite fixed coefficients or Wald inference")
            if ((values["p_value"] < 0) | (values["p_value"] > 1)).any():
                raise RuntimeError("Invalid Wald p-values")
            coefficients = pd.DataFrame({"term": fixed_terms, **values})
            coefficients["inference_method"] = "asymptotic_normal_wald"
            status["status"] = "ok"
    except Exception as error:
        # Import failures and numerical optimizer/covariance errors remain visible
        # per feature; system-exiting exceptions are intentionally not caught.
        status["error"] = str(error)
        status["error_type"] = type(error).__name__
    finally:
        messages = [f"{warning.category.__name__}: {warning.message}" for warning in captured]
        if status.get("n_sex_levels", 0) is not None and 0 < status["n_sex_levels"] < 5:
            messages.append(f"FewLevelsWarning: sex random intercept has only {status['n_sex_levels']} levels")
        status["warnings"] = " | ".join(messages)
    return coefficients, status


def fit_feature(frame: pd.DataFrame, settings: FitSettings | None = None) -> tuple[pd.DataFrame, dict]:
    """Return valid Wald coefficients or an empty table and explicit failure.

    Numerical failures are local to this feature; callers can continue all other
    planned fits and export a complete status report. Boundary estimates and
    optimizer warnings are retained, while covariance/Hessian validity is
    checked directly before coefficients are published.
    """
    return _fit_model(frame, settings, feature=True)


def fit_null(frame: pd.DataFrame, settings: FitSettings | None = None) -> tuple[pd.DataFrame, dict]:
    """Fit the age/sex null using the selected random or fixed sex effect.

    No E or interaction term is constructed. Likelihood, variances, and AIC/BIC
    remain available in diagnostics even if later Wald checks reject inference.
    """
    return _fit_model(frame, settings, feature=False)
