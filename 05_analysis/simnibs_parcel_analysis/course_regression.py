"""The two assigned regressions, with one observation per treatment course."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from .lmm import SessionFields


NULL_FORMULA = "Tx ~ 1 + age + C(sex) + C(electrode_placement)"
ROI_FORMULA = "Tx ~ 1 + EF_roi + EF_scaled"
RANDOM_EFFECTS = {"age": "0 + C(age)", "sex": "0 + C(sex)"}


def _unique_mode(values: pd.Series, label: str):
    counts = values.value_counts()
    modes = counts.index[counts.eq(counts.max())]
    if len(modes) != 1:
        raise ValueError(f"{label}: a unique mode is required; counts={counts.to_dict()}")
    return modes[0]


def prepare_course_predictors(sessions: pd.DataFrame, courses: pd.DataFrame, fields: SessionFields, *, aggregation: str = "mode") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return clinical courses, course x ROI raw/scaled fields, and a session audit.

    Mode selects the modal placement, then modal ef_sf within that placement.
    Mean/sum aggregate each session's E and E*ef_sf, preserving their pairing.
    The null model always uses the course's modal clinical placement.
    """
    if aggregation not in {"mode", "mean", "sum"}:
        raise ValueError("aggregation must be mode, mean, or sum")
    required = {"treatment_course_id", "subjid_base", "cgi_start", "cgi_end", "age", "sex"}
    if missing := required.difference(courses):
        raise ValueError(f"Course table is missing {sorted(missing)}; supply course covariates or --subject-info")
    clinical = courses.copy().set_index("treatment_course_id", verify_integrity=True)
    if clinical.index.isna().any() or set(clinical.index) != set(sessions["treatment_course_id"]):
        raise ValueError("Course identities must be complete and match the session table")
    for column in ["age", "cgi_start", "cgi_end"]:
        if clinical[column].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise ValueError(f"Boolean values are invalid in {column}")
        clinical[column] = pd.to_numeric(clinical[column], errors="raise").astype(float)
        if not np.isfinite(clinical[column]).all():
            raise ValueError(f"Missing/nonfinite {column}")
    if clinical["age"].lt(0).any():
        raise ValueError("age must be nonnegative")
    sex = clinical["sex"].astype("string").str.strip()
    if sex.isna().any() or sex.str.lower().isin(["", "nan"]).any():
        raise ValueError("Missing sex category")
    clinical["sex"] = sex.astype(str)
    clinical["Tx"] = clinical["cgi_end"] - clinical["cgi_start"]
    session_ids = pd.Index(sessions["session_id"], name="session_id")
    if session_ids.has_duplicates or not fields.raw.index.equals(session_ids) or not fields.scaled.index.equals(session_ids):
        raise ValueError("Field rows must match unique session IDs in order")
    if not fields.raw.columns.equals(fields.scaled.columns) or fields.raw.shape[1] == 0:
        raise ValueError("Raw and scaled matrices must have the same nonempty ROI columns")
    if not np.isfinite(fields.raw.to_numpy()).all() or not np.isfinite(fields.scaled.to_numpy()).all():
        raise ValueError("Missing/nonfinite session ROI fields; select valid ROIs explicitly")
    if not np.allclose(fields.scaled, fields.raw.mul(sessions["ef_sf"].to_numpy(), axis=0), rtol=1e-12, atol=0):
        raise ValueError("Scaled fields must equal each session's E multiplied by its own ef_sf")

    audit = sessions.copy()
    audit["used_for_course_EF"] = False
    records, raw_rows, scaled_rows = [], {}, {}
    for course_id, group in sessions.groupby("treatment_course_id", sort=False):
        placement = _unique_mode(group["electrode_placement"], f"{course_id}: electrode placement")
        chosen = group
        if aggregation == "mode":
            chosen = group.loc[group["electrode_placement"].eq(placement)]
            factor = _unique_mode(chosen["ef_sf"], f"{course_id}: ef_sf at {placement}")
            chosen = chosen.loc[chosen["ef_sf"].eq(factor)]
            if chosen["hdf5_fn"].nunique() != 1:
                raise ValueError(f"{course_id}: modal sessions map to multiple HDF5 files")
            raw = fields.raw.loc[chosen["session_id"]]
            scaled = fields.scaled.loc[chosen["session_id"]]
            if not raw.eq(raw.iloc[0]).all(axis=None) or not scaled.eq(scaled.iloc[0]).all(axis=None):
                raise ValueError(f"{course_id}: modal sessions have inconsistent fields")
            raw_rows[course_id], scaled_rows[course_id] = raw.iloc[0], scaled.iloc[0]
        else:
            factor = group["ef_sf"].agg(aggregation)
            raw_rows[course_id] = fields.raw.loc[group["session_id"]].agg(aggregation)
            scaled_rows[course_id] = fields.scaled.loc[group["session_id"]].agg(aggregation)
        audit.loc[audit["session_id"].isin(chosen["session_id"]), "used_for_course_EF"] = True
        counts = group["electrode_placement"].value_counts().sort_index()
        records.append({"treatment_course_id": course_id, "electrode_placement": placement, "ef_sf": factor,
                        "n_sessions": len(group), "n_sessions_used_for_EF": len(chosen),
                        "n_placements": len(counts), "placement_counts": json.dumps(counts.to_dict())})
    summary = pd.DataFrame(records).set_index("treatment_course_id")
    if "electrode_placement" in clinical:
        supplied = clinical["electrode_placement"].astype("string").str.strip().str.upper()
        if not supplied.eq(summary["electrode_placement"].reindex(clinical.index)).all():
            raise ValueError("Supplied course electrode_placement disagrees with its session mode")
        clinical = clinical.drop(columns="electrode_placement")
    if overlap := set(summary).intersection(clinical):
        raise ValueError(f"Course inputs already contain derived columns: {sorted(overlap)}")
    clinical = clinical.join(summary, validate="one_to_one")
    raw = pd.DataFrame.from_dict(raw_rows, orient="index").reindex(clinical.index)
    scaled = pd.DataFrame.from_dict(scaled_rows, orient="index").reindex(clinical.index)
    raw.index.name = scaled.index.name = "treatment_course_id"
    if not np.isfinite(raw.to_numpy()).all() or not np.isfinite(scaled.to_numpy()).all():
        raise ValueError("Course exposure aggregation produced nonfinite values")
    return clinical, raw, scaled, audit


def _fit_summary(model, *, mixed: bool) -> tuple[pd.DataFrame, dict]:
    """Keep optimizer warnings visible; never publish invalid Wald inference."""
    design = np.asarray(model.exog)
    if not np.isfinite(design).all() or len(design) <= design.shape[1] or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Insufficient courses or rank-deficient/nonfinite model design")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = model.fit(reml=False, method="powell", maxiter=2000, disp=False) if mixed else model.fit()
    messages = [str(item.message) for item in captured]
    for message in messages:
        warnings.warn(message, UserWarning, stacklevel=2)
    if mixed and not result.converged:
        raise RuntimeError(f"Mixed model did not converge: {messages}")
    if mixed and any("singular" in text.lower() or "not positive definite" in text.lower() for text in messages):
        raise RuntimeError(f"Invalid mixed-model covariance/Hessian: {messages}")
    terms = model.exog_names
    covariance = result.cov_params().loc[terms, terms].to_numpy()
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
        raise RuntimeError("Invalid fixed-effect covariance; inference is unavailable")
    intervals = result.conf_int().loc[terms]
    coefficients = pd.DataFrame({"term": terms, "estimate": result.params.loc[terms].to_numpy(),
                                 "std_error": result.bse.loc[terms].to_numpy(), "p_value": result.pvalues.loc[terms].to_numpy(),
                                 "ci_low": intervals.iloc[:, 0].to_numpy(), "ci_high": intervals.iloc[:, 1].to_numpy()})
    if not np.isfinite(coefficients.drop(columns="term").to_numpy()).all():
        raise RuntimeError("Nonfinite coefficients or uncertainty")
    diagnostics = {"status": "ok", "n_courses": int(result.nobs), "log_likelihood": float(result.llf),
                   "aic": float(result.aic), "bic": float(result.bic), "warnings": " | ".join(messages)}
    if mixed:
        diagnostics.update({f"{name}_variance": float(value) for name, value in zip(model.exog_vc.names, result.vcomp)})
        diagnostics["residual_variance"] = float(result.scale)
    return coefficients, diagnostics


def fit_course_regressions(courses: pd.DataFrame, raw: pd.DataFrame, scaled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the null once, then the assigned crossed age/sex model for each ROI.

    There is no patient effect. EF_scaled represents the requested EF:ef_sf
    term, constructed within sessions before course aggregation. Age is not
    rounded/binned; its exact observed values define random-intercept groups.
    Failed fits are reported explicitly while other ROIs continue.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests

    if not courses.index.is_unique or not raw.index.equals(courses.index) or not scaled.index.equals(courses.index):
        raise ValueError("Predictors must align with unique treatment-course rows")
    if not raw.columns.equals(scaled.columns):
        raise ValueError("Raw and scaled ROI columns differ")
    coefficients, fits = [], []
    try:
        table, diagnostics = _fit_summary(smf.ols(NULL_FORMULA, data=courses, missing="raise"), mixed=False)
        coefficients.append(table.assign(model="null", roi=""))
        fits.append({"model": "null", "roi": "", **diagnostics})
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        fits.append({"model": "null", "roi": "", "status": "failed", "error": str(error)})

    age_levels, sex_levels = courses["age"].nunique(), courses["sex"].nunique()
    for name, levels in [("age", age_levels), ("sex", sex_levels)]:
        if levels < 5:
            warnings.warn(f"{name} random intercept has only {levels} levels; its variance may be poorly estimated", UserWarning)
    for roi in raw:
        try:
            if age_levels < 2 or sex_levels < 2:
                raise ValueError("Both age and sex random effects require at least two observed levels")
            if age_levels == len(courses):
                raise ValueError("Every course has a unique age: age-intercept variance is confounded with residual variance")
            data = courses.assign(EF_roi=raw[roi], EF_scaled=scaled[roi])
            model = smf.mixedlm(ROI_FORMULA, data=data, groups=np.ones(len(data)), re_formula="0", vc_formula=RANDOM_EFFECTS, missing="raise")
            table, diagnostics = _fit_summary(model, mixed=True)
            coefficients.append(table.assign(model="roi", roi=roi))
            fits.append({"model": "roi", "roi": roi, **diagnostics})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            fits.append({"model": "roi", "roi": roi, "status": "failed", "error": str(error)})

    result = pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame(columns=["term", "estimate", "p_value", "model", "roi"])
    result["q_value_bh"] = np.nan
    for term in ["EF_roi", "EF_scaled"]:
        selected = result["model"].eq("roi") & result["term"].eq(term)
        # Keep the planned ROI family, including failed fits as p=1 for correction.
        p_values = pd.Series(1.0, index=raw.columns)
        p_values.loc[result.loc[selected, "roi"]] = result.loc[selected, "p_value"].to_numpy()
        q_values = pd.Series(multipletests(p_values, method="fdr_bh")[1], index=raw.columns)
        result.loc[selected, "q_value_bh"] = result.loc[selected, "roi"].map(q_values)
    return result, pd.DataFrame(fits)
