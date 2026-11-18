"""Shared, dependency-free definitions for the supported series models."""

from __future__ import annotations


MODEL_ID = "continuous_age_sex_random_intercept_v1"
FORMULA = "cgi_change ~ 1 + E + E:ef_sf + age"
RANDOM_EFFECTS = {"sex": "0 + C(sex)"}
FIXED_TERMS = ["Intercept", "E", "E:ef_sf", "age"]
NULL_MODEL_ID = "continuous_age_sex_random_intercept_null_v1"
NULL_FORMULA = "cgi_change ~ 1 + age"
NULL_FIXED_TERMS = ["Intercept", "age"]
SEX_FIXED_MODEL_ID = "continuous_age_sex_fixed_v1"
SEX_FIXED_NULL_MODEL_ID = "continuous_age_sex_fixed_null_v1"
SEX_FIXED_FORMULA = FORMULA + " + C(sex)"
SEX_FIXED_NULL_FORMULA = NULL_FORMULA + " + C(sex)"


def get_model_spec(sex_effect="random", *, null=False, sex_levels=None) -> dict:
    """Return a fresh specification, with deterministic treatment contrasts.

    Fixed-sex terms follow Patsy's order: intercept, categorical contrasts,
    then numeric terms. The first lexicographically sorted observed sex label
    is the reference. Without supplied levels, fixed-sex term/count fields are
    None; callers must resolve them from the actual saved clinical cohort.
    """
    if not isinstance(sex_effect, str) or sex_effect not in {"random", "fixed"}:
        raise ValueError("sex_effect must be 'random' or 'fixed'")
    if not isinstance(null, bool):
        raise ValueError("null must be Boolean")
    levels = None
    if sex_levels is not None:
        if isinstance(sex_levels, (str, bytes)):
            raise ValueError("sex_levels must be a sequence of complete category strings")
        levels = list(sex_levels)
        if any(not isinstance(level, str) or not level or level != level.strip()
               or level.lower() in {"nan", "none", "null", "<na>"} for level in levels):
            raise ValueError("sex_levels must contain complete unpadded category strings")
        if len(set(levels)) != len(levels) or len(levels) < 2:
            raise ValueError("sex_levels must contain at least two distinct observed categories")
        levels = sorted(levels)
    random = sex_effect == "random"
    if random:
        model_id = NULL_MODEL_ID if null else MODEL_ID
        formula = NULL_FORMULA if null else FORMULA
        terms = list(NULL_FIXED_TERMS if null else FIXED_TERMS)
    else:
        model_id = SEX_FIXED_NULL_MODEL_ID if null else SEX_FIXED_MODEL_ID
        formula = SEX_FIXED_NULL_FORMULA if null else SEX_FIXED_FORMULA
        terms = None if levels is None else [
            "Intercept", *[f"C(sex)[T.{level}]" for level in levels[1:]],
            *(["age"] if null else ["E", "E:ef_sf", "age"]),
        ]
    p = len(terms) if terms is not None else None
    return {
        "model_id": model_id,
        "fixed_formula": formula,
        "fixed_terms": terms,
        "variance_components": dict(RANDOM_EFFECTS) if random else {},
        "age_effect": "continuous_fixed",
        "sex_effect": "categorical_random_intercept" if random else "categorical_fixed",
        "sex_effect_choice": sex_effect,
        "sex_levels": levels,
        "sex_reference": levels[0] if not random and levels is not None else None,
        "n_fixed_effects": p,
        "n_parameters": p + (2 if random else 1) if p is not None else None,
        "fit_backend": "mixedlm" if random else "ols",
        "outer_group": "constant" if random else None,
        "n_outer_groups": 1 if random else 0,
        "group_random_intercept": False,
        "n_group_random_intercepts": 0,
        "patient_random_effect": False,
        "observation_unit": "treatment series",
        "centering": "none",
        "standardization": "none",
    }
