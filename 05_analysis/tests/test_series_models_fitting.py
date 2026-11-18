"""Continuous age, categorical sex random intercepts, and numerical failures."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from series_models.fitting import (
    FIXED_TERMS, FORMULA, NULL_FIXED_TERMS, NULL_FORMULA, NULL_MODEL_ID,
    FitSettings, fit_feature, fit_null, validate_feature_design, validate_null_design,
)


def synthetic_rows():
    rng = np.random.default_rng(108)
    age = np.repeat(np.arange(25, 37), 20)
    sex = np.tile(np.repeat(["F", "M"], 10), 12)
    E = rng.normal(size=len(age))
    factor = rng.uniform(0.5, 2, len(age))
    outcome = (
        2 + 1.5 * E - 0.6 * E * factor + 0.03 * age
        + np.where(sex == "M", 0.6, -0.6) + rng.normal(0, 0.15, len(age))
    )
    return pd.DataFrame({
        "person_id": np.arange(len(age)) // 2, "series_num": np.arange(len(age)) % 2 + 1,
        "age": age, "sex": sex, "E": E, "ef_sf": factor, "cgi_change": outcome,
    })


def fake_model_result(fixed_terms=None):
    fixed_terms = FIXED_TERMS if fixed_terms is None else fixed_terms
    estimates = pd.Series({"Intercept": 1.0, "E": 2.0, "E:ef_sf": -0.5, "age": 0.02}).loc[fixed_terms]
    covariance = pd.DataFrame(np.eye(len(fixed_terms)) * 0.04, index=fixed_terms, columns=fixed_terms)
    result = SimpleNamespace(
        converged=True, llf=-10.0, aic=30.0, bic=40.0, scale=0.1,
        vcomp=np.array([0.0]), params_object=object(), fe_params=estimates,
        pvalues=pd.Series(2 * norm.sf(abs(estimates / 0.2)), index=fixed_terms),
        cov_params=Mock(return_value=covariance),
        conf_int=Mock(return_value=pd.DataFrame({0: estimates - 1.96 * 0.2, 1: estimates + 1.96 * 0.2})),
    )
    model = SimpleNamespace(
        exog_names=fixed_terms, k_re=0, group_labels=[1.0],
        exog_vc=SimpleNamespace(names=["sex"]), fit=Mock(return_value=result),
        hessian=Mock(return_value=(-np.eye(len(fixed_terms) + 1), False)),
    )
    return model, result


class TestSeriesModelFit(unittest.TestCase):
    def test_real_fit_recovers_continuous_age_and_exact_sex_random_design(self):
        import statsmodels.formula.api as smf

        rows = synthetic_rows()
        constructor = smf.mixedlm
        constructed = []

        def record(*args, **kwargs):
            model = constructor(*args, **kwargs)
            constructed.append(model)
            return model

        with patch("statsmodels.formula.api.mixedlm", side_effect=record) as factory:
            coefficients, status = fit_feature(rows)
        self.assertEqual(status["status"], "ok", status)
        self.assertEqual(status["n_observations"], 240)
        self.assertEqual(status["n_people"], 120)
        self.assertEqual(status["n_age_values"], 12)
        self.assertEqual(status["n_sex_levels"], 2)
        self.assertEqual(coefficients["term"].tolist(), FIXED_TERMS)
        estimates = coefficients.set_index("term")["estimate"]
        np.testing.assert_allclose(estimates.loc[["E", "E:ef_sf"]], [1.5, -0.6], atol=0.12)
        self.assertAlmostEqual(estimates["age"], 0.03, delta=0.01)
        # Age zero is outside the synthetic range. Check the intercept/age
        # combination at an observed age as well as the individual age slope.
        self.assertAlmostEqual(estimates["Intercept"] + estimates["age"] * rows.age.mean(),
                               2 + 0.03 * rows.age.mean(), delta=0.12)
        self.assertTrue(coefficients["p_value"].between(0, 1).all())
        model = constructed[0]
        self.assertEqual(model.k_re, 0)
        self.assertEqual(len(model.group_labels), 1)
        self.assertEqual(model.exog_vc.names, ["sex"])
        self.assertEqual([matrix[0].shape[1] for matrix in model.exog_vc.mats], [2])
        np.testing.assert_allclose(model.exog, np.column_stack([np.ones(len(rows)), rows.E, rows.E * rows.ef_sf, rows.age]))
        self.assertEqual(factory.call_args.args[0], FORMULA)
        self.assertEqual(factory.call_args.kwargs["re_formula"], "0")
        self.assertEqual(factory.call_args.kwargs["missing"], "raise")
        self.assertEqual(factory.call_args.kwargs["vc_formula"], {"sex": "0 + C(sex)"})
        self.assertNotIn("person_id", model.exog_names)
        self.assertEqual(status["age_effect"], "continuous_fixed")
        self.assertNotIn("age_variance", status)
        self.assertNotIn("age_group_sizes", status)
        self.assertEqual(status["fit_method"], "ML")
        self.assertEqual(status["n_parameters"], 6)
        self.assertIn("FewLevelsWarning", status["warnings"])

    def test_nonconvergence_retains_variances_and_warnings(self):
        model, result = fake_model_result()
        result.converged = False

        def fit(**kwargs):
            warnings.warn("optimizer exhausted", RuntimeWarning)
            return result

        model.fit.side_effect = fit
        with patch("statsmodels.formula.api.mixedlm", return_value=model):
            coefficients, status = fit_feature(synthetic_rows(), FitSettings(maxiter=2))
        self.assertTrue(coefficients.empty)
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["converged"])
        self.assertEqual(status["residual_variance"], 0.1)
        self.assertEqual(status["sex_variance"], 0.0)
        self.assertIn("did not converge", status["error"])
        self.assertIn("optimizer exhausted", status["warnings"])

    def test_optimizer_exception_preserves_captured_warning(self):
        model, _ = fake_model_result()

        def fit(**kwargs):
            warnings.warn("before failure", RuntimeWarning)
            raise np.linalg.LinAlgError("singular solve")

        model.fit.side_effect = fit
        with patch("statsmodels.formula.api.mixedlm", return_value=model):
            coefficients, status = fit_feature(synthetic_rows())
        self.assertTrue(coefficients.empty)
        self.assertEqual(status["error_type"], "LinAlgError")
        self.assertIn("singular solve", status["error"])
        self.assertIn("before failure", status["warnings"])
        for key in [
            "log_likelihood", "aic", "bic", "sex_variance",
            "residual_variance", "sex_variance_boundary",
            "any_variance_boundary", "random_effects_covariance_singular",
            "fixed_covariance_min_scaled_eigenvalue", "information_min_scaled_eigenvalue",
        ]:
            self.assertIn(key, status)
            self.assertIsNone(status[key])

    def test_fitted_observation_count_must_match_complete_input(self):
        model, result = fake_model_result()
        result.nobs = 239
        with patch("statsmodels.formula.api.mixedlm", return_value=model):
            coefficients, status = fit_feature(synthetic_rows())
        self.assertTrue(coefficients.empty)
        self.assertEqual(status["status"], "failed")
        self.assertIn("observation count", status["error"])

    def test_covariance_and_hessian_failures_withhold_coefficients(self):
        for invalid in ["covariance", "hessian", "residual", "sex_variance"]:
            with self.subTest(invalid=invalid):
                model, result = fake_model_result()
                if invalid == "covariance":
                    result.cov_params.return_value.iloc[0, 0] = -1
                elif invalid == "hessian":
                    matrix = -np.eye(5)
                    matrix[4, 4] = 1
                    model.hessian.return_value = matrix, False
                elif invalid == "residual":
                    result.scale = 0
                else:
                    result.vcomp[0] = -0.1
                with patch("statsmodels.formula.api.mixedlm", return_value=model):
                    coefficients, status = fit_feature(synthetic_rows())
                self.assertTrue(coefficients.empty)
                self.assertEqual(status["status"], "failed")
                self.assertTrue(status["error"])
                self.assertTrue(status["converged"])
                self.assertEqual(status["log_likelihood"], -10)

    def test_zero_boundary_alone_does_not_invalidate_wald_inference(self):
        model, result = fake_model_result()

        def fit(**kwargs):
            warnings.warn("The MLE may be on the boundary of the parameter space.", UserWarning)
            return result

        model.fit.side_effect = fit
        with patch("statsmodels.formula.api.mixedlm", return_value=model):
            coefficients, status = fit_feature(synthetic_rows())
        self.assertEqual(status["status"], "ok", status)
        self.assertEqual(len(coefficients), 4)
        self.assertTrue(status["sex_variance_boundary"])
        self.assertTrue(status["any_variance_boundary"])
        self.assertIn("boundary", status["warnings"])

    def test_reml_is_explicit_and_unavailable_information_criteria_are_null(self):
        model, result = fake_model_result()
        result.aic = result.bic = np.nan
        with patch("statsmodels.formula.api.mixedlm", return_value=model):
            _, status = fit_feature(synthetic_rows(), FitSettings(reml=True, optimizer="lbfgs", maxiter=5))
        self.assertEqual(status["status"], "ok", status)
        self.assertEqual(status["fit_method"], "REML")
        self.assertIsNone(status["aic"])
        self.assertIsNone(status["bic"])
        model.fit.assert_called_once_with(reml=True, method="lbfgs", maxiter=5, disp=False)


class TestSeriesDesignValidation(unittest.TestCase):
    def test_invalid_designs_fail_without_fitting_or_dropping_rows(self):
        variants = []
        rows = synthetic_rows()
        variants.append((rows.assign(ef_sf=1.0), "Rank-deficient"))
        variants.append((rows.assign(age=40), "Rank-deficient"))
        variants.append((rows.assign(age=40 + rows.E), "Rank-deficient"))
        variants.append((rows.assign(sex="F"), "two observed levels"))
        variants.append((rows.assign(E=np.nan), "finite numeric"))
        variants.append((rows.assign(cgi_change=np.inf), "finite numeric"))
        variants.append((rows.assign(ef_sf=np.nan), "finite numeric"))
        variants.append((rows.assign(age=np.nan), "finite numeric"))
        variants.append((rows.assign(sex=""), "nonempty categories"))
        variants.append((rows.assign(sex=None), "nonempty categories"))
        variants.append((rows.assign(E=True), "Boolean"))
        variants.append((rows.iloc[:4], "More than four"))
        for invalid, message in variants:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_feature_design(invalid)
                with patch("statsmodels.formula.api.mixedlm") as factory:
                    coefficients, status = fit_feature(invalid)
                factory.assert_not_called()
                self.assertTrue(coefficients.empty)
                self.assertEqual(status["n_observations"], len(invalid))
                self.assertEqual(status["status"], "failed")
                self.assertIn(message, status["error"])

    def test_unique_unrounded_ages_are_valid_continuous_predictors(self):
        rows = synthetic_rows()
        rows["age"] = np.linspace(19.125, 86.875, len(rows))
        diagnostics = validate_feature_design(rows)
        self.assertEqual(diagnostics["design_rank"], 4)
        self.assertEqual(diagnostics["n_age_values"], len(rows))
        self.assertEqual(diagnostics["random_effects"], '{"sex": "0 + C(sex)"}')
        model, _ = fake_model_result()
        with patch("statsmodels.formula.api.mixedlm", return_value=model) as factory:
            coefficients, status = fit_feature(rows)
        self.assertEqual(status["status"], "ok", status)
        self.assertEqual(coefficients.term.tolist(), FIXED_TERMS)
        np.testing.assert_array_equal(factory.call_args.kwargs["data"].age, rows.age)

    def test_settings_are_validated(self):
        for kwargs in [{"maxiter": 0}, {"maxiter": True}, {"reml": "false"}, {"optimizer": "newton"}, {"boundary_tolerance": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FitSettings(**kwargs)


class TestNullModelFit(unittest.TestCase):
    def test_real_null_fit_uses_only_age_and_same_sex_random_intercept(self):
        import statsmodels.formula.api as smf

        rng = np.random.default_rng(602)
        age = np.linspace(19.25, 86.75, 240)
        sex = np.tile(["F", "M"], 120)
        rows = pd.DataFrame({
            "person_id": np.arange(240) // 2, "series_num": np.arange(240) % 2 + 1,
            "age": age, "sex": sex,
            "cgi_change": 1.2 + 0.025 * age + np.where(sex == "M", 0.7, -0.7)
                          + rng.normal(0, 0.15, 240),
        })
        models = []
        constructor = smf.mixedlm

        def record(*args, **kwargs):
            model = constructor(*args, **kwargs)
            models.append(model)
            return model

        with patch("statsmodels.formula.api.mixedlm", side_effect=record) as factory:
            coefficients, status = fit_null(rows)
        self.assertEqual(status["status"], "ok", status)
        self.assertEqual(status["formula"], NULL_FORMULA)
        self.assertEqual(status["model_id"], NULL_MODEL_ID)
        self.assertEqual(status["n_fixed_effects"], 2)
        self.assertEqual(status["n_parameters"], 4)
        self.assertEqual(status["n_age_values"], 240)
        self.assertEqual(coefficients.term.tolist(), ["Intercept", "age"])
        estimates = coefficients.set_index("term").estimate
        self.assertAlmostEqual(estimates["age"], 0.025, delta=0.005)
        self.assertAlmostEqual(estimates["Intercept"] + estimates["age"] * age.mean(),
                               1.2 + 0.025 * age.mean(), delta=0.08)
        np.testing.assert_allclose(models[0].exog, np.column_stack([np.ones(240), age]))
        self.assertEqual(models[0].exog_vc.names, ["sex"])
        self.assertEqual(models[0].k_re, 0)
        self.assertEqual(len(models[0].group_labels), 1)
        self.assertEqual(factory.call_args.args[0], NULL_FORMULA)
        self.assertEqual(factory.call_args.kwargs["vc_formula"], {"sex": "0 + C(sex)"})
        self.assertEqual(factory.call_args.kwargs["re_formula"], "0")
        self.assertAlmostEqual(status["aic"], -2 * status["log_likelihood"] + 2 * 4)
        self.assertAlmostEqual(status["bic"], -2 * status["log_likelihood"] + 4 * np.log(240))

    def test_null_validation_requires_no_feature_or_stimulation_values(self):
        rows = synthetic_rows().drop(columns=["E", "ef_sf"])
        diagnostic = validate_null_design(rows)
        self.assertEqual(diagnostic["design_rank"], 2)
        self.assertEqual(diagnostic["fixed_terms"], '["Intercept", "age"]')
        model, _ = fake_model_result(NULL_FIXED_TERMS)
        # Feature columns, if present in the input metadata, are not used.
        with patch("statsmodels.formula.api.mixedlm", return_value=model) as factory:
            coefficients, status = fit_null(rows.assign(E=np.nan, ef_sf=np.nan))
        self.assertEqual(status["status"], "ok", status)
        self.assertEqual(coefficients.term.tolist(), NULL_FIXED_TERMS)
        self.assertEqual(factory.call_args.args[0], "cgi_change ~ 1 + age")

    def test_null_failures_keep_likelihood_and_do_not_publish_invalid_inference(self):
        rows = synthetic_rows().drop(columns=["E", "ef_sf"])
        for failure in ("nonconvergence", "covariance"):
            with self.subTest(failure=failure):
                model, result = fake_model_result(NULL_FIXED_TERMS)
                if failure == "nonconvergence":
                    result.converged = False
                else:
                    result.cov_params.return_value.iloc[0, 0] = -1
                with patch("statsmodels.formula.api.mixedlm", return_value=model):
                    coefficients, status = fit_null(rows)
                self.assertTrue(coefficients.empty)
                self.assertEqual(status["status"], "failed")
                self.assertTrue(status["error"])
                self.assertEqual(status["log_likelihood"], -10.0)
                self.assertEqual(status["aic"], 30.0)
                self.assertEqual(status["sex_variance"], 0.0)
                self.assertEqual(status["n_parameters"], 4)

    def test_invalid_null_design_is_reported_without_optimization(self):
        rows = synthetic_rows().drop(columns=["E", "ef_sf"])
        variants = [
            (rows.assign(age=40), "Rank-deficient"),
            (rows.assign(age=np.nan), "finite numeric"),
            (rows.assign(sex="F"), "two observed levels"),
            (rows.iloc[:2], "More than two"),
        ]
        for invalid, message in variants:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_null_design(invalid)
                with patch("statsmodels.formula.api.mixedlm") as factory:
                    coefficients, status = fit_null(invalid)
                factory.assert_not_called()
                self.assertTrue(coefficients.empty)
                self.assertEqual(status["n_observations"], len(invalid))
                self.assertIn(message, status["error"])


if __name__ == "__main__":
    unittest.main()
