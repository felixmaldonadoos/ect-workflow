"""Point predictions reconstruct saved fits without using Wald covariance."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from series_models.fitting import (
    FIXED_TERMS, FORMULA, MODEL_ID, NULL_FIXED_TERMS, NULL_FORMULA, NULL_MODEL_ID,
    RANDOM_EFFECTS,
)
from series_models.prediction import reconstruct_prediction


def synthetic_rows():
    rng = np.random.default_rng(754)
    n = 80
    E = rng.normal(size=n)
    sf = rng.uniform(0.5, 2, size=n)
    age = rng.uniform(20, 80, size=n)
    sex = np.array(["Female", "Male"] * (n // 2))
    y = 1.5 + 1.2 * E - 0.35 * E * sf + 0.02 * age
    y += np.where(sex == "Female", -1.5, 1.5) + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"person_id": [f"person_{index // 2}" for index in range(n)],
                         "series_num": np.arange(n) % 2 + 1, "sex": sex,
                         "E": E, "ef_sf": sf, "age": age, "cgi_change": y})


def manual_diagnostics(frame, *, feature=True, sex_variance=0.0, residual_variance=0.5, reml=False):
    """Independent likelihood calculation for exact-boundary/invalid-fit cases."""
    y = frame.cgi_change.to_numpy(float)
    X = np.column_stack([np.ones(len(frame)), frame.E, frame.E * frame.ef_sf, frame.age]) if feature else np.column_stack([np.ones(len(frame)), frame.age])
    sex = frame.sex.to_numpy()
    Z = (sex[:, None] == np.unique(sex)[None, :]).astype(float)
    V = residual_variance * np.eye(len(frame)) + sex_variance * (Z @ Z.T)
    inverse = np.linalg.inv(V)
    beta = np.linalg.solve(X.T @ inverse @ X, X.T @ inverse @ y)
    residual = y - X @ beta
    _, logdet = np.linalg.slogdet(V)
    n = len(frame) - (X.shape[1] if reml else 0)
    likelihood = -0.5 * (n * np.log(2 * np.pi) + logdet + residual @ inverse @ residual)
    if reml:
        likelihood -= 0.5 * np.linalg.slogdet(X.T @ inverse @ X)[1]
    return {"converged": True, "status": "ok", "reml": reml,
            "fit_method": "REML" if reml else "ML", "log_likelihood": likelihood,
            "sex_variance": sex_variance, "residual_variance": residual_variance,
            "formula": FORMULA if feature else NULL_FORMULA,
            "model_id": MODEL_ID if feature else NULL_MODEL_ID,
            "n_observations": len(frame), "boundary_tolerance": 1e-8}


class TestFitPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import statsmodels.formula.api as smf

        cls.frame = synthetic_rows()
        cls.fits = {}
        for feature in (True, False):
            for reml in (False, True):
                formula = FORMULA if feature else NULL_FORMULA
                model = smf.mixedlm(formula, cls.frame, groups=np.ones(len(cls.frame)),
                                    re_formula="0", vc_formula=RANDOM_EFFECTS, missing="raise")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = model.fit(reml=reml, method="powell", maxiter=2000, disp=False)
                if not result.converged:
                    raise AssertionError("Synthetic reference fit did not converge")
                diagnostics = {"converged": True, "status": "ok", "reml": reml,
                               "fit_method": "REML" if reml else "ML",
                               "log_likelihood": result.llf, "sex_variance": result.vcomp[0],
                               "residual_variance": result.scale, "n_observations": len(cls.frame),
                               "formula": formula, "fixed_formula": formula,
                               "model_id": MODEL_ID if feature else NULL_MODEL_ID,
                               "fixed_terms": json.dumps(FIXED_TERMS if feature else NULL_FIXED_TERMS),
                               "random_effects": json.dumps(RANDOM_EFFECTS),
                               "aic": result.aic}
                cls.fits[(feature, reml)] = (result, diagnostics)

    def test_reconstruction_matches_statsmodels_ml_and_reml_for_both_designs(self):
        for (feature, reml), (result, diagnostics) in self.fits.items():
            with self.subTest(feature=feature, reml=reml):
                prediction = reconstruct_prediction(self.frame, diagnostics, feature=feature)
                terms = FIXED_TERMS if feature else NULL_FIXED_TERMS
                np.testing.assert_allclose([prediction.coefficients[term] for term in terms], result.fe_params.loc[terms], atol=1e-8)
                np.testing.assert_allclose(prediction.points.fixed_prediction, result.predict(), atol=1e-8)
                np.testing.assert_allclose(prediction.points.conditional_prediction, result.fittedvalues, atol=1e-8)
                np.testing.assert_allclose(prediction.points.residual, result.resid, atol=1e-8)
                self.assertLess(abs(prediction.metadata["log_likelihood_difference"]), 1e-8)
                self.assertEqual(prediction.metadata["fit_method"], "REML" if reml else "ML")
                self.assertTrue(prediction.metadata["point_estimates_available"])

    def test_feature_adjustment_uses_mean_age_and_median_factor(self):
        _, diagnostics = self.fits[(True, False)]
        original = self.frame.copy(deep=True)
        prediction = reconstruct_prediction(self.frame, diagnostics)
        p, beta = prediction.points, prediction.coefficients
        a0, f0 = self.frame.age.mean(), self.frame.ef_sf.median()
        self.assertEqual(prediction.metadata["reference_age"], a0)
        self.assertEqual(prediction.metadata["reference_ef_sf"], f0)
        expected = p.observed_cgi_change - beta["age"] * (p.age - a0) - beta["E:ef_sf"] * p.E * (p.ef_sf - f0)
        np.testing.assert_allclose(p.adjusted_cgi_change, expected)
        reference_mean = beta["Intercept"] + beta["age"] * a0 + (beta["E"] + beta["E:ef_sf"] * f0) * p.E
        group_offsets = p.sex.map(prediction.sex_offsets)
        np.testing.assert_allclose(p.adjusted_cgi_change - reference_mean - group_offsets, p.residual, atol=1e-12)
        pd.testing.assert_frame_equal(self.frame, original)

    def test_curves_are_parallel_and_use_only_observed_group_ranges(self):
        _, diagnostics = self.fits[(True, False)]
        prediction = reconstruct_prediction(self.frame, diagnostics, grid_points=17)
        curves = prediction.curves
        slope = prediction.coefficients["E"] + prediction.coefficients["E:ef_sf"] * prediction.metadata["reference_ef_sf"]
        for (kind, sex), rows in curves.groupby(["curve_kind", "sex"], dropna=False):
            source = self.frame if kind == "fixed" else self.frame.loc[self.frame.sex.eq(sex)]
            self.assertEqual(len(rows), 17)
            self.assertEqual(rows.x.min(), source.E.min())
            self.assertEqual(rows.x.max(), source.E.max())
            np.testing.assert_allclose(np.diff(rows.predicted_cgi_change) / np.diff(rows.x), slope, atol=1e-10)
            intercept = prediction.coefficients["Intercept"] + prediction.coefficients["age"] * prediction.metadata["reference_age"]
            if kind == "sex":
                intercept += prediction.sex_offsets[sex]
            np.testing.assert_allclose(rows.predicted_cgi_change, intercept + slope * rows.x)

    def test_null_does_not_need_E_or_factor_and_uses_age_as_x(self):
        _, diagnostics = self.fits[(False, True)]
        frame = self.frame.drop(columns=["E", "ef_sf"])
        prediction = reconstruct_prediction(frame, diagnostics, feature=False)
        self.assertEqual(prediction.metadata["x_column"], "age")
        self.assertIsNone(prediction.metadata["reference_ef_sf"])
        self.assertNotIn("E", prediction.points)
        np.testing.assert_array_equal(prediction.points.adjusted_cgi_change, frame.cgi_change)
        self.assertTrue(prediction.points.ef_sf.isna().all())
        fixed = prediction.curves.loc[prediction.curves.curve_kind.eq("fixed")]
        np.testing.assert_allclose(fixed.predicted_cgi_change, prediction.coefficients["Intercept"] + prediction.coefficients["age"] * fixed.x)

    def test_zero_sex_variance_has_exactly_zero_offsets(self):
        for reml in (False, True):
            with self.subTest(reml=reml):
                diagnostics = manual_diagnostics(self.frame, reml=reml)
                prediction = reconstruct_prediction(self.frame, diagnostics)
                self.assertTrue(all(value == 0 for value in prediction.sex_offsets.values()))
                np.testing.assert_array_equal(prediction.points.fixed_prediction, prediction.points.conditional_prediction)
                self.assertTrue(prediction.metadata["provisional"])
                self.assertIn("variance_boundary", prediction.metadata["caveat"])

    def test_failed_wald_diagnostics_preserve_point_predictions_without_uncertainty(self):
        _, original = self.fits[(True, False)]
        diagnostics = {**original, "status": "failed", "warnings": "Hessian is not positive definite",
                       "error": "Invalid observed-information Hessian"}
        prediction = reconstruct_prediction(self.frame, diagnostics)
        self.assertTrue(prediction.metadata["provisional"])
        self.assertIn("wald_inference_unavailable", prediction.metadata["caveat"])
        self.assertIn("fit_warnings", prediction.metadata["caveat"])
        self.assertFalse(prediction.metadata["uncertainty_available"])
        self.assertFalse(any(column.startswith(("ci_", "std_error")) for column in prediction.curves))

    def test_nonconvergence_invalid_variances_and_likelihood_mismatch_reject(self):
        _, original = self.fits[(True, False)]
        for name, value, message in (
            ("converged", False, "converged"),
            ("sex_variance", -0.1, "variances"),
            ("residual_variance", 0, "variances"),
            ("log_likelihood", np.nan, "finite"),
            ("log_likelihood", original["log_likelihood"] + 1, "differs from saved"),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    reconstruct_prediction(self.frame, {**original, name: value})

    def test_wrong_design_metadata_and_duplicate_series_reject(self):
        _, original = self.fits[(True, False)]
        for name, value in (("n_observations", len(self.frame) - 1), ("model_id", "old_age_categorical"),
                            ("formula", NULL_FORMULA), ("fit_method", "REML"),
                            ("random_effects", json.dumps({"age": "0 + C(age)"})),
                            ("n_group_random_intercepts", 1)):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    reconstruct_prediction(self.frame, {**original, name: value})
        repeated = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate treatment series"):
            reconstruct_prediction(repeated, {**original, "n_observations": len(repeated)})

    def test_one_x_value_for_one_group_produces_one_curve_point(self):
        frame = self.frame.copy()
        frame.loc[frame.sex.eq("Female"), "E"] = 0.25
        diagnostics = manual_diagnostics(frame, sex_variance=0.2)
        prediction = reconstruct_prediction(frame, diagnostics, grid_points=9)
        female = prediction.curves.loc[prediction.curves.sex.eq("Female")]
        self.assertEqual(len(female), 1)
        self.assertEqual(female.x.iloc[0], 0.25)
        self.assertEqual(len(prediction.curves.loc[prediction.curves.curve_kind.eq("fixed")]), 9)

    def test_scaled_predictors_reconstruct_same_ml_predictions(self):
        _, diagnostics = self.fits[(True, False)]
        frame = self.frame.copy()
        frame.E *= 1e6
        frame.ef_sf *= 1000
        prediction = reconstruct_prediction(frame, diagnostics)
        original = reconstruct_prediction(self.frame, diagnostics)
        np.testing.assert_allclose(prediction.points.fixed_prediction, original.points.fixed_prediction, atol=1e-8)
        np.testing.assert_allclose(prediction.points.conditional_prediction, original.points.conditional_prediction, atol=1e-8)

    def test_invalid_grid_and_missing_model_method_reject(self):
        _, diagnostics = self.fits[(True, False)]
        for value in (True, 0, 1, 2.5):
            with self.subTest(grid=value):
                with self.assertRaisesRegex(ValueError, "grid_points"):
                    reconstruct_prediction(self.frame, diagnostics, grid_points=value)
        unknown = copy.deepcopy(diagnostics)
        unknown.pop("reml")
        unknown.pop("fit_method")
        with self.assertRaisesRegex(ValueError, "identify ML or REML"):
            reconstruct_prediction(self.frame, unknown)


if __name__ == "__main__":
    unittest.main()
