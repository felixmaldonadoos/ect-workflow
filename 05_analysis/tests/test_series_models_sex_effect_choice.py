"""Fixed-sex OLS, prediction, and comparison contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from series_models.aic import compare_analyses
from series_models.fitting import FitSettings, fit_feature, fit_null
from series_models.model_spec import get_model_spec
from series_models.prediction import reconstruct_prediction


def rows(levels=("Male", "Female")):
    rng = np.random.default_rng(671)
    sex = np.tile(np.asarray(levels), 24)
    n = len(sex)
    E, factor = rng.normal(size=n), rng.uniform(0.5, 2, size=n)
    age = rng.uniform(20, 80, size=n)
    shifts = {level: index * 1.25 for index, level in enumerate(sorted(levels))}
    y = 2 + 1.4 * E - 0.45 * E * factor + 0.025 * age
    y += np.array([shifts[level] for level in sex]) + rng.normal(0, 0.3, size=n)
    return pd.DataFrame({
        "person_id": [f"person_{index}" for index in range(n)], "series_num": 1,
        "sex": sex, "age": age, "E": E, "ef_sf": factor, "cgi_change": y,
    })


class TestSexEffectChoice(unittest.TestCase):
    def assert_ols_matches(self, frame, *, null=False):
        fitter = fit_null if null else fit_feature
        coefficients, status = fitter(frame, FitSettings(sex_effect="fixed"))
        self.assertEqual(status["status"], "ok", status)
        formula = "cgi_change ~ 1 + age + C(sex)" if null else "cgi_change ~ 1 + E + E:ef_sf + age + C(sex)"
        reference = smf.ols(formula, frame, missing="raise").fit()
        actual = coefficients.set_index("term")
        np.testing.assert_allclose(actual.loc[reference.params.index, "estimate"], reference.params, rtol=1e-8, atol=1e-9)
        variance_ml = float(np.sum(reference.resid ** 2) / len(frame))
        self.assertAlmostEqual(status["residual_variance"], variance_ml, places=10)
        self.assertAlmostEqual(status["log_likelihood"], reference.llf, places=9)
        # Statsmodels OLS.aic omits the residual scale parameter; this workflow
        # counts it explicitly, consistently with the random-sex ML fits.
        p = len(reference.params)
        self.assertEqual(status["n_parameters"], p + 1)
        self.assertAlmostEqual(status["aic"], -2 * reference.llf + 2 * (p + 1), places=9)
        self.assertAlmostEqual(status["bic"], -2 * reference.llf + np.log(len(frame)) * (p + 1), places=9)
        self.assertAlmostEqual(status["aic"], reference.aic + 2, places=9)
        expected_covariance = np.asarray(reference.normalized_cov_params) * variance_ml
        np.testing.assert_allclose(actual.loc[reference.params.index, "std_error"], np.sqrt(np.diag(expected_covariance)), rtol=1e-8, atol=1e-9)
        return status, reference

    def test_feature_and_null_match_independent_ols_for_two_and_three_categories(self):
        for levels in (("Male", "Female"), ("Zulu", "Alpha", "Other")):
            for null in (False, True):
                with self.subTest(levels=levels, null=null):
                    frame = rows(levels)
                    status, reference = self.assert_ols_matches(frame, null=null)
                    self.assertEqual(status["sex_reference"], sorted(levels)[0])
                    self.assertEqual(json.loads(status["sex_levels"]), sorted(levels))
                    self.assertEqual(json.loads(status["fixed_terms"]), reference.model.exog_names)
                    self.assertEqual(status["sex_effect"], "categorical_fixed")
                    self.assertEqual(status["fit_algorithm"], "analytic_ols")

    def test_reference_is_independent_of_row_order_and_existing_category_order(self):
        frame = rows(("Zulu", "Alpha", "Other"))
        baseline, first = fit_feature(frame, FitSettings(sex_effect="fixed"))
        shuffled = frame.sample(frac=1, random_state=54).copy()
        shuffled["sex"] = pd.Categorical(shuffled.sex, categories=["Zulu", "Other", "Alpha"], ordered=True)
        changed, second = fit_feature(shuffled, FitSettings(sex_effect="fixed"))
        self.assertEqual(first["status"], "ok", first)
        self.assertEqual(second["status"], "ok", second)
        self.assertEqual(second["sex_reference"], "Alpha")
        self.assertEqual(first["fixed_terms"], second["fixed_terms"])
        np.testing.assert_allclose(baseline.set_index("term").estimate, changed.set_index("term").estimate, atol=1e-9)

    def test_fixed_sex_has_no_random_variance_or_few_level_warning(self):
        _, status = fit_feature(rows(), FitSettings(sex_effect="fixed"))
        self.assertEqual(status["status"], "ok", status)
        self.assertEqual(json.loads(status["random_effects"]), {})
        self.assertEqual(status["n_outer_groups"], 0)
        self.assertEqual(status["n_group_random_intercepts"], 0)
        for column in ("sex_variance", "sex_variance_boundary", "any_variance_boundary"):
            self.assertIsNone(status[column])
        self.assertNotIn("FewLevelsWarning", status["warnings"])
        self.assertNotIn("boundary", status["warnings"].lower())
        with self.assertRaisesRegex(ValueError, "REML.*fixed-sex"):
            FitSettings(sex_effect="fixed", reml=True)

    def test_prediction_includes_fixed_categories_once_and_preserves_category_means(self):
        for null in (False, True):
            with self.subTest(null=null):
                frame = rows(("Zulu", "Alpha", "Other"))
                status, reference = self.assert_ols_matches(frame, null=null)
                prediction = reconstruct_prediction(frame, status, feature=not null, grid_points=9)
                np.testing.assert_allclose(prediction.points.fixed_prediction, reference.fittedvalues, atol=1e-9)
                np.testing.assert_allclose(prediction.points.conditional_prediction, reference.fittedvalues, atol=1e-9)
                # OLS with an intercept and category contrasts has zero mean
                # residual within every category; a doubled contrast breaks it.
                for _, group in prediction.points.groupby("sex"):
                    self.assertAlmostEqual(group.observed_cgi_change.mean(), group.conditional_prediction.mean(), places=9)
                self.assertEqual(prediction.sex_offsets["Alpha"], 0)
                for level in ("Other", "Zulu"):
                    self.assertAlmostEqual(prediction.sex_offsets[level], reference.params[f"C(sex)[T.{level}]"], places=9)
                self.assertFalse(prediction.metadata["provisional"])
                self.assertNotIn("few_sex_levels", prediction.metadata["caveat"])
                self.assertNotIn("variance_boundary", prediction.metadata["caveat"])
                for level, curve in prediction.curves.loc[prediction.curves.curve_kind.eq("sex")].groupby("sex"):
                    data = pd.DataFrame({"age": curve.x if null else frame.age.mean(), "sex": level}, index=curve.index)
                    if not null:
                        data["E"], data["ef_sf"] = curve.x, frame.ef_sf.median()
                    np.testing.assert_allclose(curve.predicted_cgi_change, reference.predict(data), atol=1e-9)

    def test_saved_aic_accepts_a_fixed_pair_and_rejects_mixing_sex_models(self):
        # Reuse the source/provenance fixture, while generating actual fixed-sex
        # fit diagnostics rather than inventing eligible likelihood values.
        from test_series_models_aic import TestSavedAICComparison

        fixture = TestSavedAICComparison("test_ranking_includes_null_and_preserves_identity_and_hashes")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        settings = FitSettings(sex_effect="fixed")
        for directory, null in ((fixture.null, True), (fixture.feature, False)):
            path, manifest = fixture.manifest(directory)
            prepared_path = Path(manifest["artifacts"]["model_input"])
            status_path = Path(manifest["artifacts"]["fit_status"])
            prepared = pd.read_csv(prepared_path, keep_default_na=False)
            original = pd.read_csv(status_path, keep_default_na=False)
            levels = sorted(prepared.sex.unique().tolist())
            spec = get_model_spec("fixed", null=null, sex_levels=levels)
            statuses = []
            for old in original.to_dict("records"):
                group = prepared if null else prepared.loc[prepared.feature_id.eq(old["feature_id"])]
                _, diagnostic = (fit_null if null else fit_feature)(group, settings)
                self.assertEqual(diagnostic["status"], "ok", diagnostic)
                statuses.append({**old, **diagnostic})
            pd.DataFrame(statuses).to_csv(status_path, index=False)
            for column in ("model_id", "fixed_formula", "age_effect", "sex_effect"):
                manifest[column] = spec[column]
                if not null:
                    prepared[column] = spec[column]
            prepared.to_csv(prepared_path, index=False)
            manifest["fit_settings"]["sex_effect"] = "fixed"
            manifest["model"].update(spec)
            path.write_text(json.dumps(manifest))
        result = compare_analyses(fixture.null, [fixture.feature])
        self.assertEqual(result.metadata["n_eligible"], 2)
        self.assertEqual(result.metadata["n_excluded"], 0)
        self.assertTrue(result.table.sex_effect.eq("categorical_fixed").all())
        self.assertTrue(result.table.loc[~result.table.is_null, "n_parameters"].eq(6).all())
        random = fixture.write_analysis("other_random", likelihoods=[-17])
        with self.assertRaisesRegex(ValueError, "sex|settings"):
            compare_analyses(fixture.null, [random])


if __name__ == "__main__":
    unittest.main()
