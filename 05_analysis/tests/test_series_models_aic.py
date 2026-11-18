"""Saved-model AIC accounting, cohort protection, and numerical eligibility."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import series_models.aic as aic_module
from series_models.aic import CONFIGURATION_COLUMNS, compare_analyses
from series_models.cohort import cohort_sha256
from series_models.fitting import (
    FIXED_TERMS, FORMULA, MODEL_ID, NULL_FIXED_TERMS, NULL_FORMULA,
    NULL_MODEL_ID, RANDOM_EFFECTS,
)


class TestSavedAICComparison(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.settings = {"reml": False, "optimizer": "powell", "maxiter": 2000,
                         "boundary_tolerance": 1e-8}
        self.cohort = pd.DataFrame({
            "person_id": [f"person_{index // 2}" for index in range(8)],
            "series_num": list(range(1, 9)),
            "cgi_change": [0.0, 2.0, 1.0, 3.0, -1.0, 4.0, 2.0, 1.0],
            "age": [20, 20, 35, 35, 50, 50, 65, 65],
            "sex": ["Female", "Male"] * 4,
        })
        self.null = self.write_analysis("null", null=True, likelihoods=[-20.0])
        self.feature = self.write_analysis("feature", likelihoods=[-17.0, -18.0])

    def tearDown(self):
        self.temp.cleanup()

    def write_analysis(self, name, *, null=False, likelihoods=None, kind="roi",
                       normalization="raw", boundary=False, wald_failed=False):
        directory = self.directory / name
        (directory / "data").mkdir(parents=True)
        formula, model_id = (NULL_FORMULA, NULL_MODEL_ID) if null else (FORMULA, MODEL_ID)
        terms = NULL_FIXED_TERMS if null else FIXED_TERMS
        source = self.directory / f"source_{name}.csv"
        source.write_text("saved source identifier\n")
        provenance = {
            "master_file": source.name, "source_master_path": str(source),
            "source_master_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "model_id": model_id, "fixed_formula": formula,
            "age_effect": "continuous_fixed", "sex_effect": "categorical_random_intercept",
        }
        tag = "raw" if normalization == "raw" else "normed_brain_magnE_mean_mean"
        features = ["null"] if null else [f"PC{index + 1}" if kind == "pca" else f"ROI_{index + 1}"
                                           for index in range(len(likelihoods))]
        inputs, statuses = [], []
        for index, (feature, likelihood) in enumerate(zip(features, likelihoods, strict=True)):
            identity = {
                "feature_type": "null" if null else kind, "field_name": "magnE_mean",
                "normalization": "not_applicable" if null else normalization,
                "normalization_tag": "not_applicable" if null else tag,
                "model_type": "shared_cohort" if null else "skin_single",
                "step": "all" if null else "step_0", "basis_id": "basis_0" if kind == "pca" and not null else "",
                "feature_id": feature,
            }
            frame = self.cohort.copy()
            if not null:
                for column, value in {**provenance, **identity}.items():
                    frame[column] = value
                frame["E"] = np.sin(np.arange(8) * (index + 1))
                frame["ef_sf"] = 1 + np.arange(8) % 3
                frame["value_column"] = "score" if kind == "pca" else "mean"
                frame["is_brain_normalized"] = normalization == "brain_mean"
            inputs.append(frame)
            parameters = len(terms) + 2
            statuses.append({
                **provenance, **identity, **self.settings,
                "formula": formula, "fit_method": "ML",
                "n_observations": 8, "n_people": 4, "n_age_values": 4, "n_sex_levels": 2,
                "fixed_terms": json.dumps(terms), "random_effects": json.dumps(RANDOM_EFFECTS),
                "n_outer_groups": 1, "n_group_random_intercepts": 0,
                "status": "failed" if wald_failed else "ok", "converged": True,
                "warnings": "Hessian warning" if wald_failed else "",
                "error": "Invalid observed-information Hessian" if wald_failed else "",
                "log_likelihood": likelihood, "aic": -2 * likelihood + 2 * parameters,
                "bic": -2 * likelihood + np.log(8) * parameters,
                "sex_variance": 0.0 if boundary else 0.1, "residual_variance": 1.0,
                "sex_variance_boundary": boundary, "any_variance_boundary": boundary,
                "random_effects_covariance_singular": False,
                "roi_name": feature if kind == "roi" and not null else "",
                "pc_name": feature if kind == "pca" and not null else "",
                "value_column": "score" if kind == "pca" else "mean",
                "is_brain_normalized": normalization == "brain_mean",
            })
        model_input = directory / "data" / ("null_model_input.csv" if null else "source_model_input_long.csv")
        fit_status = directory / ("fit_status.csv" if null else "source_fit_status.csv")
        prepared = pd.concat(inputs, ignore_index=True)
        prepared.to_csv(model_input, index=False)
        pd.DataFrame(statuses).to_csv(fit_status, index=False)
        manifest = {
            "schema_version": 2, "analysis_name": name, "prepare_only": False,
            **provenance,
            "model": {"fixed_formula": formula, "fixed_terms": terms,
                      "variance_components": RANDOM_EFFECTS, "outer_group": "constant",
                      "group_random_intercept": False, "patient_random_effect": False,
                      "observation_unit": "treatment series", "centering": "none", "standardization": "none"},
            "fit_settings": copy.deepcopy(self.settings), "planned_fits": len(features),
            "artifacts": {"model_input": str(model_input), "fit_status": str(fit_status)},
        }
        if null:
            manifest.update(analysis_type="null", cohort_sha256=cohort_sha256(self.cohort))
            manifest.update({column: statuses[0][column] for column in (
                "feature_type", "field_name", "normalization", "normalization_tag", "model_type",
                "step", "basis_id", "feature_id",
            )})
        else:
            manifest["feature_selection"] = {
                "feature_type": kind, "field_name": "magnE_mean", "normalization": normalization,
                "normalization_tag": tag, "value_column": "score" if kind == "pca" else "mean",
                "feature_ids": sorted(features), "n_input_rows": len(prepared),
                "n_features": len(features), "n_bases": 1 if kind == "pca" else 0,
            }
        path = directory / ("analysis.json" if null else "analysis_source.json")
        path.write_text(json.dumps(manifest))
        return directory

    def manifest(self, directory):
        path = directory / ("analysis.json" if directory == self.null else "analysis_source.json")
        return path, json.loads(path.read_text())

    def change_manifest(self, directory, change):
        path, manifest = self.manifest(directory)
        change(manifest)
        path.write_text(json.dumps(manifest))

    def change_csv(self, directory, role, change):
        _, manifest = self.manifest(directory)
        path = Path(manifest["artifacts"][role])
        frame = pd.read_csv(path, keep_default_na=False)
        updated = change(frame)
        (frame if updated is None else updated).to_csv(path, index=False)

    def compare(self, features=None):
        return compare_analyses(self.null, [self.feature] if features is None else features)

    def test_ranking_includes_null_and_preserves_identity_and_hashes(self):
        result = self.compare()
        table = result.table.set_index("feature_id")
        self.assertEqual(table.loc["ROI_1", "aic_rank_global"], 1)
        self.assertEqual(table.loc["ROI_2", "aic_rank_global"], 2)
        self.assertEqual(table.loc["null", "aic_rank_global"], 2)
        self.assertEqual(table.loc["ROI_1", "aic_improvement_over_null"], 2)
        self.assertEqual(table.loc["ROI_2", "delta_aic_best_feature"], 2)
        self.assertTrue(np.isnan(table.loc["null", "delta_aic_best_feature"]))
        self.assertEqual(table.loc["ROI_1", "n_parameters"], 6)
        self.assertEqual(table.loc["null", "n_parameters"], 4)
        self.assertEqual(result.metadata["n_candidates"], 2)
        self.assertEqual(result.metadata["n_eligible"], 2)
        self.assertEqual(result.metadata["n_excluded"], 0)
        self.assertEqual(result.metadata["null_aic"], 48)
        self.assertEqual(result.metadata["cohort_sha256"], cohort_sha256(self.cohort))
        self.assertEqual(len(result.metadata["input_files"]), 6)
        for record in result.metadata["input_files"]:
            self.assertEqual(record["sha256"], hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest())
        json.dumps(result.metadata, allow_nan=False)
        self.assertEqual(set(CONFIGURATION_COLUMNS), set(result.metadata["comparison_settings"]["configuration_columns"]))

    def test_wald_rejection_and_zero_variance_retain_provisional_aic(self):
        feature = self.write_analysis("wald_failed", likelihoods=[-17], boundary=True, wald_failed=True)
        result = self.compare([feature])
        row = result.table.loc[~result.table.is_null].iloc[0]
        self.assertTrue(row.aic_eligible)
        self.assertTrue(row.aic_provisional)
        self.assertIn("variance_boundary", row.aic_caveat)
        self.assertIn("wald_inference_unavailable", row.aic_caveat)
        self.assertEqual(row.status, "failed")
        self.assertEqual(row.error, "Invalid observed-information Hessian")
        self.assertEqual(row.warnings, "Hessian warning")

    def test_provisional_null_propagates_to_feature_comparison(self):
        self.change_csv(self.null, "fit_status", lambda frame: frame.__setitem__("status", "failed"))
        result = self.compare()
        rows = result.table.loc[~result.table.is_null]
        self.assertTrue(result.metadata["null_model_provisional"])
        self.assertTrue(rows.aic_caveat.str.contains("null_model_provisional").all())

    def test_null_can_be_best_and_all_candidates_can_be_excluded(self):
        worse = self.write_analysis("worse", likelihoods=[-19, -19])
        result = self.compare([worse])
        rows = result.table.set_index("feature_id")
        self.assertEqual(rows.loc["null", "aic_rank_global"], 1)
        self.assertTrue(rows.loc[["ROI_1", "ROI_2"], "aic_rank_global"].eq(2).all())
        self.assertTrue(rows.loc[["ROI_1", "ROI_2"], "aic_rank_within_configuration"].eq(2).all())
        self.assertTrue(rows.loc[["ROI_1", "ROI_2"], "delta_aic_within_configuration_including_null"].eq(2).all())
        self.change_csv(worse, "fit_status", lambda frame: frame.__setitem__("converged", False))
        result = self.compare([worse])
        self.assertEqual(result.metadata["n_eligible"], 0)
        self.assertIsNone(result.metadata["best_candidate"])
        self.assertTrue(result.table.loc[~result.table.is_null, "aic_rank_global"].isna().all())

    def test_rowwise_invalid_fits_remain_present_with_explicit_reasons(self):
        changes = {
            "not_converged": ("converged", False),
            "nonfinite_log_likelihood": ("log_likelihood", np.nan),
            "nonfinite_aic": ("aic", np.inf),
            "invalid_sex_variance": ("sex_variance", -0.1),
            "invalid_residual_variance": ("residual_variance", 0),
            "aic_parameter_count_mismatch": ("aic", 999),
            "bic_parameter_count_mismatch": ("bic", 999),
            "parameter_count_mismatch": ("n_parameters", 7),
        }
        for reason, (column, value) in changes.items():
            with self.subTest(reason=reason):
                path = self.write_analysis(reason, likelihoods=[-17])
                self.change_csv(path, "fit_status", lambda frame: frame.__setitem__(column, value))
                result = self.compare([path])
                row = result.table.loc[~result.table.is_null].iloc[0]
                self.assertFalse(row.aic_eligible)
                self.assertIn(reason, row.aic_exclusion_reason)
                self.assertTrue(np.isnan(row.aic_rank_global))
                self.assertTrue(np.isnan(row.aic_improvement_over_null))
                if column == "n_parameters":
                    self.assertEqual(row.reported_n_parameters, 7)
                    self.assertEqual(row.n_parameters, 6)

    def test_full_likelihood_below_nested_null_is_excluded_not_ranked(self):
        path = self.write_analysis("non_nested_likelihood", likelihoods=[-20.1])
        result = self.compare([path])
        row = result.table.loc[~result.table.is_null].iloc[0]
        self.assertFalse(row.aic_eligible)
        self.assertIn("log_likelihood_below_nested_null", row.aic_exclusion_reason)
        near = self.write_analysis("roundoff", likelihoods=[-20.0000001])
        self.assertEqual(self.compare([near]).metadata["n_eligible"], 1)

    def test_invalid_null_stops_entire_comparison(self):
        self.change_csv(self.null, "fit_status", lambda frame: frame.__setitem__("converged", False))
        with self.assertRaisesRegex(ValueError, "Null model is ineligible.*not_converged"):
            self.compare()

    def test_distinct_pca_bases_and_normalizations_remain_distinct(self):
        raw = self.write_analysis("raw_pca", likelihoods=[-17], kind="pca")
        norm = self.write_analysis("norm_pca", likelihoods=[-16], kind="pca", normalization="brain_mean")
        self.change_csv(norm, "model_input", lambda frame: frame.__setitem__("basis_id", "norm_basis"))
        self.change_csv(norm, "fit_status", lambda frame: frame.__setitem__("basis_id", "norm_basis"))
        result = self.compare([raw, norm])
        rows = result.table.loc[~result.table.is_null]
        self.assertEqual(set(rows.feature_id), {"PC1"})
        self.assertEqual(set(rows.basis_id), {"basis_0", "norm_basis"})
        self.assertEqual(set(rows.normalization), {"raw", "brain_mean"})
        self.assertTrue(rows.aic_rank_within_configuration.eq(1).all())

    def test_cohort_values_and_series_keys_must_match_for_every_feature(self):
        for column, value in (("age", 99), ("sex", "Other"), ("cgi_change", 99), ("person_id", "changed")):
            with self.subTest(column=column):
                path = self.write_analysis(f"changed_{column}", likelihoods=[-17])
                def change(frame):
                    frame.loc[0, column] = value
                self.change_csv(path, "model_input", change)
                with self.assertRaisesRegex(ValueError, "Cohort mismatch"):
                    self.compare([path])
        self.change_csv(self.feature, "model_input", lambda frame: frame.__setitem__("cgi_change", frame.cgi_change + 1e-13))
        self.assertEqual(self.compare().metadata["n_eligible"], 2)

    def test_old_age_spec_and_reml_cannot_be_compared(self):
        cases = (
            lambda manifest: manifest.__setitem__("model_id", "age_categorical_v1"),
            lambda manifest: manifest["model"].__setitem__("variance_components", {"age": "0 + C(age)", **RANDOM_EFFECTS}),
            lambda manifest: manifest["fit_settings"].__setitem__("reml", True),
        )
        for index, change in enumerate(cases):
            with self.subTest(case=index):
                path = self.write_analysis(f"bad_spec_{index}", likelihoods=[-17])
                self.change_manifest(path, change)
                with self.assertRaises(ValueError):
                    self.compare([path])
        self.change_csv(self.feature, "fit_status", lambda frame: frame.__setitem__("reml", True))
        with self.assertRaisesRegex(ValueError, "ML.*REML"):
            self.compare()

    def test_numerical_fit_settings_must_match_null(self):
        for column, value in (("optimizer", "lbfgs"), ("maxiter", 10), ("boundary_tolerance", 1e-6)):
            with self.subTest(column=column):
                path = self.write_analysis(f"settings_{column}", likelihoods=[-17])
                self.change_manifest(path, lambda manifest: manifest["fit_settings"].__setitem__(column, value))
                self.change_csv(path, "fit_status", lambda frame: frame.__setitem__(column, value))
                with self.assertRaisesRegex(ValueError, "fit settings differ from the null"):
                    self.compare([path])

    def test_duplicate_candidate_rows_inputs_and_analysis_directories_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate.*directories"):
            self.compare([self.feature, self.feature])
        duplicate = self.write_analysis("duplicate_candidates", likelihoods=[-17, -18])
        with self.assertRaisesRegex(ValueError, "Duplicate feature/configuration"):
            self.compare([self.feature, duplicate])
        self.change_csv(self.feature, "fit_status", lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "duplicate feature/configuration fit-status"):
            self.compare()
        self.change_csv(duplicate, "model_input", lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "duplicate treatment series"):
            self.compare([duplicate])

    def test_planned_coverage_source_labels_and_nobs_are_checked(self):
        for case in ("coverage", "source", "normalization", "nobs"):
            with self.subTest(case=case):
                path = self.write_analysis(f"bad_{case}", likelihoods=[-17, -18])
                if case == "coverage":
                    self.change_csv(path, "fit_status", lambda frame: frame.iloc[:1])
                elif case == "source":
                    self.change_csv(path, "fit_status", lambda frame: frame.__setitem__("source_master_sha256", "b" * 64))
                elif case == "normalization":
                    self.change_csv(path, "fit_status", lambda frame: frame.__setitem__("normalization", "brain_mean"))
                else:
                    self.change_csv(path, "fit_status", lambda frame: frame.__setitem__("n_observations", 7))
                with self.assertRaises(ValueError):
                    self.compare([path])

    def test_artifact_paths_cannot_escape_analysis_directory(self):
        self.change_manifest(self.feature, lambda manifest: manifest["artifacts"].__setitem__("model_input", str(self.null / "data/null_model_input.csv")))
        with self.assertRaisesRegex(ValueError, "Invalid saved model_input artifact"):
            self.compare()

    def test_concurrently_modified_input_is_rejected_before_returning_ranks(self):
        original = aic_module._csv
        def mutate_after_read(path):
            frame = original(path)
            if path.name == "source_model_input_long.csv":
                with path.open("a") as handle:
                    handle.write("\n")
            return frame
        with patch("series_models.aic._csv", side_effect=mutate_after_read):
            with self.assertRaisesRegex(ValueError, "input changed while being read"):
                self.compare()


if __name__ == "__main__":
    unittest.main()
