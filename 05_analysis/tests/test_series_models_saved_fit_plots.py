"""Saved-run plot inputs: full coverage, fit-local failures, and stable bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import series_models.saved_fit_plots as loader
from series_models.cohort import cohort_sha256
from series_models.fitting import (
    FIXED_TERMS, FORMULA, MODEL_ID, NULL_FIXED_TERMS, NULL_FORMULA,
    NULL_MODEL_ID, RANDOM_EFFECTS,
)


class TestSavedFitPlots(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.cohort = pd.DataFrame({
            "person_id": [f"person_{index // 2}" for index in range(8)],
            "series_num": [1, 2] * 4, "cgi_change": [0, 1, -1, 3, 2, 1, 4, 0],
            "age": [20, 21, 30, 31, 40, 41, 50, 51], "sex": ["F", "M"] * 4,
        })

    def tearDown(self):
        self.temporary.cleanup()

    def make_run(self, *, null=False, reml=False, kind="roi", name="run"):
        directory = self.base / name
        (directory / "data").mkdir(parents=True)
        formula, model_id = (NULL_FORMULA, NULL_MODEL_ID) if null else (FORMULA, MODEL_ID)
        terms = NULL_FIXED_TERMS if null else FIXED_TERMS
        source = {
            "model_id": model_id, "fixed_formula": formula,
            "age_effect": "continuous_fixed", "sex_effect": "categorical_random_intercept",
            "master_file": "source.csv", "source_master_path": "/not/read/source.csv",
            "source_master_sha256": "a" * 64,
        }
        settings = {"reml": reml, "optimizer": "powell", "maxiter": 2000, "boundary_tolerance": 1e-8}
        frames, statuses = [], []
        feature_names = ["null"] if null else (["PC1", "PC2"] if kind == "pca" else ["ROI_1", "ROI_2"])
        for feature_index, feature in enumerate(feature_names):
            identity = {
                "feature_type": "null" if null else kind, "feature_id": feature,
                "field_name": "magnE_mean", "normalization": "not_applicable" if null else "raw",
                "normalization_tag": "not_applicable" if null else "raw",
                "model_type": "shared_cohort" if null else "skin_single", "step": "all" if null else "step_0",
                "basis_id": "test_basis" if kind == "pca" and not null else "",
            }
            frame = self.cohort.copy()
            if not null:
                for column, value in {**identity, **source}.items():
                    frame[column] = value
                frame["E"] = np.sin(np.arange(8) + feature_index)
                frame["ef_sf"] = 1 + np.arange(8) % 3
                frame["value_column"] = "score" if kind == "pca" else "mean"
                frame["is_brain_normalized"] = False
            frames.append(frame)
            statuses.append({
                **identity, **source, **settings, "formula": formula, "fit_method": "REML" if reml else "ML",
                "status": "ok", "converged": True, "error": "", "warnings": "",
                "fixed_terms": json.dumps(terms), "random_effects": json.dumps(RANDOM_EFFECTS),
                "n_observations": 8, "n_people": 4, "n_age_values": 8, "n_sex_levels": 2,
                "n_outer_groups": 1, "n_group_random_intercepts": 0,
                "sex_variance": 0.1, "residual_variance": 1.0, "aic": "" if reml else 30.,
                "roi_name": feature if kind == "roi" and not null else "",
                "pc_name": feature if kind == "pca" and not null else "",
                "value_column": "score" if kind == "pca" else "mean", "is_brain_normalized": False,
            })
        prepared = pd.concat(frames, ignore_index=True)
        input_path = directory / "data" / ("null_model_input.csv" if null else "source__raw__aaaaaaaaaaaa_model_input_long.csv")
        status_path = directory / ("fit_status.csv" if null else "source__raw__aaaaaaaaaaaa_fit_status.csv")
        prepared.to_csv(input_path, index=False)
        pd.DataFrame(statuses).to_csv(status_path, index=False)
        manifest = {
            "schema_version": 2, "analysis_name": name, "prepare_only": False, **source,
            "model": {"fixed_formula": formula, "fixed_terms": terms, "variance_components": RANDOM_EFFECTS,
                      "outer_group": "constant", "group_random_intercept": False, "patient_random_effect": False,
                      "observation_unit": "treatment series", "centering": "none", "standardization": "none"},
            "fit_settings": settings, "planned_fits": len(statuses),
            "artifacts": {"model_input": str(input_path), "fit_status": str(status_path)},
        }
        if null:
            manifest.update(analysis_type="null", cohort_sha256=cohort_sha256(self.cohort), **identity)
        else:
            manifest["feature_selection"] = {
                "feature_type": kind, "field_name": "magnE_mean", "normalization": "raw", "normalization_tag": "raw",
                "feature_ids": sorted(feature_names), "n_input_rows": len(prepared), "n_features": len(feature_names),
                "n_bases": 1 if kind == "pca" else 0, "value_column": "score" if kind == "pca" else "mean",
            }
        manifest_path = directory / ("analysis.json" if null else "analysis_source.json")
        manifest_path.write_text(json.dumps(manifest))
        return directory, manifest_path, input_path, status_path

    def mutate_manifest(self, path, function):
        manifest = json.loads(path.read_text())
        function(manifest)
        path.write_text(json.dumps(manifest))

    def mutate_csv(self, path, function):
        frame = pd.read_csv(path, keep_default_na=False)
        result = function(frame)
        (frame if result is None else result).to_csv(path, index=False)

    def test_feature_identity_reconstruction_and_byte_provenance(self):
        directory, *_ = self.make_run(kind="pca")
        captured = []

        def reconstruct(frame, status, **settings):
            captured.append((frame, status, settings))
            return SimpleNamespace(feature_id=status["feature_id"])

        with patch.object(loader, "_reconstruct_prediction", side_effect=reconstruct):
            records, source_id, metadata = loader.load_saved_fit_records(directory, grid_points=31)
        self.assertEqual(len(records), 2)
        self.assertEqual(source_id, "source__raw__aaaaaaaaaaaa")
        self.assertEqual([record["identity"]["feature_id"] for record in records], ["PC1", "PC2"])
        self.assertTrue(all(record["identity"]["basis_id"] == "test_basis" for record in records))
        self.assertTrue(all(record["identity"]["aic"] == 30 for record in records))
        self.assertTrue(all(record["error"] == "" and record["prediction"] is not None for record in records))
        self.assertTrue(all(settings == {"feature": True, "grid_points": 31} for _, _, settings in captured))
        self.assertEqual(metadata["n_observations"], 8)
        self.assertEqual(metadata["n_people"], 4)
        self.assertFalse(metadata["prepare_only"])
        self.assertEqual(metadata["cohort_sha256"], cohort_sha256(self.cohort))
        for item in metadata["input_files"]:
            self.assertEqual(item["sha256"], hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest())
        json.dumps(metadata, allow_nan=False)

    def test_ml_and_reml_nulls_use_one_cohort_without_feature_inputs(self):
        for reml in (False, True):
            with self.subTest(reml=reml):
                directory, *_ = self.make_run(null=True, reml=reml, name=f"null_{reml}")
                with patch.object(loader, "_reconstruct_prediction", return_value="prediction") as reconstruct:
                    records, source_id, metadata = loader.load_saved_fit_records(directory)
                self.assertEqual(len(records), 1)
                self.assertEqual(source_id, "null__aaaaaaaaaaaa")
                self.assertEqual(records[0]["prediction"], "prediction")
                self.assertTrue(metadata["is_null"])
                self.assertEqual(metadata["fit_settings"]["reml"], reml)
                frame = reconstruct.call_args.args[0]
                self.assertNotIn("E", frame)
                self.assertNotIn("ef_sf", frame)
                self.assertFalse(reconstruct.call_args.kwargs["feature"])

    def test_reml_feature_run_is_supported_without_aic(self):
        directory, *_ = self.make_run(reml=True)
        with patch.object(loader, "_reconstruct_prediction", return_value="prediction"):
            records, _, metadata = loader.load_saved_fit_records(directory)
        self.assertEqual(len(records), 2)
        self.assertTrue(metadata["fit_settings"]["reml"])
        self.assertEqual(records[0]["identity"]["aic"], "")

    def test_nonconverged_fit_remains_present_and_is_not_reconstructed(self):
        directory, _, _, status_path = self.make_run()

        def fail_first(frame):
            frame.loc[0, ["converged", "status", "error"]] = [False, "failed", "optimizer stopped"]
        self.mutate_csv(status_path, fail_first)
        with patch.object(loader, "_reconstruct_prediction", return_value="second") as reconstruct:
            records, _, metadata = loader.load_saved_fit_records(directory)
        reconstruct.assert_called_once()
        self.assertEqual(metadata["planned_fits"], 2)
        self.assertIsNone(records[0]["prediction"])
        self.assertIn("optimizer stopped", records[0]["error"])
        self.assertEqual(records[1]["prediction"], "second")

    def test_wald_failure_still_reconstructs_and_local_errors_do_not_drop_rows(self):
        directory, _, _, status_path = self.make_run()
        self.mutate_csv(status_path, lambda frame: frame.__setitem__("status", "failed"))
        with patch.object(loader, "_reconstruct_prediction", side_effect=["first", ValueError("bad variance")]):
            records, _, _ = loader.load_saved_fit_records(directory)
        self.assertEqual(records[0]["prediction"], "first")
        self.assertEqual(records[0]["identity"]["status"], "failed")
        self.assertIsNone(records[1]["prediction"])
        self.assertEqual(records[1]["error"], "ValueError: bad variance")

    def test_invalid_model_spec_or_prepare_only_fails_before_reconstruction(self):
        mutations = {
            "prepare": lambda value: value.__setitem__("prepare_only", True),
            "model_id": lambda value: value.__setitem__("model_id", "old_categorical_age"),
            "random": lambda value: value["model"].__setitem__("variance_components", {"age": "0+C(age)"}),
            "patient": lambda value: value["model"].__setitem__("patient_random_effect", True),
            "settings": lambda value: value["fit_settings"].__setitem__("maxiter", 0),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                directory, manifest_path, *_ = self.make_run(name=name)
                self.mutate_manifest(manifest_path, mutation)
                with patch.object(loader, "_reconstruct_prediction") as reconstruct:
                    with self.assertRaises(ValueError):
                        loader.load_saved_fit_records(directory)
                reconstruct.assert_not_called()

    def test_duplicate_or_missing_fit_coverage_is_a_global_error(self):
        for kind in ("duplicate_status", "missing_status", "duplicate_series", "wrong_count"):
            with self.subTest(kind=kind):
                directory, _, input_path, status_path = self.make_run(name=kind)
                if kind == "duplicate_status":
                    self.mutate_csv(status_path, lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
                elif kind == "missing_status":
                    self.mutate_csv(status_path, lambda frame: frame.iloc[[0]])
                elif kind == "duplicate_series":
                    self.mutate_csv(input_path, lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
                else:
                    self.mutate_csv(status_path, lambda frame: frame.__setitem__("n_observations", 7))
                with patch.object(loader, "_reconstruct_prediction") as reconstruct:
                    with self.assertRaises(ValueError):
                        loader.load_saved_fit_records(directory)
                reconstruct.assert_not_called()

    def test_covariate_or_saved_spec_conflicts_are_global_errors(self):
        for kind in ("cohort", "reml", "formula", "settings", "random_effects"):
            with self.subTest(kind=kind):
                directory, _, input_path, status_path = self.make_run(name=kind)
                if kind == "cohort":
                    self.mutate_csv(input_path, lambda frame: frame.__setitem__("age", frame.age + (frame.feature_id == "ROI_2")))
                elif kind == "reml":
                    self.mutate_csv(status_path, lambda frame: frame.__setitem__("reml", True))
                elif kind == "formula":
                    self.mutate_csv(status_path, lambda frame: frame.__setitem__("formula", "cgi_change ~ age"))
                elif kind == "settings":
                    self.mutate_csv(status_path, lambda frame: frame.__setitem__("optimizer", "lbfgs"))
                else:
                    self.mutate_csv(status_path, lambda frame: frame.__setitem__("random_effects", json.dumps({"age": "0+age"})))
                with patch.object(loader, "_reconstruct_prediction") as reconstruct:
                    with self.assertRaises(ValueError):
                        loader.load_saved_fit_records(directory)
                reconstruct.assert_not_called()

    def test_null_duplicate_series_and_hash_changes_are_global_errors(self):
        directory, _, input_path, _ = self.make_run(null=True)
        self.mutate_csv(input_path, lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "Duplicate treatment series"):
            loader.load_saved_fit_records(directory)
        directory, _, input_path, _ = self.make_run(null=True, name="hash_changed")
        self.mutate_csv(input_path, lambda frame: frame.__setitem__("age", frame.age + 1))
        with self.assertRaisesRegex(ValueError, "cohort SHA256"):
            loader.load_saved_fit_records(directory)

    def test_declared_artifact_cannot_escape_the_analysis_directory(self):
        directory, manifest_path, _, status_path = self.make_run()
        outside = self.base / "outside_status.csv"
        outside.write_bytes(status_path.read_bytes())
        self.mutate_manifest(manifest_path, lambda value: value["artifacts"].__setitem__("fit_status", str(outside)))
        with self.assertRaisesRegex(ValueError, "Invalid saved fit_status artifact"):
            loader.load_saved_fit_records(directory)

    def test_changed_bytes_during_reading_or_reconstruction_are_rejected(self):
        directory, _, input_path, status_path = self.make_run()
        original_csv = loader._csv

        def unstable_read(path):
            frame = original_csv(path)
            if path == input_path:
                path.write_text(path.read_text() + "\n")
            return frame
        with patch.object(loader, "_csv", side_effect=unstable_read):
            with self.assertRaisesRegex(ValueError, "changed while being read"):
                loader.load_saved_fit_records(directory)

        def unstable_prediction(*args, **kwargs):
            status_path.write_text(status_path.read_text() + "\n")
            return "prediction"
        with patch.object(loader, "_reconstruct_prediction", side_effect=unstable_prediction):
            with self.assertRaisesRegex(ValueError, "changed during plot preparation"):
                loader.load_saved_fit_records(directory)


if __name__ == "__main__":
    unittest.main()
