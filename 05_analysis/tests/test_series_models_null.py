"""Shared null cohort, exported provenance, and command-line behavior."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from analysis_paths import SOURCE_DIR
from run_series_mixed_effects import main, parse_args
from series_models.cohort import (
    COHORT_COLUMNS, canonical_cohort, cohort_sha256, require_same_cohort,
)
from series_models.fitting import COEFFICIENT_COLUMNS, NULL_FORMULA, NULL_MODEL_ID
from series_models.null_workflow import run_null_analysis


def source_rows(kind="roi"):
    """Two series per person, repeated across two features/configurations."""
    records = []
    for step in ("step_0", "step_9"):
        for index in range(8):
            person, series = f"person_{index // 2}", index % 2 + 1
            for feature_index in (1, 2):
                record = {
                    "person_id": person, "series_num": series,
                    "age": 25.0 + index / 2, "age_scan": 20.0 + index / 2,
                    "sex": "F" if index % 2 else "M", "cgi_change": index / 4,
                    "ef_sf": 1.0 + index % 3, "field_name": "magnE_mean",
                    "model_type": "synthetic_model", "step": step,
                    "full_file_path": "shared_simulation.nii.gz",
                }
                if kind == "roi":
                    record.update(roi_org=f"ROI_{feature_index}", mean=index + feature_index / 10)
                else:
                    record.update(component=f"PC{feature_index}", score=index - feature_index / 10,
                                  component_index=feature_index, basis_id=f"basis_{step}",
                                  sample_id=f"sample_{step}_{index}",
                                  selected_for_correlation=False)
                records.append(record)
    return pd.DataFrame(records).sample(frac=1, random_state=71).reset_index(drop=True)


def arguments(path, kind="roi", analysis_name="null_test", **changes):
    values = dict(master_path=path, feature_type=kind, field_name="magnE_mean",
                  analysis_name=analysis_name, null_model=True, features=None,
                  pca_scope="selected", prepare_only=False, reml=False,
                  optimizer="powell", maxiter=2000)
    values.update(changes)
    return argparse.Namespace(**values)


def successful_null(frame, settings):
    coefficients = pd.DataFrame({
        "term": ["Intercept", "age"], "estimate": [0.5, 0.03],
        "std_error": [0.1, 0.01], "wald_z": [5.0, 3.0],
        "p_value": [0.001, 0.003], "ci_low": [0.3, 0.01],
        "ci_high": [0.7, 0.05], "inference_method": ["asymptotic_normal_wald"] * 2,
    })
    return coefficients, {
        "status": "ok", "error": "", "warnings": "", "converged": True,
        "n_observations": len(frame), "n_people": frame.person_id.nunique(),
        "n_fixed_effects": 2, "n_parameters": 4,
        "log_likelihood": -10.0, "aic": 28.0, "bic": 28.3,
        "sex_variance": 0.1, "residual_variance": 0.5,
    }


class TestNullCohort(unittest.TestCase):
    def test_only_validated_person_series_repeats_collapse(self):
        source = source_rows()
        expected = source[COHORT_COLUMNS].drop_duplicates().sort_values(
            ["person_id", "series_num"]).reset_index(drop=True)
        cohort = canonical_cohort(source)
        pd.testing.assert_frame_equal(cohort, expected)
        self.assertEqual(len(cohort), 8)
        self.assertEqual(cohort.person_id.nunique(), 4)
        self.assertTrue(cohort.groupby("person_id").size().eq(2).all())
        # Every row references the same imaging file; that is not the cohort key.
        self.assertEqual(source.full_file_path.nunique(), 1)
        self.assertEqual(list(cohort), COHORT_COLUMNS)
        self.assertEqual(cohort_sha256(source), cohort_sha256(source.sample(frac=1, random_state=4)))

    def test_clinical_conflicts_across_features_or_configurations_are_errors(self):
        for column, value in (("age", 99), ("cgi_change", 99), ("sex", "other")):
            with self.subTest(column=column):
                source = source_rows()
                source.loc[source.index[0], column] = value
                with self.assertRaisesRegex(ValueError, "Conflicting repeated series covariates"):
                    canonical_cohort(source)

    def test_invalid_keys_and_covariates_cannot_silently_change_cohort(self):
        for column, value in (("person_id", " "), ("sex", " F"),
                              ("series_num", 0), ("series_num", 1.5),
                              ("age", -1), ("age", np.nan), ("cgi_change", np.inf)):
            with self.subTest(column=column, value=value):
                source = source_rows().astype({column: object})
                source.loc[source.index[0], column] = value
                with self.assertRaises(ValueError):
                    canonical_cohort(source)
        source = source_rows().astype({"series_num": object})
        source.loc[source.index[0], "series_num"] = True
        with self.assertRaisesRegex(ValueError, "Boolean"):
            canonical_cohort(source)
        with self.assertRaises(ValueError):
            canonical_cohort(source_rows().drop(columns="age"))
        with self.assertRaises(ValueError):
            canonical_cohort(source_rows().iloc[:0])

    def test_cohort_comparison_accepts_only_same_series_and_saved_covariates(self):
        cohort = canonical_cohort(source_rows())
        require_same_cohort(cohort, cohort.sample(frac=1, random_state=4))
        roundoff = cohort.copy()
        roundoff["age"] += 1e-13
        require_same_cohort(cohort, roundoff)
        for column, value in (("person_id", "new_person"), ("series_num", 3),
                              ("age", 99), ("cgi_change", 99), ("sex", "other")):
            with self.subTest(column=column):
                changed = cohort.copy()
                changed.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, "Cohort mismatch"):
                    require_same_cohort(cohort, changed)


class TestNullWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.root = self.directory / "results"
        self.environment = patch.dict(os.environ, {"RESULTS_DIR": str(self.root)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def write_master(self, kind="roi", frame=None, name="master.csv"):
        path = self.directory / name
        (source_rows(kind) if frame is None else frame).to_csv(path, index=False)
        return path

    def test_single_null_uses_saved_age_and_exactly_five_cohort_columns(self):
        path = self.write_master()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        captured = []

        def capture(frame, settings):
            captured.append(frame.copy())
            self.assertFalse(settings.reml)
            return successful_null(frame, settings)

        with patch("series_models.null_workflow.fit_null", side_effect=capture) as fitter:
            result = run_null_analysis(arguments(path))
        fitter.assert_called_once()
        cohort = pd.read_csv(result["paths"]["model_input"])
        pd.testing.assert_frame_equal(cohort, canonical_cohort(source_rows()))
        pd.testing.assert_frame_equal(captured[0], cohort)
        self.assertEqual(list(cohort), COHORT_COLUMNS)
        self.assertNotIn("E", cohort)
        self.assertNotIn("ef_sf", cohort)
        self.assertNotIn("age_scan", cohort)
        self.assertTrue(cohort.age.ge(25).all())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        table = pd.read_csv(result["paths"]["null_results"])
        self.assertEqual(len(table), 1)
        self.assertEqual(table.n_observations.iloc[0], 8)
        self.assertEqual(table.n_people.iloc[0], 4)
        self.assertEqual(table.age_estimate.iloc[0], 0.03)
        self.assertEqual(table.intercept_estimate.iloc[0], 0.5)
        self.assertFalse(any(column.startswith("E_") for column in table))

    def test_optional_scan_age_never_replaces_required_saved_age(self):
        for include_scan in (True, False):
            with self.subTest(include_scan=include_scan):
                frame = source_rows()
                if not include_scan:
                    frame = frame.drop(columns="age_scan")
                path = self.write_master(frame=frame, name=f"age_{include_scan}.csv")
                result = run_null_analysis(arguments(path, analysis_name=f"age_{include_scan}", prepare_only=True))
                cohort = pd.read_csv(result["paths"]["model_input"])
                np.testing.assert_allclose(cohort.age, canonical_cohort(source_rows()).age)
        path = self.write_master(frame=source_rows().drop(columns="age"), name="scan_only.csv")
        with self.assertRaisesRegex(ValueError, "missing required columns.*age"):
            run_null_analysis(arguments(path, analysis_name="scan_only"))
        self.assertFalse((self.root / "lmm/scan_only/null").exists())

    def test_pca_selection_does_not_filter_null_cohort(self):
        # No PC is marked selected, but all eight saved clinical series belong
        # to the shared null; even a generic PC master without flags is valid.
        for drop_selection in (False, True):
            with self.subTest(drop_selection=drop_selection):
                frame = source_rows("pca")
                if drop_selection:
                    frame = frame.drop(columns="selected_for_correlation")
                path = self.write_master(frame=frame, name=f"pc_{drop_selection}.csv")
                with patch("series_models.null_workflow.fit_null", side_effect=successful_null) as fitter:
                    result = run_null_analysis(arguments(path, kind="pca", analysis_name=f"pc_{drop_selection}"))
                self.assertEqual(len(fitter.call_args.args[0]), 8)
                self.assertEqual(len(pd.read_csv(result["paths"]["model_input"])), 8)
                manifest = json.loads(result["paths"]["analysis"].read_text())
                self.assertEqual(manifest["source_adapter"]["feature_selection"], {"mode": "all"})

    def test_manifest_records_null_definition_source_hash_and_protected_artifacts(self):
        path = self.write_master()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch("series_models.null_workflow.fit_null", side_effect=successful_null):
            result = run_null_analysis(arguments(path))
        manifest = json.loads(result["paths"]["analysis"].read_text())
        self.assertEqual(manifest["analysis_type"], "null")
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["planned_fits"], 1)
        self.assertEqual(manifest["model_id"], NULL_MODEL_ID)
        self.assertEqual(manifest["model"]["fixed_formula"], NULL_FORMULA)
        self.assertEqual(manifest["model"]["fixed_terms"], ["Intercept", "age"])
        self.assertEqual(manifest["model"]["variance_components"], {"sex": "0 + C(sex)"})
        self.assertEqual(manifest["model"]["outer_group"], "constant")
        self.assertFalse(manifest["model"]["group_random_intercept"])
        self.assertFalse(manifest["model"]["patient_random_effect"])
        self.assertEqual(manifest["model"]["observation_unit"], "treatment series")
        self.assertEqual(manifest["source_master_path"], str(path))
        self.assertEqual(manifest["source_master_sha256"], digest)
        self.assertEqual(manifest["cohort_sha256"], cohort_sha256(source_rows()))
        self.assertEqual(manifest["normalization"], "not_applicable")
        self.assertFalse(manifest["fit_settings"]["reml"])
        self.assertIn("series_models/null_workflow.py", manifest["provenance"]["source_sha256"])
        for key, artifact in result["paths"].items():
            self.assertTrue(artifact.is_relative_to(self.root / "lmm/null_test/null"))
            self.assertTrue(artifact.is_file())
            self.assertEqual(manifest["artifacts"][key], str(artifact))
        original = {key: value.read_bytes() for key, value in result["paths"].items()}
        with self.assertRaisesRegex(FileExistsError, "new --analysis-name"):
            run_null_analysis(arguments(path))
        for key, artifact in result["paths"].items():
            self.assertEqual(artifact.read_bytes(), original[key])

    def test_conflicting_config_covariates_and_duplicate_features_fail_before_writes(self):
        for column, value in (("age", 99), ("cgi_change", 99), ("sex", "other")):
            with self.subTest(column=column):
                frame = source_rows()
                changed = frame.person_id.eq("person_0") & frame.series_num.eq(1) & frame.step.eq("step_9")
                frame.loc[changed, column] = value
                path = self.write_master(frame=frame, name=f"conflict_{column}.csv")
                with patch("series_models.null_workflow.fit_null") as fitter:
                    with self.assertRaisesRegex(ValueError, "Conflicting repeated series covariates"):
                        run_null_analysis(arguments(path))
                fitter.assert_not_called()
                self.assertFalse(self.root.exists())
        frame = source_rows()
        path = self.write_master(frame=pd.concat([frame, frame.iloc[:1]], ignore_index=True), name="duplicate.csv")
        with self.assertRaisesRegex(ValueError, "Duplicate.*observations"):
            run_null_analysis(arguments(path))
        self.assertFalse(self.root.exists())

    def test_failed_optimizer_or_wald_inference_still_export_likelihood_and_succeed(self):
        path = self.write_master()
        for converged, error in ((True, "Invalid observed-information Hessian: nonpositive diagonal"),
                                 (False, "Optimizer did not converge")):
            with self.subTest(converged=converged):
                coefficients = pd.DataFrame(columns=COEFFICIENT_COLUMNS)
                diagnostics = {
                    "status": "failed", "error": error, "warnings": "boundary warning",
                    "converged": converged, "any_variance_boundary": True,
                    "log_likelihood": -10.0, "aic": 28.0, "bic": 28.3,
                    "n_parameters": 4, "sex_variance": 1e-10, "residual_variance": 0.5,
                }
                captured = io.StringIO()
                with redirect_stdout(captured), patch("series_models.null_workflow.fit_null", return_value=(coefficients, diagnostics)):
                    code = main(["--master-path", str(path), "--feature-type", "roi", "--null-model",
                                 "--analysis-name", f"failed_{converged}"])
                self.assertEqual(code, 0)
                root = self.root / f"lmm/failed_{converged}/null"
                for name in ("fit_status.csv", "null_results.csv"):
                    table = pd.read_csv(root / name)
                    self.assertEqual(len(table), 1)
                    self.assertEqual(table.status.iloc[0], "failed")
                    self.assertEqual(table.error.iloc[0], error)
                    self.assertEqual(bool(table.converged.iloc[0]), converged)
                    self.assertEqual(table.aic.iloc[0], 28.0)
                    self.assertEqual(table.log_likelihood.iloc[0], -10.0)
                    self.assertEqual(table.warnings.iloc[0], "boundary warning")
                wide = pd.read_csv(root / "null_results.csv")
                self.assertTrue(wide[["intercept_estimate", "age_estimate", "age_p_value"]].isna().all().all())
                self.assertEqual(len(pd.read_csv(root / "coefficients.csv")), 0)
                self.assertIn("Run completed successfully (exit code 0)", captured.getvalue())

    def test_prepare_only_validates_design_and_never_calls_optimizer(self):
        path = self.write_master()
        with patch("series_models.null_workflow.fit_null") as fitter:
            result = run_null_analysis(arguments(path, prepare_only=True))
        fitter.assert_not_called()
        self.assertEqual(result["exit_code"], 0)
        table = pd.read_csv(result["paths"]["null_results"])
        self.assertEqual(table.status.iloc[0], "prepared")
        self.assertEqual(table.design_rank.iloc[0], 2)
        self.assertEqual(table.n_observations.iloc[0], 8)
        self.assertTrue(table.age_estimate.isna().all())
        self.assertEqual(len(pd.read_csv(result["paths"]["coefficients"])), 0)

    def test_invalid_output_configuration_is_rejected_before_writes(self):
        path = self.write_master()
        for root in ("", "relative/results", str(SOURCE_DIR / "series_models")):
            with self.subTest(root=root), patch.dict(os.environ, {"RESULTS_DIR": root}):
                with self.assertRaises(ValueError):
                    run_null_analysis(arguments(path, prepare_only=True))
        with self.assertRaises(ValueError):
            run_null_analysis(arguments(path, analysis_name="../escape", prepare_only=True))
        self.assertFalse(self.root.exists())

    def test_cli_null_flag_dispatches_to_null_fitter_and_global_input_errors_return_two(self):
        path = self.write_master()
        argv = ["--master-path", str(path), "--feature-type", "roi", "--null-model", "--analysis-name", "cli_null"]
        self.assertTrue(parse_args(argv).null_model)
        self.assertFalse(parse_args([value for value in argv if value != "--null-model"]).null_model)
        with patch("series_models.null_workflow.fit_null", side_effect=successful_null) as null_fitter, \
                patch("series_models.workflow.fit_feature") as feature_fitter:
            self.assertEqual(main(argv), 0)
        null_fitter.assert_called_once()
        feature_fitter.assert_not_called()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parse_args([*argv, "--features", "ROI_1"])
        self.assertEqual(error.exception.code, 2)
        invalid = self.write_master(frame=source_rows().drop(columns="age"), name="invalid.csv")
        for source in (invalid, self.directory / "missing.csv"):
            with self.subTest(source=source), redirect_stderr(io.StringIO()), \
                    patch("series_models.null_workflow.fit_null") as fitter:
                code = main(["--master-path", str(source), "--feature-type", "roi", "--null-model",
                             "--analysis-name", "invalid_input"])
            self.assertEqual(code, 2)
            fitter.assert_not_called()
            self.assertFalse((self.root / "lmm/invalid_input/null").exists())


if __name__ == "__main__":
    unittest.main()
