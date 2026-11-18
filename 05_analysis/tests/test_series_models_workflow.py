"""Source-specific result tables, per-feature accounting, and output policy."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
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
from series_models.run_layout import ensure_run_manifest, run_configuration
from series_models.workflow import run_analysis


def master_rows(kind="roi", normalization="raw"):
    records = []
    for step in ("step_0", "step_9"):
        for series in range(1, 9):
            for index, feature in enumerate(("LH_Vis_A", "RH_Vis_A") if kind == "roi" else ("PC1", "PC2"), 1):
                row = {
                    "person_id": f"person_{(series - 1) // 2}", "series_num": series,
                    "model_type": "skin_single", "step": step, "field_name": "magnE_mean",
                    "age": 20 + (series - 1) // 2, "sex": "Female" if series % 2 else "Male",
                    "ef_sf": 1 + series % 3, "cgi_change": series / 4,
                    "cgi_start": 7.0, "cgi_end": 7 - series / 4,
                    "session_num": 2, "comments": f"Retain series {series}",
                    "full_file_path": "reused_simulation.nii.gz",
                    "normalization": normalization,
                    "normalization_tag": "raw" if normalization == "raw" else "normed_brain_magnE_mean_mean",
                }
                value = np.sin(series * index)
                if kind == "roi":
                    row.update(roi_org=feature, mean=value)
                    if normalization == "brain_mean":
                        row.update(mean_raw=value * 10, whole_brain_mean=10.0,
                                   normalization_applies_to="mean")
                else:
                    row.update(component=feature, component_index=index, score=value,
                               basis_id=f"basis_{step}_{normalization}", sample_id=f"sample_{step}_{series}",
                               explained_variance=float(3 - index),
                               explained_variance_ratio=0.8 if index == 1 else 0.2,
                               cumulative_variance_ratio=0.8 if index == 1 else 1.0,
                               selected_for_correlation=index == 1)
                records.append(row)
    return pd.DataFrame(records).sample(frac=1, random_state=18).reset_index(drop=True)


def arguments(path, kind="roi", analysis_name="series_test", **changes):
    values = dict(master_path=path, feature_type=kind, field_name="magnE_mean",
                  analysis_name=analysis_name, features=None, pca_scope="selected",
                  prepare_only=False, reml=False, optimizer="powell", maxiter=2000)
    values.update(changes)
    return argparse.Namespace(**values)


def successful_fit(frame, settings):
    # The fitter's numerical behavior has its own real synthetic-fit tests.
    table = pd.DataFrame({
        "term": ["Intercept", "E", "E:ef_sf", "age"], "estimate": [0.5, 2.0, 0.25, 0.03],
        "std_error": [0.1] * 4, "wald_z": [5.0, 20.0, 2.5, 0.3],
        "p_value": [0.001, 0.0001, 0.02, 0.76], "ci_low": [0.3, 1.8, 0.05, -0.17],
        "ci_high": [0.7, 2.2, 0.45, 0.23], "inference_method": ["asymptotic_normal_wald"] * 4,
    })
    return table, {"status": "ok", "error": "", "warnings": "", "converged": True,
                   "n_observations": len(frame),
                   "sex_variance": 0.1, "residual_variance": 0.5}


class TestSeriesWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.root = self.directory / "results"
        self.environment = patch.dict(os.environ, {"RESULTS_DIR": str(self.root)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def write_master(self, name="master.csv", **kwargs):
        path = self.directory / name
        master_rows(**kwargs).to_csv(path, index=False)
        return path

    def test_each_fit_has_feature_source_normalization_and_independent_series(self):
        path = self.write_master()
        original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sizes = []

        def capture(frame, settings):
            sizes.append((frame.feature_id.nunique(), frame.step.nunique(), len(frame)))
            return successful_fit(frame, settings)

        with patch("series_models.workflow.fit_feature", side_effect=capture):
            result = run_analysis(arguments(path))
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(sizes, [(1, 1, 8)] * 4)
        table = pd.read_csv(result["paths"]["lmm_results"])
        self.assertEqual(len(table), 4)
        self.assertEqual(set(table.roi_name), {"LH_Vis_A", "RH_Vis_A"})
        self.assertTrue(table.pc_name.isna().all())
        self.assertTrue(table.normalization.eq("raw").all())
        self.assertFalse(table.is_brain_normalized.any())
        self.assertTrue(table.master_file.eq(path.name).all())
        self.assertTrue(table.source_master_path.eq(str(path)).all())
        self.assertTrue(table.source_master_sha256.eq(original_digest).all())
        self.assertTrue(table.n_observations.eq(8).all())
        self.assertTrue(table.n_people.eq(4).all())
        self.assertTrue(table.E_estimate.eq(2.0).all())
        self.assertTrue(table.E_ef_sf_estimate.eq(0.25).all())
        self.assertTrue(table.age_estimate.eq(0.03).all())
        self.assertTrue(table.age_effect.eq("continuous_fixed").all())
        self.assertNotIn("age_variance", table)
        self.assertIn("__raw__", result["paths"]["lmm_results"].name)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_digest)
        prepared = pd.read_csv(result["paths"]["model_input"])
        self.assertEqual(len(prepared), 32)
        self.assertIn("comments", prepared)
        np.testing.assert_allclose(prepared.E_x_ef_sf, prepared.E * prepared.ef_sf)
        for artifact in result["paths"].values():
            self.assertTrue(artifact.is_relative_to(self.root / "lmm/series_test"))
        manifest = json.loads(result["paths"]["analysis"].read_text())
        self.assertEqual(manifest["planned_fits"], 4)
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["model"]["fixed_formula"], "cgi_change ~ 1 + E + E:ef_sf + age")
        self.assertEqual(manifest["model"]["variance_components"], {"sex": "0 + C(sex)"})
        self.assertEqual(manifest["model"]["fixed_terms"], ["Intercept", "E", "E:ef_sf", "age"])
        self.assertEqual(manifest["model"]["age_effect"]["type"], "continuous_fixed")
        self.assertEqual(manifest["fit_settings"]["optimizer"], "powell")
        self.assertEqual(len(manifest["source_master_sha256"]), 64)

    def test_normalized_and_raw_masters_get_separate_tables_in_same_analysis(self):
        raw = self.write_master("master raw.csv")
        normed = self.write_master("master normalized.csv", normalization="brain_mean")
        configuration = run_configuration("random", {
            "reml": False, "optimizer": "powell", "maxiter": 2000,
            "boundary_tolerance": 1e-8, "sex_effect": "random",
        })
        run_manifest = ensure_run_manifest("series_test", configuration, inputs=[str(raw), str(normed)])
        with patch("series_models.workflow.fit_feature", side_effect=successful_fit):
            first = run_analysis(arguments(raw, output_component="roi_raw"))
            second = run_analysis(arguments(normed, output_component="roi_brain_mean"))
        self.assertEqual(run_manifest, self.root / "lmm/series_test/run.json")
        self.assertEqual(json.loads(run_manifest.read_text())["input_masters"], sorted([str(raw), str(normed)]))
        self.assertTrue(first["paths"]["lmm_results"].is_relative_to(self.root / "lmm/series_test/roi_raw"))
        self.assertTrue(second["paths"]["lmm_results"].is_relative_to(self.root / "lmm/series_test/roi_brain_mean"))
        self.assertNotEqual(first["paths"]["lmm_results"], second["paths"]["lmm_results"])
        self.assertIn("__brain_mean__", second["paths"]["lmm_results"].name)
        table = pd.read_csv(second["paths"]["lmm_results"])
        self.assertTrue(table.is_brain_normalized.all())
        self.assertTrue(table.normalization_tag.eq("normed_brain_magnE_mean_mean").all())

    def test_pc_names_and_bases_are_explicit_without_pooling(self):
        path = self.write_master(kind="pca")
        with patch("series_models.workflow.fit_feature", side_effect=successful_fit):
            result = run_analysis(arguments(path, kind="pca"))
        table = pd.read_csv(result["paths"]["lmm_results"])
        self.assertEqual(len(table), 2)
        self.assertTrue(table.pc_name.eq("PC1").all())
        self.assertEqual(table.basis_id.nunique(), 2)
        self.assertTrue(table.n_observations.eq(8).all())
        prepared = pd.read_csv(result["paths"]["model_input"])
        self.assertTrue(prepared.E.lt(0).any())
        np.testing.assert_allclose(prepared.E, prepared.score)

    def test_failed_feature_retains_status_and_other_features_keep_inference(self):
        path = self.write_master()

        def partial(frame, settings):
            if frame.feature_id.iloc[0] == "RH_Vis_A":
                return pd.DataFrame(), {"status": "failed", "error": "Optimizer failure", "converged": False}
            return successful_fit(frame, settings)

        with patch("series_models.workflow.fit_feature", side_effect=partial):
            result = run_analysis(arguments(path))
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["failed_fits"], 2)
        statuses = pd.read_csv(result["paths"]["fit_status"])
        table = pd.read_csv(result["paths"]["lmm_results"])
        coefficients = pd.read_csv(result["paths"]["coefficients"])
        self.assertEqual(len(statuses), 4)
        self.assertEqual(len(table), 4)
        self.assertEqual(len(coefficients), 8)
        self.assertTrue(table.loc[table.status.eq("failed"), "E_p_value"].isna().all())
        self.assertTrue(table.loc[table.status.eq("ok"), "E_p_value"].notna().all())

    def test_converged_inference_failures_are_explained_after_complete_exports(self):
        path = self.write_master()

        def boundary_failure(frame, settings):
            if frame.feature_id.iloc[0] == "RH_Vis_A":
                return pd.DataFrame(), {
                    "status": "failed", "converged": True, "any_variance_boundary": True,
                    "error": "Invalid observed-information Hessian: nonpositive diagonal",
                }
            return successful_fit(frame, settings)

        captured = io.StringIO()
        with redirect_stdout(captured), patch("series_models.workflow.fit_feature", side_effect=boundary_failure):
            result = run_analysis(arguments(path))
        self.assertEqual(result["exit_code"], 0)
        output = captured.getvalue()
        self.assertIn("Fits with valid Wald inference: 2/4", output)
        self.assertIn("Optimizer converged: 4/4", output)
        self.assertIn("Converged fits rejected by inference/validity checks: 2", output)
        self.assertIn("Failed fits with a variance at the recorded boundary: 2", output)
        self.assertIn("2: Invalid observed-information Hessian: nonpositive diagonal", output)
        self.assertIn("Run completed successfully (exit code 0)", output)
        status = pd.read_csv(result["paths"]["fit_status"])
        table = pd.read_csv(result["paths"]["lmm_results"])
        self.assertEqual(len(status), 4)
        self.assertEqual(len(table), 4)
        self.assertTrue(table.loc[table.status.eq("failed"), "E_p_value"].isna().all())
        self.assertEqual(len(pd.read_csv(result["paths"]["coefficients"])), 8)

    def test_prepare_only_validates_each_feature_and_never_calls_fitter(self):
        path = self.write_master(kind="pca")
        with patch("series_models.workflow.fit_feature") as fitter:
            result = run_analysis(arguments(path, kind="pca", prepare_only=True, pca_scope="all"))
        fitter.assert_not_called()
        self.assertEqual(result["planned_fits"], 4)
        table = pd.read_csv(result["paths"]["lmm_results"])
        self.assertTrue(table.status.eq("prepared").all())
        self.assertTrue(table.design_rank.eq(4).all())
        self.assertTrue(table.E_estimate.isna().all())
        self.assertEqual(len(pd.read_csv(result["paths"]["coefficients"])), 0)

    def test_prepare_only_rank_failure_is_exported_for_each_planned_fit(self):
        path = self.directory / "constant_factor.csv"
        master_rows().assign(ef_sf=1).to_csv(path, index=False)
        result = run_analysis(arguments(path, prepare_only=True))
        self.assertEqual(result["exit_code"], 0)
        table = pd.read_csv(result["paths"]["lmm_results"])
        self.assertEqual(len(table), 4)
        self.assertTrue(table.status.eq("failed").all())
        self.assertTrue(table.error.str.contains("rank", case=False).all())

    def test_existing_run_cannot_be_overwritten(self):
        path = self.write_master()
        result = run_analysis(arguments(path, prepare_only=True))
        prior = {key: value.read_bytes() for key, value in result["paths"].items()}
        with self.assertRaisesRegex(FileExistsError, "new --analysis-name"):
            run_analysis(arguments(path, prepare_only=True))
        for key, value in result["paths"].items():
            self.assertEqual(value.read_bytes(), prior[key])

    def test_invalid_results_configuration_creates_no_artifacts(self):
        path = self.write_master()
        for root in ("", "relative/results", str(SOURCE_DIR / "series_models")):
            with patch.dict(os.environ, {"RESULTS_DIR": root}):
                with self.assertRaises(ValueError):
                    run_analysis(arguments(path, prepare_only=True))
        with self.assertRaises(ValueError):
            run_analysis(arguments(path, analysis_name="../escape", prepare_only=True))
        self.assertFalse(self.root.exists())

    def test_reserved_input_metadata_is_rejected_before_any_output(self):
        for column in ("master_file", "source_master_path", "source_master_sha256",
                       "is_brain_normalized", "roi_name_used", "pc_name_used"):
            with self.subTest(column=column):
                path = self.directory / f"reserved_{column}.csv"
                master_rows().assign(**{column: "conflicting original metadata"}).to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "reserved modeling columns"):
                    run_analysis(arguments(path, prepare_only=True))
                self.assertFalse(self.root.exists())

    def test_cli_end_to_end_synthetic_model_exports_valid_inference(self):
        rng = np.random.default_rng(108)
        age = np.repeat(np.arange(25, 37), 20)
        sex = np.tile(np.repeat(["F", "M"], 10), 12)
        field = rng.normal(size=len(age))
        factor = rng.uniform(0.5, 2, len(age))
        outcome = (2 + 1.5 * field - 0.6 * field * factor + 0.03 * age
                   + np.where(sex == "M", 0.6, -0.6) + rng.normal(0, 0.15, len(age)))
        path = self.directory / "synthetic_master.csv"
        pd.DataFrame({
            "person_id": np.arange(len(age)) // 2, "series_num": np.arange(len(age)) % 2 + 1,
            "age": age, "sex": sex, "mean": field, "ef_sf": factor, "cgi_change": outcome,
            "model_type": "synthetic_model", "step": "step_0", "field_name": "magnE_mean",
            "roi_org": "LH_synthetic_ROI",
        }).to_csv(path, index=False)
        code = main(["--master-path", str(path), "--feature-type", "roi",
                     "--analysis-name", "synthetic_actual_fit"])
        self.assertEqual(code, 0)
        table_path, = (self.root / "lmm/synthetic_actual_fit").glob("*_lmm_results.csv")
        table = pd.read_csv(table_path)
        self.assertEqual(len(table), 1)
        self.assertEqual(table.status.iloc[0], "ok")
        self.assertEqual(table.n_observations.iloc[0], 240)
        self.assertEqual(table.n_people.iloc[0], 120)
        self.assertEqual(table.roi_name.iloc[0], "LH_synthetic_ROI")
        np.testing.assert_allclose(table[["E_estimate", "E_ef_sf_estimate"]].iloc[0],
                                   [1.5, -0.6], atol=0.12)
        self.assertAlmostEqual(table.age_estimate.iloc[0], 0.03, delta=0.01)
        self.assertAlmostEqual(table.intercept_estimate.iloc[0] + table.age_estimate.iloc[0] * age.mean(),
                               2 + 0.03 * age.mean(), delta=0.12)
        self.assertTrue(table.age_effect.eq("continuous_fixed").all())
        self.assertTrue(table.fixed_formula.eq("cgi_change ~ 1 + E + E:ef_sf + age").all())
        self.assertNotIn("age_variance", table)
        self.assertEqual(len(pd.read_csv(next(table_path.parent.glob("*_coefficients.csv")))), 4)
        self.assertTrue(table[["intercept_p_value", "E_p_value", "E_ef_sf_p_value"]].notna().all(axis=None))

    def test_cli_environment_validation_precedes_scientific_imports(self):
        environment = dict(os.environ, RESULTS_DIR="", PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(
            [sys.executable, "-B", "-S", str(SOURCE / "run_series_mixed_effects.py"),
             "--master-path", str(self.directory / "missing.csv"), "--feature-type", "roi",
             "--analysis-name", "test"], env=environment, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("RESULTS_DIR must be set", completed.stderr)
        self.assertNotIn("ModuleNotFoundError", completed.stderr)
        self.assertFalse(self.root.exists())

    def test_cli_returns_success_after_exporting_failed_designs(self):
        path = self.directory / "collinear.csv"
        master_rows().assign(ef_sf=1).to_csv(path, index=False)
        self.assertEqual(main(["--master-path", str(path), "--feature-type", "roi",
                               "--analysis-name", "cli_test", "--prepare-only"]), 0)
        self.assertEqual(len(list((self.root / "lmm/cli_test").glob("*_fit_status.csv"))), 1)

    def test_cli_nonconvergence_is_success_with_failure_saved_for_every_feature(self):
        path = self.write_master()

        def nonconverged(frame, settings):
            return pd.DataFrame(), {
                "status": "failed", "converged": False,
                "error": "Mixed model did not converge", "warnings": "Optimizer exhausted",
            }

        with patch("series_models.workflow.fit_feature", side_effect=nonconverged):
            code = main(["--master-path", str(path), "--feature-type", "roi",
                         "--analysis-name", "nonconverged_cli"])
        self.assertEqual(code, 0)
        output = self.root / "lmm/nonconverged_cli"
        status = pd.read_csv(next(output.glob("*_fit_status.csv")))
        results = pd.read_csv(next(output.glob("*_lmm_results.csv")))
        self.assertEqual(len(status), 4)
        self.assertEqual(len(results), 4)
        self.assertTrue(results.status.eq("failed").all())
        self.assertFalse(results.converged.any())
        self.assertTrue(results.error.eq("Mixed model did not converge").all())
        self.assertTrue(results.warnings.eq("Optimizer exhausted").all())
        self.assertTrue(results.E_p_value.isna().all())
        self.assertTrue(pd.read_csv(next(output.glob("*_coefficients.csv"))).empty)

    def test_cli_missing_master_keeps_nonzero_exit_code_and_creates_no_output(self):
        code = main(["--master-path", str(self.directory / "missing_master.csv"),
                     "--feature-type", "roi", "--analysis-name", "missing_master"])
        self.assertEqual(code, 2)
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
