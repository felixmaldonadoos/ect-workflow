"""Automatic plot integration and saved-run export protection."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from run_series_mixed_effects import parse_args
from run_series_mixed_effects_plots import main as plots_main
from series_models.fit_plot_workflow import export_fit_plots, run_saved_fit_plots
from series_models.prediction import FitPrediction
from series_models.null_workflow import run_null_analysis
from series_models.workflow import run_analysis
from test_series_models_workflow import arguments, master_rows


def plot_record(identity=None, *, available=True):
    saved = dict(feature_type="roi", field_name="magnE_mean", normalization="raw", normalization_tag="raw",
                 model_type="skin_single", step="step_0", basis_id="", feature_id="ROI_1", status="failed",
                 converged=True, aic=42., sex_variance=0., residual_variance=1.)
    saved.update(identity or {})
    if not available:
        return {"identity": saved, "prediction": None, "error": "Optimizer did not converge"}
    points = pd.DataFrame({"person_id": ["p1", "p2"], "series_num": [1, 1], "sex": ["F", "M"],
                           "E": [0., 1.], "age": [30., 40.], "ef_sf": [1., 1.],
                           "observed_cgi_change": [1., 2.], "adjusted_cgi_change": [1., 2.],
                           "fixed_prediction": [1., 2.], "conditional_prediction": [1., 2.], "residual": [0., 0.]})
    curves = pd.DataFrame({"curve_kind": ["fixed", "fixed", "sex", "sex"], "sex": ["", "", "F", "M"],
                           "x": [0., 1., 0., 1.], "predicted_cgi_change": [1., 2., 1., 2.]})
    prediction = FitPrediction(points, curves, {"Intercept": 1., "E": 1., "E:ef_sf": 0., "age": 0.},
                               {"F": 0., "M": 0.},
                               {"x_column": "E", "reference_age": 35., "reference_ef_sf": 1.,
                                "provisional": True, "caveat": "wald_inference_unavailable;variance_boundary",
                                "status": "failed", "converged": True, "n_observations": 2, "aic": 42.,
                                "fit_method": "ML", "log_likelihood_difference": 0.})
    return {"identity": saved, "prediction": prediction, "error": ""}


def saved_source_metadata():
    return {
        "fit_settings": {"reml": False, "optimizer": "powell", "maxiter": 2000,
                         "boundary_tolerance": 1e-8, "sex_effect": "random"},
        "source_master_path": "/input/saved_master.csv",
    }


class TestFitPlotWorkflow(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "results"
        self.environment = patch.dict(os.environ, {"RESULTS_DIR": str(self.root)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_export_accounts_for_provisional_and_unavailable_fits(self):
        records = [plot_record(), plot_record({"feature_id": "ROI_2", "converged": False}, available=False)]
        with redirect_stdout(io.StringIO()):
            output = export_fit_plots(records, "plots", "master__raw__123", dpi=60)
        manifest = pd.read_csv(output["paths"]["manifest"], keep_default_na=False)
        self.assertEqual(manifest.plot_status.tolist(), ["provisional", "skipped"])
        self.assertEqual(manifest.pdf_page.tolist(), [1, 2])
        self.assertEqual(len(output["reports"]), 1)
        self.assertEqual(output["skipped"], 1)
        self.assertTrue(all(path.stat().st_size > 0 for path in output["reports"] + output["pngs"]))
        self.assertEqual(len(pd.read_csv(output["paths"]["observations"])), 2)
        parameters = pd.read_csv(output["paths"]["parameters"])
        self.assertTrue(parameters.estimate_type.eq("point_estimate_only").all())
        self.assertNotIn("p_value", parameters)
        self.assertNotIn("std_error", parameters)
        metadata = json.loads(output["paths"]["analysis"].read_text())
        self.assertEqual(len(metadata["fits"]), 2)
        self.assertFalse(metadata["fits"][1]["prediction_available"])
        self.assertTrue((self.root / "plots/logs/matplotlib_cache").is_dir())
        original = output["paths"]["manifest"].read_bytes()
        with self.assertRaises(FileExistsError):
            export_fit_plots(records, "plots", "master__raw__123", dpi=60)
        self.assertEqual(output["paths"]["manifest"].read_bytes(), original)

    def test_full_feature_workflow_automatically_calls_plot_export(self):
        source = self.base / "master.csv"
        data = master_rows()
        data = data.loc[data.step.eq("step_0") & data.roi_org.eq("LH_Vis_A")]
        data.to_csv(source, index=False)
        original = source.read_bytes()
        status = {"status": "failed", "converged": True, "warnings": "Hessian warning", "error": "Wald rejected"}
        with patch("series_models.workflow.fit_feature", return_value=(pd.DataFrame(), status)):
            with patch("series_models.fit_plot_workflow.make_fit_record",
                       side_effect=lambda frame, identity: plot_record(identity)) as prediction:
                with redirect_stdout(io.StringIO()):
                    result = run_analysis(arguments(source, plot_dpi=60))
        prediction.assert_called_once()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["failed_fits"], 1)
        self.assertEqual(len(result["fit_plots"]["pngs"]), 1)
        metadata = json.loads(result["paths"]["analysis"].read_text())
        self.assertTrue(metadata["fit_plots"]["enabled"])
        self.assertEqual(metadata["fit_plots"]["dpi"], 60)
        self.assertEqual(source.read_bytes(), original)
        self.assertTrue(pd.read_csv(result["paths"]["coefficients"]).empty)

    def test_disabled_and_prepare_only_do_not_attempt_plotting(self):
        source = self.base / "master.csv"
        master_rows().to_csv(source, index=False)
        for name, changes in (("disabled", {"no_fit_plots": True}), ("prepare", {"prepare_only": True})):
            with self.subTest(name=name), patch("series_models.workflow.fit_feature", return_value=(pd.DataFrame(), {"status": "failed", "error": "Optimizer failure"})):
                with patch("series_models.fit_plot_workflow.make_fit_record") as prediction:
                    with redirect_stdout(io.StringIO()):
                        result = run_analysis(arguments(source, analysis_name=name, **changes))
                prediction.assert_not_called()
                self.assertIsNone(result["fit_plots"])
                self.assertFalse((self.root / "lmm" / name / "plots").exists())

    def test_full_null_workflow_automatically_exports_age_plot(self):
        source = self.base / "master.csv"
        master_rows().to_csv(source, index=False)
        args = arguments(source, analysis_name="null_auto", plot_dpi=60)
        args.null_model = True
        status = {"status": "failed", "converged": True, "warnings": "Hessian warning", "error": "Wald rejected"}

        def null_record(frame, identity, *, feature):
            self.assertFalse(feature)
            record = plot_record(identity)
            record["prediction"].metadata["x_column"] = "age"
            record["prediction"].metadata["reference_ef_sf"] = None
            record["prediction"].curves["x"] = [30., 40., 30., 40.]
            return record

        with patch("series_models.null_workflow.fit_null", return_value=(pd.DataFrame(), status)):
            with patch("series_models.fit_plot_workflow.make_fit_record", side_effect=null_record) as prediction:
                with redirect_stdout(io.StringIO()):
                    result = run_null_analysis(args)
        prediction.assert_called_once()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(len(result["fit_plots"]["reports"]), 1)
        manifest = pd.read_csv(result["fit_plots"]["paths"]["manifest"], keep_default_na=False)
        self.assertEqual(manifest.feature_type.iloc[0], "null")
        self.assertEqual(manifest.plot_status.iloc[0], "provisional")

    def test_saved_cli_exports_without_rewriting_its_input(self):
        saved = self.base / "saved"
        saved.mkdir()
        sentinel = saved / "unchanged.csv"
        sentinel.write_text("original\n")
        with patch("series_models.saved_fit_plots.load_saved_fit_records",
                   return_value=([plot_record()], "source", saved_source_metadata())):
            with redirect_stdout(io.StringIO()):
                code = plots_main(["--analysis-dirs", str(saved), "--analysis-name", "saved_plots", "--dpi", "60"])
        self.assertEqual(code, 0)
        metadata = json.loads((self.root / "lmm/saved_plots/plots/analysis.json").read_text())
        self.assertFalse(metadata["model_refitting"])
        self.assertIn("run_series_mixed_effects_plots.py", metadata["provenance"]["source_sha256"])
        summary = pd.read_csv(self.root / "lmm/saved_plots/plots/fit_plot_summary.csv")
        self.assertEqual(summary.n_plotted.iloc[0], 1)
        self.assertEqual(sentinel.read_text(), "original\n")

    def test_invalid_later_source_leaves_no_output(self):
        with patch("series_models.saved_fit_plots.load_saved_fit_records",
                   side_effect=[([plot_record()], "first", saved_source_metadata()), ValueError("Invalid second source")]):
            with redirect_stderr(io.StringIO()) as error:
                code = plots_main(["--analysis-dirs", str(self.base / "first"), str(self.base / "second"),
                                   "--analysis-name", "invalid"])
        self.assertEqual(code, 2)
        self.assertIn("Invalid second source", error.getvalue())
        self.assertFalse(self.root.exists())

    def test_cli_validates_plot_options(self):
        args = parse_args(["--master-path", "master.csv", "--feature-type", "roi", "--analysis-name", "fit"])
        self.assertFalse(args.no_fit_plots)
        self.assertEqual(args.plot_dpi, 120)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--master-path", "master.csv", "--feature-type", "roi", "--analysis-name", "fit", "--plot-dpi", "0"])


if __name__ == "__main__":
    unittest.main()
