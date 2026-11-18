"""AIC report export, output protection, and command-line contracts."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from run_series_aic_comparison import main, parse_args
from series_models.aic_workflow import run_aic_comparison


def comparison_fixture():
    common = dict(field_name="magnE_mean", model_type="model", step="step_1", basis_id="",
                  normalization="raw", aic_provisional=True, status="failed", converged=True,
                  aic_exclusion_reason="", aic_caveat="Wald inference unavailable")
    table = pd.DataFrame([
        {**common, "analysis_dir": "/input/null", "feature_type": "null", "feature_id": "null",
         "source_master_path": "/input/null_master.csv",
         "is_null": True, "aic_eligible": True, "aic": 100., "aic_improvement_over_null": 0.},
        {**common, "analysis_dir": "/input/roi", "feature_type": "roi", "feature_id": "ROI_better",
         "source_master_path": "/input/roi_master.csv",
         "is_null": False, "aic_eligible": True, "aic": 98., "aic_improvement_over_null": 2.},
        {**common, "analysis_dir": "/input/roi", "feature_type": "roi", "feature_id": "ROI_failed",
         "source_master_path": "/input/roi_master.csv",
         "is_null": False, "aic_eligible": False, "aic": float("nan"),
         "aic_exclusion_reason": "optimizer did not converge", "aic_improvement_over_null": float("nan")},
    ])
    settings = {"reml": False, "optimizer": "powell", "maxiter": 2000,
                "boundary_tolerance": 1e-8, "sex_effect": "random"}
    return SimpleNamespace(table=table, metadata={
        "null_aic": 100., "n_candidates": 2,
        "analysis_settings": [{"fit_settings": settings}],
        "null_model_spec": {"sex_effect_choice": "random"},
    })


class TestAICWorkflow(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "results"
        self.environment = patch.dict(os.environ, {"RESULTS_DIR": str(self.root)})
        self.environment.start()
        self.args = argparse.Namespace(null_analysis_dir=self.base / "null", feature_analysis_dirs=[self.base / "roi"],
                                       analysis_name="comparison", no_plots=True, top_n=20, dpi=150)

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_no_plots_exports_all_candidates_separate_eligibility_and_provenance(self):
        with patch("series_models.aic_workflow.compare_analyses", return_value=comparison_fixture()):
            with redirect_stdout(io.StringIO()):
                result = run_aic_comparison(self.args)
        self.assertEqual(result["exit_code"], 0)
        table = pd.read_csv(result["paths"]["comparison"])
        self.assertEqual(len(table), 3)
        self.assertEqual(table.status.tolist(), ["failed"] * 3)
        ranked = pd.read_csv(result["paths"]["ranked"], keep_default_na=False)
        self.assertEqual(ranked.feature_id.tolist(), ["ROI_better", "null"])
        excluded = pd.read_csv(result["paths"]["excluded"])
        self.assertEqual(excluded.feature_id.tolist(), ["ROI_failed"])
        self.assertIn("converge", excluded.aic_exclusion_reason.iloc[0])
        summary = pd.read_csv(result["paths"]["summary"], keep_default_na=False)
        self.assertEqual(summary.n_candidates.iloc[0], 2)
        self.assertEqual(summary.n_eligible.iloc[0], 1)
        self.assertEqual(summary.best_improvement_over_null.iloc[0], 2.)
        self.assertEqual(summary.best_including_null.iloc[0], "feature")
        manifest = json.loads(result["paths"]["analysis"].read_text())
        self.assertEqual(manifest["analysis_type"], "aic_comparison")
        self.assertIn("run_series_aic_comparison.py", manifest["provenance"]["source_sha256"])
        self.assertEqual(manifest["planned_plots"], [])
        plots = pd.read_csv(result["paths"]["plot_manifest"])
        self.assertTrue(plots.empty)
        self.assertFalse((self.root / "lmm/comparison/aic/logs/matplotlib_cache").exists())

    def test_input_validation_failure_creates_no_report_artifacts(self):
        with patch("series_models.aic_workflow.compare_analyses", side_effect=ValueError("Cohort mismatch")):
            with self.assertRaisesRegex(ValueError, "Cohort mismatch"):
                run_aic_comparison(self.args)
        self.assertFalse(self.root.exists())

    def test_existing_artifacts_are_protected(self):
        output = self.root / "lmm/comparison/aic"
        output.mkdir(parents=True)
        sentinel = output / "aic_ranked.csv"
        sentinel.write_text("existing content\n")
        with patch("series_models.aic_workflow.compare_analyses", return_value=comparison_fixture()):
            with self.assertRaises(FileExistsError):
                run_aic_comparison(self.args)
        self.assertEqual(sentinel.read_text(), "existing content\n")
        self.assertFalse((output / "analysis.json").exists())

    def test_report_cannot_share_an_input_analysis_directory(self):
        self.args.null_analysis_dir = self.root / "lmm/comparison/aic"
        with patch("series_models.aic_workflow.compare_analyses") as compare:
            with self.assertRaisesRegex(ValueError, "distinct"):
                run_aic_comparison(self.args)
        compare.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_null_wins_summary_when_every_feature_has_higher_aic(self):
        fixture = comparison_fixture()
        fixture.table.loc[1, ["aic", "aic_improvement_over_null"]] = [102., -2.]
        with patch("series_models.aic_workflow.compare_analyses", return_value=fixture):
            with redirect_stdout(io.StringIO()):
                result = run_aic_comparison(self.args)
        summary = pd.read_csv(result["paths"]["summary"], keep_default_na=False)
        self.assertEqual(summary.best_including_null.iloc[0], "null")
        self.assertEqual(summary.best_improvement_over_null.iloc[0], -2.)

    def test_cli_returns_two_for_validation_failure(self):
        with patch("series_models.aic_workflow.compare_analyses", side_effect=ValueError("Cohort mismatch")):
            with redirect_stderr(io.StringIO()) as error:
                code = main(["--null-analysis-dir", str(self.args.null_analysis_dir),
                             "--feature-analysis-dirs", str(self.args.feature_analysis_dirs[0]),
                             "--analysis-name", "comparison", "--no-plots"])
        self.assertEqual(code, 2)
        self.assertIn("Cohort mismatch", error.getvalue())
        self.assertFalse(self.root.exists())

    def test_cli_rejects_invalid_output_before_scientific_work(self):
        arguments = ["--null-analysis-dir", "null", "--feature-analysis-dirs", "roi", "--analysis-name", "../bad"]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parse_args(arguments)
        self.assertEqual(error.exception.code, 2)
        for name in ("--top-n", "--dpi"):
            with self.subTest(option=name), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(arguments[:-1] + ["comparison", name, "0"])


if __name__ == "__main__":
    unittest.main()
