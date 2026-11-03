"""Prep and ancillary CLI contracts for results-only artifact writes."""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from analysis_paths import validate_output_path
from networks import add_region_means as ancillary
from networks import mni_yeo2011_prep_data as prep
from networks._results import normalization_metadata


class TestPrepOutputPaths(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="yeo_prep_paths_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / "results"
        self.environment = patch.dict(os.environ, {"RESULTS_DIR": str(self.results)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.master = pd.DataFrame({
            "person_id": ["person_a"], "series_num": [1],
            "model_type": ["skin_single"], "step": ["step_0"],
            "field_name": ["magnE_mean"], "roi_name": ["7Networks_LH_Vis"],
            "mean": [2.5], "n_voxels": [123], "cgi_change": [1.0],
            "comments": ["preserve clinical metadata"],
        })

    def test_prep_default_and_explicit_selected_analysis_outputs(self):
        with patch.object(prep, "build_series_roi_master", return_value=self.master) as build:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prep.main(["--analysis-name", "roi_test"]), 0)
        build.assert_called_once()
        target = self.results / "roi_test/data/yeo2011_magnE_mean_series_ROI_master.csv"
        actual = pd.read_csv(target)
        self.assertEqual(actual["mean"].iloc[0], 2.5)
        self.assertEqual(actual["n_voxels"].iloc[0], 123)
        self.assertEqual(actual["roi_org"].iloc[0], "7Networks_LH_Vis")
        self.assertEqual(actual["comments"].iloc[0], "preserve clinical metadata")
        self.assertIn("roi_name", self.master)
        custom = self.results / "roi_test/data/custom.csv"
        with patch.object(prep, "build_series_roi_master", return_value=self.master):
            with contextlib.redirect_stdout(io.StringIO()):
                prep.main(["--analysis-name", "roi_test", "--output-path", str(custom)])
        self.assertTrue(custom.is_file())

    def test_prep_invalid_destinations_rejected_before_loading(self):
        destinations = [self.root / "escaped.csv", self.results / "another/data/master.csv"]
        for destination in destinations:
            with self.subTest(destination=destination):
                with patch.object(prep, "build_series_roi_master") as build:
                    with self.assertRaises(ValueError):
                        prep.main(["--analysis-name", "roi_test", "--output-path", str(destination)])
                    build.assert_not_called()
                self.assertFalse(destination.exists())

    def test_prep_library_saver_rejects_source_tree(self):
        with self.assertRaises(ValueError):
            prep.save_roi_master(self.master, SOURCE / "networks/data/should_not_exist.csv")

    def test_ancillary_default_output_and_legacy_input(self):
        source = self.root / "input.csv"
        pd.DataFrame({"subject": ["0001"], "full_file_path": ["field.nii.gz"]}).to_csv(source, index=False)
        extracted = {"mean": pd.DataFrame({"E_mean__ROI": [2.5]})}
        with patch.object(ancillary, "extract_region_stats", return_value=extracted):
            with contextlib.redirect_stdout(io.StringIO()):
                ancillary.main([str(source), "atlas.nii.gz", "lut.txt", "--analysis-name", "roi_test"])
        target = self.results / "roi_test/data/efield_by_region.csv"
        actual = pd.read_csv(target, dtype=str)
        self.assertEqual(actual["subject"].iloc[0], "0001")
        self.assertEqual(float(actual["E_mean__ROI"].iloc[0]), 2.5)
        with patch.object(ancillary, "extract_region_stats") as extraction:
            with self.assertRaises(ValueError):
                ancillary.main([str(source), "atlas", "lut", str(self.root / "escape.csv")])
            extraction.assert_not_called()

    def test_all_standalone_clis_require_env_before_heavy_imports(self):
        names = (
            "mni_yeo2011_prep_data.py", "mni_yeo2011_correlations.py",
            "mni_yeo2011_correlations_perm.py", "mni_yeo2011_normalize_brain.py",
            "mni_yeo2011_plot_correlations.py", "add_region_means.py",
        )
        environment = dict(os.environ, RESULTS_DIR="", PYTHONDONTWRITEBYTECODE="1")
        for name in names:
            with self.subTest(name=name):
                result = subprocess.run(
                    [sys.executable, "-B", "-S", str(SOURCE / "networks" / name)],
                    cwd=self.root, env=environment, text=True, capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("RESULTS_DIR must be set", result.stderr)
                self.assertNotIn("No module named", result.stderr)

    def test_normalization_metadata_and_selected_analysis_policy(self):
        raw = pd.DataFrame({"field_name": ["magnE_mean"]})
        info = normalization_metadata(raw, "magnE_mean")
        self.assertEqual(info["normalization_tag"], "raw")
        tag = "normed_brain_magnE_mean_mean"
        legacy = normalization_metadata(raw, "magnE_mean", source_path=f"legacy_{tag}_annotated.csv")
        self.assertEqual(legacy["normalization"], "brain_mean")
        mixed = pd.DataFrame({"field_name": ["magnE_mean"] * 2, "normalization": ["raw", "brain_mean"]})
        with self.assertRaisesRegex(ValueError, "Mixed"):
            normalization_metadata(mixed, "magnE_mean")
        with self.assertRaisesRegex(ValueError, "selected analysis"):
            validate_output_path(self.results / "other/data/test.csv", "roi_test")


if __name__ == "__main__":
    unittest.main()
