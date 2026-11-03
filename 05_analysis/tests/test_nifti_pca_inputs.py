"""Input identity, metadata preservation, and results-location contracts."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifti_pca.inputs import load_series_inputs, validate_series_inputs
from nifti_pca.paths import SOURCE_DIR, resolve_analysis_dir, validate_output_path


def roi_input():
    rows = []
    for series in (1, 2, 3):
        for roi, mean in (("LH_Vis", 2.0), ("RH_Vis", 4.0)):
            rows.append({
                "person_id": "person_a", "series_num": series,
                "model_type": "skin_single", "step": "step_0",
                "field_name": "magnE_mean", "full_file_path": f"field_{series}.nii.gz",
                "cgi_change": series, "roi_org": roi, "mean": mean, "n_voxels": 100,
                "date_start": f"2025-01-0{series}", "pulse_width_ms": 0.5,
                "attending": "clinician", "comments": "preserve all clinical metadata",
            })
    return pd.DataFrame(rows)


class TestSeriesInputs(unittest.TestCase):
    def test_roi_repeats_collapse_without_collapsing_independent_series(self):
        original = roi_input()
        result = validate_series_inputs(original, "magnE_mean")
        self.assertEqual(len(result), 3)
        self.assertEqual(result["person_id"].nunique(), 1)
        self.assertEqual(result["sample_id"].nunique(), 3)
        self.assertTrue(result["comments"].eq("preserve all clinical metadata").all())
        self.assertIn("date_start", result)
        self.assertIn("pulse_width_ms", result)
        self.assertNotIn("mean", result)
        self.assertNotIn("roi_org", result)
        again = validate_series_inputs(original.iloc[::-1], "magnE_mean")
        self.assertEqual(
            dict(zip(result["series_num"], result["sample_id"])),
            dict(zip(again["series_num"], again["sample_id"])),
        )
        self.assertIn("mean", original)

    def test_conflicting_metadata_is_not_silently_deduplicated(self):
        frame = roi_input()
        frame.loc[1, "comments"] = "conflicting clinical information"
        with self.assertRaisesRegex(ValueError, "Conflicting metadata"):
            validate_series_inputs(frame, "magnE_mean")

    def test_cross_configuration_outcomes_must_agree(self):
        first = roi_input()
        second = first.assign(step="step_9", cgi_change=99)
        with self.assertRaisesRegex(ValueError, "Inconsistent CGI"):
            validate_series_inputs(pd.concat([first, second]), "magnE_mean")

    def test_normalized_roi_master_still_uses_original_nifti_paths(self):
        frame = roi_input().assign(
            mean_raw=lambda df: df["mean"], normalization="brain_mean",
            whole_brain_mean=10.0,
        )
        result = validate_series_inputs(frame, "magnE_mean")
        self.assertNotIn("mean_raw", result)
        self.assertNotIn("normalization", result)
        self.assertNotIn("whole_brain_mean", result)
        self.assertEqual(result["full_file_path"].tolist(), [f"field_{i}.nii.gz" for i in (1, 2, 3)])

    def test_relative_field_paths_are_relative_to_input_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "series.csv"
            roi_input().to_csv(source, index=False)
            result = load_series_inputs(field_name="magnE_mean", series_csv=source)
            self.assertEqual(result["full_file_path"].iloc[0], str(Path(directory) / "field_1.nii.gz"))

    def test_invalid_series_number_and_duplicate_sample_ids(self):
        frame = roi_input()
        frame.loc[0:1, "series_num"] = -1
        with self.assertRaisesRegex(ValueError, "positive integers"):
            validate_series_inputs(frame, "magnE_mean")
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_series_inputs(roi_input().assign(sample_id="same_id"), "magnE_mean")


class TestResultsLocation(unittest.TestCase):
    def test_no_results_root_fallback(self):
        with patch.dict(os.environ, {"RESULTS_DIR": ""}):
            with self.assertRaisesRegex(ValueError, "RESULTS_DIR must be set"):
                resolve_analysis_dir("test")

    def test_invalid_names_source_tree_and_escaping_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"RESULTS_DIR": directory}):
                root = resolve_analysis_dir("analysis_test")
                self.assertEqual(root, Path(directory) / "analysis_test")
                self.assertEqual(validate_output_path(root / "data/master.csv"), root / "data/master.csv")
                for name in ("../escape", "a/b", ".", "", "bad name"):
                    with self.assertRaises(ValueError):
                        resolve_analysis_dir(name)
                with self.assertRaises(ValueError):
                    validate_output_path(Path(directory) / "direct.csv")
                with self.assertRaises(ValueError):
                    validate_output_path(Path(directory).parent / "escaped.csv")
            with patch.dict(os.environ, {"RESULTS_DIR": str(SOURCE_DIR / "data")}):
                with self.assertRaisesRegex(ValueError, "source tree"):
                    resolve_analysis_dir("test")


if __name__ == "__main__":
    unittest.main()
