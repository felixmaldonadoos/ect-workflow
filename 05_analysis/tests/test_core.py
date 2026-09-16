from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from simnibs_parcel_analysis import (
    atlas_masks_to_labels,
    build_scan_table,
    components_for_variance,
    correlate_predictors,
    expand_predictors_to_courses,
    fit_parcel_pca,
    global_metrics_from_parcels,
    prepare_scan_outcomes,
    resolve_freesurfer_subjects,
    sample_volume_labels,
    select_volume_roi_labels,
    weighted_percentile,
    validate_treatment_courses,
)


class TestWeightedPercentile(unittest.TestCase):
    def test_weight_changes_empirical_percentile(self) -> None:
        values = np.array([1.0, 2.0, 10.0])
        self.assertEqual(weighted_percentile(values, np.array([1.0, 1.0, 20.0]), 50), 10.0)
        self.assertEqual(weighted_percentile(values, np.ones(3), 50), 2.0)


class TestVolumeLabels(unittest.TestCase):
    def test_nearest_neighbor_sampling(self) -> None:
        data = np.zeros((4, 4, 4), dtype=int)
        data[1, 2, 3] = 17
        labels, inside = sample_volume_labels(np.array([[1.1, 2.2, 2.9], [9.0, 9.0, 9.0]]), data, np.eye(4))
        np.testing.assert_array_equal(labels, [17, 0])
        np.testing.assert_array_equal(inside, [True, False])

    def test_cortical_and_subcortical_selection(self) -> None:
        lut = {
            2: "Left-Cerebral-White-Matter",
            10: "Left-Thalamus",
            17: "Left-Hippocampus",
            1002: "ctx-lh-caudalanteriorcingulate",
            2002: "ctx-rh-caudalanteriorcingulate",
            11101: "ctx_lh_G_and_S_frontomargin",
        }
        selected = select_volume_roi_labels(lut, lut)
        self.assertEqual(
            set(selected.values()),
            {
                "Left-Thalamus",
                "Left-Hippocampus",
                "ctx-lh-caudalanteriorcingulate",
                "ctx-rh-caudalanteriorcingulate",
                "ctx_lh_G_and_S_frontomargin",
            },
        )


class TestSubjectResolution(unittest.TestCase):
    def test_unique_base_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "subj-cat-001-001").mkdir()
            scans = build_scan_table(["subj-cat-001-001", "subj-cat-001-002"])
            result = resolve_freesurfer_subjects(scans, directory)
            self.assertEqual(result["fs_subjid"].tolist(), ["subj-cat-001-001", "subj-cat-001-001"])

    def test_ambiguous_base_requires_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "subj-cat-001-001").mkdir()
            Path(directory, "subj-cat-001-003").mkdir()
            scans = build_scan_table(["subj-cat-001-002"])
            with self.assertRaisesRegex(ValueError, "Ambiguous recon-all match"):
                resolve_freesurfer_subjects(scans, directory)
            result = resolve_freesurfer_subjects(scans, directory, subject_map={"subj-cat-001-002": "subj-cat-001-001"})
            self.assertEqual(result.loc[0, "fs_subjid"], "subj-cat-001-001")


class TestPCAAndOutcomes(unittest.TestCase):
    def test_requested_normalization_and_pca(self) -> None:
        parcels = pd.DataFrame(
            [[1, 2, 4], [2, 5, 3], [4, 1, 2], [3, 6, 2]],
            index=["subj-cat-001-001", "subj-cat-002-001", "subj-cat-003-001", "subj-cat-004-001"],
            columns=["A", "B", "C"],
            dtype=float,
        )
        result = fit_parcel_pca(parcels)
        np.testing.assert_allclose(result.relative_parcels.mean(axis=1), 1)
        np.testing.assert_allclose(result.subject_demeaned_parcels.mean(axis=1), 0, atol=1e-12)
        self.assertGreaterEqual(components_for_variance(result, 0.96), 1)

    def test_courses_expand_across_scans_by_base_id(self) -> None:
        scans = build_scan_table(["subj-cat-001-001", "subj-cat-001-002", "subj-cat-002-001"])
        cognitive = pd.DataFrame(
            {
                "subjid": ["subj-cat-001", "subj-cat-001", "subj-cat-002"],
                "date_start": ["2025-01-01", "2025-06-01", "2025-02-01"],
                "cgi_change": [-2, -3, -1],
            }
        )
        result = prepare_scan_outcomes(scans, cognitive)
        self.assertEqual(len(result), 5)
        self.assertEqual(result.groupby("subjid").size().to_dict(), {"subj-cat-001-001": 2, "subj-cat-001-002": 2, "subj-cat-002-001": 1})
        self.assertFalse(result["observation_id"].duplicated().any())

    def test_scans_without_courses_are_excluded_only_from_outcomes(self) -> None:
        scans = build_scan_table(["subj-cat-001-001", "subj-cat-002-001"])
        cognitive = pd.DataFrame(
            {
                "subjid": ["subj-cat-001", "subj-cat-001"],
                "date_start": ["2025-01-01", "2025-06-01"],
                "cgi_change": [-2.0, -1.0],
            }
        )
        with self.assertWarnsRegex(UserWarning, "subj-cat-002-001"):
            outcomes = prepare_scan_outcomes(scans, cognitive)
        self.assertEqual(outcomes["subjid"].tolist(), ["subj-cat-001-001", "subj-cat-001-001"])

        predictors = pd.DataFrame(
            {"global": [1.0, 2.0]},
            index=pd.Index(["subj-cat-001-001", "subj-cat-002-001"], name="subjid"),
        )
        expanded = expand_predictors_to_courses(predictors, outcomes)
        self.assertEqual(expanded["subjid"].tolist(), ["subj-cat-001-001", "subj-cat-001-001"])
        self.assertEqual(expanded["global"].tolist(), [1.0, 1.0])

    def test_same_base_and_start_date_conflicts(self) -> None:
        cognitive = pd.DataFrame(
            {
                "subjid": ["subj-cat-001-001", "subj-cat-001-002"],
                "date_start": ["2025-01-01", "2025-01-01"],
                "cgi_change": [-2, -3],
            }
        )
        with self.assertRaisesRegex(ValueError, "same subject and treatment-course start date"):
            validate_treatment_courses(cognitive)

    def test_predictors_expand_before_correlation(self) -> None:
        predictors = pd.DataFrame(
            {"global": [1.0, 2.0]},
            index=pd.Index(["subj-cat-001-001", "subj-cat-002-001"], name="subjid"),
        )
        scans = build_scan_table(predictors.index.tolist())
        cognitive = pd.DataFrame(
            {
                "subjid": ["subj-cat-001", "subj-cat-001", "subj-cat-002"],
                "date_start": ["2025-01-01", "2025-06-01", "2025-02-01"],
                "cgi_change": [-3.0, -2.0, -1.0],
            }
        )
        outcomes = prepare_scan_outcomes(scans, cognitive)
        expanded = expand_predictors_to_courses(predictors, outcomes)
        self.assertEqual(expanded["global"].tolist(), [1.0, 1.0, 2.0])
        correlations = correlate_predictors(predictors, outcomes)
        self.assertEqual(int(correlations.loc["global", "n_observations"]), 3)
        self.assertEqual(int(correlations.loc["global", "n_treatment_courses"]), 3)

    def test_global_metrics(self) -> None:
        parcels = pd.DataFrame([[1.0, 3.0]], index=["scan"], columns=["A", "B"])
        sizes = pd.DataFrame([[1.0, 3.0]], index=["scan"], columns=["A", "B"])
        metrics = global_metrics_from_parcels(parcels, sizes)
        self.assertEqual(metrics.loc["scan", "global_mean_E_unweighted"], 2.0)
        self.assertEqual(metrics.loc["scan", "global_mean_E_weighted"], 2.5)


class TestSurfaceAtlas(unittest.TestCase):
    def test_masks_to_labels(self) -> None:
        masks = {"lh.A": np.array([1, 1, 0, 0], dtype=bool), "rh.A": np.array([0, 0, 1, 1], dtype=bool)}
        labels, names = atlas_masks_to_labels(masks, 4)
        np.testing.assert_array_equal(labels, [1, 1, 2, 2])
        self.assertEqual(names, {1: "lh.A", 2: "rh.A"})


if __name__ == "__main__":
    unittest.main()
