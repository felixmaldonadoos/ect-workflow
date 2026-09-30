"""Numerical, extraction, and export checks for the additive demeaning analyses."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import example_usage_global_E_weighted as wrapper
from simnibs_parcel_analysis import (
    DEMEAN_REFERENCES, ParcelSummary, build_scan_table, build_surface_roi_summary,
    build_volume_roi_summary, fit_demeaned_parcel_pca, fit_parcel_pca, plot_subject_parcel_heatmap,
)
from simnibs_parcel_analysis.summary import mean_labeled_field, weighted_percentile


def make_summary() -> ParcelSummary:
    ids = [f"subj-cat-{number:03d}-001" for number in range(1, 5)]
    index = pd.Index(ids, name="subjid")
    parcels = pd.DataFrame([[1, 2, 4], [2, 5, 3], [4, 1, 2], [3, 6, 2]], index=index, columns=["A", "B", "C"], dtype=float)
    means = pd.DataFrame([[0.5, 1, 2], [1, 3, 2], [3, 0.5, 1], [1, 4, 0.5]], index=index, columns=parcels.columns)
    sizes = pd.DataFrame([[1, 2, 3]] * 4, index=index, columns=parcels.columns, dtype=float)
    return ParcelSummary(build_scan_table(ids), parcels, parcels + 0.25, sizes, sizes.astype(int),
                         pd.DataFrame({"n_rois": [3] * 4}, index=index), "aparc", "volume", "mm3", means)


class TestDemeaningNumerics(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = make_summary()
        self.parcels = self.summary.p95_unweighted
        self.brain_mean = pd.Series([2.2, 3.8, 1.4, 5.1], index=self.parcels.index)

    def test_three_references_reach_pca_without_rescaling(self) -> None:
        expected = {
            "brain_mean": self.parcels.sub(self.brain_mean, axis=0),
            "parcel_p95_mean": self.parcels.sub(self.parcels.mean(axis=1), axis=0),
            "parcel_mean": self.parcels - self.summary.parcel_mean_e,
        }
        for reference, matrix in expected.items():
            with self.subTest(reference=reference):
                result = fit_demeaned_parcel_pca(self.parcels, demean_by=reference,
                                                global_mean_brain_e=self.brain_mean, parcel_mean_e=self.summary.parcel_mean_e)
                pd.testing.assert_frame_equal(result.pca_input, matrix)
                np.testing.assert_allclose(result.model.mean_, matrix.mean(axis=0))
                np.testing.assert_allclose(result.model.inverse_transform(result.scores), matrix, atol=1e-12)
                np.testing.assert_allclose(result.variance.explained_variance_ratio, PCA().fit(matrix).explained_variance_ratio_)
        self.assertTrue((expected["brain_mean"] < 0).any().any())
        self.assertFalse(np.allclose(expected["brain_mean"].mean(axis=1), 0))

    def test_original_global_p95_weighted_formula_is_preserved(self) -> None:
        weights = pd.Series([2, 3, 4, 5], index=self.parcels.index, dtype=float)
        result = fit_parcel_pca(self.parcels, weights)
        relative = self.parcels.div(self.parcels.mean(axis=1), axis=0)
        expected = relative.sub(relative.mean(axis=1), axis=0).mul(weights, axis=0)
        pd.testing.assert_frame_equal(result.global_p95_weighted_parcels, expected, check_names=False)
        np.testing.assert_allclose(result.scores, PCA(svd_solver="full").fit_transform(expected))

    def test_original_one_argument_call_is_compatible(self) -> None:
        result = fit_parcel_pca(self.parcels)
        np.testing.assert_allclose(result.pca_input, self.parcels.div(self.parcels.mean(axis=1), axis=0) - 1)
        self.assertIsNone(result.global_p95_e)

    def test_subject_scaling_is_retained_by_raw_demeaning(self) -> None:
        before = fit_demeaned_parcel_pca(self.parcels, demean_by="parcel_p95_mean")
        after = fit_demeaned_parcel_pca(self.parcels * 3, demean_by="parcel_p95_mean")
        np.testing.assert_allclose(after.pca_input, before.pca_input * 3)

    def test_reordered_reference_labels_align_without_mutating_inputs(self) -> None:
        original = self.summary.parcel_mean_e.copy(deep=True)
        shuffled = original.iloc[::-1, ::-1]
        result = fit_demeaned_parcel_pca(self.parcels, demean_by="parcel_mean", parcel_mean_e=shuffled)
        pd.testing.assert_frame_equal(result.pca_input, self.parcels - original)
        pd.testing.assert_frame_equal(self.summary.parcel_mean_e, original)
        result = fit_demeaned_parcel_pca(self.parcels, demean_by="brain_mean", global_mean_brain_e=self.brain_mean.iloc[::-1])
        pd.testing.assert_frame_equal(result.pca_input, self.parcels.sub(self.brain_mean, axis=0))

    def test_invalid_or_missing_references_fail_explicitly(self) -> None:
        with self.assertRaises(ValueError):
            fit_demeaned_parcel_pca(self.parcels, demean_by="unknown")
        for reference in ("brain_mean", "parcel_mean"):
            with self.subTest(reference=reference), self.assertRaises(TypeError):
                fit_demeaned_parcel_pca(self.parcels, demean_by=reference)
        for bad in (self.brain_mean.iloc[:-1], pd.concat([self.brain_mean, self.brain_mean.iloc[:1]]),
                    self.brain_mean.rename(index={self.brain_mean.index[0]: "extra"}), self.brain_mean * np.nan,
                    self.brain_mean * -1):
            with self.subTest(reference=bad), self.assertRaises(ValueError):
                fit_demeaned_parcel_pca(self.parcels, demean_by="brain_mean", global_mean_brain_e=bad)
        for bad in (self.summary.parcel_mean_e.drop(columns="A"), self.summary.parcel_mean_e * np.inf):
            with self.assertRaises(ValueError):
                fit_demeaned_parcel_pca(self.parcels, demean_by="parcel_mean", parcel_mean_e=bad)

    def test_zero_between_scan_variance_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "between-scan variance"):
            fit_demeaned_parcel_pca(self.parcels, demean_by="parcel_mean", parcel_mean_e=self.parcels)

    def test_heatmap_uses_signed_input_and_matching_label(self) -> None:
        result = fit_demeaned_parcel_pca(self.parcels, demean_by="brain_mean", global_mean_brain_e=self.brain_mean)
        figure, axes = plot_subject_parcel_heatmap(result)
        try:
            np.testing.assert_allclose(axes.images[0].get_array(), result.pca_input)
            self.assertIn("mean E across selected brain tetrahedra", axes.get_title())
        finally:
            plt.close(figure)


class TestRawFieldMeans(unittest.TestCase):
    def test_percentile_translation_equivalence_for_both_p95_definitions(self) -> None:
        values = np.array([1.0, 2.0, 4.0, 30.0])
        weights = np.array([1.0, 2.0, 4.0, 1.0])
        mean = values.mean()
        self.assertAlmostEqual(np.percentile(values - mean, 95), np.percentile(values, 95) - mean)
        self.assertAlmostEqual(weighted_percentile(values - mean, weights, 95), weighted_percentile(values, weights, 95) - mean)

    def test_empty_parcel_mean_is_not_imputed(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty parcel"):
            mean_labeled_field(np.array([1.0, 2.0]), np.array([1, 1]), {1: "A", 2: "B"})

    def test_volume_and_surface_extract_actual_sample_means(self) -> None:
        field = SimpleNamespace(value=np.array([1.0, 9.0, 2.0, 6.0]), field_name="magnE_mean")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subject = "subj-cat-001-001"
            hdf5 = root / f"{subject}.hdf5"
            hdf5.touch()
            atlas_path = root / subject / "mri" / "aparc+aseg.mgz"
            atlas_path.parent.mkdir(parents=True)
            atlas_path.touch()
            data = np.array([10, 10, 17, 17]).reshape(4, 1, 1)
            mesh = SimpleNamespace(field={"magnE_mean": field}, elmdata=[field], nodedata=[],
                                   elm=SimpleNamespace(nr=4, elm_type=np.array([4] * 4)),
                                   elements_baricenters=lambda: np.column_stack([np.arange(4), np.zeros((4, 2))]),
                                   elements_volumes_and_areas=lambda: np.array([1.0, 9.0, 1.0, 3.0]))
            volume = build_volume_roi_summary([hdf5], root, mesh_loader=lambda *args: mesh,
                                              atlas_loader=lambda path: (data, np.eye(4)),
                                              lut={10: "Left-Thalamus", 17: "Left-Hippocampus"}, min_elements_per_roi=1)
            self.assertEqual(volume.parcel_mean_e.loc[subject, "Left-Thalamus"], 5.0)
            self.assertEqual(volume.parcel_mean_e.loc[subject, "Left-Hippocampus"], 4.0)
            surface_file = root / "surface.msh"
            surface_file.touch()
            surface_mesh = SimpleNamespace(field={"magnE_mean": field}, elmdata=[], nodedata=[field],
                                           nodes=SimpleNamespace(nr=4), nodes_areas=lambda: np.array([1.0, 9.0, 1.0, 3.0]))
            surface = build_surface_roi_summary([hdf5], surface_files=[surface_file], mesh_loader=lambda path: surface_mesh,
                                                atlas_loader=lambda name: {"A": [1, 1, 0, 0], "B": [0, 0, 1, 1]}, min_nodes_per_roi=1)
            np.testing.assert_allclose(surface.parcel_mean_e.loc[subject], [5, 4])

    def test_global_mean_uses_selected_brain_tetrahedra_and_preserves_p95(self) -> None:
        field = SimpleNamespace(value=np.array([1.0, 9.0, 100.0, 50.0]), field_name="magnE_mean")
        mesh = SimpleNamespace(field={"magnE_mean": field}, elmdata=[field],
                               elm=SimpleNamespace(nr=4, tag1=np.array([1, 2, 5, 1]), elm_type=np.array([4, 4, 4, 2])))
        reader = unittest.mock.Mock(return_value=mesh)
        fake_simnibs = SimpleNamespace(Msh=SimpleNamespace(read_hdf5=reader))
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"simnibs": fake_simnibs}):
            path = Path(directory) / "subj-cat-001-001.hdf5"
            path.touch()
            with contextlib.redirect_stdout(io.StringIO()):
                statistics = wrapper.build_global_e_statistics([path], mesh_key="custom_mesh")
                old_result = wrapper.build_global_p95_e([path])
                restricted = wrapper.build_global_e_statistics([path], brain_tags=(2,))
            reader.assert_any_call(str(path), "custom_mesh")
            self.assertEqual(statistics.iloc[0].global_mean_brain_E, 5.0)
            self.assertAlmostEqual(statistics.iloc[0].global_p95_E, np.percentile([1.0, 9.0], 95))
            pd.testing.assert_series_equal(old_result, statistics.global_p95_E)
            self.assertEqual(restricted.iloc[0].global_mean_brain_E, 9.0)

    def test_pooling_keeps_mean_rows_aligned(self) -> None:
        full = make_summary()
        def subset(start: int, end: int) -> ParcelSummary:
            fields = {name: getattr(full, name).iloc[start:end].copy() for name in
                      ("scans", "p95_unweighted", "p95_spatial_weighted", "parcel_size", "parcel_count", "qc", "parcel_mean_e")}
            return replace(full, **fields)
        first, second = subset(0, 2), subset(2, 4)
        combined = wrapper.combine_parcel_summaries([second, first])
        pd.testing.assert_frame_equal(combined.parcel_mean_e, pd.concat([second.parcel_mean_e, first.parcel_mean_e]))
        with self.assertRaisesRegex(ValueError, "with and without"):
            wrapper.combine_parcel_summaries([first, replace(second, parcel_mean_e=None)])


class TestDemeaningIntegration(unittest.TestCase):
    def test_existing_and_added_runs_export_distinct_correct_inputs(self) -> None:
        summary = make_summary()
        ids = summary.scans.subjid.tolist()
        cognitive = pd.DataFrame({"subjid": ids, "date_start": ["2025-01-01"] * 4, "cgi_change": [-1, -2, -4, -3]})
        statistics = pd.DataFrame({"global_p95_E": [3.0, 5.0, 6.0, 7.0], "global_mean_brain_E": [2.2, 3.8, 1.4, 5.1]},
                                  index=summary.p95_unweighted.index)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"{subject}.hdf5" for subject in ids]
            sources = {path: root for path in paths}
            for path in paths:
                path.touch()
            patches = [patch.object(wrapper, "build_global_e_statistics", return_value=statistics),
                       patch.object(wrapper, "inspect_hdf5_mesh", return_value={}),
                       patch.object(wrapper, "build_volume_roi_summary", side_effect=lambda *args, **kw: replace(summary, atlas_name=kw["atlas_name"])),
                       patch.object(wrapper, "build_surface_roi_summary", return_value=replace(summary, atlas_name="HCP_MMP1", domain="surface", size_unit="mm2")),
                       patch("matplotlib.figure.Figure.savefig")]
            with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
                for item in patches:
                    stack.enter_context(item)
                results = wrapper.run_all_atlases_global_p95_weighted(paths, cognitive, root / "results",
                                                                      subjects_dir_by_hdf5=sources, demean_references=DEMEAN_REFERENCES)
                # Defaults still produce the original six results only.
                defaults = wrapper.run_all_atlases_global_p95_weighted(paths, cognitive, root / "defaults", subjects_dir_by_hdf5=sources)
            self.assertEqual(len(results), 24)
            self.assertEqual(len(defaults), 6)
            for key, old_result in defaults.items():
                pd.testing.assert_frame_equal(results[key].pca.pca_input, old_result.pca.pca_input)
                pd.testing.assert_frame_equal(results[key].correlations, old_result.correlations)
            for prefix in ("volume_aparc", "volume_a2009s", "surface_HCP_MMP1"):
                for label, method, folder in (("weighted", "spatial_weighted", "weighted_p95"),
                                              ("unweighted", "unweighted", "unweighted_p95_sensitivity")):
                    for reference in DEMEAN_REFERENCES:
                        with self.subTest(atlas=prefix, method=method, reference=reference):
                            result = results[f"{prefix}_{label}_demean_{reference}"]
                            output = root / "results" / prefix / folder / f"demean_{reference}"
                            exported = pd.read_csv(output / "parcel_pca_input.csv", index_col=0)
                            np.testing.assert_allclose(exported, result.pca.pca_input)
                            references = pd.read_csv(output / "parcel_demeaning_reference_E.csv", index_col=0)
                            np.testing.assert_allclose(result.pca.pca_input + references, summary.p95(method))
                            pd.testing.assert_frame_equal(result.global_metrics, summary.global_metrics(method))
                            settings = json.loads((output / "analysis_settings.json").read_text())
                            self.assertEqual(settings["demeaning_reference"], reference)
                            self.assertFalse(settings["global_p95_weighting"]["enabled"])
                            self.assertFalse(settings["pca_divide_by_global_mean"])
                            self.assertEqual(set(settings["components_by_threshold"]), {"0.96", "0.97", "0.98", "0.99"})
                            self.assertEqual(len(result.outcomes), 4)


if __name__ == "__main__":
    unittest.main()
