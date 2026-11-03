"""Small artifact-level checks for saved PCA plotting."""

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nifti_pca import plotting as pca_plotting


class TestNiftiPcaPlotting(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / "results"
        self.output = self.results / "example" / "plots"
        environment = patch.dict(os.environ, {"RESULTS_DIR": str(self.results)})
        environment.start()
        self.addCleanup(environment.stop)
        self.variance = pd.DataFrame({
            "component": ["PC1", "PC2"],
            "component_index": [1, 2],
            "explained_variance": [3.0, 1.0],
            "explained_variance_ratio": [0.75, 0.25],
            "cumulative_variance_ratio": [0.75, 1.0],
            "selected_for_correlation": [True, False],
        })
        self.master = pd.DataFrame({
            "component": ["PC1"] * 5 + ["PC2"] * 5,
            "score": [-2, -1, 0, 1, 2, 0, 1, -1, 1, -1],
            "cgi_change": [-3, -1, -0.5, 0, 2] * 2,
            "person_id": ["a", "a", "b", "c", "d"] * 2,
            "series_num": [1, 2, 1, 1, 1] * 2,
            "basis_id": ["skin_step0_raw"] * 10,
            "model_type": ["skin"] * 10,
            "step": ["step_0"] * 10,
            "field_name": ["magnE"] * 10,
            "normalization": ["raw"] * 10,
        })
        self.correlations = pd.DataFrame({
            "component": ["PC1"], "rho": [1.0], "p_value": [0.0],
            "p_permutation": [0.08], "null_distribution": ["[0.5, -0.5]"],
        })
        weights = np.zeros((9, 9, 9), dtype=np.float32)
        weights[2:5, 2:5, 2:5] = 0.005
        weights[5:7, 5:7, 5:7] = -0.02
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = -8
        self.map_path = self.root / "PC1.nii.gz"
        nib.save(nib.Nifti1Image(weights, affine), self.map_path)

    def test_variance_plot_uses_all_saved_pcs_in_numeric_order(self):
        variance = pd.DataFrame({
            "component": ["PC3", "PC1", "PC2"],
            "component_index": ["3", "1", "2"],
            "explained_variance": [1.0, 6.0, 3.0],
            "explained_variance_ratio": [0.1, 0.6, 0.3],
            "cumulative_variance_ratio": [1.0, 0.6, 0.9],
            "selected_for_correlation": ["False", "True", "0"],
        })
        original = variance.copy(deep=True)
        with patch.object(
            pca_plotting.plt, "close", wraps=pca_plotting.plt.close,
        ) as close:
            manifest = pca_plotting.plot_variance(
                variance=variance, output_dir=self.output, dpi=40,
                metadata={"basis_id": "saved_basis", "normalization": "raw"},
            )
        figure = next(
            call.args[0] for call in close.call_args_list
            if call.args and isinstance(call.args[0], pca_plotting.plt.Figure)
        )
        bars, cumulative = figure.axes
        np.testing.assert_allclose(
            [bar.get_height() for bar in bars.patches], [60, 30, 10],
        )
        self.assertEqual(
            [tick.get_text() for tick in bars.get_xticklabels()],
            ["PC1", "PC2", "PC3"],
        )
        np.testing.assert_allclose(
            [bar.get_facecolor() for bar in bars.patches],
            [to_rgba("#2676a5"), to_rgba("#bfc4c8"), to_rgba("#bfc4c8")],
        )
        np.testing.assert_allclose(cumulative.lines[0].get_ydata(), [60, 90, 100])
        source = pd.read_csv(self.output / "variance_source.csv")
        self.assertEqual(source["component"].tolist(), ["PC1", "PC2", "PC3"])
        self.assertEqual(source["selected_for_correlation"].tolist(), [True, False, False])
        self.assertTrue(source["basis_id"].eq("saved_basis").all())
        self.assertEqual(set(manifest["type"]), {"variance_source", "explained_variance"})
        self.assertTrue((self.output / "explained_variance.png").is_file())
        self.assertTrue((self.output / "variance_plot_manifest.csv").is_file())
        pd.testing.assert_frame_equal(variance, original)
        self.assertEqual(pca_plotting.plt.get_fignums(), [])

    def test_invalid_variance_is_rejected_before_writing(self):
        cases = {
            "missing_column": self.variance.drop(columns="explained_variance_ratio"),
            "empty": self.variance.iloc[:0],
            "duplicate_pc": self.variance.assign(component=["PC1", "PC1"]),
            "nonfinite_ratio": self.variance.assign(explained_variance_ratio=[np.nan, 0.25]),
            "fractional_index": self.variance.assign(component_index=[1, 1.5]),
            "invalid_selection": self.variance.assign(selected_for_correlation=["yes", "False"]),
            "mixed_basis": self.variance.assign(basis_id=["first", "second"]),
        }
        for name, variance in cases.items():
            with self.subTest(case=name):
                output = self.output / name
                with self.assertRaises(ValueError):
                    pca_plotting.plot_variance(variance=variance, output_dir=output)
                self.assertFalse(output.exists())
        self.assertEqual(pca_plotting.plt.get_fignums(), [])

    def test_variance_plot_validates_all_output_destinations(self):
        for filename in (
            "variance_source.csv", "explained_variance.png", "variance_plot_manifest.csv",
        ):
            with self.subTest(filename=filename):
                output = self.output / filename.replace(".", "_")
                output.mkdir(parents=True)
                outside = self.root / filename
                outside.write_text("keep this file")
                (output / filename).symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "below RESULTS_DIR"):
                    pca_plotting.plot_variance(
                        variance=self.variance, output_dir=output, dpi=40,
                    )
                self.assertEqual(outside.read_text(), "keep this file")
        self.assertEqual(pca_plotting.plt.get_fignums(), [])

    def test_artifacts_and_signed_weight_scale(self):
        with patch.object(
            pca_plotting.plotting, "plot_stat_map",
            wraps=pca_plotting.plotting.plot_stat_map,
        ) as render:
            manifest = pca_plotting.plot_basis(
                variance=self.variance, master=self.master,
                correlations=self.correlations,
                component_maps={"PC1": self.map_path},
                background_path=None, output_dir=self.output, dpi=40,
            )
        self.assertAlmostEqual(render.call_args.kwargs["vmax"], 0.02, places=7)
        self.assertTrue(render.call_args.kwargs["symmetric_cbar"])
        self.assertIsNone(render.call_args.kwargs["bg_img"])
        for artifact in manifest["path"]:
            artifact = Path(artifact)
            self.assertTrue(artifact.is_file())
            self.assertTrue(artifact.is_relative_to(self.output))
        self.assertEqual(
            set(manifest["component"].dropna()), {"PC1"}
        )
        self.assertFalse((self.output / "PC2_signed_weights.png").exists())
        source = pd.read_csv(self.output / "PC1_score_source.csv")
        self.assertEqual(source["used_in_scatter"].sum(), 5)
        self.assertEqual(source["person_id"].tolist(), ["a", "a", "b", "c", "d"])
        saved = pd.read_csv(self.output / "PC1_correlation_source.csv")
        self.assertEqual(saved.loc[0, "null_distribution"], "[0.5, -0.5]")
        self.assertTrue((self.output / "plot_manifest.csv").is_file())
        self.assertEqual(len(pca_plotting.plt.get_fignums()), 0)

    def test_mixed_bases_rejected_before_writing(self):
        mixed = self.master.copy()
        mixed.loc[0, "basis_id"] = "another_basis"
        with self.assertRaisesRegex(ValueError, "multiple basis_id"):
            pca_plotting.plot_basis(
                variance=self.variance, master=mixed,
                correlations=self.correlations,
                component_maps={"PC1": self.map_path},
                background_path=None, output_dir=self.output,
            )
        self.assertFalse(self.output.exists())

    def test_output_outside_results_rejected(self):
        with self.assertRaisesRegex(ValueError, "below RESULTS_DIR"):
            pca_plotting.plot_basis(
                variance=self.variance, master=self.master,
                correlations=self.correlations,
                component_maps={"PC1": self.map_path},
                background_path=None, output_dir=self.root / "wrong",
            )

    def test_nonfinite_master_rejected_before_writing(self):
        for column in ("score", "cgi_change"):
            for value in (np.nan, np.inf):
                with self.subTest(column=column, value=value):
                    invalid = self.master.copy()
                    invalid[column] = invalid[column].astype(float)
                    invalid.loc[0, column] = value
                    with self.assertRaisesRegex(ValueError, f"master.{column}"):
                        pca_plotting.plot_basis(
                            variance=self.variance, master=invalid,
                            correlations=self.correlations,
                            component_maps={"PC1": self.map_path},
                            background_path=None, output_dir=self.output,
                        )
                    self.assertFalse(self.output.exists())

    def test_symlink_csv_and_png_destinations_rejected(self):
        for filename in ("variance_source.csv", "explained_variance.png"):
            with self.subTest(filename=filename):
                output = self.output / Path(filename).suffix.lstrip(".")
                output.mkdir(parents=True)
                outside = self.root / filename
                outside.write_text("keep this file")
                (output / filename).symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "below RESULTS_DIR"):
                    pca_plotting.plot_basis(
                        variance=self.variance, master=self.master,
                        correlations=self.correlations,
                        component_maps={"PC1": self.map_path},
                        background_path=None, output_dir=output, dpi=40,
                    )
                self.assertEqual(outside.read_text(), "keep this file")
                self.assertEqual(len(pca_plotting.plt.get_fignums()), 0)

    def test_correlation_bars_fall_back_to_analytic_p(self):
        correlations = self.correlations.copy()
        correlations["p_permutation"] = np.nan
        correlations["p_value"] = 0.01
        self.output.mkdir(parents=True)
        try:
            with patch.object(pca_plotting.plt, "close"):
                pca_plotting._save_correlation_bars(
                    correlations, self.output / "analytic.png", "", 40
                )
                labels = [text.get_text() for text in
                          pca_plotting.plt.gcf().axes[0].texts]
            self.assertIn("p=0.01", labels)
        finally:
            pca_plotting.plt.close("all")


if __name__ == "__main__":
    unittest.main()
