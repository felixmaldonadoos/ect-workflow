"""Artifact checks for RESULTS_DIR ROI normalization and plotting outputs."""

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from networks import mni_yeo2011_normalize_brain as normalizer
from networks import mni_yeo2011_plot_correlations as roi_plotting


class _Display:
    def savefig(self, filename, **kwargs):
        Path(filename).write_bytes(b"synthetic plot")

    def close(self):
        pass


class TestRoiNormalizationPaths(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / "results"
        environment = patch.dict(os.environ, {"RESULTS_DIR": str(self.results)})
        environment.start()
        self.addCleanup(environment.stop)
        self.analysis = self.results / "example"

    def test_legacy_input_uses_canonical_results_output(self):
        field_path = self.root / "field.nii.gz"
        mask_path = self.root / "mask.nii.gz"
        field = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
        mask = np.zeros_like(field)
        mask[0] = 10
        nib.save(nib.Nifti1Image(field, np.eye(4)), field_path)
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
        source_path = self.root / "legacy_master.csv"
        master = pd.DataFrame({
            "field_name": ["magnE_mean"] * 2,
            "full_file_path": [str(field_path)] * 2,
            "mean": [3.0, 6.0],
            "n_voxels": [2, 4],
            "person_id": ["same_patient"] * 2,
            "series_num": [1, 2],
        })
        master.to_csv(source_path, index=False)
        normalizer.main([
            "--analysis-name", "example", "--master-path", str(source_path),
            "--brain-mask", str(mask_path),
        ])
        output = self.analysis / "data" / (
            "yeo2011_magnE_mean_series_ROI_master_"
            "normed_brain_magnE_mean_mean.csv"
        )
        saved = pd.read_csv(output)
        np.testing.assert_allclose(saved["mean"], [2, 4])
        np.testing.assert_allclose(saved["mean_raw"], [3, 6])
        np.testing.assert_allclose(saved["whole_brain_mean"], [1.5, 1.5])
        self.assertEqual(saved["whole_brain_n_voxels"].tolist(), [4, 4])
        self.assertEqual(saved["n_voxels"].tolist(), [2, 4])
        self.assertEqual(saved["series_num"].tolist(), [1, 2])
        self.assertTrue(saved["raw_master_path"].eq(str(source_path)).all())
        pd.testing.assert_frame_equal(pd.read_csv(source_path), master)
        self.assertFalse(source_path.with_name(
            "legacy_master_normed_brain_magnE_mean_mean.csv"
        ).exists())

    def test_reject_output_escape_before_reading_images_or_csv(self):
        for output in (
            self.root / "wrong.csv",
            self.results / "other_analysis" / "wrong.csv",
        ):
            with self.subTest(output=output), patch.object(
                normalizer, "normalize_master"
            ) as normalize, patch.object(normalizer.pd, "read_csv") as read:
                with self.assertRaises(ValueError):
                    normalizer.main([
                        "--analysis-name", "example",
                        "--master-path", str(self.root / "missing.csv"),
                        "--output-path", str(output),
                    ])
                normalize.assert_not_called()
                read.assert_not_called()

    def test_reject_normalized_csv_symlink_before_reading(self):
        output = self.analysis / "data" / "normalized.csv"
        output.parent.mkdir(parents=True)
        outside = self.root / "outside.csv"
        outside.write_text("keep")
        output.symlink_to(outside)
        with patch.object(normalizer, "normalize_master") as normalize:
            with self.assertRaises(ValueError):
                normalizer.main([
                    "--analysis-name", "example",
                    "--master-path", str(self.root / "missing.csv"),
                    "--output-path", str(output),
                ])
            normalize.assert_not_called()
        self.assertEqual(outside.read_text(), "keep")


class TestRoiPlotPaths(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / "results"
        environment = patch.dict(os.environ, {"RESULTS_DIR": str(self.results)})
        environment.start()
        self.addCleanup(environment.stop)
        self.analysis = self.results / "example"
        self.output = self.analysis / "plots" / "raw"
        labels = np.zeros((5, 5, 5), dtype=np.int16)
        labels[:2] = 1
        labels[3:] = 2
        self.atlas = nib.Nifti1Image(labels, np.eye(4))
        self.atlas_path = self.root / "atlas.nii.gz"
        self.template_path = self.root / "template.nii.gz"
        self.lut_path = self.root / "atlas.txt"
        nib.save(self.atlas, self.atlas_path)
        nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), np.eye(4)), self.template_path)
        self.lut_path.write_text(
            "1 7Networks_LH_Vis_1\n2 7Networks_RH_Vis_1\n"
        )
        self.results_frame = roi_plotting.annotate_correlation_features(pd.DataFrame({
            "x": [
                "magnE_mean_7Networks_LH_Vis_1",
                "magnE_mean_7Networks_RH_Vis_1",
                "magnE_mean_Vis_both_hemis",
            ],
            "y": ["cgi_change"] * 3,
            "n": [5] * 3,
            "rho": [0.0, -0.5, 0.75],
            "p_value": [1, 0.4, 0.2],
            "p_permutation": [1, 0.5, 0.3],
            "n_permutations": [9999] * 3,
            "permutation_seed": [42] * 3,
            "field_name": ["magnE_mean"] * 3,
            "metric": ["mean"] * 3,
            "model_type": ["skin"] * 3,
            "step": ["step_0"] * 3,
        }))

    def arguments(self, source):
        return [
            "--analysis-name", "example", "--correlations-path", str(source),
            "--atlas-path", str(self.atlas_path), "--lut-path", str(self.lut_path),
            "--template-path", str(self.template_path), "--no-overviews", "--dpi", "30",
        ]

    def test_raw_and_normalized_artifacts_preserve_source_parameters(self):
        for normalization, tag in (
            ("raw", "raw"),
            ("brain_mean", "normed_brain_magnE_mean_mean"),
        ):
            with self.subTest(normalization=normalization):
                frame = self.results_frame.copy()
                frame["normalization"] = normalization
                frame["normalization_tag"] = tag
                source = self.root / f"legacy_{tag}.csv"
                frame.to_csv(source, index=False)
                with patch.object(
                    roi_plotting.plotting, "plot_stat_map", return_value=_Display()
                ) as render:
                    roi_plotting.main(self.arguments(source))
                self.assertEqual(render.call_count, 2)
                self.assertIsNone(render.call_args.kwargs["threshold"])
                self.assertEqual(render.call_args.kwargs["vmin"], -1)
                self.assertEqual(render.call_args.kwargs["vmax"], 1)
                output = self.analysis / "plots" / tag
                manifest = pd.read_csv(output / "plot_manifest.csv")
                self.assertEqual(len(manifest), 2)
                self.assertTrue(manifest["normalization_tag"].eq(tag).all())
                self.assertTrue(manifest["correlations_path"].eq(str(source)).all())
                for row in manifest.itertuples():
                    for filename in (row.csv, row.png, row.nifti):
                        self.assertTrue(Path(filename).is_file())
                        self.assertTrue(Path(filename).is_relative_to(output))
                    saved = pd.read_csv(row.csv)
                    self.assertTrue(saved["n_permutations"].eq(9999).all())
                    self.assertTrue(saved["permutation_seed"].eq(42).all())
                    np.testing.assert_allclose(
                        saved["p_permutation"],
                        frame.set_index("x").loc[saved["x"], "p_permutation"],
                    )
                    data = nib.load(row.nifti).get_fdata()
                    expected = [0, -0.5] if row.n_features == 2 else [0.75, 0.75]
                    np.testing.assert_allclose(data[[0, 4], 0, 0], expected)
                self.assertEqual(len(roi_plotting.plt.get_fignums()), 0)

    def test_reject_every_map_symlink_before_writing_any_artifacts(self):
        atlas, labels, metadata = roi_plotting.load_label_atlas(
            self.atlas_path, self.lut_path
        )
        jobs, _ = roi_plotting.build_plot_jobs(
            atlas, labels, metadata, "magnE_mean", plot_overviews=False
        )
        for extension in ("csv", "png", "nifti"):
            with self.subTest(extension=extension):
                output = self.output / extension
                paths = roi_plotting.map_output_paths(
                    output, jobs[-1], field_name="magnE_mean", metric="mean",
                    outcome="cgi_change", model_type="skin", step="step_0",
                    analysis_name="example",
                )
                outside = self.root / f"outside_{extension}"
                outside.write_text("keep")
                paths[extension].parent.mkdir(parents=True)
                paths[extension].symlink_to(outside)
                with patch.object(roi_plotting.plotting, "plot_stat_map") as render:
                    with self.assertRaises(ValueError):
                        roi_plotting.plot_correlation_configurations(
                            self.results_frame, atlas_path=self.atlas_path,
                            lut_path=self.lut_path, template_path=self.template_path,
                            output_dir=output, plot_overviews=False,
                            analysis_name="example",
                        )
                    render.assert_not_called()
                self.assertEqual(outside.read_text(), "keep")
                self.assertEqual(len(list(output.rglob("*.*"))), 1)

    def test_reject_manifest_symlink_before_plotting(self):
        self.output.mkdir(parents=True)
        outside = self.root / "outside.csv"
        outside.write_text("keep")
        (self.output / "plot_manifest.csv").symlink_to(outside)
        source = self.root / "legacy.csv"
        self.results_frame.to_csv(source, index=False)
        with patch.object(roi_plotting, "plot_correlation_configurations") as render:
            with self.assertRaises(ValueError):
                roi_plotting.main(self.arguments(source))
            render.assert_not_called()
        self.assertEqual(outside.read_text(), "keep")

    def test_reject_other_analysis_output_before_loading_input(self):
        with patch.object(roi_plotting, "load_correlation_results") as load:
            with self.assertRaises(ValueError):
                roi_plotting.main(self.arguments(self.root / "missing.csv") + [
                    "--output-dir", str(self.results / "different" / "plots"),
                ])
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
