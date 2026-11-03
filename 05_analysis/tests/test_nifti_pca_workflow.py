"""Exercise series-to-voxel PCA and saved-master correlation boundaries."""

from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nifti_pca.workflow import artifact_paths, run_analysis
from run_mni_nifti_pca import parse_args


class TestNiftiPCAWorkflow(unittest.TestCase):
    def test_prepare_saves_variance_and_variance_stage_needs_only_saved_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            mask_path = directory / "mask.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((1, 1, 2)), np.eye(4)), mask_path)
            rows = []
            for index, values in enumerate(([1, 1], [2, 1], [1, 3], [3, 2])):
                path = directory / f"field_{index}.nii.gz"
                nib.save(nib.Nifti1Image(np.array(values, dtype=float).reshape(1, 1, 2), np.eye(4)), path)
                rows.append({
                    "person_id": f"person_{index}", "series_num": 1,
                    "model_type": "model_a", "step": "step_0",
                    "field_name": "magnE_mean", "full_file_path": str(path),
                    "cgi_change": index - 1,
                })
            series_path = directory / "series.csv"
            pd.DataFrame(rows).to_csv(series_path, index=False)
            args = parse_args([
                "--stage", "prepare", "--analysis-name", "variance_only",
                "--series-csv", str(series_path), "--brain-mask", str(mask_path),
                "--n-components", "1", "--dpi", "40",
            ])
            with patch.dict(os.environ, {"RESULTS_DIR": str(directory / "results")}):
                with patch("nifti_pca.workflow.correlate_analysis") as correlate, patch(
                    "nifti_pca.plotting._save_component_map",
                ) as component_map:
                    analysis_dir = run_analysis(args)
                    correlate.assert_not_called()
                    component_map.assert_not_called()
                manifest_path = analysis_dir / "run_manifest.json"
                self.assertEqual(
                    set(json.loads(manifest_path.read_text())["stages"]),
                    {"prepare", "variance"},
                )
                bases = pd.read_csv(analysis_dir / "data/basis_manifest.csv")
                self.assertEqual(set(bases["normalization"]), {"raw", "brain_mean"})
                plots = pd.read_csv(analysis_dir / "plots/variance_plot_manifest.csv")
                self.assertEqual(len(plots), 4)
                self.assertEqual(plots["basis_id"].nunique(), 2)
                for path in plots["path"]:
                    self.assertTrue(Path(path).is_file())

                # A saved-basis plot must not depend on the clinical master,
                # original fields, model object, or any NIfTI artifacts.
                for mode in args.normalizations:
                    paths = artifact_paths(analysis_dir, "magnE_mean", mode)
                    paths["master"].unlink()
                    paths["scores"].unlink()
                    self.assertFalse(paths["permutations"].exists())
                for path in bases["model_path"]:
                    Path(path).unlink()
                for path in list(directory.glob("*.nii.gz")) + list(analysis_dir.rglob("*.nii.gz")):
                    path.unlink()
                series_path.unlink()
                variance_mtimes = {
                    path: Path(path).stat().st_mtime_ns for path in bases["variance_path"]
                }
                raw_figure = analysis_dir / "plots/raw/model_a/step_0/explained_variance.png"
                raw_figure.unlink()
                normalized_figure = next(
                    Path(path) for path in plots.loc[
                        plots["normalization"].eq("brain_mean")
                        & plots["type"].eq("explained_variance"), "path"
                    ]
                )
                normalized_mtime = normalized_figure.stat().st_mtime_ns
                full_manifest = analysis_dir / "plots/plot_manifest.csv"
                full_manifest.write_text("preserve full plotting records\n")
                args = parse_args([
                    "--stage", "variance", "--analysis-name", "variance_only",
                    "--normalizations", "raw", "--dpi", "40",
                    "--background-path", str(directory / "missing_background.nii.gz"),
                ])
                with patch("nifti_pca.workflow.fit_pca") as fit, patch(
                    "nifti_pca.workflow.load_masked_fields",
                ) as load_fields, patch("nifti_pca.workflow.correlate_analysis") as correlate, patch(
                    "nifti_pca.plotting._save_component_map",
                ) as component_map:
                    self.assertEqual(run_analysis(args), analysis_dir)
                    for operation in (fit, load_fields, correlate, component_map):
                        operation.assert_not_called()
                self.assertTrue(raw_figure.is_file())
                self.assertEqual(normalized_figure.stat().st_mtime_ns, normalized_mtime)
                self.assertEqual(full_manifest.read_text(), "preserve full plotting records\n")
                for path, mtime in variance_mtimes.items():
                    self.assertEqual(Path(path).stat().st_mtime_ns, mtime)
                plots = pd.read_csv(analysis_dir / "plots/variance_plot_manifest.csv")
                self.assertEqual(set(plots["normalization"]), {"raw"})

    def test_changed_master_blocks_stale_plots_and_prepare_invalidates_correlation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            mask_path = directory / "mask.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((1, 1, 2)), np.eye(4)), mask_path)
            rows = []
            for index, values in enumerate(([1, 1], [2, 1], [1, 3], [1, 1])):
                path = directory / f"field_{index}.nii.gz"
                nib.save(nib.Nifti1Image(np.array(values, dtype=float).reshape(1, 1, 2), np.eye(4)), path)
                rows.append({
                    "person_id": f"person_{index}", "series_num": 1,
                    "model_type": "model_a", "step": "step_0",
                    "field_name": "magnE_mean", "full_file_path": str(path),
                    "cgi_change": index - 1,
                })
            csv_path = directory / "series.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            args = parse_args([
                "--analysis-name", "stale_regression", "--series-csv", str(csv_path),
                "--brain-mask", str(mask_path), "--normalizations", "raw",
                "--n-components", "1", "--n-permutations", "0",
            ])
            with patch.dict(os.environ, {"RESULTS_DIR": str(directory / "results")}):
                analysis_dir = run_analysis(args)
                manifest_path = analysis_dir / "run_manifest.json"
                self.assertIn("correlate", json.loads(manifest_path.read_text())["stages"])
                paths = artifact_paths(analysis_dir, "magnE_mean", "raw")
                master = pd.read_csv(paths["master"])
                changed_sample = master["sample_id"].iloc[0]
                master.loc[master["sample_id"].eq(changed_sample), "cgi_change"] += 1
                master.to_csv(paths["master"], index=False)
                args.stage = "plot"
                with patch("nifti_pca.plotting.plot_basis") as plot_basis:
                    with self.assertRaisesRegex(ValueError, "PC master changed"):
                        run_analysis(args)
                    plot_basis.assert_not_called()
                self.assertTrue((analysis_dir / "plots/raw/model_a/step_0/explained_variance.png").is_file())
                self.assertFalse((analysis_dir / "plots/plot_manifest.csv").exists())
                args.stage = "prepare"
                run_analysis(args)
                stages = json.loads(manifest_path.read_text())["stages"]
                self.assertIn("prepare", stages)
                self.assertIn("variance", stages)
                self.assertNotIn("correlate", stages)
                self.assertNotIn("plot", stages)

    def test_roi_manifest_to_grouped_bases_and_saved_master_correlations(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = directory / "inputs"
            inputs.mkdir()
            mask_path = inputs / "mask.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), mask_path)
            sample_fields = [0, 0, 1, 2, 3]
            people = ["person_a", "person_a", "person_b", "person_c", "person_d"]
            series = [1, 2, 1, 1, 1]
            outcomes = [3, 1, -1, 2, 0]
            rows = []
            for model_index, model_type in enumerate(["model_a", "model_b"]):
                for step_index, step in enumerate(["step_0", "step_9"]):
                    file_paths = []
                    for image_index in range(4):
                        field = (
                            np.arange(8, dtype=np.float32) + 10
                            + (image_index + model_index) * np.array([1, 0, 2, 1, 2, 0, 1, 0])
                            + (image_index ** 2 + step_index) * np.array([0, 1, 0, 2, 1, 1, 0, 2])
                        ).reshape(2, 2, 2)
                        path = inputs / f"{model_type}_{step}_{image_index}.nii.gz"
                        nib.save(nib.Nifti1Image(field, np.eye(4)), path)
                        file_paths.append(path)
                    for sample_index, image_index in enumerate(sample_fields):
                        for roi in (1, 2):
                            rows.append({
                                "person_id": people[sample_index],
                                "series_num": series[sample_index],
                                "model_type": model_type,
                                "step": step,
                                "field_name": "magnE_mean",
                                "full_file_path": file_paths[image_index].name,
                                "cgi_change": outcomes[sample_index],
                                "clinical_note": f"series metadata {sample_index}",
                                "roi": roi,
                                "mean": roi * 100,  # Never used for voxel PCA.
                                "whole_brain_mean": -999,  # Recomputed from NIfTI.
                            })
            csv_path = inputs / "roi_manifest.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            args = parse_args([
                "--analysis-name", "synthetic_pca",
                "--series-csv", str(csv_path), "--brain-mask", str(mask_path),
                "--n-components", "1", "--n-permutations", "19", "--batch-size", "7",
            ])
            with patch.dict(os.environ, {"RESULTS_DIR": str(directory / "results")}):
                analysis_dir = run_analysis(args)
                self.assertEqual(analysis_dir, directory / "results" / "synthetic_pca")
                samples = pd.read_csv(analysis_dir / "data/series_inputs.csv")
                self.assertEqual(len(samples), 20)
                self.assertTrue(samples["sample_id"].is_unique)
                self.assertNotIn("roi", samples)
                bases = pd.read_csv(analysis_dir / "data/basis_manifest.csv")
                self.assertEqual(len(bases), 8)
                self.assertTrue(bases["basis_id"].is_unique)
                self.assertTrue(bases["n_unique_fields"].eq(4).all())
                self.assertTrue(bases["n_series"].eq(5).all())
                self.assertTrue(bases["n_components_tested"].eq(1).all())
                model_mtimes = {path: Path(path).stat().st_mtime_ns for path in bases["model_path"]}

                for mode in ["raw", "brain_mean"]:
                    paths = artifact_paths(analysis_dir, "magnE_mean", mode)
                    master = pd.read_csv(paths["master"])
                    wide = pd.read_csv(paths["scores"])
                    correlations = pd.read_csv(paths["permutations"])
                    self.assertEqual(len(wide), 20)
                    self.assertEqual(len(correlations), 4)
                    self.assertTrue(correlations["component"].eq("PC1").all())
                    self.assertTrue(correlations["n"].eq(5).all())
                    self.assertTrue(correlations["permutation_n_resamples_actual"].eq(19).all())
                    self.assertTrue(correlations["p_permutation"].between(0, 1).all())
                    self.assertIn("samples_assumed_independent", correlations)
                    self.assertNotIn("samples_assumed_independent_x", correlations)
                    self.assertTrue(master["whole_brain_mean"].gt(0).all())
                    self.assertIn("clinical_note", master)
                    self.assertEqual(set(master["component"]), {"PC1", "PC2"})
                    self.assertEqual(len(master), 40)
                    for basis_id, group in master.groupby("basis_id"):
                        reference = group.iloc[0]
                        fitting = pd.read_csv(Path(reference["pca_settings_path"]).parent / "fitting_samples.csv")
                        values = np.stack([
                            nib.load(path).get_fdata().reshape(-1)
                            for path in fitting["full_file_path"]
                        ])
                        if mode == "brain_mean":
                            values /= values.mean(axis=1, keepdims=True)
                        model = joblib.load(reference["pca_model_path"])
                        expected_scores = model.transform(values)
                        saved_scores = wide.set_index("sample_id").loc[
                            fitting["sample_id"], ["PC1", "PC2"]
                        ].to_numpy()
                        np.testing.assert_allclose(saved_scores, expected_scores, atol=1e-12)
                        selected = group.loc[group["component"].eq("PC1")]
                        expected_rho, expected_p = spearmanr(selected["score"], selected["cgi_change"])
                        result = correlations.loc[correlations["basis_id"].eq(basis_id)].iloc[0]
                        self.assertAlmostEqual(result["rho"], expected_rho)
                        self.assertAlmostEqual(result["p_value"], expected_p)
                        self.assertEqual(selected["sample_id"].nunique(), 5)
                        self.assertEqual(selected["person_id"].nunique(), 4)

                # Correlate resumes from stored scores and selection flags, even
                # if a later CLI invocation requests a different PCA selection.
                args.stage = "correlate"
                args.n_permutations = 0
                args.n_components = 2
                run_analysis(args)
                for mode in ["raw", "brain_mean"]:
                    paths = artifact_paths(analysis_dir, "magnE_mean", mode)
                    correlations = pd.read_csv(paths["permutations"])
                    self.assertEqual(len(correlations), 4)
                    self.assertTrue(correlations["component"].eq("PC1").all())
                    self.assertTrue(correlations["p_permutation"].isna().all())
                for path, previous_mtime in model_mtimes.items():
                    self.assertEqual(Path(path).stat().st_mtime_ns, previous_mtime)


if __name__ == "__main__":
    unittest.main()
