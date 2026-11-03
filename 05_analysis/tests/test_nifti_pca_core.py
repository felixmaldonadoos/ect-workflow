"""Small image and matrix checks for the NIfTI PCA scientific contract."""

from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import joblib
import nibabel as nib
import numpy as np

from nifti_pca.images import load_masked_fields
from nifti_pca.pca import fit_pca, save_pca_artifacts


class TestMaskedFields(unittest.TestCase):
    def test_fixed_voxel_order_means_and_duplicate_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mask = np.zeros((2, 2, 2), dtype=np.float32)
            mask[0] = 10
            mask_path = directory / "mask.nii.gz"
            field_path = directory / "field.nii.gz"
            data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            data[1] = np.nan  # Non-brain NaNs do not alter the feature vectors.
            nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
            nib.save(nib.Nifti1Image(data[..., None], np.eye(4)), field_path)
            with patch("nifti_pca.images.nib.load", wraps=nib.load) as load:
                fields = load_masked_fields([field_path, field_path], mask_path)
            self.assertEqual(load.call_count, 2)  # Mask plus one unique field.
            np.testing.assert_array_equal(fields.values, [[0, 1, 2, 3]] * 2)
            np.testing.assert_allclose(fields.brain_means, [1.5, 1.5])
            self.assertEqual(len(fields.unique_paths), 1)
            np.testing.assert_array_equal(fields.mask_img.get_fdata() > 0, mask > 0)

    def test_grid_mismatch_requires_explicit_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mask_path = directory / "mask.nii.gz"
            field_path = directory / "field.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4)), mask_path)
            affine = np.diag([2.0, 2.0, 2.0, 1.0])
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), affine), field_path)
            with self.assertRaisesRegex(ValueError, "grid does not match"):
                load_masked_fields([field_path], mask_path)
            fields = load_masked_fields([field_path], mask_path, resample=True)
            np.testing.assert_allclose(fields.values, 1)
            np.testing.assert_allclose(fields.brain_means, 1)

    def test_nonfinite_masked_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mask_path = directory / "mask.nii.gz"
            field_path = directory / "field.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), mask_path)
            values = np.ones((2, 2, 2))
            values[0, 0, 0] = np.nan
            nib.save(nib.Nifti1Image(values, np.eye(4)), field_path)
            with self.assertRaisesRegex(ValueError, "non-finite values inside"):
                load_masked_fields([field_path], mask_path)


class TestVoxelPCA(unittest.TestCase):
    def test_selection_full_rank_scores_and_reloaded_transform(self):
        values = 10 + np.array([[-3, -0.1], [3, -0.1], [-3, 0.1], [3, 0.1]])
        result = fit_pca(values)
        self.assertEqual(result.scores.shape, (4, 2))
        self.assertEqual(result.variance["selected_for_correlation"].tolist(), [True, False])
        np.testing.assert_allclose(result.model.inverse_transform(result.scores), values)
        np.testing.assert_allclose(result.mean, values.mean(axis=0))
        self.assertTrue(
            all(axis[np.argmax(np.abs(axis))] > 0 for axis in result.components)
        )
        selected = fit_pca(values, n_components=2)
        self.assertTrue(selected.variance["selected_for_correlation"].all())
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mask = np.zeros((2, 2, 2), dtype=np.uint8)
            mask[0, 0, :] = 1
            mask_img = nib.Nifti1Image(mask, np.eye(4))
            with patch.dict(os.environ, {"RESULTS_DIR": str(directory / "results")}):
                artifacts = save_pca_artifacts(
                    result, mask_img, directory / "results" / "analysis" / "models"
                )
            restored_model = joblib.load(artifacts["model_path"])
            np.testing.assert_allclose(restored_model.transform(values), result.scores)
            for name, path in artifacts["component_maps"].items():
                index = int(name[2:]) - 1
                data = nib.load(path).get_fdata()
                np.testing.assert_allclose(data[mask > 0], result.components[index])
                np.testing.assert_array_equal(data[mask == 0], 0)

    def test_repeated_rows_keep_sample_weight_but_exclude_null_components(self):
        unique = np.array([[1, 2, 4], [2, 5, 3], [4, 1, 2]], dtype=float)
        values = unique[[0, 0, 1, 2, 2]]
        result = fit_pca(values, variance_threshold=1)
        self.assertEqual(result.components.shape, (2, 3))
        np.testing.assert_allclose(result.model.transform(values), result.scores)
        np.testing.assert_allclose(result.model.inverse_transform(result.scores), values)
        np.testing.assert_allclose(result.mean, values.mean(axis=0))
        np.testing.assert_allclose(result.scores[0], result.scores[1])
        with self.assertRaisesRegex(ValueError, "only 2 nonzero"):
            fit_pca(values, n_components=3)

    def test_zero_variance_and_nonfinite_inputs_are_rejected(self):
        for constant in (1.0, 0.1):
            with self.assertRaisesRegex(ValueError, "no nonzero variance"):
                fit_pca(np.full((3, 4), constant))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            fit_pca(np.array([[1, 2], [3, np.nan]]))

    def test_artifact_children_cannot_symlink_outside_results(self):
        result = fit_pca(np.array([[1, 2], [2, 1], [3, 4]], dtype=float))
        mask_img = nib.Nifti1Image(np.ones((1, 1, 2), dtype=np.uint8), np.eye(4))
        for artifact in ("model.joblib", "components", "components/PC1.nii.gz"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                results = directory / "results"
                output = results / "analysis" / "models"
                output.mkdir(parents=True)
                outside = directory / "outside"
                outside.mkdir()
                child = output / artifact
                child.parent.mkdir(exist_ok=True)
                if artifact == "components":
                    child.symlink_to(outside, target_is_directory=True)
                else:
                    child.symlink_to(outside / "unexpected_output")
                with patch.dict(os.environ, {"RESULTS_DIR": str(results)}):
                    with self.assertRaisesRegex(ValueError, "below RESULTS_DIR"):
                        save_pca_artifacts(result, mask_img, output)
                self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
