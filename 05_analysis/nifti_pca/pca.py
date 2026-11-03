"""PCA of sample-by-voxel matrices and export of signed component maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .paths import validate_output_path


@dataclass
class PCAResult:
    """All numerically nonzero components and their correlation selection."""

    scores: np.ndarray
    components: np.ndarray
    mean: np.ndarray
    variance: pd.DataFrame
    model: PCA


def fit_pca(
    values: np.ndarray,
    *,
    variance_threshold: float = 0.96,
    n_components: int | None = None,
) -> PCAResult:
    """Fit full SVD without scaling, whitening, or subtracting each row's mean.

    Standard PCA centers each voxel across sample rows. Repeated rows contribute
    separately. Every numerically nonzero component is retained; ``n_components``
    selects a fixed number for correlation, overriding ``variance_threshold``.
    Otherwise select the smallest leading set reaching that variance fraction.

    Component signs are fixed so each axis's largest absolute weight is positive.
    The fitted model uses the same signs and retained components as the scores.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("PCA requires a 2D matrix with >= 2 samples and >= 1 voxel.")
    if not np.isfinite(matrix).all():
        raise ValueError("PCA input contains non-finite values.")
    if not np.isfinite(variance_threshold) or not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be in (0, 1].")
    if n_components is not None and (
        isinstance(n_components, (bool, np.bool_))
        or not isinstance(n_components, (int, np.integer))
        or n_components < 1
    ):
        raise ValueError("n_components must be a positive integer.")
    if not np.any(np.ptp(matrix, axis=0)):
        raise ValueError("PCA input has no nonzero variance across samples.")

    # Use the economy SVD, never a voxel-by-voxel covariance matrix. Float64 is
    # important for recognizing duplicate-image null directions at brain scale.
    model = PCA(svd_solver="full", whiten=False)
    model.fit(matrix)
    singular_values = model.singular_values_
    tolerance = np.finfo(matrix.dtype).eps * max(matrix.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    rank = min(rank, matrix.shape[0] - 1)
    if rank == 0:
        raise ValueError("PCA input has no nonzero variance across samples.")
    if n_components is not None and n_components > rank:
        raise ValueError(
            f"Requested {n_components} correlation components, but the centered "
            f"input has only {rank} nonzero components."
        )

    omitted_variance = model.explained_variance_[rank:]
    model.noise_variance_ = (
        float(omitted_variance.mean()) if omitted_variance.size else 0.0
    )
    model.components_ = model.components_[:rank].copy()
    model.explained_variance_ = model.explained_variance_[:rank].copy()
    model.explained_variance_ratio_ = model.explained_variance_ratio_[:rank].copy()
    model.singular_values_ = model.singular_values_[:rank].copy()
    model.n_components_ = rank

    for component in model.components_:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    scores = model.transform(matrix)
    cumulative = np.cumsum(model.explained_variance_ratio_)
    selected_count = (
        int(n_components)
        if n_components is not None
        else min(int(np.searchsorted(cumulative, variance_threshold)) + 1, rank)
    )
    variance = pd.DataFrame(
        {
            "component": [f"PC{index}" for index in range(1, rank + 1)],
            "component_index": np.arange(1, rank + 1),
            "explained_variance": model.explained_variance_,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_variance_ratio": cumulative,
            "selected_for_correlation": np.arange(rank) < selected_count,
        }
    )
    return PCAResult(
        scores=scores,
        components=model.components_,
        mean=model.mean_,
        variance=variance,
        model=model,
    )


def save_pca_artifacts(
    result: PCAResult,
    mask_img: nib.spatialimages.SpatialImage,
    output_dir: str | Path,
) -> dict[str, object]:
    """Save the model, variance table, mask, mean, and signed component images.

    ``output_dir`` must be below ``$RESULTS_DIR/<analysis_name>/``. Component and
    mean volumes are float32 with zeros outside the mask; model weights retain
    their fitted precision. The caller writes run/basis provenance separately.
    """
    output_dir = validate_output_path(output_dir)
    mask_values = np.asanyarray(mask_img.dataobj)
    if mask_values.ndim != 3 or not np.isfinite(mask_values).all():
        raise ValueError("Artifact mask must be a finite 3D image.")
    mask = mask_values > 0
    voxel_count = int(mask.sum())
    if (
        voxel_count == 0
        or result.components.ndim != 2
        or result.components.shape[1] != voxel_count
        or result.mean.shape != (voxel_count,)
        or len(result.variance) != len(result.components)
    ):
        raise ValueError("PCA component/mean dimensions do not match the artifact mask.")

    # Resolve every artifact before writing: an existing child file/directory
    # can itself be a symlink even when its parent is safely under RESULTS_DIR.
    components_dir = validate_output_path(output_dir / "components")
    component_maps = {
        name: validate_output_path(components_dir / f"{name}.nii.gz")
        for name in result.variance["component"]
    }
    paths: dict[str, object] = {
        "model_path": validate_output_path(output_dir / "model.joblib"),
        "variance_path": validate_output_path(output_dir / "explained_variance.csv"),
        "mean_field_path": validate_output_path(output_dir / "mean_field.nii.gz"),
        "mask_path": validate_output_path(output_dir / "brain_mask.nii.gz"),
        "component_maps": component_maps,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    components_dir.mkdir(exist_ok=True)

    def save_vector(vector: np.ndarray, path: Path) -> None:
        volume = np.zeros(mask.shape, dtype=np.float32)
        volume[mask] = vector
        header = mask_img.header.copy()
        header.set_data_dtype(np.float32)
        header["cal_min"] = float(volume.min())
        header["cal_max"] = float(volume.max())
        nib.save(nib.Nifti1Image(volume, mask_img.affine, header), str(path))

    joblib.dump(result.model, paths["model_path"])
    result.variance.to_csv(paths["variance_path"], index=False)
    save_vector(result.mean, paths["mean_field_path"])
    mask_header = mask_img.header.copy()
    mask_header.set_data_dtype(np.uint8)
    mask_header["cal_min"] = 0
    mask_header["cal_max"] = 1
    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8), mask_img.affine, mask_header),
        str(paths["mask_path"]),
    )
    for component_name, vector in zip(result.variance["component"], result.components):
        save_vector(vector, component_maps[component_name])
    return paths
