"""Read scalar NIfTI fields into a common, masked voxel feature space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


@dataclass
class MaskedFields:
    """Field vectors in input sample order; repeated images remain repeated rows."""

    values: np.ndarray
    brain_means: np.ndarray
    mask_img: nib.Nifti1Image
    reference_img: nib.Nifti1Image
    unique_paths: list[Path]
    sample_paths: list[Path]


def _load_scalar_image(path: Path) -> nib.spatialimages.SpatialImage:
    image = nib.load(str(path))
    if len(image.shape) == 4 and image.shape[-1] == 1:
        image = image.slicer[..., 0]
    if len(image.shape) != 3:
        raise ValueError(
            f"Expected a 3D or singleton-4D scalar image: {path}; "
            f"got shape {image.shape}."
        )
    if not np.isfinite(image.affine).all():
        raise ValueError(f"Image affine contains non-finite values: {path}")
    return image


def load_masked_fields(
    paths: Sequence[str | Path],
    mask_path: str | Path,
    *,
    resample: bool = False,
) -> MaskedFields:
    """Load each unique field once, then expand vectors to the sample rows.

    All features use the supplied mask's grid, and brain means include zero-valued
    field voxels. A template can be supplied as the mask: values greater than zero
    define the brain. Affines must agree within 1e-5 mm unless ``resample=True``;
    resampling uses linear interpolation of the field onto the fixed mask grid.
    Inputs must already share anatomical space; resampling is not registration.

    Finite values are required inside the mask. Means need not be positive here;
    callers must require positive means before dividing by them for normalization.
    """
    if isinstance(paths, (str, Path)):
        raise TypeError("paths must be a sequence of field paths, not one path.")
    sample_paths = [Path(path).expanduser().resolve() for path in paths]
    if not sample_paths:
        raise ValueError("At least one field image is required.")

    mask_source_path = Path(mask_path).expanduser().resolve()
    mask_source = _load_scalar_image(mask_source_path)
    mask_values = mask_source.get_fdata(dtype=np.float32)
    if not np.isfinite(mask_values).all():
        raise ValueError(f"Brain mask contains non-finite values: {mask_source_path}")
    mask = mask_values > 0
    voxel_count = int(mask.sum())
    if voxel_count == 0:
        raise ValueError("Brain mask is empty after thresholding at > 0.")
    header = mask_source.header.copy()
    header.set_data_dtype(np.uint8)
    header["cal_min"] = 0
    header["cal_max"] = 1
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), mask_source.affine, header)

    unique_paths = list(dict.fromkeys(sample_paths))
    unique_values = np.empty((len(unique_paths), voxel_count), dtype=np.float32)
    unique_means = np.empty(len(unique_paths), dtype=np.float64)
    path_index = {path: index for index, path in enumerate(unique_paths)}

    for index, path in enumerate(unique_paths):
        image = _load_scalar_image(path)
        same_grid = image.shape == mask.shape and np.allclose(
            image.affine, mask_img.affine, rtol=0.0, atol=1e-5
        )
        if not same_grid:
            if not resample:
                raise ValueError(
                    f"Field grid does not match the brain mask: {path}; "
                    f"field shape {image.shape}, mask shape {mask.shape}. "
                    "Use --resample only if the images already share "
                    "anatomical space."
                )
            image = resample_from_to(
                image,
                (mask_img.shape, mask_img.affine),
                order=1,
                mode="constant",
                cval=0.0,
            )
        field_values = image.get_fdata(dtype=np.float32)[mask]
        if not np.isfinite(field_values).all():
            raise ValueError(f"Field contains non-finite values inside the brain: {path}")
        unique_values[index] = field_values
        unique_means[index] = np.mean(field_values, dtype=np.float64)

    if len(unique_paths) == len(sample_paths):
        values = unique_values
        brain_means = unique_means
    else:
        sample_index = np.array([path_index[path] for path in sample_paths], dtype=int)
        values = unique_values[sample_index]
        brain_means = unique_means[sample_index]

    return MaskedFields(
        values=values,
        brain_means=brain_means,
        mask_img=mask_img,
        reference_img=mask_img,
        unique_paths=unique_paths,
        sample_paths=sample_paths,
    )
