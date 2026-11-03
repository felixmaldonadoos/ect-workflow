"""Extract regional E-field statistics as a dictionary of DataFrames.

Dependencies: pip install numpy pandas nibabel scipy 'nilearn>=0.14'

In Python (df_query must already be filtered to the E-field files):
    from add_region_means import extract_region_stats
    stats = extract_region_stats(df_query, 'atlas.nii.gz', 'labels.txt')
    result = pd.concat([df_query, *stats.values()], axis=1)
    # Save result inside $RESULTS_DIR/<analysis_name>/data/.

From the command line (saves one CSV containing metadata and all statistics):
    python -B add_region_means.py input.csv atlas.nii.gz labels.txt --analysis-name mni_yeo2011
"""

import argparse
from pathlib import Path

if __package__ in (None, ""):
    from _results import add_analysis_argument, bootstrap_cli, data_dir, validate_output_path
else:
    from ._results import add_analysis_argument, bootstrap_cli, data_dir, validate_output_path

if __name__ == "__main__":
    bootstrap_cli()

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import math_img, resample_to_img
from nilearn.maskers import NiftiMasker
from nilearn.masking import apply_mask
from scipy.stats import mode


STATISTICS = (
    "mean", "mode", "median", "min", "max",
    "95th_percentile", "99th_percentile", "n_voxels",
)


def extract_region_stats(
    df,
    atlas_path,
    lut_path,
    path_column="full_file_path",
    *,
    mode_decimals=None,
):
    """Return dict[str, DataFrame], keyed by the names in STATISTICS.

    The atlas must be a 3D integer-valued label image; 0 is background.
    lut_path is a FreeSurfer/Freeview LUT whose first two columns contain the
    numeric label ID and region name. The attached glossary is not required.

    Each input must be a finite scalar image in the atlas's anatomical space:
    either 3D or 4D with shape (X, Y, Z, 1). A singleton fourth dimension is
    removed while preserving the spatial axes and affine. Nilearn resamples
    the atlas to each E-field grid using nearest-neighbor interpolation. This
    does not perform registration or change E-field values. NiftiMasker
    extracts the atlas voxels once per image, with no smoothing, detrending,
    or standardization. Consecutive images with the same grid reuse the masker.

    Every returned DataFrame has exactly df's row order and index (including
    duplicate indices), plus one E_<statistic>__<region> column per atlas region.
    Input metadata are not included in these DataFrames and df is not modified.
    Join all statistics with pd.concat([df, *stats.values()], axis=1). This also
    preserves duplicate indices without the row multiplication of an index join.

    All statistics include zero E-fields. n_voxels counts labeled voxels on the
    E-field grid, not on the original atlas grid. If a region is absent after
    resampling, n_voxels is 0 and its other statistics are NaN. Partial coverage
    uses only the part of the region inside the E-field image's grid.

    Pandas GroupBy computes the regional summaries and linearly interpolated
    percentiles. SciPy computes mode, using exact values unless
    mode_decimals is an integer, in which case values are rounded to that many
    decimal places FOR MODE ONLY. Ties choose the smallest modal value; if all
    values are unique, this is the minimum. Rounding does not affect any other
    statistic. Missing files and non-finite E-field values raise errors.
    """
    if path_column not in df.columns:
        raise KeyError(f"No path column named {path_column!r}")
    if mode_decimals is not None and (
        isinstance(mode_decimals, (bool, np.bool_))
        or not isinstance(mode_decimals, (int, np.integer))
    ):
        raise TypeError("mode_decimals must be an integer or None.")

    atlas = nib.load(atlas_path)
    atlas_data = np.asanyarray(atlas.dataobj)
    if atlas_data.ndim != 3:
        raise ValueError("The atlas must be a 3D label image.")
    if not np.isfinite(atlas_data).all() or not np.equal(
        atlas_data, np.rint(atlas_data)
    ).all():
        raise ValueError("Atlas labels must be finite integer values.")
    region_ids = np.unique(atlas_data)
    region_ids = region_ids[region_ids != 0].astype(int)
    if not len(region_ids):
        raise ValueError("The atlas contains no non-background regions.")

    lut = pd.read_csv(
        lut_path, sep=r"\s+", comment="#", header=None,
        usecols=[0, 1], names=["id", "name"],
    )
    if lut["id"].duplicated().any():
        raise ValueError("The lookup table contains duplicate label IDs.")
    names = lut.set_index("id")["name"].to_dict()
    missing = set(region_ids) - set(names)
    if missing:
        raise ValueError(f"Atlas label IDs missing from lookup table: {sorted(missing)}")
    region_names = [names[i] for i in region_ids]
    if len(set(region_names)) != len(region_names):
        raise ValueError("Region names must be unique.")
    columns = {
        stat: [f"E_{stat}__{name}" for name in region_names]
        for stat in STATISTICS
    }
    overlap = set().union(*columns.values()) & set(df.columns)
    if overlap:
        raise ValueError(f"Output columns already exist: {sorted(overlap)}")

    shape = (len(df), len(region_ids))
    output = {stat: np.full(shape, np.nan) for stat in STATISTICS[:-1]}
    output["n_voxels"] = np.zeros(shape, dtype=np.int64)
    cached_grid = None
    masker = None
    voxel_labels = None

    for row, path in enumerate(df[path_column]):
        image = nib.load(path)
        if len(image.shape) == 4 and image.shape[-1] == 1:
            image = image.slicer[..., 0]
        if len(image.shape) != 3:
            raise ValueError(
                "Expected a 3D scalar E-field image or shape (X, Y, Z, 1); "
                f"got {image.shape}: {path}"
            )
        # Check before masking, which can otherwise replace non-finite values.
        if not np.isfinite(image.get_fdata(dtype=np.float64)).all():
            raise ValueError(f"E-field image contains NaN or infinity: {path}")

        grid = (image.shape, tuple(image.affine.ravel()))
        if grid != cached_grid:
            aligned_atlas = resample_to_img(
                atlas, image, interpolation="nearest",
                force_resample=True, copy_header=True,
            )
            mask_img = math_img("img != 0", img=aligned_atlas)
            masker = None
            if np.asanyarray(mask_img.dataobj).any():
                masker = NiftiMasker(
                    mask_img=mask_img,
                    dtype=np.float64,
                    standardize=None,
                    detrend=False,
                    smoothing_fwhm=None,
                    reports=False,
                ).fit()
                # apply_mask and NiftiMasker use the same voxel ordering.
                voxel_labels = apply_mask(
                    aligned_atlas, masker.mask_img_,
                ).astype(np.int64)
            cached_grid = grid

        if masker is None:
            continue
        values = pd.Series(np.asarray(masker.transform(image)).ravel())
        grouped = values.groupby(voxel_labels, sort=True)
        summary = grouped.agg(
            mean="mean", median="median", min="min", max="max", n_voxels="size",
        )
        quantiles = grouped.quantile([0.95, 0.99], interpolation="linear").unstack()
        summary["95th_percentile"] = quantiles[0.95]
        summary["99th_percentile"] = quantiles[0.99]
        mode_values = values if mode_decimals is None else values.round(mode_decimals)
        summary["mode"] = mode_values.groupby(voxel_labels, sort=True).agg(
            lambda voxels: float(mode(voxels, axis=None, keepdims=False).mode),
        )
        summary = summary.reindex(region_ids)
        summary["n_voxels"] = summary["n_voxels"].fillna(0).astype(np.int64)
        for stat in STATISTICS:
            output[stat][row] = summary[stat].to_numpy()

    return {
        stat: pd.DataFrame(output[stat], index=df.index.copy(), columns=columns[stat])
        for stat in STATISTICS
    }


# The original function name remains available, with the new dictionary return.
add_region_means = extract_region_stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_analysis_argument(parser)
    parser.add_argument("input_csv", help="CSV already filtered to E-field NIfTIs")
    parser.add_argument("atlas", help="3D label atlas NIfTI")
    parser.add_argument("lut", help="Freeview/FreeSurfer color lookup table")
    parser.add_argument(
        "output_csv", nargs="?", type=Path,
        help="Destination inside the selected analysis directory; default: data/efield_by_region.csv.",
    )
    parser.add_argument("--mode-decimals", type=int, default=None)
    args = parser.parse_args(argv)
    output_path = validate_output_path(
        args.output_csv if args.output_csv is not None else data_dir(args.analysis_name) / "efield_by_region.csv",
        args.analysis_name,
    )

    # Preserve text fields such as IDs, leading zeros, and empty cells.
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    stats = extract_region_stats(
        df, args.atlas, args.lut, mode_decimals=args.mode_decimals,
    )
    result = pd.concat([df, *stats.values()], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(
        f"Saved {len(result)} rows and {result.shape[1] - df.shape[1]} "
        f"region-statistic columns ({len(stats)} statistics)."
    )


if __name__ == "__main__":
    main()
