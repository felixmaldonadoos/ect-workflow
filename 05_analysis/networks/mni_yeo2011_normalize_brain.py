#!/usr/bin/env python3
"""
Normalize regional E-field means by each GM/WM-only field's nonzero mean.

Reads the raw ROI master produced by mni_yeo2011_prep_data.py.

For every unique E-field file:
    1. Load the GM/WM-only MNI field (SimNIBS tissue labels 1 and 2).
    2. Validate all field values are finite.
    3. Select nonzero field values, treating zeros as excluded tissue/background.
    4. Calculate their arithmetic mean independently for each field file.

Output:
    mean_raw          Original regional mean.
    whole_brain_mean  Mean nonzero E-field in the GM/WM-only input.
    mean              mean_raw / whole_brain_mean.

Dividing each ROI mean by this per-file scalar is equivalent to averaging
field_nifti_values_gmwm / mean_values_gmwm over the same ROI voxels. The
normalized field need not be materialized as another NIfTI.

n_voxels and other spatial statistics remain unchanged. Metadata identifies
the GM/WM nonzero domain. No external brain template/mask is used. Inputs
must already be restricted to GM/WM; genuine zero-valued tissue cannot be
distinguished from background by this selection rule.

Requires numpy, pandas, and nibabel.
"""

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    from _results import (
        add_analysis_argument,
        bootstrap_cli,
        data_dir,
        default_master_path,
        validate_output_path,
    )
else:
    from ._results import (
        add_analysis_argument,
        bootstrap_cli,
        data_dir,
        default_master_path,
        validate_output_path,
    )

if __name__ == "__main__":
    bootstrap_cli()

import nibabel as nib
import numpy as np
import pandas as pd


DEFAULT_FIELD_NAME = "magnE_mean"

RAW_SPATIAL_STATISTICS = (
    "mode",
    "median",
    "min",
    "max",
    "p95",
    "p99",
)


def load_scalar_img(path):
    """Load a 3D image, accepting a singleton fourth dimension."""
    img = nib.load(str(path))

    if len(img.shape) == 4 and img.shape[-1] == 1:
        img = img.slicer[..., 0]

    if len(img.shape) != 3:
        raise ValueError(
            f"Expected a 3D or singleton-4D image: {path}; "
            f"got {img.shape}"
        )

    return img


def normalize_master(
    master,
    field_name=DEFAULT_FIELD_NAME,
):
    """Return a normalized copy of the selected field's master rows."""
    required = {"field_name", "full_file_path", "mean"}
    missing = required - set(master.columns)

    if missing:
        raise KeyError(
            f"Missing master columns: {sorted(missing)}"
        )

    result = master.loc[
        master["field_name"].eq(field_name)
    ].copy()

    if result.empty:
        raise ValueError(
            f"No rows for field_name={field_name!r}."
        )

    # Prevent accidental division of previously normalized values.
    if "mean_raw" in result.columns:
        raise ValueError(
            "Input already contains mean_raw. "
            "Use the original raw master CSV."
        )

    if "normalization" in result.columns:
        labels = (
            result["normalization"]
            .fillna("none")
            .astype(str)
            .str.lower()
        )

        if not labels.isin(["none", "raw", ""]).all():
            raise ValueError("Input is already normalized.")

    paths_column = result["full_file_path"]

    if (
        paths_column.isna().any()
        or paths_column.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Missing E-field file paths.")

    raw_mean = pd.to_numeric(
        result["mean"],
        errors="raise",
    )

    if not np.isfinite(
        raw_mean.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "ROI means contain non-finite values."
        )

    # Resolve equivalent paths so each unique field is loaded once.
    paths = paths_column.map(
        lambda value: str(
            Path(value).expanduser().resolve()
        )
    )
    unique_paths = paths.drop_duplicates().tolist()

    brain_means = {}
    brain_counts = {}

    for position, path in enumerate(unique_paths, start=1):
        field = load_scalar_img(path)
        data = field.get_fdata(dtype=np.float64)
        if not np.isfinite(data).all():
            raise ValueError(
                f"Non-finite E-field values in GM/WM input: {path}"
            )

        # Foreground varies per image, even when shape and affine are identical.
        brain_values = data[data != 0]
        if not brain_values.size:
            raise ValueError(f"GM/WM input contains no nonzero field values: {path}")
        brain_mean = float(brain_values.mean())

        if not np.isfinite(brain_mean) or brain_mean <= 0:
            raise ValueError(
                "GM/WM mean must be finite and positive: "
                f"{path}; got {brain_mean}"
            )

        brain_means[path] = brain_mean
        brain_counts[path] = int(brain_values.size)

        print(
            f"GM/WM mean {position}/{len(unique_paths)}: "
            f"{brain_mean:.8g} "
            f"({brain_counts[path]:,} voxels) | {path}",
            flush=True,
        )

    result["mean_raw"] = raw_mean
    result["whole_brain_mean"] = paths.map(brain_means)
    result["whole_brain_n_voxels"] = paths.map(brain_counts)

    # Existing correlation scripts consume the mean column.
    result["mean"] = (
        result["mean_raw"] / result["whole_brain_mean"]
    )

    if not np.isfinite(
        result["mean"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Normalized ROI means contain non-finite values."
        )

    result["normalization"] = "brain_mean"
    result["normalization_tag"] = (
        f"normed_brain_{field_name}_mean"
    )
    result["normalization_applies_to"] = "mean"
    result["normalization_domain"] = "gmwm_nonzero"
    result["normalization_mask_path"] = ""
    result["normalization_mask_rule"] = (
        "field != 0; GM/WM-only input (SimNIBS labels 1,2); zeros excluded"
    )
    result["unnormalized_spatial_statistics"] = json.dumps([
        column
        for column in RAW_SPATIAL_STATISTICS
        if column in result.columns
    ])

    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Divide each ROI mean by the mean of nonzero values in the "
            "corresponding GM/WM-only MNI field (SimNIBS labels 1,2)."
        ),
        allow_abbrev=False,
    )
    add_analysis_argument(parser)

    parser.add_argument(
        "--master-path",
        type=Path,
        help="Raw ROI master CSV.",
    )
    parser.add_argument(
        "--field-name",
        default=DEFAULT_FIELD_NAME,
        help="Source field name (default: %(default)s).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Normalized master CSV under RESULTS_DIR/<analysis-name>/. "
            "Default: data/yeo2011_<field-name>_series_ROI_master_"
            "normed_brain_<field-name>_mean.csv."
        ),
    )

    args = parser.parse_args(argv)

    if (
        not args.field_name
        or args.field_name != args.field_name.strip()
    ):
        parser.error(
            "--field-name must be nonempty without surrounding whitespace."
        )

    return args


def main(argv=None):
    args = parse_args(argv)

    master_path = args.master_path
    if master_path is None:
        master_path = default_master_path(args.analysis_name, args.field_name)
    master_path = master_path.expanduser().resolve()

    norm_tag = f"normed_brain_{args.field_name}_mean"

    output_path = args.output_path
    if output_path is None:
        output_path = data_dir(args.analysis_name) / (
            f"yeo2011_{args.field_name}_series_ROI_master_{norm_tag}.csv"
        )
    output_path = validate_output_path(output_path, args.analysis_name)

    if output_path == master_path:
        raise ValueError(
            "Output path must differ from the raw master CSV."
        )

    if not master_path.is_file():
        raise FileNotFoundError(
            f"Raw master CSV does not exist: {master_path}"
        )

    master = pd.read_csv(master_path)

    normalized = normalize_master(
        master,
        field_name=args.field_name,
    )
    normalized["raw_master_path"] = str(master_path)

    output_path = validate_output_path(output_path, args.analysis_name)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    normalized.to_csv(output_path, index=False)

    print(
        f"Saved {len(normalized):,} normalized ROI rows "
        f"to {output_path}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
