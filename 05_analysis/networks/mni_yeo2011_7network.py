#!/usr/bin/env python3
"""Run the MNI152 Yeo2011 seven-network analysis outside Jupyter.

This is the executable counterpart of mni_yeo2011_7network.ipynb. It uses the
same project clinical loaders, strict series alignment, exact-first BT/BL
fallback, raw E-field ROI statistics, equally weighted regional network means,
and Spearman correlations. Missing/invalid data and unresolved mappings raise.
No notebook, IPython state, or notebook execution package is needed at runtime.

Defaults reproduce the notebook's four model/step configurations and session 2.
Paths default relative to this script's project location; output defaults to
its sibling data directory, regardless of the current working directory.
The full run saves both whole-atlas maps and the two explicit per-parent maps
(hemisphere-specific regional coefficients and the bilateral network mean),
with PNG/NIfTI/source-CSV sets for each parent network.

Run with the notebook's SimNIBS Python environment, which must also provide
numpy, pandas, scipy, matplotlib, nibabel, nilearn, and openpyxl::

    python mni_yeo2011_7network.py --help
    python mni_yeo2011_7network.py --output-dir /path/to/results
    python mni_yeo2011_7network.py --skip-plots --output-dir /path/to/tables

No multiple-comparison correction or E-field dose scaling is introduced.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

# Select the backend before project imports, which can also import pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import image, masking, plotting
from nilearn.maskers import NiftiMasker
from scipy.stats import mode, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent
WORKFLOW_DIR = ANALYSIS_DIR.parent
PROJECT_DIR = WORKFLOW_DIR.parent

if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from simnibs_parcel_analysis.utils.loader import (
    load_course_data,
    load_query_results,
    load_stimulation_data,
    prepare_session_data,
)
from simnibs_parcel_analysis.utils.prepare_session_file_table import MissingMappingError

MODELS = ("skin_single", "skin_double")
STEPS = ("step_0", "step_9")
NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")

# Analysis functions are retained from the notebook. Workflow execution is in
# main(), after their definitions, so importing this module never runs analysis.


def load_scalar_img(niimg):
    """Load a 3D image, accepting a singleton fourth dimension."""
    img = image.load_img(niimg)

    if len(img.shape) == 4 and img.shape[-1] == 1:
        img = img.slicer[..., 0]

    if len(img.shape) != 3:
        raise ValueError(f"Expected 3D or (X,Y,Z,1); got {img.shape}")

    return img


def load_label_atlas(atlas_path, lut_path):
    """Return the atlas and a Series mapping label IDs to ROI names."""
    atlas = load_scalar_img(atlas_path)
    ids = np.unique(image.get_data(atlas))

    if not np.isfinite(ids).all() or not np.equal(ids, np.rint(ids)).all():
        raise ValueError("The atlas must contain finite integer labels.")

    ids = ids[ids != 0].astype(int)

    lut = pd.read_csv(
        lut_path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1],
        names=["id", "name"],
    ).set_index("id")["name"]

    if not lut.index.is_unique:
        raise ValueError("Duplicate IDs in the LUT.")

    names = lut.reindex(ids)

    if names.empty or names.isna().any() or not names.is_unique:
        raise ValueError("Every atlas ROI needs a unique name in the LUT.")

    return atlas, names


def extract_region_stats(df,
    atlas_path,
    lut_path,
    path_column="full_file_path",
    *,
    mode_decimals=None,):
    """Return {statistic: DataFrame}, preserving df's index and row order.

    Column names use one underscore between statistic and ROI:
    E_mean_7Networks_LH_Vis.

    Zeros are included. Non-finite E-fields raise an error.
    Missing regions receive count 0 and NaN for other statistics.
    mode_decimals affects only the mode calculation.
    """
    atlas, names = load_label_atlas(atlas_path, lut_path)

    statistics = (
        "mean", "mode", "median", "min", "max",
        "95th_percentile", "99th_percentile", "n_voxels",
    )

    shape = (len(df), len(names))
    arrays = {s: np.full(shape, np.nan) for s in statistics[:-1]}
    arrays["n_voxels"] = np.zeros(shape, dtype=np.int64)

    cached_grid, masker = None, None

    for row, path in enumerate(df[path_column]):
        img = load_scalar_img(path)

        if not np.isfinite(img.get_fdata()).all():
            raise ValueError(f"Non-finite E-field values: {path}")

        grid = (img.shape, tuple(img.affine.ravel()))

        if grid != cached_grid:
            # Align categorical labels to the E-field grid.
            labels = image.resample_to_img(
                atlas,
                img,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            roi_mask = image.math_img("labels != 0", labels=labels)
            cached_grid, masker = grid, None

            if image.get_data(roi_mask).any():
                masker = NiftiMasker(
                    mask_img=roi_mask,
                    dtype=np.float64,
                    standardize=None,
                    detrend=False,
                    smoothing_fwhm=None,
                    reports=False,
                ).fit()

                # Same voxel ordering as masker.transform().
                ids = masking.apply_mask(
                    labels, masker.mask_img_
                ).astype(int)

        if masker is None:
            continue

        values = pd.Series(np.asarray(masker.transform(img)).ravel())
        grouped = values.groupby(ids)

        summary = grouped.agg(
            mean="mean",
            median="median",
            min="min",
            max="max",
            n_voxels="size",
        )

        quantiles = grouped.quantile(
            [0.95, 0.99], interpolation="linear"
        ).unstack()

        summary["95th_percentile"] = quantiles[0.95]
        summary["99th_percentile"] = quantiles[0.99]

        mode_values = (
            values if mode_decimals is None
            else values.round(mode_decimals)
        )

        summary["mode"] = mode_values.groupby(ids).agg(
            lambda v: float(mode(v, axis=None, keepdims=False).mode)
        )

        summary = summary.reindex(names.index)
        summary["n_voxels"] = (
            summary["n_voxels"].fillna(0).astype(np.int64)
        )

        for stat in statistics:
            arrays[stat][row] = summary[stat].to_numpy()

    return {
        stat: pd.DataFrame(
            arrays[stat],
            index=df.index.copy(),
            columns=[f"E_{stat}_{name}" for name in names],
        )
        for stat in statistics
    }


def merge_series_data(df_cgi: pd.DataFrame, df_stim_collapsed: pd.DataFrame,) -> pd.DataFrame:
    keys = ["person_id", "series_num"]
    problems = []

    try:
        for name, table in [
            ("df_stim_collapsed", df_stim_collapsed),
            ("df_cgi", df_cgi),
        ]:
            missing_columns = set(keys) - set(table.columns)
            if missing_columns:
                raise KeyError(
                    f"{name} is missing columns: {sorted(missing_columns)}"
                )

            missing_keys = table.loc[table[keys].isna().any(axis=1)]
            if not missing_keys.empty:
                problems.append(f"{name} contains missing merge-key values")
                print(f"\n{name}: rows with missing merge keys:")
                print(missing_keys.to_string(index=True))

            duplicates = table.loc[table.duplicated(keys, keep=False)]
            if not duplicates.empty:
                problems.append(f"{name} contains duplicate series keys")
                print(f"\n{name}: rows with duplicate merge keys:")
                print(duplicates.to_string(index=True))

        stim_keys = pd.MultiIndex.from_frame(df_stim_collapsed[keys])
        cgi_keys = pd.MultiIndex.from_frame(df_cgi[keys])

        stim_only = df_stim_collapsed.loc[~stim_keys.isin(cgi_keys)]
        cgi_only = df_cgi.loc[~cgi_keys.isin(stim_keys)]

        if not stim_only.empty:
            problems.append(
                f"{len(stim_only)} stimulation rows have no matching CGI series"
            )
            print("\nStimulation rows without matching CGI series:")
            print(stim_only.to_string(index=True))

        if not cgi_only.empty:
            problems.append(
                f"{len(cgi_only)} CGI rows have no matching stimulation series"
            )
            print("\nCGI rows without matching stimulation series:")
            print(cgi_only.to_string(index=True))

        if problems:
            raise pd.errors.MergeError(
                "Perfect series alignment required: " + "; ".join(problems)
            )

        shared_metadata = [
            col for col in ["age", "sex", "dob"]
            if col in df_stim_collapsed.columns and col in df_cgi.columns
        ]

        return df_stim_collapsed.drop(columns=shared_metadata).merge(
            df_cgi,
            on=keys,
            how="left",
            sort=False,
            validate="one_to_one",
            suffixes=("_stim", "_cgi"),
        )

    except (KeyError, ValueError) as error:
        # pandas MergeError is also a ValueError.
        print(
            f"\nMerge failed: {error}\n"
            f"Stimulation rows: {len(df_stim_collapsed)}; "
            f"CGI rows: {len(df_cgi)}"
        )
        raise


def load_series_data(
    subjects_exclude: list,
    fn_query,
    fn_cgi,
    query_filters: dict,
    fn_field: str,
    *,
    subject_scan_map=None,
    session_num=2,) -> pd.DataFrame:
    # Load and filter the simulation inventory.
    df_query = load_query_results(fn_query)

    for column, value in query_filters.items():
        df_query = df_query.loc[df_query[column].eq(value)]

    df_query = df_query.loc[
        ~df_query["subject_dir"].isin(subjects_exclude)
    ].copy()

    if df_query.empty:
        raise ValueError(
            f"No query rows remain after filtering: {query_filters}"
        )

    # Load and filter the per-series outcomes.
    df_cgi = load_course_data(fn_cgi)
    df_cgi = df_cgi.loc[
        ~df_cgi["subjid"].isin(subjects_exclude)
    ].copy()

    # Select the representative session from each series.
    df_stim = load_stimulation_data(fn_cgi)
    df_stim = df_stim.loc[
        ~df_stim["subjid"].isin(subjects_exclude)
        & df_stim["person_id"].isin(df_cgi["person_id"])
        & df_stim["session_num"].eq(session_num)
    ].copy()

    if df_stim.empty:
        raise ValueError(
            f"No stimulation rows remain for session_num={session_num}."
        )

    # Temporary indices make fallback assignment independent of original indices.
    sessions = df_stim.reset_index(drop=True)

    mapping_options = dict(
        df_query=df_query,
        df_cgi=df_cgi,
        filename=fn_field,
        subject_scan_map=subject_scan_map,
    )

    # prepare_session_data also computes ef_sf.
    try:
        df_stim_mapped = prepare_session_data(
            df_stim=sessions,
            **mapping_options,
        )

    except MissingMappingError as error:
        placements = (
            error.unmatched["electrode_placement"]
            .str.strip()
            .str.upper()
        )

        if not placements.isin(["BT", "BL"]).all():
            raise

        fallback_sessions = sessions.copy()
        fallback_sessions.loc[
            error.unmatched.index, "electrode_placement"
        ] = placements.replace({"BT": "BL", "BL": "BT"})

        # Unresolved or ambiguous mappings still raise.
        df_stim_mapped = prepare_session_data(
            df_stim=fallback_sessions,
            **mapping_options,
        )

    # Restore the original clinical labels and indices.
    df_stim_mapped["electrode_placement"] = sessions.loc[
        df_stim_mapped.index, "electrode_placement"
    ]
    df_stim_mapped.index = df_stim.index.take(df_stim_mapped.index)

    # Uses the strict alignment checks and diagnostics defined earlier.
    return merge_series_data(
        df_cgi=df_cgi,
        df_stim_collapsed=df_stim_mapped,
    )


def keep_cgi_and_field(df: pd.DataFrame) -> pd.DataFrame:
    mean_columns = [
        col for col in df.columns
        if col.startswith("E_mean_7Networks_")
    ]

    if not mean_columns:
        raise ValueError("No E_mean_ columns found.")

    return df.loc[:, ["cgi_change", *mean_columns]].copy()


def add_network_means_both_hemis(df: pd.DataFrame) -> pd.DataFrame:
    networks = [
        "Vis", "SomMot", "DorsAttn", "SalVentAttn",
        "Limbic", "Cont", "Default",
    ]

    result = df.copy()

    for network in networks:
        columns = []

        for hemisphere in ("LH", "RH"):
            prefix = f"E_mean_7Networks_{hemisphere}_{network}"

            region_columns = [
                col for col in df.columns
                if col == prefix or col.startswith(prefix + "_")
            ]

            if not region_columns:
                raise ValueError(
                    f"No regional mean columns for {hemisphere} {network}."
                )

            columns.extend(region_columns)

        result[f"E_mean_{network}_both_hemis"] = df[columns].mean(
            axis=1,
            skipna=False,
        )

    return result


def save_field_roi_tables(tables: dict, filename) -> pd.DataFrame:
    if not tables:
        raise ValueError("No feature tables supplied.")

    combined = pd.concat(
        [
            df.assign(model_type=model_type, step=step)
            for (model_type, step), df in tables.items()
        ],
        ignore_index=True,
    )

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(
        f"Saved {len(combined)} rows and "
        f"{len(combined.columns)} columns to {output_path}"
    )

    return combined


def run_spearman(df: pd.DataFrame, y: str, x: list[str] | str | None = None,) -> pd.DataFrame:
    if df.columns.has_duplicates:
        raise ValueError("Dataframe contains duplicate column names.")

    if y not in df.columns:
        raise KeyError(f"Outcome column {y!r} is missing.")

    if x is None:
        features = [col for col in df.columns if col != y]
    elif isinstance(x, str):
        features = [x]
    else:
        features = list(x)

    if not features:
        raise ValueError("No feature columns selected.")

    if y in features or len(features) != len(set(features)):
        raise ValueError("Features must be unique and must not include y.")

    data = df[[y, *features]].apply(pd.to_numeric, errors="raise")

    if len(data) < 3:
        raise ValueError("At least three observations are required.")

    invalid = data.columns[
        ~np.isfinite(data.to_numpy(dtype=float)).all(axis=0)
    ].tolist()
    if invalid:
        raise ValueError(f"Missing or non-finite values in: {invalid}")

    constant = data.columns[data.nunique().lt(2)].tolist()
    if constant:
        raise ValueError(f"Constant columns cannot be correlated: {constant}")

    results = []
    for feature in features:
        rho, p_value = spearmanr(
            data[feature],
            data[y],
            nan_policy="raise",
        )
        results.append({
            "x": feature,
            "y": y,
            "n": len(data),
            "rho": float(rho),
            "p_value": float(p_value),
        })

    return pd.DataFrame(results)


def save_spearman_table(df: pd.DataFrame, filename) -> pd.DataFrame:
    columns = ["x", "y", "n", "rho", "p_value", "model_type", "step"]

    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    result = df[columns].copy()

    networks = "Vis|SomMot|DorsAttn|SalVentAttn|Limbic|Cont|Default"

    # Single underscore after E_mean.
    regional_pattern = (
        rf"E_mean_7Networks_(LH|RH)_({networks})(?:_(.+))?"
    )
    bilateral_pattern = rf"E_mean_({networks})_both_hemis"

    metadata = []

    for feature in result["x"]:
        if not isinstance(feature, str):
            raise ValueError(f"Invalid feature name: {feature!r}")

        regional = re.fullmatch(regional_pattern, feature)
        bilateral = re.fullmatch(bilateral_pattern, feature)

        if regional:
            hemi, network, roi = regional.groups()
            roi = roi or network
        elif bilateral:
            network = bilateral.group(1)
            hemi = "both"
            roi = network
        else:
            raise ValueError(f"Unrecognized feature name: {feature!r}")

        metadata.append((roi, hemi, network))

    result["roi"] = [row[0] for row in metadata]
    result["hemi"] = [row[1] for row in metadata]
    result["network"] = [row[2] for row in metadata]

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return result


def network_correlations_to_image(
    results,
    atlas_path,
    lut_path,
    *,
    level="network",
):
    """Map either 'roi' or 'network' correlations onto the atlas."""
    if level not in ("roi", "network"):
        raise ValueError("level must be 'roi' or 'network'.")

    results = results.copy()

    # Normalize names from older saved correlation tables.
    results["x"] = results["x"].str.replace(
        r"^E_mean__7Networks_",
        "E_mean_7Networks_",
        regex=True,
    )

    if results["x"].duplicated().any():
        raise ValueError("Duplicate features: select one configuration only.")

    if results["y"].nunique(dropna=False) != 1:
        raise ValueError("Results must contain exactly one outcome.")

    atlas, names = load_label_atlas(atlas_path, lut_path)
    labels = image.get_data(atlas)

    networks = names.str.extract(
        r"^7Networks_(?:LH|RH)_([^_]+)(?:_|$)",
        expand=False,
    )

    if networks.isna().any():
        raise ValueError(
            f"Unrecognized atlas labels: {names[networks.isna()].tolist()}"
        )

    # Each Series maps atlas label IDs to correlation feature names.
    roi_features = "E_mean_" + names
    network_features = "E_mean_" + networks + "_both_hemis"

    known_features = set(roi_features) | set(network_features)
    unexpected = set(results["x"]) - known_features

    if unexpected:
        raise ValueError(f"Unrecognized features: {sorted(unexpected)}")

    features = roi_features if level == "roi" else network_features
    required = set(features)

    selected = results.loc[results["x"].isin(required)]
    missing = required - set(selected["x"])

    if missing:
        raise ValueError(f"Missing {level} features: {sorted(missing)}")

    rho = pd.to_numeric(
        selected.set_index("x")["rho"],
        errors="raise",
    )

    if not np.isfinite(rho.to_numpy(dtype=float)).all():
        raise ValueError("Correlation coefficients contain non-finite values.")

    if rho.abs().gt(1).any():
        raise ValueError("Correlation coefficients must be within [-1, 1].")

    rho_volume = np.zeros(labels.shape, dtype=np.float32)

    for label_id, feature in features.items():
        rho_volume[labels == label_id] = rho.loc[feature]

    rho_img = image.new_img_like(atlas, rho_volume, copy_header=True)
    rho_img.set_data_dtype(np.float32)

    return rho_img


def plot_save_parent_network(
    results,
    *,
    network,
    model_type,
    step,
    atlas_path,
    lut_path,
    template_path,
    output_dir,
    y="cgi_change",
):
    """Save regional and bilateral-network maps for one parent/configuration."""
    atlas, names = load_label_atlas(atlas_path, lut_path)
    labels = image.get_data(atlas)

    # Select this parent's atlas regions, keeping both hemispheres.
    roi_names = names.loc[
        names.str.split("_").str[2].eq(network)
    ]

    hemispheres = set(roi_names.str.split("_").str[1])
    if roi_names.empty or hemispheres != {"LH", "RH"}:
        raise ValueError(
            f"Atlas must contain LH and RH regions for {network!r}."
        )

    roi_features = ("E_mean_" + roi_names).tolist()
    bilateral_feature = f"E_mean_{network}_both_hemis"
    required = [*roi_features, bilateral_feature]

    # Select one configuration and outcome.
    selected = results.loc[
        results["model_type"].eq(model_type)
        & results["step"].eq(step)
        & results["y"].eq(y)
    ].copy()

    # Support older CSVs; normalized duplicate names still raise below.
    selected["x"] = selected["x"].str.replace(
        r"^E_mean__7Networks_",
        "E_mean_7Networks_",
        regex=True,
    )

    selected = selected.loc[selected["x"].isin(required)]

    if selected["x"].duplicated().any():
        duplicates = selected.loc[
            selected["x"].duplicated(keep=False), "x"
        ].tolist()
        raise ValueError(f"Duplicate correlation features: {duplicates}")

    missing = set(required) - set(selected["x"])
    if missing:
        raise ValueError(
            f"{model_type}/{step}/{network}: "
            f"missing correlation features: {sorted(missing)}"
        )

    selected = selected.set_index("x").loc[required].copy()
    rho = pd.to_numeric(selected["rho"], errors="raise")

    if not np.isfinite(rho.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite correlation coefficients.")

    if rho.abs().gt(1).any():
        raise ValueError("Correlation coefficients must be within [-1, 1].")

    selected["rho"] = rho

    parent_mask = np.isin(labels, roi_names.index)
    regional_volume = np.zeros(labels.shape, dtype=np.float32)

    # Choose a representative axial slice for every component ROI.
    # Both map types use these same deterministic slice coordinates.
    cuts = []

    for label_id, roi_name in roi_names.items():
        roi_mask = labels == label_id
        regional_volume[roi_mask] = rho.loc[f"E_mean_{roi_name}"]

        k = int(roi_mask.sum(axis=(0, 1)).argmax())
        i, j = np.argwhere(roi_mask[:, :, k])[0]
        world_z = (atlas.affine @ [i, j, k, 1])[2]
        cuts.append(float(world_z))

    cuts = sorted(set(cuts))

    bilateral_volume = np.zeros(labels.shape, dtype=np.float32)
    bilateral_volume[parent_mask] = rho.loc[bilateral_feature]

    # Transparency hides everything outside this parent network,
    # while allowing a genuine rho=0 region to remain visible.
    mask_img = image.new_img_like(
        atlas,
        parent_mask.astype(np.uint8),
        copy_header=False,
    )
    background = image.load_img(template_path)

    jobs = [
        (
            "regional_ROIs_LH_RH_separate",
            regional_volume,
            roi_features,
            "Each LH/RH region has its own correlation coefficient",
        ),
        (
            "network_mean_both_hemis",
            bilateral_volume,
            [bilateral_feature],
            "One network coefficient: regional E means combined across LH + RH",
        ),
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    for tag, volume, features, description in jobs:
        prefix = (
            f"spearman_rho_{model_type}_{step}_{y}_"
            f"parent_{network}_{tag}"
        )

        paths = {
            "nifti": output_dir / f"{prefix}.nii.gz",
            "png": output_dir / f"{prefix}.png",
            "csv": output_dir / f"{prefix}.csv",
        }

        rho_img = image.new_img_like(atlas, volume, copy_header=True)
        rho_img.set_data_dtype(np.float32)
        rho_img.header["cal_min"] = -1
        rho_img.header["cal_max"] = 1
        rho_img.to_filename(paths["nifti"])

        # Regional CSV: one row per component ROI.
        # Bilateral CSV: exactly one network-level correlation row.
        selected.loc[features].reset_index().to_csv(
            paths["csv"],
            index=False,
        )

        figure = plt.figure(
            figsize=(max(10, 2 * len(cuts) + 1), 4)
        )

        display = plotting.plot_stat_map(
            rho_img,
            bg_img=background,
            figure=figure,
            display_mode="z",
            cut_coords=cuts,
            threshold=None,
            transparency=mask_img,
            transparency_range=[0, 1],
            resampling_interpolation="nearest",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            symmetric_cbar=True,
            colorbar=True,
            cbar_tick_format="%.2f",
            title=(
                f"Parent network: {network} | {model_type} | {step}\n"
                f"{description}\nSpearman rho vs {y}"
            ),
        )

        display.savefig(paths["png"], dpi=300)
        display.close()

        saved[tag] = paths

    return saved


def parser() -> argparse.ArgumentParser:
    """Command-line defaults mirror the source notebook."""
    result = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    atlas_dir = WORKFLOW_DIR / "atlases" / "Yeo2011"
    result.add_argument("--query", type=Path, default=PROJECT_DIR / "ect-results-tools" / "query_result_2.csv")
    result.add_argument(
        "--clinical", type=Path,
        default=PROJECT_DIR / "datasets" / "STU00225089" / ".clincal_notes" / "cgi_batch_latest_20261029_1207.xlsx",
        help="Workbook containing both Sheet1 (courses) and stimulus (sessions).",
    )
    result.add_argument("--atlas", type=Path, default=atlas_dir / "Yeo2011_7Networks_N1000.split_components.FSL_MNI152_1mm.nii.gz")
    result.add_argument("--lut", type=Path, default=atlas_dir / "7Networks_ColorLUT_freeview.txt")
    result.add_argument(
        "--template", type=Path,
        default=PROJECT_DIR / "simnibs_dev" / "simnibs" / "simnibs" / "resources" / "templates" / "MNI152_T1_1mm.nii.gz",
        help="FSL MNI152 T1 background matching the atlas's template variant.",
    )
    result.add_argument("--subject-scan-map", type=Path, default=ANALYSIS_DIR / "subject_scan_map.json")
    result.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "data")
    result.add_argument("--field", default="tdcs_uq_gpc_MNI_magnE_mean.nii.gz", help="Exact NIfTI filename in the query inventory.")
    result.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    result.add_argument("--steps", nargs="+", default=list(STEPS))
    result.add_argument("--networks", nargs="+", choices=NETWORKS, default=list(NETWORKS), help="Parent networks to plot; all seven remain in the statistical analysis.")
    result.add_argument("--session-num", type=int, default=2)
    result.add_argument("--threshold", type=int, default=95)
    result.add_argument("--asf", default="asf_5")
    result.add_argument("--step-result", default="mni_volumes")
    result.add_argument("--exclude-subjects", nargs="*", default=["subj-cat-010-001"], help="Exact subject IDs to exclude; pass this flag alone for none.")
    result.add_argument("--mode-decimals", type=int, default=None, help="Optional rounding for the mode calculation only.")
    result.add_argument("--skip-plots", action="store_true", help="Save tables and correlations without producing any images.")
    result.add_argument("--skip-overview-plots", action="store_true", help="Skip whole-atlas figures; retain per-parent network figures.")
    result.add_argument("--skip-parent-plots", action="store_true", help="Skip per-parent network figures; retain whole-atlas figures.")
    return result


def series_stem(field_filename: str, session_num: int, model_type: str, step: str) -> str:
    """Preserve notebook names for defaults; describe alternate scalar fields."""
    filename = Path(field_filename).name
    if filename != field_filename:
        raise ValueError("--field must be an exact basename without directories.")
    if filename.endswith(".nii.gz"):
        field = filename[:-7]
    elif filename.endswith(".nii"):
        field = filename[:-4]
    else:
        raise ValueError("--field must identify a .nii or .nii.gz scalar image.")
    field = field.removeprefix("tdcs_uq_gpc_MNI_")
    return f"series_{field}_MNI152_session{session_num}_{step.replace('_', '')}_{model_type}"


def save_overview_plots(results: pd.DataFrame, args: argparse.Namespace, destination: Path) -> None:
    """Save the notebook's two whole-atlas summaries for each configuration."""
    destination.mkdir(parents=True, exist_ok=True)
    background = load_scalar_img(args.template)
    descriptions = {
        "roi": "Regional ROIs within networks: each LH/RH ROI has its own rho",
        "network": "Whole networks: regional E means combined across LH + RH",
    }
    for (model_type, step), table in results.groupby(["model_type", "step"], sort=False):
        for level, description in descriptions.items():
            rho_img = network_correlations_to_image(table, args.atlas, args.lut, level=level)
            stem = f"spearman_yeo2011_{model_type}_{step}_{level}"
            rho_img.to_filename(destination / f"{stem}_rho.nii.gz")
            display = plotting.plot_stat_map(
                rho_img,
                bg_img=background,
                display_mode="z",
                cut_coords=[-20, 0, 20, 40, 60],
                threshold=0,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                symmetric_cbar=True,
                resampling_interpolation="nearest",
                colorbar=True,
                cbar_tick_format="%.2f",
                title=f"{model_type} | {step} | Spearman rho vs cgi_change\n{description}",
            )
            try:
                display.savefig(destination / f"{stem}_rho.png", dpi=300)
            finally:
                display.close()


def run(args: argparse.Namespace) -> Path:
    """Execute preparation, extraction, correlations, and optional saved plots."""
    # Reject duplicate configuration arguments instead of repeating work or
    # silently overwriting a dictionary entry or output file.
    for name in ("models", "steps", "networks"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            raise ValueError(f"--{name} must contain unique values.")

    if not args.steps or any(not re.fullmatch(r"step_\d+", step) for step in args.steps):
        raise ValueError("--steps must contain labels such as step_0 and step_9.")
    # Check the filename before expensive input loading.
    series_stem(args.field, args.session_num, args.models[0], args.steps[0])

    required_paths = [("query", args.query), ("clinical", args.clinical), ("atlas", args.atlas), ("lut", args.lut)]
    if args.subject_scan_map is not None:
        required_paths.append(("subject scan map", args.subject_scan_map))
    plots_enabled = not args.skip_plots and not (args.skip_overview_plots and args.skip_parent_plots)
    if plots_enabled:
        required_paths.append(("MNI template", args.template))
    for label, path in required_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {label} input: {path}")

    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    common_filters = {"threshold": args.threshold, "step_result": args.step_result, "asf": args.asf}

    print("[1/5] Preparing clinical series and mapping simulation files", flush=True)
    series_tables = {}
    for model_type in args.models:
        for step in args.steps:
            table = load_series_data(
                subjects_exclude=args.exclude_subjects,
                fn_query=args.query,
                fn_cgi=args.clinical,
                query_filters={**common_filters, "model_type_dir": model_type, "step": step},
                fn_field=args.field,
                subject_scan_map=args.subject_scan_map,
                session_num=args.session_num,
            )
            series_tables[(model_type, step)] = table
            print(f"  {model_type}/{step}: {len(table)} aligned series", flush=True)

    # The notebook writes the prepared tables before loading them for ROI work.
    for (model_type, step), table in series_tables.items():
        stem = series_stem(args.field, args.session_num, model_type, step)
        table.to_csv(destination / f"{stem}.csv", index=False)

    print("[2/5] Extracting ROI statistics once per unique field file/configuration", flush=True)
    series_tables_with_roi = {}
    for model_type, step in series_tables:
        stem = series_stem(args.field, args.session_num, model_type, step)
        table = pd.read_csv(destination / f"{stem}.csv")
        unique_fields = table[["full_file_path"]].drop_duplicates().reset_index(drop=True)
        roi_stats = extract_region_stats(
            unique_fields,
            atlas_path=args.atlas,
            lut_path=args.lut,
            path_column="full_file_path",
            mode_decimals=args.mode_decimals,
        )
        field_stats = pd.concat([unique_fields, *roi_stats.values()], axis=1)
        result = table.merge(field_stats, on="full_file_path", how="left", sort=False, validate="many_to_one")
        result.to_csv(destination / f"{stem}_yeo2011_7network_ROI.csv", index=False)
        series_tables_with_roi[(model_type, step)] = result
        print(f"  {model_type}/{step}: {len(result)} series, {len(unique_fields)} unique field files", flush=True)

    print("[3/5] Adding seven network means alongside the regional mean features", flush=True)
    feature_tables = {
        configuration: add_network_means_both_hemis(keep_cgi_and_field(table))
        for configuration, table in series_tables_with_roi.items()
    }
    save_field_roi_tables(feature_tables, destination / "yeo2011_field_ROIs_for_spearman_all_configurations.csv")

    print("[4/5] Computing Spearman correlations for every regional and network feature", flush=True)
    spearman_results = {
        configuration: run_spearman(table, y="cgi_change")
        for configuration, table in feature_tables.items()
    }
    combined = pd.concat(
        [table.assign(model_type=model_type, step=step)
         for (model_type, step), table in spearman_results.items()],
        ignore_index=True,
    )
    correlation_path = destination / "spearman_yeo2011_ROI_and_network_annotated.csv"
    save_spearman_table(combined, correlation_path)
    print(f"  Saved {len(combined)} correlations to {correlation_path}", flush=True)

    if plots_enabled:
        print("[5/5] Saving MNI152 correlation maps and figures", flush=True)
        # Preserve the notebook's CSV roundtrip before plotting its coefficients.
        all_results = pd.read_csv(correlation_path)
        if not args.skip_overview_plots:
            save_overview_plots(all_results, args, destination / "spearman_maps")
        if not args.skip_parent_plots:
            count = 0
            for model_type in args.models:
                for step in args.steps:
                    for network in args.networks:
                        plot_save_parent_network(
                            all_results,
                            network=network,
                            model_type=model_type,
                            step=step,
                            atlas_path=args.atlas,
                            lut_path=args.lut,
                            template_path=args.template,
                            output_dir=destination / "spearman_parent_network_maps",
                            y="cgi_change",
                        )
                        count += 2
                        print(f"  Saved regional and bilateral maps: {model_type}/{step}/{network}", flush=True)
            print(f"  Saved {count} parent-network NIfTI/PNG/CSV sets", flush=True)
    else:
        print("[5/5] Plotting skipped", flush=True)

    print(f"Analysis output: {destination}", flush=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
