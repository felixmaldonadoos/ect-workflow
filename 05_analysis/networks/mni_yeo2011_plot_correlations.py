#!/usr/bin/env python3
"""
Standalone version of mni_yeo2011_plot_correlations.ipynb.

Save as:
    mni_yeo2011_plot_correlations.py

Examples:
    python mni_yeo2011_plot_correlations.py

    python mni_yeo2011_plot_correlations.py \
        --correlations-path /path/to/permutation_results.csv

    python mni_yeo2011_plot_correlations.py \
        --networks Default Cont --no-overviews

Reads saved correlations without recomputing statistics.
Plots every observed model_type × step for the selected field/metric/outcome.
Saves NIfTI maps, PNGs, source-subset CSVs, and a plot manifest.
Extra columns, including permutation results, remain in source-subset CSVs.
No p-value mask is applied.
"""

import argparse
from pathlib import Path
import re

if __package__ in (None, ""):
    from _results import (
        add_analysis_argument,
        bootstrap_cli,
        data_dir,
        normalization_metadata,
        plot_output_dir,
        validate_output_path,
    )
else:
    from ._results import (
        add_analysis_argument,
        bootstrap_cli,
        data_dir,
        normalization_metadata,
        plot_output_dir,
        validate_output_path,
    )

if __name__ == "__main__":
    bootstrap_cli()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import image, plotting


BASE_DIR = Path("/projects/p32903/Alex2/ect-workflow")

DEFAULT_ATLAS = (
    BASE_DIR
    / "atlases/Yeo2011"
    / "Yeo2011_7Networks_N1000.split_components.FSL_MNI152_1mm.nii.gz"
)
DEFAULT_TEMPLATE = Path(
    "/projects/p32903/Alex2/simnibs_dev/simnibs/simnibs/"
    "resources/templates/MNI152_T1_1mm.nii.gz"
)

PARENTS = "Vis|SomMot|DorsAttn|SalVentAttn|Limbic|Cont|Default"
REGIONAL_RE = re.compile(rf"7Networks_(LH|RH)_({PARENTS})(?:_(.+))?")
BILATERAL_RE = re.compile(rf"({PARENTS})_both_hemis")


def annotate_correlation_features(results):
    """Validate source-aware feature names and their saved ROI metadata."""
    required = {"x", "field_name", "metric"}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"Missing feature identity columns: {sorted(missing)}")

    result = results.copy()
    records = []

    for feature, field_name, metric in result[
        ["x", "field_name", "metric"]
    ].itertuples(index=False, name=None):
        if (
            not isinstance(field_name, str)
            or not field_name
            or field_name != field_name.strip()
        ):
            raise ValueError(f"Invalid field_name: {field_name!r}")

        if not isinstance(metric, str) or not metric:
            raise ValueError(f"Invalid metric: {metric!r}")

        prefix = f"{field_name}_"
        if not isinstance(feature, str) or not feature.startswith(prefix):
            raise ValueError(
                f"Feature {feature!r} does not preserve field_name={field_name!r}."
            )

        label = feature[len(prefix):]
        regional = REGIONAL_RE.fullmatch(label)
        bilateral = BILATERAL_RE.fullmatch(label)

        if regional:
            hemi, network, component = regional.groups()
            records.append((label, component or network, hemi, network))
        elif bilateral:
            network = bilateral.group(1)
            records.append((None, network, "both", network))
        else:
            raise ValueError(f"Unrecognized feature: {feature!r}")

    metadata = pd.DataFrame(
        records,
        columns=["roi_org", "roi", "hemi", "network"],
        index=result.index,
    )

    for column in metadata:
        if column in result.columns:
            agrees = result[column].fillna("").eq(
                metadata[column].fillna("")
            )
            if not agrees.all():
                raise ValueError(
                    f"Saved {column!r} metadata conflicts with feature names."
                )
        result[column] = metadata[column]

    return result


def load_correlation_results(filename):
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Correlation CSV does not exist: {path}")

    results = pd.read_csv(path, low_memory=False)
    required = {
        "x", "y", "n", "rho", "p_value", "field_name", "metric",
        "model_type", "step", "roi_org", "roi", "hemi", "network",
    }
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"Missing correlation columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("Correlation CSV is empty.")

    keys = ["field_name", "metric", "model_type", "step", "y", "x"]
    if results[keys].isna().any().any():
        raise ValueError("Missing field/configuration/outcome/feature identifiers.")
    if results.duplicated(keys).any():
        raise ValueError("Duplicate field/configuration/outcome/feature rows.")

    results = annotate_correlation_features(results)
    results["rho"] = pd.to_numeric(results["rho"], errors="raise")

    if not np.isfinite(results["rho"].to_numpy(dtype=float)).all():
        raise ValueError("Correlation coefficients contain non-finite values.")
    if results["rho"].abs().gt(1).any():
        raise ValueError("Correlation coefficients must be within [-1, 1].")

    return results


def load_scalar_img(filename):
    img = image.load_img(filename)
    if len(img.shape) == 4 and img.shape[-1] == 1:
        img = img.slicer[..., 0]
    if len(img.shape) != 3:
        raise ValueError(f"Expected a 3D image; got {img.shape}")
    return img


def load_label_atlas(atlas_path, lut_path):
    atlas = load_scalar_img(atlas_path)
    labels = image.get_data(atlas)
    ids = np.unique(labels)

    if not np.isfinite(ids).all() or not np.equal(ids, np.rint(ids)).all():
        raise ValueError("Atlas labels must be finite integers.")

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
        raise ValueError("Duplicate IDs in the atlas LUT.")

    names = lut.reindex(ids)
    if names.empty or names.isna().any() or not names.is_unique:
        raise ValueError("Every atlas ROI needs a unique LUT name.")

    parsed = names.str.extract(REGIONAL_RE)
    parsed.columns = ["hemi", "network", "roi"]
    if parsed[["hemi", "network"]].isna().any().any():
        raise ValueError("Unrecognized regional labels in the atlas LUT.")

    parsed["roi"] = parsed["roi"].fillna(parsed["network"])
    metadata = names.rename("roi_org").to_frame().join(parsed)

    return atlas, labels, metadata


def parent_cut_coordinates(atlas, labels, region_ids):
    """Choose the original notebook's representative axial slice per ROI."""
    cuts = []

    for label_id in region_ids:
        mask = labels == label_id
        k = int(mask.sum(axis=(0, 1)).argmax())
        i, j = np.argwhere(mask[:, :, k])[0]
        world_z = (atlas.affine @ [i, j, k, 1])[2]
        cuts.append(float(world_z))

    return sorted(set(cuts))


def build_plot_jobs(
    atlas,
    labels,
    metadata,
    field_name,
    networks=None,
    plot_overviews=True,
    plot_parents=True,
):
    available = metadata["network"].drop_duplicates().tolist()
    parents = available if networks is None else list(networks)

    if plot_parents:
        if not parents or len(parents) != len(set(parents)):
            raise ValueError("networks must contain unique parent names.")
        unknown = set(parents) - set(available)
        if unknown:
            raise ValueError(f"Unknown networks: {sorted(unknown)}")

    regional_mapping = {
        label_id: f"{field_name}_{row.roi_org}"
        for label_id, row in metadata.iterrows()
    }
    bilateral_mapping = {
        label_id: f"{field_name}_{row.network}_both_hemis"
        for label_id, row in metadata.iterrows()
    }

    jobs = []

    if plot_overviews:
        jobs.extend([
            {
                "family": "overview",
                "network": "all_networks",
                "plot_type": "all_networks_regional_ROIs_LH_RH_separate",
                "mapping": regional_mapping,
                "cuts": [-20, 0, 20, 40, 60],
                "description": (
                    "All parent networks: each LH/RH regional ROI has its own rho"
                ),
            },
            {
                "family": "overview",
                "network": "all_networks",
                "plot_type": "all_networks_network_means_both_hemis",
                "mapping": bilateral_mapping,
                "cuts": [-20, 0, 20, 40, 60],
                "description": (
                    "All parent networks: one bilateral network rho per parent"
                ),
            },
        ])

    if plot_parents:
        for parent in parents:
            regions = metadata.loc[metadata["network"].eq(parent)]

            if set(regions["hemi"]) != {"LH", "RH"}:
                raise ValueError(
                    f"Atlas must contain LH and RH regions for {parent!r}."
                )

            cuts = parent_cut_coordinates(atlas, labels, regions.index)

            jobs.extend([
                {
                    "family": "parent",
                    "network": parent,
                    "plot_type": "regional_ROIs_LH_RH_separate",
                    "mapping": {
                        label_id: regional_mapping[label_id]
                        for label_id in regions.index
                    },
                    "cuts": cuts,
                    "description": (
                        "Each LH/RH region has its own correlation coefficient"
                    ),
                },
                {
                    "family": "parent",
                    "network": parent,
                    "plot_type": "network_mean_both_hemis",
                    "mapping": {
                        label_id: bilateral_mapping[label_id]
                        for label_id in regions.index
                    },
                    "cuts": cuts,
                    "description": (
                        "One network coefficient: regional field means "
                        "combined across LH + RH"
                    ),
                },
            ])

    return jobs, set(regional_mapping.values()) | set(bilateral_mapping.values())


def map_output_paths(
    output_dir,
    job,
    *,
    field_name,
    metric,
    outcome,
    model_type,
    step,
    analysis_name=None,
):
    """Plan and validate every artifact before any map files are written."""
    output_dir = validate_output_path(output_dir, analysis_name)
    is_parent = job["family"] == "parent"
    family_dir = "parent_network_maps" if is_parent else "overview_maps"
    destination = validate_output_path(output_dir / family_dir, analysis_name)
    suffix = job["plot_type"]
    if is_parent:
        suffix = f"parent_{job['network']}_{suffix}"
    prefix = (
        f"spearman_rho_field-{field_name}_metric-{metric}_"
        f"{model_type}_{step}_{outcome}_{suffix}"
    )
    return {
        key: validate_output_path(
            destination / f"{prefix}{extension}", analysis_name
        )
        for key, extension in (
            ("nifti", ".nii.gz"), ("png", ".png"), ("csv", ".csv")
        )
    }


def save_map_set(
    table,
    job,
    *,
    atlas,
    labels,
    background,
    output_dir,
    field_name,
    metric,
    outcome,
    model_type,
    step,
    dpi=300,
    analysis_name=None,
):
    mapping = job["mapping"]
    features = list(dict.fromkeys(mapping.values()))
    missing = set(features) - set(table.index)
    if missing:
        raise ValueError(
            f"{model_type}/{step}/{job['network']}: "
            f"missing features: {sorted(missing)}"
        )

    paths = map_output_paths(
        output_dir,
        job,
        field_name=field_name,
        metric=metric,
        outcome=outcome,
        model_type=model_type,
        step=step,
        analysis_name=analysis_name,
    )

    source = table.loc[features].copy()
    volume = np.zeros(labels.shape, dtype=np.float32)

    # Bilateral maps use the saved network coefficient directly.
    # Regional coefficients are never averaged to construct network maps.
    for label_id, feature in mapping.items():
        volume[labels == label_id] = source.loc[feature, "rho"]

    rho_img = image.new_img_like(atlas, volume, copy_header=True)
    rho_img.set_data_dtype(np.float32)
    rho_img.header["cal_min"] = -1
    rho_img.header["cal_max"] = 1

    is_parent = job["family"] == "parent"
    paths["nifti"].parent.mkdir(parents=True, exist_ok=True)

    rho_img.to_filename(paths["nifti"])
    source.reset_index().to_csv(paths["csv"], index=False)

    plot_options = {"threshold": 0}
    if is_parent:
        parent_mask = np.isin(labels, list(mapping))
        mask_img = image.new_img_like(
            atlas,
            parent_mask.astype(np.uint8),
            copy_header=False,
        )
        plot_options = {
            "threshold": None,
            "transparency": mask_img,
            "transparency_range": [0, 1],
        }

    figure = plt.figure(
        figsize=(max(10, 2 * len(job["cuts"]) + 1), 4)
    )
    display = None

    try:
        display = plotting.plot_stat_map(
            rho_img,
            bg_img=background,
            figure=figure,
            display_mode="z",
            cut_coords=job["cuts"],
            resampling_interpolation="nearest",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            symmetric_cbar=True,
            colorbar=True,
            cbar_tick_format="%.2f",
            title=(
                f"{field_name} | metric: {metric} | {model_type} | {step}\n"
                f"Spearman rho vs {outcome} | parent: {job['network']}\n"
                f"{job['description']}"
            ),
            **plot_options,
        )
        display.savefig(paths["png"], dpi=dpi)
    finally:
        if display is not None:
            display.close()
        plt.close(figure)

    return {
        "field_name": field_name,
        "metric": metric,
        "model_type": model_type,
        "step": step,
        "outcome": outcome,
        "network": job["network"],
        "family": job["family"],
        "plot_type": job["plot_type"],
        "n_features": len(features),
        **{key: str(value) for key, value in paths.items()},
    }


def plot_correlation_configurations(
    results,
    *,
    atlas_path,
    lut_path,
    template_path,
    output_dir,
    field_name="magnE_mean",
    metric="mean",
    outcome="cgi_change",
    networks=None,
    plot_overviews=True,
    plot_parents=True,
    dpi=300,
    analysis_name=None,
):
    if not plot_overviews and not plot_parents:
        raise ValueError("Enable at least one plot family.")
    output_dir = validate_output_path(output_dir, analysis_name)

    selected = results.loc[
        results["field_name"].eq(field_name)
        & results["metric"].eq(metric)
        & results["y"].eq(outcome)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No correlations for field={field_name!r}, "
            f"metric={metric!r}, outcome={outcome!r}."
        )

    atlas, labels, metadata = load_label_atlas(atlas_path, lut_path)
    background = load_scalar_img(template_path)

    jobs, known_features = build_plot_jobs(
        atlas,
        labels,
        metadata,
        field_name,
        networks=networks,
        plot_overviews=plot_overviews,
        plot_parents=plot_parents,
    )

    groups = list(
        selected.groupby(["model_type", "step"], sort=False, observed=True)
    )

    # Validate all requested map sets before writing outputs.
    for (model_type, step), group in groups:
        features = set(group["x"])
        unexpected = features - known_features
        if unexpected:
            raise ValueError(
                f"{model_type}/{step}: unrecognized features: "
                f"{sorted(unexpected)}"
            )

        for job in jobs:
            missing = set(job["mapping"].values()) - features
            if missing:
                raise ValueError(
                    f"{model_type}/{step}/{job['network']}: "
                    f"missing features: {sorted(missing)}"
                )
            map_output_paths(
                output_dir,
                job,
                field_name=field_name,
                metric=metric,
                outcome=outcome,
                model_type=model_type,
                step=step,
                analysis_name=analysis_name,
            )

    records = []

    for (model_type, step), group in groups:
        print(
            f"Plotting {field_name} / {metric} / {model_type} / {step}",
            flush=True,
        )
        table = group.set_index("x")

        for job in jobs:
            records.append(
                save_map_set(
                    table,
                    job,
                    atlas=atlas,
                    labels=labels,
                    background=background,
                    output_dir=output_dir,
                    field_name=field_name,
                    metric=metric,
                    outcome=outcome,
                    model_type=model_type,
                    step=step,
                    dpi=dpi,
                    analysis_name=analysis_name,
                )
            )

    return pd.DataFrame(records)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_analysis_argument(parser)
    parser.add_argument("--field-name", default="magnE_mean")
    parser.add_argument("--metric", default="mean")
    parser.add_argument("--outcome", default="cgi_change")
    parser.add_argument(
        "--correlations-path",
        type=Path,
        help="Original or permutation correlation CSV.",
    )
    parser.add_argument("--atlas-path", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--lut-path", type=Path)
    parser.add_argument(
        "--template-path", type=Path, default=DEFAULT_TEMPLATE
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Plot directory under RESULTS_DIR/<analysis-name>/.",
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        help="Parents to plot, e.g. Default Cont. Overview maps remain complete.",
    )
    parser.add_argument("--no-overviews", action="store_true")
    parser.add_argument("--no-parents", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args(argv)

    if args.no_overviews and args.no_parents:
        parser.error("Enable at least one plot family.")
    if args.dpi < 1:
        parser.error("--dpi must be positive.")

    return args


def main(argv=None):
    args = parse_args(argv)

    correlations_path = args.correlations_path
    if correlations_path is None:
        correlations_path = data_dir(args.analysis_name) / (
            f"spearman_{args.field_name}_ROImean_and_network_annotated.csv"
        )
    correlations_path = correlations_path.expanduser().resolve()

    lut_path = args.lut_path
    if lut_path is None:
        lut_path = args.atlas_path.parent / "7Networks_ColorLUT_freeview.txt"

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir = validate_output_path(output_dir, args.analysis_name)

    results = load_correlation_results(correlations_path)
    metadata = normalization_metadata(
        results, args.field_name, source_path=correlations_path
    )
    if output_dir is None:
        output_dir = plot_output_dir(args.analysis_name, metadata)
    output_dir = validate_output_path(output_dir, args.analysis_name)
    manifest_path = validate_output_path(
        output_dir / "plot_manifest.csv", args.analysis_name
    )
    print(
        f"Loaded {len(results):,} correlations from {correlations_path}",
        flush=True,
    )

    manifest = plot_correlation_configurations(
        results,
        atlas_path=args.atlas_path,
        lut_path=lut_path,
        template_path=args.template_path,
        output_dir=output_dir,
        field_name=args.field_name,
        metric=args.metric,
        outcome=args.outcome,
        networks=args.networks,
        plot_overviews=not args.no_overviews,
        plot_parents=not args.no_parents,
        dpi=args.dpi,
        analysis_name=args.analysis_name,
    )

    for column, value in metadata.items():
        manifest[column] = value
    manifest["correlations_path"] = str(correlations_path.resolve())
    manifest_path = validate_output_path(manifest_path, args.analysis_name)
    manifest.to_csv(manifest_path, index=False)

    summary = (
        manifest.groupby(
            ["field_name", "metric", "model_type", "step", "family"],
            sort=False,
        )
        .size()
        .unstack("family", fill_value=0)
        .rename_axis(columns=None)
    )

    print(summary.to_string())
    print(f"Saved {len(manifest):,} map sets under {output_dir}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
