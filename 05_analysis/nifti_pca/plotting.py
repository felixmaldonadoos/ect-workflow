"""Plot saved NIfTI PCA bases and their component-score correlations.

No PCA or correlation is recomputed here.  Spatial maps show signed PCA
eigenvector weights, and scatter plots show each treatment series separately.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting

from .paths import validate_output_path


_IDENTITY_COLUMNS = (
    "basis_id", "model_type", "step", "field_name", "normalization"
)
_VARIANCE_COLUMNS = {
    "component", "component_index", "explained_variance",
    "explained_variance_ratio", "cumulative_variance_ratio",
    "selected_for_correlation",
}


def _selection_mask(values: pd.Series) -> pd.Series:
    """Accept boolean columns read either directly or back from a CSV."""
    normalized = values.astype(str).str.strip().str.lower()
    parsed = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if parsed.isna().any():
        invalid = values[parsed.isna()].unique().tolist()
        raise ValueError(f"Invalid selected_for_correlation values: {invalid}")
    return parsed.astype(bool)


def _filename(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not result:
        raise ValueError(f"Component cannot be used as a filename: {value!r}")
    return result


def _finite_numeric(values: pd.Series, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    return numeric


def _metadata(*tables: pd.DataFrame) -> dict[str, object]:
    result = {}
    for column in _IDENTITY_COLUMNS:
        values = pd.concat(
            [table[column] for table in tables if column in table],
            ignore_index=True,
        ) if any(column in table for table in tables) else pd.Series(dtype=object)
        unique = values.dropna().astype(str).unique()
        if len(unique) > 1:
            raise ValueError(
                f"plot_basis accepts one PCA basis; found multiple {column}: "
                f"{unique.tolist()}"
            )
        if len(unique):
            result[column] = unique[0]
    return result


def _title(metadata: dict[str, object]) -> str:
    return " | ".join(
        str(metadata[column]) for column in
        ("model_type", "step", "field_name", "normalization")
        if column in metadata
    )


def _p_label(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{numeric:.3g}" if np.isfinite(numeric) else "NA"


def _stat_label(row: pd.Series | None) -> str:
    if row is None:
        return ""
    parts = [f"Spearman rho={_p_label(row.get('rho'))}"]
    if "p_value" in row:
        parts.append(f"analytic p={_p_label(row['p_value'])}")
    if "p_permutation" in row:
        parts.append(f"permutation p={_p_label(row['p_permutation'])}")
    return "\n".join(parts)


def _validated_variance(variance: pd.DataFrame) -> pd.DataFrame:
    missing = _VARIANCE_COLUMNS.difference(variance.columns)
    if missing:
        raise ValueError(f"variance is missing columns: {sorted(missing)}")
    if variance.empty:
        raise ValueError("variance must contain at least one component")
    if variance["component"].isna().any() or variance["component"].duplicated().any():
        raise ValueError("variance component names must be nonmissing and unique")
    variance = variance.copy()
    indices = _finite_numeric(variance["component_index"], "variance.component_index")
    if (indices < 1).any() or (indices != np.floor(indices)).any() or len(set(indices)) != len(indices):
        raise ValueError("variance.component_index must contain unique positive integers")
    variance["component_index"] = indices.astype(int)
    for column in ("explained_variance", "explained_variance_ratio", "cumulative_variance_ratio"):
        values = _finite_numeric(variance[column], f"variance.{column}")
        if (values < 0).any() or (column.endswith("ratio") and (values > 1 + 1e-8).any()):
            raise ValueError(f"Invalid variance.{column} values")
        variance[column] = values
    variance["selected_for_correlation"] = _selection_mask(
        variance["selected_for_correlation"]
    )
    return variance.sort_values("component_index", kind="stable")


def _save_variance_plot(
    variance: pd.DataFrame, path: Path, title: str, dpi: int
) -> None:
    positions = np.arange(len(variance))
    selected = variance["selected_for_correlation"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(max(7.5, min(16, len(variance) * 0.3)), 4.8))
    try:
        ax.bar(
            positions,
            variance["explained_variance_ratio"].to_numpy() * 100,
            color=np.where(selected, "#2676a5", "#bfc4c8"),
            label="Component variance (blue = selected for correlation)",
        )
        tick_step = 1 if len(variance) <= 40 else int(np.ceil(len(variance) / 20))
        ax.set_xticks(positions[::tick_step])
        ax.set_xticklabels(variance["component"].iloc[::tick_step], rotation=45)
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance (%)")
        ax.set_ylim(bottom=0)
        cumulative_ax = ax.twinx()
        cumulative_ax.plot(
            positions,
            variance["cumulative_variance_ratio"].to_numpy() * 100,
            color="#ab4c32", marker=".", label="Cumulative variance",
        )
        cumulative_ax.set_ylim(0, 105)
        cumulative_ax.set_ylabel("Cumulative explained variance (%)")
        handles, labels = ax.get_legend_handles_labels()
        other_handles, other_labels = cumulative_ax.get_legend_handles_labels()
        fig.legend(handles + other_handles, labels + other_labels,
                   loc="lower center", frameon=False, fontsize=8)
        ax.set_title("PCA explained variance" + (f"\n{title}" if title else ""))
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(validate_output_path(path), dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_variance(
    *,
    variance: pd.DataFrame,
    output_dir: Path,
    metadata: Mapping[str, object] | None = None,
    dpi: int = 150,
) -> pd.DataFrame:
    """Plot every saved PC's variance without scores, outcomes, or NIfTI maps."""
    directory = validate_output_path(output_dir)
    if dpi < 1:
        raise ValueError("dpi must be positive")
    variance = _validated_variance(variance)
    identity = _metadata(variance, pd.DataFrame([dict(metadata or {})]))
    for column, value in identity.items():
        variance[column] = value
    source_path = validate_output_path(directory / "variance_source.csv")
    figure_path = validate_output_path(directory / "explained_variance.png")
    manifest_path = validate_output_path(directory / "variance_plot_manifest.csv")
    directory.mkdir(parents=True, exist_ok=True)
    variance.to_csv(source_path, index=False)
    _save_variance_plot(variance, figure_path, _title(identity), dpi)
    manifest = pd.DataFrame([
        {"path": str(source_path), "type": "variance_source", "component": None, **identity},
        {"path": str(figure_path), "type": "explained_variance", "component": None, **identity},
    ])
    manifest.to_csv(manifest_path, index=False)
    return manifest


def _save_component_map(
    component: str, map_path: Path, background_path: Path | None,
    destination: Path, title: str, dpi: int,
) -> None:
    img = nib.load(str(map_path))
    weights = np.asarray(img.dataobj)
    if weights.ndim != 3 or not np.isfinite(weights).all():
        raise ValueError(f"Expected a finite 3D component map: {map_path}")
    vmax = float(np.max(np.abs(weights)))
    if vmax == 0:
        raise ValueError(f"Component map contains only zero weights: {map_path}")
    fig = plt.figure(figsize=(10, 3.6))
    display = None
    try:
        display = plotting.plot_stat_map(
            img,
            bg_img=str(background_path) if background_path is not None else None,
            display_mode="ortho",
            cut_coords=None,
            threshold=None,
            cmap="RdBu_r",
            symmetric_cbar=True,
            vmax=vmax,
            colorbar=True,
            cbar_tick_format="%.2g",
            draw_cross=False,
            black_bg=False,
            figure=fig,
            title=f"{component}: signed PCA weight" + (f"\n{title}" if title else ""),
        )
        display.savefig(str(validate_output_path(destination)), dpi=dpi)
    finally:
        if display is not None:
            display.close()
        plt.close(fig)


def _save_scatter(
    component: str, rows: pd.DataFrame, correlation: pd.Series | None,
    path: Path, title: str, dpi: int,
) -> pd.DataFrame:
    source = rows.copy()
    scores = _finite_numeric(source["score"], "master.score")
    outcome = _finite_numeric(source["cgi_change"], "master.cgi_change")
    source["used_in_scatter"] = True
    fig, ax = plt.subplots(figsize=(6.5, 5))
    try:
        ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
        ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
        ax.scatter(scores, outcome, s=32, alpha=0.7,
                   color="#2676a5", edgecolors="none")
        annotation = f"n={len(source)} series"
        statistics = _stat_label(correlation)
        if statistics:
            annotation += "\n" + statistics
        ax.text(0.02, 0.98, annotation, transform=ax.transAxes,
                ha="left", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set_xlabel(f"{component} score")
        ax.set_ylabel("CGI change")
        ax.set_title(f"{component} score versus CGI change" +
                     (f"\n{title}" if title else ""))
        fig.tight_layout()
        fig.savefig(validate_output_path(path), dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return source


def _save_correlation_bars(
    rows: pd.DataFrame, path: Path, title: str, dpi: int
) -> None:
    rho = pd.to_numeric(rows["rho"], errors="coerce").to_numpy(dtype=float)
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(6.5, len(rows) * 0.75), 5))
    try:
        ax.axhline(0, color="0.6", linewidth=0.8)
        ax.bar(positions, rho, color=np.where(rho >= 0, "#ab4c32", "#2676a5"))
        ax.set_xticks(positions)
        ax.set_xticklabels(rows["component"], rotation=45)
        ax.set_ylim(-1.18, 1.18)
        ax.set_ylabel("Spearman rho with CGI change")
        ax.set_xlabel("Selected principal component")
        ax.set_title("Component-score correlations" + (f"\n{title}" if title else ""))
        has_permutation = "p_permutation" in rows and np.isfinite(
            pd.to_numeric(rows["p_permutation"], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
        ).any()
        p_column = "p_permutation" if has_permutation else "p_value"
        p_prefix = "perm p" if p_column == "p_permutation" else "p"
        for position, value, (_, row) in zip(positions, rho, rows.iterrows()):
            label = f"{p_prefix}={_p_label(row.get(p_column))}"
            if np.isfinite(value):
                ax.text(position, value + (0.04 if value >= 0 else -0.04), label,
                        ha="center", va="bottom" if value >= 0 else "top",
                        fontsize=8, rotation=0)
            else:
                ax.text(position, 0.04, "rho=NA\n" + label,
                        ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(validate_output_path(path), dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_basis(
    *,
    variance: pd.DataFrame,
    master: pd.DataFrame,
    correlations: pd.DataFrame,
    component_maps: Mapping[str, Path],
    background_path: Path | None,
    output_dir: Path,
    dpi: int = 150,
) -> pd.DataFrame:
    """Save plots and source CSVs for one already fitted PCA basis.

    ``master`` has one row per series and component.  All variance components
    appear on the scree plot; only components selected for correlation receive
    spatial maps and score scatter plots.  Correlation labels use saved values,
    without a significance filter.  Returned paths are absolute artifact paths;
    ``plot_manifest.csv`` contains the same records.
    """
    directory = validate_output_path(output_dir)
    if dpi < 1:
        raise ValueError("dpi must be positive")
    for table, required, name in (
        (variance, _VARIANCE_COLUMNS, "variance"),
        (master, {"component", "score", "cgi_change"}, "master"),
        (correlations, {"component", "rho"}, "correlations"),
    ):
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    variance = _validated_variance(variance)
    if correlations["component"].duplicated().any():
        raise ValueError("correlations must contain at most one row per component")
    _finite_numeric(master["score"], "master.score")
    _finite_numeric(master["cgi_change"], "master.cgi_change")
    metadata = _metadata(variance, master, correlations)
    title = _title(metadata)
    selected = variance.loc[variance["selected_for_correlation"], "component"].tolist()
    names = [_filename(str(component)) for component in selected]
    if len(set(names)) != len(names):
        raise ValueError("Component names produce conflicting output filenames")
    for component in selected:
        if component not in component_maps:
            raise ValueError(f"Missing NIfTI map for selected component {component}")
        if not Path(component_maps[component]).is_file():
            raise FileNotFoundError(component_maps[component])
        if not master["component"].eq(component).any():
            raise ValueError(f"No master rows for selected component {component}")
    if background_path is not None and not Path(background_path).is_file():
        raise FileNotFoundError(background_path)
    directory.mkdir(parents=True, exist_ok=True)
    records = plot_variance(
        variance=variance, output_dir=directory, metadata=metadata, dpi=dpi,
    ).to_dict("records")

    def record(path: Path, kind: str, component: str | None = None) -> None:
        records.append({"path": str(path.resolve()), "type": kind,
                        "component": component, **metadata})

    for table, filename, kind in (
        (master, "master_source.csv", "master_source"),
        (correlations, "correlations_source.csv", "correlations_source"),
    ):
        path = validate_output_path(directory / filename)
        table.to_csv(path, index=False)
        record(path, kind)
    indexed_correlations = correlations.set_index("component", drop=False)
    for component, filename in zip(selected, names):
        component = str(component)
        map_path = directory / f"{filename}_signed_weights.png"
        _save_component_map(component, Path(component_maps[component]),
                            background_path, map_path, title, dpi)
        record(map_path, "component_weights", component)
        rows = master.loc[master["component"].eq(component)].copy()
        correlation = (indexed_correlations.loc[component]
                       if component in indexed_correlations.index else None)
        scatter_path = directory / f"{filename}_score_vs_cgi_change.png"
        source = _save_scatter(component, rows, correlation, scatter_path, title, dpi)
        record(scatter_path, "score_scatter", component)
        source_path = validate_output_path(directory / f"{filename}_score_source.csv")
        source.to_csv(source_path, index=False)
        record(source_path, "score_source", component)
        if correlation is not None:
            source_path = validate_output_path(
                directory / f"{filename}_correlation_source.csv"
            )
            correlation.to_frame().T.to_csv(source_path, index=False)
            record(source_path, "correlation_source", component)
    selected_correlations = indexed_correlations.reindex(
        [component for component in selected if component in indexed_correlations.index]
    )
    if not selected_correlations.empty:
        path = directory / "component_correlations.png"
        _save_correlation_bars(selected_correlations, path, title, dpi)
        record(path, "correlation_bars")
    columns = ["path", "type", "component", *metadata]
    manifest = pd.DataFrame(records, columns=columns)
    manifest.to_csv(validate_output_path(directory / "plot_manifest.csv"), index=False)
    return manifest
