"""Exploratory plots of saved, comparable maximum-likelihood AIC results.

Matplotlib is imported only after the selected results directory and its cache
have been validated. No plot pools PCA bases or averages model comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from analysis_paths import resolve_analysis_dir, validate_output_path


GROUP_COLUMNS = ["feature_type", "field_name", "normalization"]
REQUIRED_COLUMNS = [
    *GROUP_COLUMNS, "model_type", "step", "basis_id", "feature_id", "is_null",
    "aic_eligible", "aic_provisional", "aic_caveat", "aic_improvement_over_null",
    "aic_rank_global", "delta_aic_best_including_null",
]
MANIFEST_COLUMNS = ["path", "kind", *GROUP_COLUMNS, "notes"]


@dataclass(frozen=True)
class _PlotSpec:
    path: Path
    kind: str
    feature_type: str
    field_name: str
    normalization: str


def _segment(value: str) -> str:
    """Keep readable safe labels, with a digest whenever conversion loses data."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    if safe and safe == value and len(safe) <= 48 and safe not in {".", ".."}:
        return safe
    stem = safe[:36].strip("._-") or "value"
    return f"{stem}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:10]}"


def _validated(table: pd.DataFrame) -> pd.DataFrame:
    if table.columns.has_duplicates:
        raise ValueError("Duplicate AIC plot-table column names")
    if missing := set(REQUIRED_COLUMNS) - set(table):
        raise ValueError(f"AIC plot table is missing columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("AIC plot table is empty")
    data = table.copy()
    for column in ("is_null", "aic_eligible", "aic_provisional"):
        if not data[column].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f"AIC plot flag must be Boolean: {column}")
    if data.is_null.sum() != 1:
        raise ValueError("AIC plots require exactly one shared null row")
    full = data.loc[~data.is_null]
    for column in [*GROUP_COLUMNS, "model_type", "step", "feature_id"]:
        if not full[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Incomplete AIC feature identity: {column}")
    data["basis_id"] = data.basis_id.fillna("").astype(str)
    for column in ("aic_improvement_over_null", "aic_rank_global", "delta_aic_best_including_null"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    eligible_features = ~data.is_null & data.aic_eligible
    if not np.isfinite(data.loc[eligible_features, "aic_improvement_over_null"]).all():
        raise ValueError("Eligible AIC feature rows need a finite null comparison")
    return data


def _plot_specs(table: pd.DataFrame, analysis_name: str) -> list[_PlotSpec]:
    data = _validated(table)
    root = resolve_analysis_dir(analysis_name)
    specifications = []
    for suffix in ("png", "pdf"):
        path = validate_output_path(root / "plots" / f"aic_ranked_features.{suffix}", analysis_name)
        specifications.append(_PlotSpec(path, "ranked_features", "all", "all", "all"))
    for values, _ in data.loc[~data.is_null].groupby(GROUP_COLUMNS, sort=True, dropna=False):
        stem = "aic_heatmap_" + "__".join(_segment(value) for value in values)
        for suffix in ("png", "pdf"):
            path = validate_output_path(root / "plots" / f"{stem}.{suffix}", analysis_name)
            specifications.append(_PlotSpec(path, "heatmap", *values))
    if len({spec.path for spec in specifications}) != len(specifications):
        raise ValueError("AIC plot labels produce conflicting artifact paths")
    return specifications


def plan_plot_paths(table: pd.DataFrame, analysis_name: str) -> list[Path]:
    """Return every PNG/PDF path without creating directories or plotting."""
    return [spec.path for spec in _plot_specs(table, analysis_name)]


def _natural_key(value: str) -> tuple:
    return tuple((0, int(piece)) if piece.isdigit() else (1, piece.casefold())
                 for piece in re.split(r"(\d+)", str(value)))


def _feature_name(row: pd.Series) -> str:
    column = "roi_name" if row.feature_type == "roi" else "pc_name"
    value = row.get(column, "")
    return value if isinstance(value, str) and value.strip() else row.feature_id


def _short_basis(value: str) -> str:
    if len(value) <= 20:
        return value
    return value[:10] + "…" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:7]


def _feature_label(row: pd.Series) -> str:
    label = (f"{_feature_name(row)} | {row.feature_type}/{row.field_name}/{row.normalization}"
             f" | {row.model_type}/{row.step}")
    if row.basis_id:
        label += f" | basis={_short_basis(row.basis_id)}"
    return label


def _ranked_rows(table: pd.DataFrame, top_n: int) -> pd.DataFrame:
    features = table.loc[~table.is_null & table.aic_eligible].copy()
    if not features.empty:
        features["_stable_identity"] = features.apply(_feature_label, axis=1)
        features = features.sort_values(
            ["aic_improvement_over_null", "_stable_identity"], ascending=[False, True], kind="stable",
        ).head(top_n).drop(columns="_stable_identity")
    null = table.loc[table.is_null].copy()
    null["aic_improvement_over_null"] = 0.0
    # Retaining the reference as its own row makes it visible even when it is
    # outside the top N, or its fitted likelihood is marked provisional.
    return pd.concat([features, null], ignore_index=True)


def _heatmap_data(group: pd.DataFrame) -> dict:
    """Map each candidate to one cell; duplicate identities are never averaged."""
    names = {}
    for _, row in group.iterrows():
        name = _feature_name(row)
        if row.feature_id in names and names[row.feature_id] != name:
            raise ValueError(f"Conflicting full feature names for {row.feature_id}")
        names[row.feature_id] = name
    features = sorted(names, key=lambda value: (_natural_key(names[value]), _natural_key(value)))
    configs = sorted(set(zip(group.model_type, group.step, group.basis_id)),
                     key=lambda values: tuple(_natural_key(value) for value in values))
    row_index = {value: index for index, value in enumerate(features)}
    col_index = {value: index for index, value in enumerate(configs)}
    values = np.full((len(features), len(configs)), np.nan)
    provisional = np.zeros(values.shape, dtype=bool)
    present = np.zeros(values.shape, dtype=bool)
    for _, row in group.iterrows():
        index = (row_index[row.feature_id], col_index[(row.model_type, row.step, row.basis_id)])
        if present[index]:
            raise ValueError("Duplicate AIC heatmap cell; feature/configuration/basis rows cannot be pooled")
        present[index] = True
        if row.aic_eligible:
            values[index] = float(row.aic_improvement_over_null)
            provisional[index] = row.aic_provisional
    labels = [f"{model}/{step}" + (f"\nbasis={_short_basis(basis)}" if basis else "")
              for model, step, basis in configs]
    return {"values": values, "provisional": provisional, "present": present,
            "row_labels": [names[value] for value in features], "column_labels": labels}


def _ranked_figure(table, top_n, plt, Patch, Line2D):
    selected = _ranked_rows(table, top_n)
    height = max(4.3, 0.38 * len(selected) + 2.0)
    fig, ax = plt.subplots(figsize=(14, height))
    labels = []
    for index, (_, row) in enumerate(selected.iterrows()):
        improvement = row.aic_improvement_over_null
        if row.is_null:
            labels.append("Shared null | age + sex random intercept")
            ax.plot(0, index, marker="D", markersize=7, linestyle="none",
                    markerfacecolor="white" if row.aic_provisional else "black", markeredgecolor="black")
        else:
            labels.append(_feature_label(row))
            color = "#2878a0" if improvement >= 0 else "#d78236"
            ax.barh(index, improvement, color=color, edgecolor="black", linewidth=0.6,
                    hatch="///" if row.aic_provisional else None, alpha=0.85)
            if row.aic_provisional:
                ax.plot(improvement, index, marker="o", markersize=5, linestyle="none",
                        markerfacecolor="white", markeredgecolor="black")
    ax.set_yticks(np.arange(len(selected)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    left, right = ax.get_xlim()
    padding = max(right - left, 1.0) * 0.03
    ax.set_xlim(min(left, -padding), max(right, padding))
    ax.set_xlabel("AIC(null) − AIC(feature): positive values favor the feature model")
    ax.set_title(f"Top {min(top_n, len(selected) - 1)} eligible feature models and shared null\n"
                 "Exploratory within-dataset ML comparison", fontsize=12)
    ax.xaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor="#2878a0", label="Eligible ML comparison"),
                       Patch(facecolor="white", edgecolor="black", hatch="///",
                             label="Provisional: boundary or invalid Wald inference"),
                       Line2D([], [], marker="D", color="black", linestyle="none", label="Shared null (0)")],
              loc="best", fontsize=8)
    fig.text(0.01, 0.01, "AIC compares relative fit with a complexity penalty; these are not p-values. "
             "Excluded fits are omitted from this ranking and remain visible in the heatmaps.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def _heatmap_figure(group, identity, plt, Patch, Line2D, TwoSlopeNorm):
    matrix = _heatmap_data(group)
    values = matrix["values"]
    finite = values[np.isfinite(values)]
    limit = max(float(np.max(np.abs(finite))) if finite.size else 0.0, 1.0)
    height = min(55, max(4.7, 0.25 * len(matrix["row_labels"]) + 2.8))
    width = min(32, max(9.5, 1.3 * len(matrix["column_labels"]) + 4.8))
    fig, ax = plt.subplots(figsize=(width, height))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#d3d3d3")
    plotted = ax.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="none", cmap=cmap,
                        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    ax.set_yticks(np.arange(values.shape[0]), matrix["row_labels"], fontsize=7)
    ax.set_xticks(np.arange(values.shape[1]), matrix["column_labels"], rotation=60, ha="right", fontsize=7)
    marked_rows, marked_columns = np.where(matrix["provisional"])
    ax.scatter(marked_columns + 0.32, marked_rows - 0.28, marker="o", s=26,
               facecolors="white", edgecolors="black", linewidths=0.9)
    if values.size <= 120:
        for (row, column), value in np.ndenumerate(values):
            if np.isfinite(value):
                ax.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.65 * limit else "black")
    feature_type, field, normalization = identity
    ax.set_title(f"{feature_type.upper()} | {field} | {normalization}\n"
                 "Exploratory within-dataset ML comparison", fontsize=12)
    fig.colorbar(plotted, ax=ax, label="AIC(null) − AIC(feature); positive favors the feature", shrink=0.75)
    fig.legend(handles=[Patch(facecolor="#d3d3d3", label="Excluded or unavailable comparison"),
                        Line2D([], [], marker="o", color="black", markerfacecolor="white", linestyle="none",
                               label="Provisional: boundary or invalid Wald inference")],
               loc="upper left", bbox_to_anchor=(0.01, 1.02), ncol=2, fontsize=8)
    caption = "Each cell is one feature/model/step/basis fit; no averaging. Colors are centered on the shared null (0)."
    if feature_type == "pca":
        caption += "\nPC indices represent different axes across PCA bases; matching PC numbers do not imply matching spatial patterns."
    fig.text(0.01, 0.01, caption, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def render_comparison_plots(
    table: pd.DataFrame, analysis_name: str, *, top_n: int = 20, dpi: int = 150,
) -> pd.DataFrame:
    """Export ranked and per-source heatmap PNG/PDF pairs below the analysis."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    data = _validated(table)
    specifications = _plot_specs(data, analysis_name)
    if existing := [str(spec.path) for spec in specifications if spec.path.exists()]:
        raise FileExistsError("AIC plots already exist; choose a new analysis name: " + ", ".join(existing))
    # Validate all cell identities before generating any artifacts.
    groups = {identity: group for identity, group in data.loc[~data.is_null].groupby(GROUP_COLUMNS, sort=True)}
    for group in groups.values():
        _heatmap_data(group)
    root = resolve_analysis_dir(analysis_name)
    cache = validate_output_path(root / "logs" / "matplotlib_cache", analysis_name)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    # An embedding process may already have imported matplotlib with another
    # cache setting; clear its memoized path before using the selected cache.
    for name in ("get_cachedir", "get_configdir"):
        resolver = getattr(matplotlib, name)
        if hasattr(resolver, "cache_clear"):
            resolver.cache_clear()
        elif hasattr(resolver, "__wrapped__"):
            # Matplotlib 3.10 uses a closure rather than functools.cache. A
            # fresh wrapper retains memoization while resolving the new env.
            setattr(matplotlib, name, functools.lru_cache(maxsize=1)(resolver.__wrapped__))
        if validate_output_path(getattr(matplotlib, name)(), analysis_name) != cache:
            raise RuntimeError("Matplotlib did not honor the selected analysis cache")
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    manifest = []
    figures = {}
    try:
        for spec in specifications:
            identity = (spec.feature_type, spec.field_name, spec.normalization)
            key = (spec.kind, *identity)
            if key not in figures:
                if spec.kind == "ranked_features":
                    figures[key] = _ranked_figure(data, top_n, plt, Patch, Line2D)
                else:
                    figures[key] = _heatmap_figure(groups[identity], identity, plt, Patch, Line2D, TwoSlopeNorm)
            path = validate_output_path(spec.path, analysis_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                figures[key].savefig(stream, format=path.suffix[1:], dpi=dpi, bbox_inches="tight")
            notes = (f"Top {top_n} eligible feature fits plus the shared null; provisional fits are hatched."
                     if spec.kind == "ranked_features" else
                     "One fit per cell; excluded/unavailable gray; provisional fits marked with open circles.")
            if spec.feature_type == "pca":
                notes += " PC indices denote different axes across PCA bases."
            manifest.append({"path": str(path), "kind": spec.kind,
                             **dict(zip(GROUP_COLUMNS, identity)), "notes": notes})
    finally:
        for figure in figures.values():
            plt.close(figure)
    return pd.DataFrame(manifest, columns=MANIFEST_COLUMNS)
