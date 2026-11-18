"""Render fitted series-model means and accounted-for unavailable fit pages."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd

from analysis_paths import resolve_analysis_dir, validate_output_path


IDENTITY_COLUMNS = [
    "feature_type", "field_name", "normalization", "model_type", "step", "basis_id", "feature_id",
]
MANIFEST_COLUMNS = [
    *IDENTITY_COLUMNS, "pdf_path", "pdf_page", "png_path", "plot_status", "plot_error", "provisional",
]


@dataclass(frozen=True)
class _FitSpec:
    index: int
    identity: dict
    report: Path
    png: Path
    page: int = 0


def _segment(value: str, limit: int = 48) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    if safe == value and safe not in {"", ".", ".."} and len(safe) <= limit:
        return safe
    stem = safe[:limit - 12].strip("._-") or "value"
    return f"{stem}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:10]}"


def _natural_key(value: str) -> tuple:
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold())
                 for part in re.split(r"(\d+)", str(value)))


def _plans(records, analysis_name: str, source_id: str):
    root = resolve_analysis_dir(analysis_name)
    if not isinstance(source_id, str) or not source_id.strip() or source_id != source_id.strip():
        raise ValueError("Fit-plot source_id must be a nonempty, unpadded string")
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("Fit plots need a nonempty list of planned-fit records")
    specifications = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("identity"), dict):
            raise ValueError("Every fit-plot record needs an identity dictionary")
        identity = dict(record["identity"])
        identity["basis_id"] = "" if identity.get("basis_id") is None else identity.get("basis_id", "")
        for column in IDENTITY_COLUMNS:
            if column not in identity or not isinstance(identity[column], str) or (
                column != "basis_id" and not identity[column].strip()
            ):
                raise ValueError(f"Invalid fit-plot identity column: {column}")
        # Hash the full plotting identity, including source distinctions that
        # might share the same readable feature/configuration labels.
        digest_identity = {column: identity[column] for column in IDENTITY_COLUMNS}
        digest = hashlib.sha256(json.dumps(digest_identity, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()[:12]
        directory = (root / "plots" / "model_fits" / _segment(source_id, 96)
                     / _segment(identity["model_type"]) / _segment(identity["step"]))
        report = validate_output_path(directory / "models.pdf", analysis_name)
        png = validate_output_path(directory / f"{_segment(identity['feature_id'], 80)}__{digest}.png", analysis_name)
        specifications.append(_FitSpec(index, identity, report, png))
    if len({spec.png for spec in specifications}) != len(specifications):
        raise ValueError("Duplicate planned fit identities would overwrite the same fit PNG")
    grouped = defaultdict(list)
    for spec in specifications:
        grouped[spec.report].append(spec)
    reports = sorted(grouped, key=lambda path: tuple(_natural_key(part) for part in path.parts))
    ordered = []
    for report in reports:
        page_specs = sorted(grouped[report], key=lambda spec: (
            _natural_key(spec.identity["feature_id"]), _natural_key(spec.identity["basis_id"]), str(spec.png),
        ))
        ordered.extend(_FitSpec(spec.index, spec.identity, spec.report, spec.png, page)
                       for page, spec in enumerate(page_specs, 1))
    cache = validate_output_path(root / "logs" / "matplotlib_cache", analysis_name)
    return specifications, ordered, reports, cache


def plan_fit_plot_paths(records, analysis_name: str, source_id: str) -> dict:
    """List exact report/PNG/cache paths without creating directories."""
    specifications, _, reports, cache = _plans(records, analysis_name, source_id)
    return {"reports": reports, "pngs": [spec.png for spec in specifications], "cache": cache}


def _prepare_matplotlib(cache: Path, analysis_name: str):
    cache = validate_output_path(cache, analysis_name)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    for name in ("get_cachedir", "get_configdir"):
        resolver = getattr(matplotlib, name)
        if hasattr(resolver, "cache_clear"):
            resolver.cache_clear()
        elif hasattr(resolver, "__wrapped__"):
            setattr(matplotlib, name, functools.lru_cache(maxsize=1)(resolver.__wrapped__))
        if validate_output_path(getattr(matplotlib, name)(), analysis_name) != cache:
            raise RuntimeError("Matplotlib did not honor the selected analysis cache")
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    return plt, PdfPages


def _title(identity: dict) -> str:
    name_column = "roi_name" if identity["feature_type"] == "roi" else "pc_name"
    name = identity.get(name_column)
    if not isinstance(name, str) or not name.strip():
        name = identity["feature_id"]
    lines = [str(name), f"{identity['feature_type'].upper()} | {identity['field_name']} | {identity['normalization']}",
             f"{identity['model_type']} / {identity['step']}"]
    if identity["basis_id"]:
        lines.append(f"PCA basis: {identity['basis_id']}")
    if identity.get("sex_effect") == "categorical_fixed":
        reference = identity.get("sex_reference")
        lines.append("Sex: fixed categories" + (f"; reference = {reference}" if reference else ""))
    return "\n".join(textwrap.fill(line, 105) for line in lines)


def _finite_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"Prediction table is missing {column}")
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Prediction table has nonfinite {column}")
    return values


def _prediction_data(prediction, identity):
    metadata = prediction.metadata
    if not isinstance(metadata, dict):
        raise ValueError("Fit-prediction metadata must be a dictionary")
    converged = metadata.get("converged", identity.get("converged"))
    if isinstance(converged, (bool, np.bool_)) and not converged:
        raise ValueError("Optimizer did not converge; fitted lines are unavailable")
    points, curves = prediction.points, prediction.curves
    if not isinstance(points, pd.DataFrame) or points.empty or not isinstance(curves, pd.DataFrame) or curves.empty:
        raise ValueError("Fit-prediction points and curves must be nonempty tables")
    x_column = metadata.get("x_column")
    expected_x = "age" if identity["feature_type"] == "null" else "E"
    if x_column != expected_x:
        raise ValueError(f"Expected fit-plot x column {expected_x}, received {x_column!r}")
    x = _finite_column(points, x_column)
    y_column = "observed_cgi_change" if x_column == "age" else "adjusted_cgi_change"
    y = _finite_column(points, y_column)
    if "sex" not in points or points.sex.isna().any() or not points.sex.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("Fit-prediction points need complete saved sex categories")
    if not {"curve_kind", "sex", "x", "predicted_cgi_change"}.issubset(curves):
        raise ValueError("Fit-prediction curves have an incomplete schema")
    if not curves.curve_kind.isin(["fixed", "sex"]).all() or not curves.curve_kind.eq("fixed").any():
        raise ValueError("Fit-prediction curves need a fixed-effect reference and known curve kinds")
    _finite_column(curves, "x")
    _finite_column(curves, "predicted_cgi_change")
    for (kind, sex), curve in curves.groupby(["curve_kind", "sex"], dropna=False):
        if kind == "fixed":
            reference = metadata.get("sex_reference")
            observed_x = (x[points.sex.eq(reference).to_numpy()]
                          if metadata.get("sex_effect") == "categorical_fixed" else x)
            if not len(observed_x):
                raise ValueError("Fixed-sex reference curve has no corresponding observed category")
        else:
            observed_x = x[points.sex.eq(sex).to_numpy()]
            if not len(observed_x):
                raise ValueError(f"Sex curve has no corresponding observed points: {sex!r}")
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(observed_x))))
        if curve.x.min() < observed_x.min() - tolerance or curve.x.max() > observed_x.max() + tolerance:
            raise ValueError("Fit-prediction curves extrapolate beyond their observed x range")
    return metadata, points, curves, x, y


def _number(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    return f"{number:.4g}" if np.isfinite(number) else "unavailable"


def _draw_prediction(fig, prediction, identity, plt):
    metadata, points, curves, x, y = _prediction_data(prediction, identity)
    provisional = bool(metadata.get("provisional", identity.get("provisional", False)))
    sex_effect = metadata.get("sex_effect", identity.get("sex_effect", "categorical_random_intercept"))
    if sex_effect not in {"categorical_fixed", "categorical_random_intercept"}:
        raise ValueError(f"Unknown sex-effect plot specification: {sex_effect!r}")
    fixed_sex = sex_effect == "categorical_fixed"
    sex_reference = metadata.get("sex_reference", identity.get("sex_reference"))
    if fixed_sex and (not isinstance(sex_reference, str) or sex_reference not in points.sex.unique()):
        raise ValueError("Fixed-sex plots require the saved reference sex category")
    ax = fig.add_subplot(111)
    sexes = sorted(points.sex.unique(), key=_natural_key)
    colors = {sex: plt.get_cmap("tab10")(index % 10) for index, sex in enumerate(sexes)}
    for sex in sexes:
        selected = points.sex.eq(sex).to_numpy()
        ax.scatter(x[selected], y[selected], color=colors[sex], edgecolors="white", linewidths=0.5,
                   alpha=0.85, s=32, label=f"{sex}: observations")
    for (kind, sex), curve in curves.groupby(["curve_kind", "sex"], dropna=False, sort=True):
        curve = curve.sort_values("x")
        if kind == "fixed":
            ax.plot(curve.x, curve.predicted_cgi_change, color="black", linestyle="--", linewidth=1.6, zorder=3,
                    label=(f"Fixed-effect reference (sex = {sex_reference})" if fixed_sex else
                           "Fixed-effect reference (zero random effect)"))
        else:
            ax.plot(curve.x, curve.predicted_cgi_change, color=colors[sex], linewidth=1.8,
                    label=f"{sex}: {'fixed category mean' if fixed_sex else 'conditional model mean'}")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Age (saved years)" if metadata["x_column"] == "age" else
                  f"{identity['feature_id']} (saved {'PCA score' if identity['feature_type'] == 'pca' else 'ROI mean'})")
    ax.set_ylabel("Observed CGI change" if metadata["x_column"] == "age" else
                  "CGI change adjusted to reference age and ef_sf")
    title_identity = {**identity, "sex_effect": sex_effect, "sex_reference": sex_reference}
    ax.set_title(_title(title_identity) + ("\nPROVISIONAL MODEL FIT" if provisional else ""), fontsize=11)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8, loc="best")
    references = "Saved age varies along the x-axis"
    if metadata["x_column"] == "E":
        references = (f"Reference age: {_number(metadata.get('reference_age'))} years; "
                      f"reference ef_sf: {_number(metadata.get('reference_ef_sf'))}")
    fit_method = metadata.get("fit_method", "ML")
    status = metadata.get("status", identity.get("status", "unavailable"))
    converged = metadata.get("converged", identity.get("converged", "unavailable"))
    aic = metadata.get("aic", identity.get("aic"))
    completion = "fit completed" if fixed_sex else "optimizer converged"
    curve_description = (f"Colored curves: fixed sex-category intercepts over each category's observed x range. "
                         f"Dashed curve: fixed-effect mean for reference sex {sex_reference}." if fixed_sex else
                         "Colored curves: sex random intercepts over each sex's observed x range. "
                         "Dashed curve: fixed-effect reference.")
    caption = (references + ".\n"
               f"{fit_method}; n = {metadata.get('n_observations', len(points))}; AIC = {_number(aic)}; "
               f"fit status = {status}; {completion} = {converged}.\n"
               f"{curve_description} No confidence intervals.")
    caveat = metadata.get("caveat", "")
    if caveat:
        caption += "\nCaveat: " + str(caveat)
    fig.text(0.02, 0.02, "\n".join(textwrap.fill(line, 145) for line in caption.splitlines()), fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.23 if caveat else 0.19, 1, 1))
    return provisional


def _draw_placeholder(fig, identity, error):
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_title(_title(identity) + "\nFIT PLOT UNAVAILABLE", fontsize=11)
    ax.text(0.05, 0.6, textwrap.fill(str(error), 105), transform=ax.transAxes, fontsize=11,
            va="center", wrap=True)
    fig.text(0.02, 0.03, "This planned fit is retained in the report. No fitted line is shown.", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))


def render_fit_plots(records, analysis_name: str, source_id: str, *, dpi: int = 120) -> pd.DataFrame:
    """Write bounded-memory PDF reports and one PNG for every planned fit.

    Invalid individual predictions become placeholder pages and recorded skips.
    Global path, identity, collision, and write errors remain fatal.
    """
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("Fit-plot dpi must be a positive integer")
    specifications, ordered, reports, cache = _plans(records, analysis_name, source_id)
    paths = [*reports, *(spec.png for spec in specifications)]
    if existing := [str(path) for path in paths if path.exists()]:
        raise FileExistsError("Fit plots already exist; choose a new analysis name: " + ", ".join(existing))
    plt, PdfPages = _prepare_matplotlib(cache, analysis_name)
    by_report = defaultdict(list)
    for spec in ordered:
        by_report[spec.report].append(spec)
    manifest = {}
    for report in reports:
        report = validate_output_path(report, analysis_name)
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("xb") as stream, PdfPages(stream) as pdf:
            for spec in by_report[report]:
                record = records[spec.index]
                prediction = record.get("prediction")
                error = ""
                provisional = False
                fig = plt.figure(figsize=(10.5, 7.6))
                try:
                    try:
                        if prediction is None:
                            raise ValueError(record.get("error") or "No finite, converged model prediction is available")
                        provisional = _draw_prediction(fig, prediction, spec.identity, plt)
                        status = "provisional" if provisional else "ok"
                    except Exception as failure:
                        error = str(failure) or type(failure).__name__
                        status = "skipped"
                        fig.clear()
                        _draw_placeholder(fig, spec.identity, error)
                    png = validate_output_path(spec.png, analysis_name)
                    with png.open("xb") as png_stream:
                        fig.savefig(png_stream, format="png", dpi=dpi, bbox_inches="tight")
                    pdf.savefig(fig, bbox_inches="tight")
                    manifest[spec.index] = {
                        **{column: spec.identity[column] for column in IDENTITY_COLUMNS},
                        "pdf_path": str(report), "pdf_page": spec.page, "png_path": str(png),
                        "plot_status": status, "plot_error": error, "provisional": provisional,
                    }
                finally:
                    plt.close(fig)
    return pd.DataFrame([manifest[index] for index in range(len(records))], columns=MANIFEST_COLUMNS)
