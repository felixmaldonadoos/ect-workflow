"""Export one model per feature, with continuous age and fixed or random sex."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess

import numpy as np
import pandas as pd

from analysis_paths import SOURCE_DIR, lmm_analysis_name, resolve_analysis_dir, validate_output_path
from . import __version__
from .adapters import read_and_prepare
from .fitting import (
    COEFFICIENT_COLUMNS, FIXED_TERMS, FORMULA, MODEL_ID, RANDOM_EFFECTS,
    FitSettings, fit_feature, sample_counts, validate_feature_design,
)
from .model_spec import get_model_spec
from .run_layout import ensure_run_manifest, run_configuration


FIXED_FORMULA = FORMULA
GROUP_COLUMNS = [
    "feature_type", "field_name", "normalization", "normalization_tag",
    "model_type", "step", "basis_id", "feature_id",
]
WIDE_TERMS = {"Intercept": "intercept", "E": "E", "E:ef_sf": "E_ef_sf", "age": "age"}
WIDE_VALUES = ["estimate", "std_error", "wald_z", "p_value", "ci_low", "ci_high"]


def exported_fit_settings(settings: FitSettings) -> dict:
    """Keep short CLI choices distinct from semantic model metadata columns."""
    values = asdict(settings)
    values["sex_effect_choice"] = values.pop("sex_effect", "random")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_paths(analysis_name: str, master_path: Path, digest: str, normalization: str) -> dict:
    """Source-specific filenames also make raw versus brain-normalized runs visible."""
    root = resolve_analysis_dir(analysis_name)
    safe_source = re.sub(r"[^A-Za-z0-9_.-]", "_", master_path.stem)[:110]
    stem = f"{safe_source}__{normalization}__{digest[:12]}"
    names = {
        "model_input": root / "data" / f"{stem}_model_input_long.csv",
        "coefficients": root / f"{stem}_coefficients.csv",
        "fit_status": root / f"{stem}_fit_status.csv",
        "lmm_results": root / f"{stem}_lmm_results.csv",
        "analysis": root / f"analysis_{stem}.json",
    }
    return {key: validate_output_path(path, analysis_name) for key, path in names.items()}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value).__name__} in analysis metadata")


def _write_json(path: Path, value: dict, analysis_name: str) -> None:
    target = validate_output_path(path, analysis_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing runs, including a concurrent invocation.
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=_json_default, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, frame: pd.DataFrame, analysis_name: str) -> None:
    target = validate_output_path(path, analysis_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def _provenance() -> dict:
    versions = {}
    for package in ("numpy", "pandas", "scipy", "statsmodels", "patsy", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE_DIR, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=SOURCE_DIR, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    files = sorted((SOURCE_DIR / "series_models").glob("*.py")) + sorted(SOURCE_DIR.glob("run_series*.py")) + [
        SOURCE_DIR / "analysis_paths.py",
        SOURCE_DIR / "networks/_results.py",
    ]
    return {
        "package_version": __version__, "python_version": platform.python_version(),
        "dependencies": versions, "git_revision": revision, "git_dirty": dirty,
        "source_sha256": {str(path.relative_to(SOURCE_DIR)): sha256_file(path)
                          for path in files if path.is_file()},
    }


def _feature_identity(group: pd.DataFrame, source: dict) -> dict:
    row = group.iloc[0]
    identity = {column: row[column] for column in GROUP_COLUMNS}
    identity.update(source)
    identity.update({
        "roi_name": row["feature_id"] if row["feature_type"] == "roi" else "",
        "pc_name": row["feature_id"] if row["feature_type"] == "pca" else "",
        "value_column": "mean" if row["feature_type"] == "roi" else "score",
        "is_brain_normalized": row["normalization"] == "brain_mean",
        "n_observations": len(group), "n_people": group["person_id"].nunique(),
    })
    # Feature/basis metadata are carried through without aggregating series data.
    for column in (
        "component_index", "explained_variance", "explained_variance_ratio",
        "cumulative_variance_ratio", "selected_for_correlation", "hemi", "network",
        "normalization_applies_to", "normalization_mask_path", "normalization_mask_rule",
    ):
        if column in group:
            values = group[column].drop_duplicates()
            if len(values) != 1:
                raise ValueError(f"Conflicting feature metadata in {column}: {identity['feature_id']}")
            identity[column] = values.iloc[0]
    return identity


def _wide_result(status: dict, coefficients: pd.DataFrame) -> dict:
    result = dict(status)
    for prefix in WIDE_TERMS.values():
        result.update({f"{prefix}_{value}": np.nan for value in WIDE_VALUES})
    result["inference_method"] = "asymptotic_normal_wald"
    terms = status.get("fixed_terms", "[]")
    terms = json.loads(terms) if isinstance(terms, str) else terms
    for term in terms or []:
        if term.startswith("C(sex)["):
            result.update({f"{term.replace('C(sex)', 'sex')}_{value}": np.nan for value in WIDE_VALUES})
    for row in coefficients.to_dict("records"):
        prefix = WIDE_TERMS.get(row["term"], row["term"].replace("C(sex)", "sex"))
        for value in WIDE_VALUES:
            result[f"{prefix}_{value}"] = row[value]
    return result


def _print_fit_summary(statuses: pd.DataFrame, *, prepare_only: bool) -> None:
    """Explain partial model failures separately from process-level crashes."""
    failed = statuses["status"].eq("failed")
    accepted_label = "Prepared designs" if prepare_only else "Fits with valid Wald inference"
    accepted_status = "prepared" if prepare_only else "ok"
    print(f"{accepted_label}: {int(statuses['status'].eq(accepted_status).sum())}/{len(statuses)}", flush=True)
    if not prepare_only:
        converged = statuses.get("converged", pd.Series(False, index=statuses.index)).eq(True)
        print(f"Optimizer converged: {int(converged.sum())}/{len(statuses)}", flush=True)
        print(
            "Converged fits rejected by inference/validity checks: "
            f"{int((failed & converged).sum())}", flush=True,
        )
        print(f"Failures without confirmed convergence: {int((failed & ~converged).sum())}", flush=True)
        if "any_variance_boundary" in statuses:
            boundary = statuses["any_variance_boundary"].eq(True)
            print(f"Failed fits with a variance at the recorded boundary: {int((failed & boundary).sum())}", flush=True)
    if failed.any():
        reasons = statuses.loc[failed, "error"].fillna("Unspecified failure").replace("", "Unspecified failure")
        print("Failure reasons:", flush=True)
        for reason, count in reasons.value_counts().items():
            print(f"  {count}: {reason}", flush=True)
        print(
            "All result tables were saved. Run completed successfully (exit code 0); "
            "per-feature failures remain recorded in the results and fit-status tables.",
            flush=True,
        )


def run_analysis(args) -> dict:
    """Return success after complete exports; retain every feature's fit status."""
    # Validate the output policy before reading data or creating any directories.
    args.analysis_name = lmm_analysis_name(args.analysis_name, getattr(args, "output_component", None))
    root = resolve_analysis_dir(args.analysis_name)
    settings = FitSettings(reml=args.reml, optimizer=args.optimizer, maxiter=args.maxiter,
                           sex_effect=getattr(args, "sex_effect", "random"))
    row_settings = exported_fit_settings(settings)
    source_path = Path(args.master_path).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"Master path is not a file: {source_path}")
    digest = sha256_file(source_path)
    prepared = read_and_prepare(
        source_path, args.feature_type, args.field_name,
        features=args.features, pca_scope=args.pca_scope,
    )
    if sha256_file(source_path) != digest:
        raise ValueError("Source master changed during preparation; rerun with a stable input")
    frame = prepared.frame.copy()
    spec = get_model_spec(settings.sex_effect, sex_levels=sorted(frame.sex.astype(str).unique()))
    normalization = prepared.metadata["normalization"]
    component = args.analysis_name.split("/")[2] if len(args.analysis_name.split("/")) == 3 else None
    if component is not None and component != f"{args.feature_type}_{normalization}":
        raise ValueError("Output component does not match the master feature type and normalization")
    paths = artifact_paths(args.analysis_name, source_path, digest, normalization)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Run artifacts already exist; choose a new --analysis-name: " + ", ".join(existing)
        )
    source = {
        "master_file": source_path.name, "source_master_path": str(source_path),
        "source_master_sha256": digest,
        "model_id": spec["model_id"], "fixed_formula": spec["fixed_formula"],
        "age_effect": "continuous_fixed", "sex_effect": spec["sex_effect"],
        "sex_effect_choice": settings.sex_effect,
        "sex_reference": spec["sex_reference"], "sex_levels": json.dumps(spec["sex_levels"]),
    }
    reserved = {*source, "is_brain_normalized", "roi_name_used", "pc_name_used"}
    if conflicts := reserved.intersection(frame):
        raise ValueError(f"Source master contains reserved modeling columns: {sorted(conflicts)}")
    for name, value in source.items():
        frame[name] = value
    frame["is_brain_normalized"] = normalization == "brain_mean"
    frame["roi_name_used"] = np.where(frame["feature_type"].eq("roi"), frame["feature_id"], "")
    frame["pc_name_used"] = np.where(frame["feature_type"].eq("pca"), frame["feature_id"], "")

    groups = list(frame.groupby(GROUP_COLUMNS, sort=True, dropna=False))
    if not groups:
        raise ValueError("No features selected for modeling")
    # Validate feature metadata before the first artifact is written.
    identities = [_feature_identity(group, source) for _, group in groups]
    plot_enabled = not args.prepare_only and not getattr(args, "no_fit_plots", False)
    plot_dpi = getattr(args, "plot_dpi", 120)
    if isinstance(plot_dpi, bool) or not isinstance(plot_dpi, int) or plot_dpi < 1:
        raise ValueError("plot_dpi must be a positive integer")
    source_id = paths["fit_status"].name.removesuffix("_fit_status.csv")
    if plot_enabled:
        from .fit_plot_workflow import preflight_fit_plots
        preflight_fit_plots(
            [{"identity": identity, "prediction": None, "error": "Fit pending"} for identity in identities],
            args.analysis_name, source_id,
        )
    manifest = {
        "schema_version": 2, "analysis_name": args.analysis_name,
        "analysis_dir": str(root), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), **source, "feature_selection": prepared.metadata,
        "model": {
            "fixed_formula": spec["fixed_formula"],
            "fixed_terms": spec["fixed_terms"],
            "variance_components": spec["variance_components"],
            "fit_backend": spec["fit_backend"],
            "sex_levels": spec["sex_levels"], "sex_reference": spec["sex_reference"],
            "age_effect": {"type": "continuous_fixed", "source_column": "age", "units": "years"},
            "sex_effect": spec["sex_effect"],
            "outer_group": spec["outer_group"], "group_random_intercept": False,
            "observation_unit": "treatment series", "patient_random_effect": False,
            "feature_models": "one ROI mean or PCA score per fit",
            "outcome": "saved cgi_change; unchanged (current masters: cgi_start - cgi_end)",
            "ef_sf": "saved factor from the same series row; no aggregation",
            "centering": "none", "standardization": "none",
            "inference": "asymptotic normal Wald; no small-sample correction",
            "multiplicity_correction": None,
        },
        "fit_settings": asdict(settings), "provenance": _provenance(),
        "planned_fits": len(groups), "artifacts": paths,
        "prepare_only": args.prepare_only,
        "fit_plots": {"enabled": plot_enabled, "dpi": plot_dpi,
                      "source_id": source_id, "reference_age": "within-fit mean",
                      "reference_ef_sf": "within-fit median"},
    }
    run_manifest = ensure_run_manifest(args.analysis_name,
        run_configuration(settings.sex_effect, asdict(settings), args.field_name, prepare_only=args.prepare_only),
        inputs=[str(source_path)])
    manifest["run_manifest"] = str(run_manifest)
    # Reserve the run before fitting, and always retain the full prepared master.
    _write_json(paths["analysis"], manifest, args.analysis_name)
    _write_csv(paths["model_input"], frame, args.analysis_name)
    coefficient_tables, statuses, wide, plot_records = [], [], [], []
    for index, ((_, group), identity) in enumerate(zip(groups, identities), 1):
        print(
            f"[{index}/{len(groups)}] {identity['model_type']}/{identity['step']} "
            f"{identity['feature_id']} ({len(group)} series)", flush=True,
        )
        coefficients = pd.DataFrame(columns=COEFFICIENT_COLUMNS)
        if args.prepare_only:
            diagnostics = sample_counts(group)
            try:
                diagnostics.update(validate_feature_design(group, sex_effect=settings.sex_effect))
                diagnostics.update(status="prepared", error="", warnings="")
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                diagnostics.update(status="failed", error=str(error), warnings="")
        else:
            coefficients, diagnostics = fit_feature(group, settings)
        status = {**identity, **diagnostics, **row_settings}
        if plot_enabled:
            from .fit_plot_workflow import make_fit_record
            plot_records.append(make_fit_record(group, status))
        statuses.append(status)
        wide.append(_wide_result(status, coefficients))
        if not coefficients.empty:
            for column, value in {**identity, **row_settings}.items():
                coefficients[column] = value
            coefficients["status"] = diagnostics["status"]
            coefficient_tables.append(coefficients)
    coefficient_frame = (
        pd.concat(coefficient_tables, ignore_index=True) if coefficient_tables else
        pd.DataFrame(columns=list(dict.fromkeys([*identities[0], *row_settings, "status", *COEFFICIENT_COLUMNS])))
    )
    status_frame = pd.DataFrame(statuses)
    result_frame = pd.DataFrame(wide)
    _write_csv(paths["coefficients"], coefficient_frame, args.analysis_name)
    _write_csv(paths["fit_status"], status_frame, args.analysis_name)
    _write_csv(paths["lmm_results"], result_frame, args.analysis_name)
    failed = int(status_frame["status"].eq("failed").sum())
    print(f"Results table: {paths['lmm_results']}", flush=True)
    print(f"Planned fits: {len(groups)}; failures: {failed}", flush=True)
    _print_fit_summary(status_frame, prepare_only=args.prepare_only)
    plots = None
    if plot_enabled:
        from .fit_plot_workflow import export_fit_plots
        plots = export_fit_plots(plot_records, args.analysis_name, source_id, dpi=plot_dpi,
                                 source_metadata={"analysis_manifest": str(paths["analysis"]),
                                                  "source_master_sha256": digest})
    return {"analysis_dir": root, "paths": paths, "planned_fits": len(groups),
            "failed_fits": failed, "fit_plots": plots, "exit_code": 0}
