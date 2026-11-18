"""Fit the shared age/sex null from saved series-master covariates."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import json

from analysis_paths import lmm_analysis_name, resolve_analysis_dir, validate_output_path
from .adapters import read_and_prepare
from .cohort import canonical_cohort, cohort_sha256
from .fitting import (
    NULL_FIXED_TERMS, NULL_FORMULA, NULL_MODEL_ID, RANDOM_EFFECTS,
    FitSettings, fit_null, validate_null_design,
)
from .workflow import _print_fit_summary, _provenance, _wide_result, _write_csv, _write_json, exported_fit_settings, sha256_file
from .model_spec import get_model_spec
from .run_layout import ensure_run_manifest, run_configuration


def run_null_analysis(args) -> dict:
    args.analysis_name = lmm_analysis_name(args.analysis_name, "null")
    root = resolve_analysis_dir(args.analysis_name)
    settings = FitSettings(reml=args.reml, optimizer=args.optimizer, maxiter=args.maxiter,
                           sex_effect=getattr(args, "sex_effect", "random"))
    row_settings = exported_fit_settings(settings)
    source_path = Path(args.master_path).expanduser().resolve(strict=True)
    digest = sha256_file(source_path)
    # Selection concerns E, which is absent in the null. Validate the entire
    # supplied master before reducing its repeated feature/configuration rows.
    prepared = read_and_prepare(source_path, args.feature_type, args.field_name, pca_scope="all")
    cohort = canonical_cohort(prepared.frame)
    spec = get_model_spec(settings.sex_effect, null=True, sex_levels=sorted(cohort.sex.astype(str).unique()))
    if sha256_file(source_path) != digest:
        raise ValueError("Source master changed during null preparation")
    paths = {
        "analysis": root / "analysis.json", "model_input": root / "data/null_model_input.csv",
        "coefficients": root / "coefficients.csv", "fit_status": root / "fit_status.csv",
        "null_results": root / "null_results.csv",
    }
    paths = {name: validate_output_path(path, args.analysis_name) for name, path in paths.items()}
    if existing := [str(path) for path in paths.values() if path.exists()]:
        raise FileExistsError("Null artifacts already exist; choose a new --analysis-name: " + ", ".join(existing))
    identity = {
        "model_id": spec["model_id"], "fixed_formula": spec["fixed_formula"],
        "age_effect": "continuous_fixed", "sex_effect": spec["sex_effect"],
        "sex_effect_choice": settings.sex_effect,
        "sex_reference": spec["sex_reference"], "sex_levels": json.dumps(spec["sex_levels"]),
        "feature_type": "null", "feature_id": "null", "field_name": args.field_name,
        "normalization": "not_applicable", "normalization_tag": "not_applicable",
        "model_type": "shared_cohort", "step": "all", "basis_id": "",
        "master_file": source_path.name, "source_master_path": str(source_path),
        "source_master_sha256": digest, "cohort_sha256": cohort_sha256(cohort),
        "n_observations": len(cohort), "n_people": cohort.person_id.nunique(),
    }
    plot_enabled = not args.prepare_only and not getattr(args, "no_fit_plots", False)
    plot_dpi = getattr(args, "plot_dpi", 120)
    if isinstance(plot_dpi, bool) or not isinstance(plot_dpi, int) or plot_dpi < 1:
        raise ValueError("plot_dpi must be a positive integer")
    source_id = f"null__{digest[:12]}"
    if plot_enabled:
        from .fit_plot_workflow import preflight_fit_plots
        preflight_fit_plots([{"identity": identity, "prediction": None, "error": "Fit pending"}],
                            args.analysis_name, source_id)
    manifest = {
        "schema_version": 2, "analysis_type": "null", "analysis_name": args.analysis_name,
        "analysis_dir": str(root), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), **identity,
        "model": {
            "fixed_formula": spec["fixed_formula"], "fixed_terms": spec["fixed_terms"],
            "variance_components": spec["variance_components"], "outer_group": spec["outer_group"],
            "fit_backend": spec["fit_backend"], "sex_levels": spec["sex_levels"], "sex_reference": spec["sex_reference"],
            "group_random_intercept": False, "patient_random_effect": False,
            "observation_unit": "treatment series", "centering": "none", "standardization": "none",
            "outcome": "saved cgi_change unchanged", "feature_models": "null without E or E:ef_sf",
        },
        "fit_settings": asdict(settings), "provenance": _provenance(),
        "source_adapter": prepared.metadata, "planned_fits": 1,
        "artifacts": paths, "prepare_only": args.prepare_only,
        "fit_plots": {"enabled": plot_enabled, "dpi": plot_dpi, "source_id": source_id,
                      "x": "age", "y": "saved CGI change"},
    }
    run_manifest = ensure_run_manifest(args.analysis_name,
        run_configuration(settings.sex_effect, asdict(settings), args.field_name, prepare_only=args.prepare_only),
        inputs=[str(source_path)])
    manifest["run_manifest"] = str(run_manifest)
    _write_json(paths["analysis"], manifest, args.analysis_name)
    _write_csv(paths["model_input"], cohort, args.analysis_name)
    if args.prepare_only:
        coefficients = pd.DataFrame(columns=["term", "estimate", "std_error", "wald_z", "p_value", "ci_low", "ci_high", "inference_method"])
        try:
            diagnostics = {**validate_null_design(cohort, sex_effect=settings.sex_effect), "status": "prepared", "error": "", "warnings": ""}
        except ValueError as error:
            diagnostics = {"status": "failed", "error": str(error), "warnings": ""}
    else:
        coefficients, diagnostics = fit_null(cohort, settings)
    status = {**identity, **diagnostics, **row_settings}
    for column, value in identity.items():
        coefficients[column] = value
    coefficients["status"] = diagnostics["status"]
    statuses = pd.DataFrame([status])
    result = _wide_result(status, coefficients)
    for prefix in ("E", "E_ef_sf"):
        for value in ("estimate", "std_error", "wald_z", "p_value", "ci_low", "ci_high"):
            result.pop(f"{prefix}_{value}", None)
    _write_csv(paths["coefficients"], coefficients, args.analysis_name)
    _write_csv(paths["fit_status"], statuses, args.analysis_name)
    _write_csv(paths["null_results"], pd.DataFrame([result]), args.analysis_name)
    print(f"Null model: {spec['fixed_formula']}; sex {settings.sex_effect}; {len(cohort)} series", flush=True)
    print(f"Null results: {paths['null_results']}", flush=True)
    _print_fit_summary(statuses, prepare_only=args.prepare_only)
    plots = None
    if plot_enabled:
        from .fit_plot_workflow import export_fit_plots, make_fit_record
        plots = export_fit_plots([make_fit_record(cohort, status, feature=False)], args.analysis_name,
                                 source_id, dpi=plot_dpi,
                                 source_metadata={"analysis_manifest": str(paths["analysis"]),
                                                  "source_master_sha256": digest})
    return {"analysis_dir": root, "paths": paths, "planned_fits": 1,
            "failed_fits": int(diagnostics["status"] == "failed"), "fit_plots": plots, "exit_code": 0}
