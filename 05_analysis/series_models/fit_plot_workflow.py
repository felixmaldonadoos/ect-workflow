"""Export fitted-model plots and the exact point predictions behind them."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd

from analysis_paths import lmm_analysis_name, resolve_analysis_dir, validate_output_path
from .workflow import _provenance, _write_csv, _write_json
from .run_layout import ensure_run_manifest, run_configuration


IDENTITY_COLUMNS = [
    "feature_type", "field_name", "normalization", "normalization_tag", "model_type", "step",
    "basis_id", "feature_id", "roi_name", "pc_name", "master_file", "source_master_path",
    "source_master_sha256", "model_id", "fixed_formula", "status", "converged", "aic", "bic",
    "sex_variance", "residual_variance",
    "sex_effect", "sex_effect_choice", "sex_reference",
]


def make_fit_record(frame: pd.DataFrame, status: dict, *, feature: bool = True) -> dict:
    """Keep every fit in the plot index, including unavailable predictions."""
    from .prediction import reconstruct_prediction
    try:
        prediction = reconstruct_prediction(frame, status, feature=feature)
        return {"identity": dict(status), "prediction": prediction, "error": ""}
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return {"identity": dict(status), "prediction": None, "error": str(error)}


def fit_plot_artifact_paths(analysis_name: str, source_id: str) -> dict[str, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", source_id):
        raise ValueError("Invalid source identifier for fit plots")
    root = resolve_analysis_dir(analysis_name)
    paths = {
        "analysis": root / "plots/model_fits" / f"{source_id}_analysis.json",
        "manifest": root / "plots/model_fits" / f"{source_id}_manifest.csv",
        "observations": root / "data/fit_plots" / f"{source_id}_observations.csv",
        "curves": root / "data/fit_plots" / f"{source_id}_curves.csv",
        "parameters": root / "data/fit_plots" / f"{source_id}_parameters.csv",
    }
    return {key: validate_output_path(path, analysis_name) for key, path in paths.items()}


def preflight_fit_plots(records: list[dict], analysis_name: str, source_id: str) -> dict:
    from .fit_plotting import plan_fit_plot_paths
    paths = fit_plot_artifact_paths(analysis_name, source_id)
    plan = plan_fit_plot_paths(records, analysis_name, source_id)
    targets = [*paths.values(), *plan["reports"], *plan["pngs"]]
    if existing := [str(path) for path in targets if path.exists()]:
        raise FileExistsError("Fit-plot artifacts already exist; choose a new analysis name: " + ", ".join(existing))
    return {"paths": paths, "plan": plan}


def _identified(frame: pd.DataFrame, identity: dict) -> pd.DataFrame:
    result = frame.copy()
    for name in IDENTITY_COLUMNS:
        if name in identity:
            result[name] = identity[name]
    return result


def export_fit_plots(
    records: list[dict], analysis_name: str, source_id: str, *, dpi: int = 120,
    source_metadata: dict | None = None,
) -> dict:
    from .fit_plotting import render_fit_plots
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("Plot DPI must be a positive integer")
    prepared = preflight_fit_plots(records, analysis_name, source_id)
    paths, plan = prepared["paths"], prepared["plan"]
    observations, curves, parameters, checks = [], [], [], []
    for record in records:
        prediction, identity = record["prediction"], record["identity"]
        check = {"identity": {key: identity[key] for key in IDENTITY_COLUMNS if key in identity},
                 "prediction_available": prediction is not None, "error": record.get("error", "")}
        if prediction is not None:
            observations.append(_identified(prediction.points, identity))
            curves.append(_identified(prediction.curves, identity))
            rows = [{"term": name, "sex": "", "estimate": value, "effect": "fixed"}
                    for name, value in prediction.coefficients.items()]
            fixed_sex = prediction.metadata.get("sex_effect") == "categorical_fixed"
            rows.extend({"term": "sex_category_offset" if fixed_sex else "sex_random_intercept",
                         "sex": name, "estimate": value, "effect": "fixed_sex_offset" if fixed_sex else "random"}
                        for name, value in prediction.sex_offsets.items())
            parameter_frame = pd.DataFrame(rows)
            parameter_frame["estimate_type"] = "point_estimate_only"
            parameter_frame["provisional"] = prediction.metadata["provisional"]
            parameters.append(_identified(parameter_frame, identity))
            check["prediction"] = prediction.metadata
        checks.append(check)
    sex_effect = records[0]["identity"].get("sex_effect", "categorical_random_intercept")
    manifest = {
        "schema_version": 1, "analysis_type": "fit_plots", "analysis_name": analysis_name,
        "source_id": source_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dpi": dpi, "planned_fits": len(records), "artifacts": paths,
        "planned_reports": plan["reports"], "planned_pngs": plan["pngs"],
        "source": source_metadata or {}, "provenance": _provenance(), "fits": checks,
        "plot_definition": {
            "feature_x": "saved ROI value or PC score", "null_x": "age in years",
            "feature_y": "CGI change adjusted to within-fit mean age and median ef_sf",
            "null_y": "saved CGI change unchanged", "sex_effect": sex_effect,
            "random_effect": "sex random intercept" if sex_effect == "categorical_random_intercept" else None,
            "reference_curve": ("fixed-effect mean with sex random effect set to zero" if sex_effect == "categorical_random_intercept"
                                else "fixed-effect mean at the reference sex category"),
            "sex_curve_support": "observed x range within the respective sex group",
            "uncertainty_bands": "none; point estimates only",
        },
    }
    # Reserve the plot run before generating figures, protecting previous runs.
    _write_json(paths["analysis"], manifest, analysis_name)
    empty_points = pd.DataFrame(columns=[*IDENTITY_COLUMNS, "person_id", "series_num", "sex",
                                        "observed_cgi_change", "adjusted_cgi_change", "fixed_prediction",
                                        "conditional_prediction", "residual"])
    empty_curves = pd.DataFrame(columns=[*IDENTITY_COLUMNS, "curve_kind", "sex", "x", "predicted_cgi_change"])
    empty_parameters = pd.DataFrame(columns=[*IDENTITY_COLUMNS, "term", "sex", "estimate", "effect",
                                            "estimate_type", "provisional"])
    _write_csv(paths["observations"], pd.concat(observations, ignore_index=True) if observations else empty_points, analysis_name)
    _write_csv(paths["curves"], pd.concat(curves, ignore_index=True) if curves else empty_curves, analysis_name)
    _write_csv(paths["parameters"], pd.concat(parameters, ignore_index=True) if parameters else empty_parameters, analysis_name)
    plot_manifest = render_fit_plots(records, analysis_name, source_id, dpi=dpi)
    if len(plot_manifest) != len(records):
        raise RuntimeError("Fit-plot manifest does not account for every planned fit")
    if {str(path.resolve()) for path in plan["reports"]} != set(plot_manifest.pdf_path):
        raise RuntimeError("Fit-plot PDF outputs disagree with the validated plan")
    if {str(path.resolve()) for path in plan["pngs"]} != set(plot_manifest.png_path):
        raise RuntimeError("Fit-plot PNG outputs disagree with the validated plan")
    _write_csv(paths["manifest"], plot_manifest, analysis_name)
    skipped = int(plot_manifest.plot_status.eq("skipped").sum())
    print(f"Fit plots: {len(records) - skipped}/{len(records)} drawn; {skipped} unavailable (placeholder pages)", flush=True)
    print(f"Fit-plot reports: {len(plan['reports'])}; index: {paths['manifest']}", flush=True)
    return {"paths": paths, "reports": plan["reports"], "pngs": plan["pngs"], "skipped": skipped}


def run_saved_fit_plots(args) -> dict:
    """Backfill plot artifacts in a fresh analysis, preserving saved model runs."""
    from .saved_fit_plots import load_saved_fit_records
    args.analysis_name = lmm_analysis_name(args.analysis_name, "plots")
    root = resolve_analysis_dir(args.analysis_name)
    inputs = [Path(path).expanduser().resolve() for path in args.analysis_dirs]
    if not inputs or len(inputs) != len(set(inputs)):
        raise ValueError("Supply unique saved analysis directories")
    if root in inputs:
        raise ValueError("Choose a plotting analysis name distinct from every saved input analysis")
    paths = {
        "analysis": validate_output_path(root / "analysis.json", args.analysis_name),
        "summary": validate_output_path(root / "fit_plot_summary.csv", args.analysis_name),
    }
    if existing := [str(path) for path in paths.values() if path.exists()]:
        raise FileExistsError("Plot run already exists; choose a new analysis name: " + ", ".join(existing))
    # Read and validate every source before reserving output or importing pyplot.
    sources = [load_saved_fit_records(path) for path in inputs]
    if len({source_id for _, source_id, _ in sources}) != len(sources):
        raise ValueError("Saved sources produce duplicate plot identifiers")
    planned = [preflight_fit_plots(records, args.analysis_name, source_id)
               for records, source_id, _ in sources]
    configurations = []
    for records, _, metadata in sources:
        settings = metadata["fit_settings"]
        configurations.append(run_configuration(settings.get("sex_effect", "random"), settings,
                                                records[0]["identity"]["field_name"]))
    if any(value != configurations[0] for value in configurations[1:]):
        raise ValueError("Saved plot sources use different model/run settings; use separate analysis run names")
    run_manifest = ensure_run_manifest(args.analysis_name, configurations[0],
                                      inputs=[metadata["source_master_path"] for _, _, metadata in sources])
    manifest = {
        "schema_version": 1, "analysis_type": "saved_fit_plots", "analysis_name": args.analysis_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
        "artifacts": paths, "provenance": _provenance(),
        "run_manifest": str(run_manifest),
        "sources": [{"source_id": source_id, "metadata": metadata, "plot_artifacts": plan["paths"],
                     "reports": plan["plan"]["reports"], "pngs": plan["plan"]["pngs"]}
                    for (_, source_id, metadata), plan in zip(sources, planned)],
        "model_refitting": False,
    }
    _write_json(paths["analysis"], manifest, args.analysis_name)
    outputs, summaries = [], []
    for (records, source_id, metadata), analysis_dir in zip(sources, inputs):
        print(f"Plotting saved analysis: {analysis_dir}", flush=True)
        output = export_fit_plots(records, args.analysis_name, source_id, dpi=args.dpi, source_metadata=metadata)
        outputs.append(output)
        summaries.append({"input_analysis_dir": str(analysis_dir), "source_id": source_id,
                          "n_fits": len(records), "n_plotted": len(records) - output["skipped"],
                          "n_skipped": output["skipped"], "n_reports": len(output["reports"]),
                          "manifest": str(output["paths"]["manifest"])})
    _write_csv(paths["summary"], pd.DataFrame(summaries), args.analysis_name)
    print(f"Fit-plot summary: {paths['summary']}", flush=True)
    return {"analysis_dir": root, "paths": paths, "sources": outputs, "exit_code": 0}
