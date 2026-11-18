"""Export a cohort-validated comparison of existing feature-model likelihoods."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analysis_paths import SOURCE_DIR, lmm_analysis_name, resolve_analysis_dir, validate_output_path
from .aic import compare_analyses
from .workflow import _provenance, _write_csv, _write_json, sha256_file
from .run_layout import ensure_run_manifest, run_configuration


PLOT_MANIFEST_COLUMNS = ["path", "kind", "feature_type", "field_name", "normalization", "notes"]
SUMMARY_GROUPS = [
    "analysis_dir", "feature_type", "field_name", "normalization", "model_type", "step", "basis_id",
]


def _configuration_summary(table: pd.DataFrame) -> pd.DataFrame:
    null_aic = float(table.loc[table.is_null, "aic"].iloc[0])
    records = []
    features = table.loc[~table.is_null].copy()
    for identity, group in features.groupby(SUMMARY_GROUPS, dropna=False, sort=True):
        eligible = group.loc[group.aic_eligible].sort_values("aic", kind="stable")
        best = eligible.iloc[0] if len(eligible) else None
        record = dict(zip(SUMMARY_GROUPS, identity))
        record.update({
            "n_candidates": len(group), "n_eligible": len(eligible),
            "n_excluded": int((~group.aic_eligible).sum()),
            "n_provisional": int(group.aic_provisional.sum()), "null_aic": null_aic,
            "best_feature_id": best.feature_id if best is not None else "",
            "best_feature_aic": best.aic if best is not None else float("nan"),
            "best_improvement_over_null": best.aic_improvement_over_null if best is not None else float("nan"),
            "best_feature_provisional": bool(best.aic_provisional) if best is not None else False,
            "best_including_null": ("feature" if best is not None and best.aic < null_aic
                                    else "tie" if best is not None and best.aic == null_aic else "null"),
        })
        records.append(record)
    return pd.DataFrame(records)


def run_aic_comparison(args) -> dict:
    args.analysis_name = lmm_analysis_name(args.analysis_name, "aic")
    root = resolve_analysis_dir(args.analysis_name)
    inputs = [Path(args.null_analysis_dir).expanduser().resolve(),
              *(Path(path).expanduser().resolve() for path in args.feature_analysis_dirs)]
    if root in inputs:
        raise ValueError("Choose a comparison --analysis-name distinct from every input analysis directory")
    # Validate all saved specifications, fit identities, and cohorts before
    # creating output files. No fitting or source-master rereading occurs here.
    comparison = compare_analyses(inputs[0], inputs[1:])
    table = comparison.table
    paths = {
        "analysis": root / "analysis.json", "comparison": root / "aic_comparison.csv",
        "ranked": root / "aic_ranked.csv", "excluded": root / "aic_excluded.csv",
        "summary": root / "aic_summary.csv", "plot_manifest": root / "plots/plot_manifest.csv",
    }
    paths = {key: validate_output_path(path, args.analysis_name) for key, path in paths.items()}
    plot_paths = []
    if not args.no_plots:
        from .aic_plotting import plan_plot_paths
        plot_paths = plan_plot_paths(table, args.analysis_name)
    if existing := [str(path) for path in [*paths.values(), *plot_paths] if path.exists()]:
        raise FileExistsError("Comparison artifacts already exist; choose a new --analysis-name: " + ", ".join(existing))
    saved_settings = comparison.metadata["analysis_settings"][0]["fit_settings"]
    choice = comparison.metadata["null_model_spec"]["sex_effect_choice"]
    run_manifest = ensure_run_manifest(args.analysis_name,
        run_configuration(choice, saved_settings, str(table.field_name.iloc[0])),
        inputs=table.source_master_path.drop_duplicates().tolist())
    provenance = _provenance()
    runner = SOURCE_DIR / "run_series_aic_comparison.py"
    provenance["source_sha256"][runner.name] = sha256_file(runner)
    manifest = {
        "schema_version": 1, "analysis_type": "aic_comparison", "analysis_name": args.analysis_name,
        "analysis_dir": str(root), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), "comparison": comparison.metadata,
        "provenance": provenance, "artifacts": paths, "planned_plots": plot_paths,
        "run_manifest": str(run_manifest),
        "interpretation": "Exploratory support among supplied models on the same saved series; not validated prediction or a significance test.",
    }
    _write_json(paths["analysis"], manifest, args.analysis_name)
    _write_csv(paths["comparison"], table, args.analysis_name)
    ranked = table.loc[table.aic_eligible].sort_values("aic", kind="stable")
    _write_csv(paths["ranked"], ranked, args.analysis_name)
    _write_csv(paths["excluded"], table.loc[~table.aic_eligible], args.analysis_name)
    _write_csv(paths["summary"], _configuration_summary(table), args.analysis_name)
    plots = pd.DataFrame(columns=PLOT_MANIFEST_COLUMNS)
    if not args.no_plots:
        from .aic_plotting import render_comparison_plots
        plots = render_comparison_plots(table, args.analysis_name, top_n=args.top_n, dpi=args.dpi)
        rendered = {str(Path(path).resolve()) for path in plots.path}
        if rendered != {str(path.resolve()) for path in plot_paths}:
            raise RuntimeError("Rendered plot paths disagree with the validated plot plan")
    _write_csv(paths["plot_manifest"], plots, args.analysis_name)
    candidates = table.loc[~table.is_null]
    print(f"AIC comparison: {int(candidates.aic_eligible.sum())}/{len(candidates)} feature models eligible; "
          f"{int((~candidates.aic_eligible).sum())} excluded", flush=True)
    print(f"Null AIC: {float(table.loc[table.is_null, 'aic'].iloc[0]):.6f}", flush=True)
    print(f"Comparison table: {paths['comparison']}", flush=True)
    print(f"Figures: {len(plots)}; plot manifest: {paths['plot_manifest']}", flush=True)
    return {"analysis_dir": root, "paths": paths, "planned_fits": 0, "exit_code": 0}
