"""Stage orchestration for NIfTI PCA, correlations, and saved-result plots."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .correlations import correlate_scores
from .images import load_masked_fields
from .inputs import SERIES_KEYS, load_series_inputs
from .paths import SOURCE_DIR, resolve_analysis_dir, validate_output_path
from .pca import fit_pca, save_pca_artifacts


def normalization_tag(field_name: str, normalization: str) -> str:
    return "raw" if normalization == "raw" else f"normed_brain_{field_name}_mean"


def artifact_paths(analysis_dir: Path, field_name: str, normalization: str) -> dict:
    tag = normalization_tag(field_name, normalization)
    data = analysis_dir / "data"
    return {
        "master": data / f"pca_{field_name}_{tag}_series_PC_master.csv",
        "scores": data / f"pca_{field_name}_{tag}_series_scores.csv",
        "correlations": data / f"spearman_{field_name}_{tag}_annotated.csv",
        "permutations": data / f"spearman_{field_name}_{tag}_permutation_annotated.csv",
    }


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value).__name__} in analysis metadata.")


def write_json(path: Path, value: Any) -> None:
    path = validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, default=_json_default, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _source_provenance() -> dict:
    versions = {}
    for dependency in ("numpy", "pandas", "scipy", "scikit-learn", "nibabel", "nilearn"):
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE_DIR,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=SOURCE_DIR,
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    files = sorted((SOURCE_DIR / "nifti_pca").glob("*.py"))
    files += [SOURCE_DIR / "run_mni_nifti_pca.py"]
    # Preserve the exact clinical/file-selection implementation we currently reuse.
    files += [
        SOURCE_DIR / "networks/mni_yeo2011_prep_data.py",
        SOURCE_DIR / "simnibs_parcel_analysis/utils/loader.py",
        SOURCE_DIR / "simnibs_parcel_analysis/utils/prepare_session_file_table.py",
    ]
    return {
        "package_version": __version__, "python_version": platform.python_version(),
        "dependencies": versions, "git_revision": revision, "git_dirty": dirty,
        "source_sha256": {str(path.relative_to(SOURCE_DIR)): _sha256(path) for path in files if path.is_file()},
    }


def record_stage(args, analysis_dir: Path, stage: str, details: dict) -> None:
    path = analysis_dir / "run_manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else {
        "schema_version": 1, "analysis_name": args.analysis_name,
        "analysis_dir": str(analysis_dir), "stages": {},
    }
    if stage == "prepare":
        manifest["stages"].pop("variance", None)
        manifest["stages"].pop("correlate", None)
        manifest["stages"].pop("plot", None)
    elif stage == "correlate":
        manifest["stages"].pop("plot", None)
    manifest["stages"][stage] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), "provenance": _source_provenance(), **details,
    }
    write_json(path, manifest)


def _safe_segment(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe model/step/field path segment: {value!r}")
    return value


def prepare_analysis(args, analysis_dir: Path) -> None:
    samples = load_series_inputs(
        field_name=args.field_name, series_csv=args.series_csv,
        query_path=args.query_path, clinical_path=args.clinical_path,
        subject_scan_map=args.subject_scan_map,
        steps=None if args.all_steps else args.steps,
        exclude_subjects=args.exclude_subjects,
    ).sort_values(SERIES_KEYS).reset_index(drop=True)
    series_path = analysis_dir / "data/series_inputs.csv"
    write_csv(series_path, samples)
    source_mask = args.brain_mask.expanduser().resolve()
    mask_digest = _sha256(source_mask)
    masters = {mode: [] for mode in args.normalizations}
    wides = {mode: [] for mode in args.normalizations}
    bases = []

    for (model_type, step), group in samples.groupby(["model_type", "step"], sort=False):
        model_type, step = _safe_segment(model_type), _safe_segment(step)
        group = group.reset_index(drop=True)
        print(f"Loading {model_type}/{step}: {len(group)} independent series", flush=True)
        fields = load_masked_fields(
            group["full_file_path"].tolist(), source_mask, resample=args.resample,
        )
        fingerprints = []
        for path in fields.unique_paths:
            stat = Path(path).stat()
            fingerprints.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})

        for mode in args.normalizations:
            tag = normalization_tag(args.field_name, mode)
            if mode == "brain_mean":
                if not np.isfinite(fields.brain_means).all() or (fields.brain_means <= 0).any():
                    raise ValueError(f"{model_type}/{step}: brain means must be finite and positive.")
                matrix = fields.values / fields.brain_means[:, None]
            else:
                matrix = fields.values
            print(f"Fitting PCA: {model_type}/{step}/{tag}", flush=True)
            result = fit_pca(
                matrix, variance_threshold=args.variance_threshold,
                n_components=args.n_components,
            )
            signature = {
                "field_name": args.field_name, "model_type": model_type, "step": step,
                "normalization": mode, "mask_sha256": mask_digest,
                "sample_ids": group["sample_id"].tolist(), "input_files": fingerprints,
                "resample": args.resample,
                "selection": {"n_components": args.n_components, "variance_threshold": args.variance_threshold},
                "fit_implementation_sha256": {
                    name: _sha256(SOURCE_DIR / "nifti_pca" / name)
                    for name in ("images.py", "pca.py")
                },
            }
            digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
            basis_id = f"{args.field_name}__{model_type}__{step}__{tag}__{digest}"
            basis_dir = analysis_dir / "models" / tag / model_type / step
            artifacts = save_pca_artifacts(result, fields.mask_img, basis_dir)
            settings_path = basis_dir / "analysis_settings.json"
            settings = {
                **signature, "basis_id": basis_id,
                "n_fit_samples": len(group), "n_unique_fields": len(fields.unique_paths),
                "n_brain_voxels": matrix.shape[1], "n_components_nonzero": len(result.variance),
                "n_components_tested": int(result.variance["selected_for_correlation"].sum()),
                "fit_unit": "treatment_series", "samples_assumed_independent": True,
                "brain_mask_source": str(source_mask), "brain_mask_rule": "mask > 0",
                "voxel_centering": "across fitting samples", "standardize_voxels": False,
                "whiten": False, "within_image_demeaning": False, "global_p95_weighting": False,
                "solver": "full", "sign_convention": "largest absolute component weight positive",
                "loaded_voxel_dtype": str(fields.values.dtype), "pca_dtype": "float64",
                "selection": {"n_components": args.n_components, "variance_threshold": args.variance_threshold},
                "provenance": _source_provenance(),
            }
            write_json(settings_path, settings)

            sample_table = group.copy()
            sample_table["normalization"] = mode
            sample_table["normalization_tag"] = tag
            sample_table["basis_id"] = basis_id
            sample_table["whole_brain_mean"] = fields.brain_means
            sample_table["n_brain_voxels"] = matrix.shape[1]
            sample_table["n_fit_samples"] = len(group)
            sample_table["n_unique_fields"] = len(fields.unique_paths)
            sample_table["pca_fit_unit"] = "treatment_series"
            sample_table["samples_assumed_independent"] = True
            sample_table["pca_model_path"] = str(artifacts["model_path"])
            sample_table["pca_variance_path"] = str(artifacts["variance_path"])
            sample_table["pca_settings_path"] = str(settings_path)
            sample_table["brain_mask_path"] = str(artifacts["mask_path"])
            components = result.variance["component"].tolist()
            scores = pd.DataFrame(result.scores, columns=components)
            scores.insert(0, "sample_id", group["sample_id"].to_numpy())
            wide = sample_table.merge(scores, on="sample_id", validate="one_to_one", sort=False)
            wides[mode].append(wide)
            long_scores = scores.melt(id_vars="sample_id", var_name="component", value_name="score")
            long = sample_table.merge(long_scores, on="sample_id", validate="one_to_many", sort=False)
            variance = result.variance.copy()
            variance["component_map_path"] = variance["component"].map(
                lambda component: str(artifacts["component_maps"][component])
            )
            long = long.merge(variance, on="component", validate="many_to_one", sort=False)
            if len(long) != len(group) * len(components):
                raise RuntimeError("Joining PC scores changed the expected series/component count.")
            masters[mode].append(long)
            write_csv(basis_dir / "fitting_samples.csv", sample_table)
            bases.append({
                "basis_id": basis_id, "field_name": args.field_name,
                "model_type": model_type, "step": step, "normalization": mode,
                "normalization_tag": tag, "n_series": len(group),
                "n_unique_fields": len(fields.unique_paths), "n_brain_voxels": matrix.shape[1],
                "n_components": len(components),
                "n_components_tested": settings["n_components_tested"],
                "settings_path": str(settings_path), "model_path": str(artifacts["model_path"]),
                "variance_path": str(artifacts["variance_path"]),
            })
        del fields

    for mode in args.normalizations:
        paths = artifact_paths(analysis_dir, args.field_name, mode)
        write_csv(paths["master"], pd.concat(masters[mode], ignore_index=True))
        write_csv(paths["scores"], pd.concat(wides[mode], ignore_index=True))
    write_csv(analysis_dir / "data/basis_manifest.csv", pd.DataFrame(bases))
    record_stage(args, analysis_dir, "prepare", {"n_series": len(samples), "n_bases": len(bases)})


def _boolean_column(values: pd.Series, name: str) -> pd.Series:
    mapped = values.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        raise ValueError(f"Invalid Boolean values in {name!r}.")
    return mapped.astype(bool)


def read_master(path: Path, field_name: str, normalization: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "sample_id", "person_id", "series_num", "model_type", "step", "field_name",
        "normalization", "normalization_tag", "basis_id", "component", "score", "cgi_change",
        "selected_for_correlation", "component_index", "component_map_path",
        "explained_variance", "explained_variance_ratio", "cumulative_variance_ratio",
        "pca_model_path", "pca_variance_path", "pca_settings_path",
    }
    missing = required - set(frame)
    if missing or frame.empty:
        raise ValueError(f"Empty or invalid PC master {path}; missing columns: {sorted(missing)}")
    if not frame["field_name"].eq(field_name).all() or not frame["normalization"].eq(normalization).all():
        raise ValueError(f"Unexpected field/normalization in PC master: {path}")
    if frame[["sample_id", "basis_id", "component"]].isna().any().any():
        raise ValueError("Missing sample/basis/component identity in PC master.")
    if frame.duplicated(["basis_id", "sample_id", "component"]).any():
        raise ValueError("Duplicate sample/component rows in PC master.")
    frame["selected_for_correlation"] = _boolean_column(
        frame["selected_for_correlation"], "selected_for_correlation"
    )
    return frame


FEATURE_METADATA = [
    "component", "component_index", "field_name", "model_type", "step", "normalization",
    "normalization_tag", "basis_id", "explained_variance", "explained_variance_ratio",
    "cumulative_variance_ratio", "selected_for_correlation", "component_map_path",
    "pca_model_path", "pca_variance_path", "pca_settings_path", "brain_mask_path",
    "n_brain_voxels", "n_fit_samples", "n_unique_fields", "pca_fit_unit",
    "samples_assumed_independent",
]


def correlate_analysis(args, analysis_dir: Path) -> None:
    total = 0
    for mode in args.normalizations:
        paths = artifact_paths(analysis_dir, args.field_name, mode)
        master = read_master(paths["master"], args.field_name, mode)
        master_digest = _sha256(paths["master"])
        tables = []
        for basis_id, group in master.groupby("basis_id", sort=False):
            for column in ["model_type", "step", "normalization"]:
                if group[column].nunique(dropna=False) != 1:
                    raise ValueError(f"{basis_id}: mixed {column} values.")
            if group.groupby("sample_id")["cgi_change"].nunique(dropna=False).ne(1).any():
                raise ValueError(f"{basis_id}: inconsistent outcomes across PC rows.")
            available_metadata = [column for column in FEATURE_METADATA if column in group]
            metadata = group[available_metadata].drop_duplicates()
            if metadata["component"].duplicated().any():
                raise ValueError(f"{basis_id}: inconsistent component metadata.")
            metadata = metadata.loc[metadata["selected_for_correlation"]].sort_values("component_index")
            if metadata.empty:
                raise ValueError(f"{basis_id}: no components selected for correlation.")
            selected = group.loc[group["component"].isin(metadata["component"])]
            scores = selected.pivot(index="sample_id", columns="component", values="score")
            scores = scores.reindex(columns=metadata["component"].tolist()).sort_index()
            outcome = group.drop_duplicates("sample_id").set_index("sample_id")["cgi_change"]
            print(f"Correlating {basis_id}: {len(scores)} series, {len(scores.columns)} PCs", flush=True)
            stats = correlate_scores(
                scores, outcome, n_permutations=args.n_permutations,
                seed=args.seed, batch_size=args.batch_size,
            )
            overlaps = (set(stats) & set(metadata)) - {"component"}
            for column in overlaps:
                if metadata[column].nunique(dropna=False) != 1 or not stats[column].eq(metadata[column].iloc[0]).all():
                    raise ValueError(f"Conflicting statistic/feature metadata: {column}")
            stats = stats.merge(metadata.drop(columns=list(overlaps)), on="component", how="left", validate="one_to_one")
            stats.insert(0, "x", args.field_name + "_" + stats["component"])
            stats.insert(1, "y", "cgi_change")
            stats["metric"] = "pc_score"
            stats["master_path"] = str(paths["master"])
            stats["master_sha256"] = master_digest
            stats["multiple_testing_adjustment"] = "none"
            tables.append(stats)
        results = pd.concat(tables, ignore_index=True)
        analytic_columns = [
            column for column in results
            if column != "p_permutation" and not column.startswith("permutation_")
        ]
        write_csv(paths["correlations"], results[analytic_columns])
        write_csv(paths["permutations"], results)
        total += len(results)
    record_stage(args, analysis_dir, "correlate", {"n_tests": total})


def plot_variance_analysis(args, analysis_dir: Path) -> None:
    """Render saved explained variance without loading scores or image maps."""
    from .plotting import plot_variance

    bases = pd.read_csv(analysis_dir / "data/basis_manifest.csv")
    required = {"basis_id", "field_name", "model_type", "step", "normalization", "variance_path"}
    missing = required.difference(bases.columns)
    if missing:
        raise ValueError(f"basis_manifest is missing columns: {sorted(missing)}")
    bases = bases.loc[
        bases["field_name"].eq(args.field_name)
        & bases["normalization"].isin(args.normalizations)
    ]
    if bases.empty:
        raise ValueError("No saved PCA bases match the requested field and normalizations.")
    if bases.duplicated(["field_name", "model_type", "step", "normalization"]).any():
        raise ValueError("basis_manifest contains duplicate PCA groups.")
    records = []
    for _, basis in bases.iterrows():
        model_type = _safe_segment(str(basis["model_type"]))
        step = _safe_segment(str(basis["step"]))
        tag = normalization_tag(args.field_name, basis["normalization"])
        print(f"Plotting explained variance: {model_type}/{step}/{tag}", flush=True)
        records.append(plot_variance(
            variance=pd.read_csv(basis["variance_path"]),
            metadata={column: basis[column] for column in required if column != "variance_path"},
            output_dir=analysis_dir / "plots" / tag / model_type / step,
            dpi=args.dpi,
        ))
    combined = pd.concat(records, ignore_index=True)
    write_csv(analysis_dir / "plots/variance_plot_manifest.csv", combined)
    record_stage(args, analysis_dir, "variance", {"n_bases": len(bases), "n_artifacts": len(combined)})


def plot_analysis(args, analysis_dir: Path) -> None:
    from .plotting import plot_basis

    records = []
    for mode in args.normalizations:
        paths = artifact_paths(analysis_dir, args.field_name, mode)
        master = read_master(paths["master"], args.field_name, mode)
        correlations = pd.read_csv(paths["permutations"])
        master_digest = _sha256(paths["master"])
        if "master_sha256" not in correlations or not correlations["master_sha256"].eq(master_digest).all():
            raise ValueError(
                "The PC master changed after the saved correlations were computed. "
                "Run --stage correlate before plotting this analysis."
            )
        for basis_id, group in master.groupby("basis_id", sort=False):
            first = group.iloc[0]
            subset = correlations.loc[correlations["basis_id"].eq(basis_id)]
            if subset.empty:
                raise ValueError(f"No correlation results for basis {basis_id}.")
            expected = set(group.loc[group["selected_for_correlation"], "component"])
            if set(subset["component"]) != expected:
                raise ValueError(f"Correlation/component selection mismatch for {basis_id}; rerun --stage correlate.")
            maps = group[["component", "component_map_path"]].drop_duplicates()
            if maps["component"].duplicated().any():
                raise ValueError(f"Conflicting component-map paths for {basis_id}.")
            manifest = plot_basis(
                variance=pd.read_csv(first["pca_variance_path"]), master=group,
                correlations=subset,
                component_maps=dict(zip(maps["component"], maps["component_map_path"])),
                background_path=args.background_path,
                output_dir=analysis_dir / "plots" / first["normalization_tag"] / first["model_type"] / first["step"],
                dpi=args.dpi,
            )
            records.append(manifest)
    combined = pd.concat(records, ignore_index=True)
    write_csv(analysis_dir / "plots/plot_manifest.csv", combined)
    record_stage(args, analysis_dir, "plot", {"n_artifacts": len(combined)})


def run_analysis(args) -> Path:
    analysis_dir = resolve_analysis_dir(args.analysis_name)
    # Configure caches before the legacy clinical loader or optional plotting
    # imports pyplot. This also applies to direct Python runs outside Slurm.
    cache_dir = validate_output_path(analysis_dir / "logs/matplotlib_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ["MPLBACKEND"] = "Agg"
    if args.stage in {"all", "prepare"}:
        prepare_analysis(args, analysis_dir)
    if args.stage in {"all", "prepare", "variance"}:
        plot_variance_analysis(args, analysis_dir)
    if args.stage in {"all", "correlate"}:
        correlate_analysis(args, analysis_dir)
    if args.stage == "plot" or (args.stage == "all" and args.plot):
        plot_analysis(args, analysis_dir)
    return analysis_dir
