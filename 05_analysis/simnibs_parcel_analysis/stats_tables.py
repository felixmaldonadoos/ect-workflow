"""Load and combine FreeSurfer/SimNIBS regional-statistics tables."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .clinical import prepare_scan_course_outcomes
from .identifiers import add_subject_id_columns


APARC_MEASURES = frozenset({"volume", "thickness", "meancurv"})


def load_aparcaseg_table(path: str | Path) -> pd.DataFrame:
    """Load an aparc+aseg E-field table while retaining unique modeled scans."""
    path = _require_file(path, "aparc+aseg table")
    df = pd.read_csv(path, sep="\t")
    if "subject_id" in df.columns and "subjid" not in df.columns:
        df = df.rename(columns={"subject_id": "subjid"})
    if "subjid" not in df.columns:
        raise KeyError("Expected either 'subjid' or 'subject_id' column")

    df = add_subject_id_columns(df, require_scan=True)
    duplicate_mask = df["subjid"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, "subjid"].tolist()
        raise ValueError(f"Duplicate modeled-scan IDs in {path}: {duplicates}")
    roi_columns = [column for column in df.columns if column not in {"subjid", "subjid_base", "scan_id"}]
    df[roi_columns] = df[roi_columns].apply(pd.to_numeric, errors="raise")
    return df


def get_stats_table_files(
    results_dir: str | Path,
    *,
    field_name: str = "magnE_mean",
    measure: str = "mean",
    skin_models: Sequence[str] = ("skin_single", "skin_double"),
    model_steps: Mapping[str, int] | None = None,
) -> dict[str, dict[str, str]]:
    """Resolve expected regional-statistics tables without global path state."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results directory does not exist: {results_dir}")
    if model_steps is None:
        model_steps = {"static": 0, "adaptive": 2}
    if not skin_models or not model_steps:
        raise ValueError("skin_models and model_steps cannot be empty")

    filename = f"aparc+aseg.{field_name}.{measure}.stats.table"
    files = {}
    missing = []
    for skin_model in skin_models:
        files[skin_model] = {}
        for model_type, step in model_steps.items():
            path = results_dir / skin_model / "stats" / f"step_{step}" / filename
            if not path.is_file():
                missing.append(path)
            files[skin_model][model_type] = str(path)
    if missing:
        raise FileNotFoundError("Missing expected stats tables:\n" + "\n".join(str(path) for path in missing))
    return files


def attach_clinical_to_modeled_scans(
    stats: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    clinical_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Expand each modeled scan to treatment courses matched by base ID."""
    if "subjid" not in stats.columns:
        raise KeyError("stats must contain 'subjid'")
    scans = add_subject_id_columns(stats, require_scan=True)
    if scans["subjid"].duplicated().any():
        duplicates = scans.loc[scans["subjid"].duplicated(keep=False), "subjid"].tolist()
        raise ValueError(f"Modeled-scan IDs must be unique: {duplicates}")

    if clinical_columns is not None:
        required = {"subjid", "date_start", "cgi_change"}
        missing_required = sorted(required.difference(clinical_columns))
        if missing_required:
            raise ValueError(f"clinical_columns must include course keys and outcome: {missing_required}")
        clinical = clinical[list(clinical_columns)].copy()

    matched = prepare_scan_course_outcomes(scans[["subjid"]], clinical)
    course_columns = [column for column in matched.columns if column not in {"subjid_base", "scan_id"}]
    result = scans.merge(matched[course_columns], on="subjid", how="inner", validate="one_to_many")
    return result


def build_model_roi(
    clinical: pd.DataFrame,
    files_agg: Mapping[str, Mapping[str, Mapping[str, str]]],
    *,
    skin_model: str = "skin_single",
    model_type: str = "static",
    dataset_is_synth: Mapping[str, bool] | None = None,
    clinical_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the requested table from each dataset and attach person-level data."""
    if dataset_is_synth is None:
        dataset_is_synth = {"org": False, "synthsr": True}
    frames = []
    for dataset, is_synth in dataset_is_synth.items():
        try:
            path = files_agg[dataset][skin_model][model_type]
        except KeyError as exc:
            raise KeyError(f"Missing files_agg entry for {dataset}/{skin_model}/{model_type}") from exc
        stats = load_aparcaseg_table(path)
        combined = attach_clinical_to_modeled_scans(stats, clinical, clinical_columns=clinical_columns)
        combined["dataset"] = dataset
        combined["is_synth"] = bool(is_synth)
        frames.append(combined)
    if not frames:
        raise ValueError("dataset_is_synth is empty")
    return pd.concat(frames, ignore_index=True)


def load_aparc_stats(stats_dir: str | Path, *, measure: str = "volume") -> pd.DataFrame:
    """Load and join left/right FreeSurfer aparc tables from one stats directory."""
    if measure not in APARC_MEASURES:
        raise ValueError(f"measure must be one of {sorted(APARC_MEASURES)}")
    stats_dir = Path(stats_dir)
    if not stats_dir.is_dir():
        raise NotADirectoryError(f"Structural stats directory does not exist: {stats_dir}")

    hemispheres = []
    for hemisphere in ("lh", "rh"):
        path = _require_file(stats_dir / f"{hemisphere}.aparc.{measure}.stats", f"{hemisphere} aparc table")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Aparc table is empty: {path}")
        df = df.rename(columns={df.columns[0]: "subjid"})
        df = add_subject_id_columns(df, require_scan=True)
        if df["subjid"].duplicated().any():
            duplicates = df.loc[df["subjid"].duplicated(keep=False), "subjid"].tolist()
            raise ValueError(f"Duplicate modeled scans in {path}: {duplicates}")

        id_columns = {"subjid", "subjid_base", "scan_id"}
        renamed = {}
        for column in df.columns:
            if column not in id_columns:
                roi = column.removeprefix(f"{hemisphere}_").removesuffix(f"_{measure}")
                renamed[column] = f"ctx-{hemisphere}-{roi}"
        df = df.rename(columns=renamed)
        roi_columns = [column for column in df.columns if column not in id_columns]
        df[roi_columns] = df[roi_columns].apply(pd.to_numeric, errors="raise")
        hemispheres.append(df)

    left, right = hemispheres
    left_ids, right_ids = set(left["subjid"]), set(right["subjid"])
    if left_ids != right_ids:
        raise ValueError(f"Left/right subject mismatch; only LH={sorted(left_ids - right_ids)}, only RH={sorted(right_ids - left_ids)}")
    return left.merge(right, on=["subjid", "subjid_base", "scan_id"], how="inner", validate="one_to_one")


def _require_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path
