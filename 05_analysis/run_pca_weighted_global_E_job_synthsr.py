"""Run one pooled global-P95-weighted analysis across both data sources."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from example_usage_global_E_weighted import run_all_atlases_global_p95_weighted
from simnibs_parcel_analysis.identifiers import infer_subject_id
import argparse

MODEL_TYPES = ("skin_single", "skin_double")
ANALYSIS_STEP_BY_MODE = {
    "static": "step_0",
    "adaptive": "step_2",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pooled atlas PCA analysis.")
    parser.add_argument("--model-type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--analysis-mode", choices=tuple(ANALYSIS_STEP_BY_MODE), required=True)
    return parser.parse_args()


QUERY_PATH = Path(
    "/gpfs/projects/p32903/Alex2/ect-workflow/05_analysis/query_result.csv"
)

OUTPUT_ROOT = Path(
    "/projects/p32903/Alex2/results/STU00225089/stats/"
    "atlas_pca_global_p95_weighted"
)

COGNITIVE_SCORES_PATH = Path(
    "/projects/p32903/Alex2/datasets/STU00225089/"
    ".clincal_notes/cgi_batch_agg_2_clean.xlsx"
)

FREESURFER_LUT = Path(
    "/projects/p32903/Alex2/ecstim/resources/FreeSurferColorLUT.txt"
)

# These are FreeSurfer SUBJECTS_DIR roots. The query table's "subject_dir"
# column is a modeled-subject folder name and is not used as either root.
SUBJECTS_DIR_BY_DATASET = {
    "STU00225089": Path(
        "/projects/p32903/Alex2/datasets/STU00225089/subjectsdir"
    ),
    "STU00225089_synthsr": Path(
        "/projects/p32903/Alex2/datasets/STU00225089/synthsr/subjectsdir"
    ),
}

MIN_VOLUME_ELEMENTS_PER_ROI = 9

SUBJECT_MAP = {
    # Add entries only when automatic matching is ambiguous.
    # "subj-cat-001-002": "subj-cat-001-001",
}

# Exclusions are atlas-specific and are applied uniformly to every modeled scan.
EXCLUDE_VOLUME_ROIS_BY_ATLAS = {
    "a2009s": ("ctx_lh_Medial_wall", "ctx_rh_Medial_wall", "ctx_lh_G_cingul-Post-ventral", "ctx_rh_G_cingul-Post-ventral", 
               "ctx_rh_S_collat_transv_post", "ctx_lh_S_collat_transv_post"),
}

EXCLUDED_SUBJIDS = ("subj-cat-002-003",)

def resolve_analysis_config(model_type: str, analysis_mode: str) -> tuple[str, Path, Path]:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type!r}")

    if analysis_mode not in ANALYSIS_STEP_BY_MODE:
        raise ValueError(f"Unsupported analysis mode: {analysis_mode!r}")

    analysis_step = ANALYSIS_STEP_BY_MODE[analysis_mode]
    output_dir = OUTPUT_ROOT / model_type / analysis_mode

    return analysis_step, QUERY_PATH, output_dir

def add_base_subjid(df: pd.DataFrame, source_col: str = "subjid") -> pd.DataFrame:
    """Add the person-level base subject ID."""
    df = df.copy()

    if source_col not in df.columns:
        raise KeyError(f"Required subject-ID column {source_col!r} is missing")

    df[source_col] = df[source_col].astype("string").str.strip()

    if df[source_col].isna().any():
        bad_rows = df.index[df[source_col].isna()].tolist()
        raise ValueError(
            f"Missing subject IDs in column {source_col!r} at rows: {bad_rows}"
        )

    if df[source_col].eq("").any():
        bad_rows = df.index[df[source_col].eq("")].tolist()
        raise ValueError(
            f"Empty subject IDs in column {source_col!r} at rows: {bad_rows}"
        )

    df["subjid_base"] = df[source_col].str.replace(
        r"-\d{3}$",
        "",
        regex=True,
    )

    return df


def load_cogscores_batch(filename: str | Path) -> pd.DataFrame:
    """Load clinical treatment courses with complete CGI values."""
    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(f"Cognitive-score file does not exist: {path}")

    df = pd.read_excel(path)
    cleaned_columns = df.columns.astype(str).str.strip()

    if cleaned_columns.duplicated().any():
        duplicates = cleaned_columns[cleaned_columns.duplicated()].tolist()
        raise ValueError(
            f"Cleaning whitespace creates duplicate columns: {duplicates}"
        )

    df.columns = cleaned_columns

    required_columns = [
        "subjid",
        "cgi_start",
        "cgi_end",
        "date_start",
        "date_end (acute)",
        "mrn",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Cognitive-score table is missing required columns: {missing_columns}"
        )

    complete_mask = df[["cgi_start", "cgi_end"]].notna().all(axis=1)
    excluded_rows = df.index[~complete_mask].tolist()

    if excluded_rows:
        print(
            f"Dropping {len(excluded_rows)} treatment-course rows "
            f"without complete CGI values: {excluded_rows}"
        )

    df = df.loc[complete_mask].copy()

    if df.empty:
        raise ValueError("No treatment courses have both CGI start and CGI end")

    df[["cgi_start", "cgi_end"]] = df[["cgi_start", "cgi_end"]].apply(
        pd.to_numeric,
        errors="raise",
    )

    df["subjid_org"] = df["subjid"].copy()
    df = add_base_subjid(df, source_col="subjid")

    return df.drop(columns=["mrn"]).reset_index(drop=True)


def load_hdf5_inventory(
    filename: str | Path,
    subjects_dirs: Mapping[str, Path],
    model_type_dir: str,
    step: str,) -> pd.DataFrame:
    """Select and validate one HDF5 file per modeled scan."""
    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(f"Query-result file does not exist: {path}")

    query = pd.read_csv(path)
    required_columns = {
        "dataset_root",
        "model_type_dir",
        "subject_dir",
        "step",
        "full_file_path",
    }
    missing_columns = sorted(required_columns.difference(query.columns))

    if missing_columns:
        raise KeyError(
            f"Query table is missing required columns: {missing_columns}"
        )

    selected = query.loc[
        query["dataset_root"].isin(subjects_dirs)
        & query["model_type_dir"].eq(model_type_dir)
        & query["step"].eq(step)
        & query["full_file_path"].str.endswith(".hdf5", na=False)
    ].copy()

    if selected.empty:
        raise ValueError(
            "No HDF5 rows matched the configured dataset roots, "
            f"model_type_dir={model_type_dir!r}, and step={step!r}"
        )

    selected_roots = set(selected["dataset_root"].unique())
    missing_roots = sorted(set(subjects_dirs).difference(selected_roots))

    if missing_roots:
        raise ValueError(
            f"No matching HDF5 rows were found for dataset roots: {missing_roots}"
        )

    validation_columns = [
        "dataset_root",
        "subject_dir",
        "full_file_path",
    ]
    missing_value_mask = selected[validation_columns].isna()

    if missing_value_mask.any(axis=None):
        bad_rows = selected.index[missing_value_mask.any(axis=1)].tolist()
        raise ValueError(
            f"Selected query rows contain missing identifiers at rows: {bad_rows}"
        )

    duplicate_path_mask = selected["full_file_path"].duplicated(keep=False)

    if duplicate_path_mask.any():
        duplicate_paths = sorted(
            selected.loc[duplicate_path_mask, "full_file_path"].unique()
        )
        raise ValueError(f"Duplicate HDF5 paths were found: {duplicate_paths}")

    selected["inferred_subjid"] = selected["full_file_path"].map(
        infer_subject_id
    )
    subject_mismatch_mask = selected["subject_dir"].ne(
        selected["inferred_subjid"]
    )

    if subject_mismatch_mask.any():
        mismatches = selected.loc[
            subject_mismatch_mask,
            ["dataset_root", "subject_dir", "inferred_subjid", "full_file_path"],
        ]
        raise ValueError(
            "The query subject_dir does not match the subject ID inferred "
            f"from full_file_path:\n{mismatches.to_string(index=False)}"
        )

    duplicate_subject_mask = selected["inferred_subjid"].duplicated(
        keep=False
    )

    if duplicate_subject_mask.any():
        duplicates = selected.loc[
            duplicate_subject_mask,
            ["dataset_root", "inferred_subjid", "full_file_path"],
        ]
        raise ValueError(
            "Each modeled-scan ID must identify exactly one HDF5 file:\n"
            f"{duplicates.to_string(index=False)}"
        )

    missing_files = sorted(
        full_file_path
        for full_file_path in selected["full_file_path"]
        if not Path(full_file_path).is_file()
    )

    if missing_files:
        raise FileNotFoundError(
            f"Selected HDF5 files do not exist: {missing_files}"
        )

    return selected.sort_values(
        ["dataset_root", "subject_dir", "full_file_path"]
    ).reset_index(drop=True)


def validate_analysis_paths(output_dir: Path) -> None:
    """Validate shared resources and create the analysis output directory."""
    for dataset_root, subjects_dir in SUBJECTS_DIR_BY_DATASET.items():
        if not subjects_dir.is_dir():
            raise NotADirectoryError(f"SUBJECTS_DIR does not exist for {dataset_root}: {subjects_dir}")

    if not FREESURFER_LUT.is_file():
        raise FileNotFoundError(f"FreeSurfer lookup table does not exist: {FREESURFER_LUT}")

    output_dir.mkdir(parents=True, exist_ok=True)

def exclude_subject_results(
    inventory: pd.DataFrame,
    subjids: list[str],
    subject_col: str = "inferred_subjid") -> pd.DataFrame:
    """Exclude result rows belonging to specified modeled subjects."""
    if subject_col not in inventory.columns:
        raise KeyError(f"Inventory is missing column {subject_col!r}")

    missing_subjids = sorted(set(subjids).difference(inventory[subject_col]))

    if missing_subjids:
        raise ValueError(f"Subjects were not found in the inventory: {missing_subjids}")

    excluded_mask = inventory[subject_col].isin(subjids)
    filtered = inventory.loc[~excluded_mask].copy()

    if filtered.empty:
        raise ValueError("No result files remain after subject exclusions")

    print(f"Excluded {excluded_mask.sum()} result files for: {sorted(subjids)}")

    return filtered.reset_index(drop=True)

def main() -> None:
    """Load both sources and run one pooled analysis."""
    args = parse_args()
    analysis_step, query_path, output_dir = resolve_analysis_config(args.model_type, args.analysis_mode)

    print(f"Model type: {args.model_type}")
    print(f"Analysis mode: {args.analysis_mode}")
    print(f"Simulation results step: {analysis_step}")
    print(f"Query table: {query_path}")
    print(f"Output directory: {output_dir}")

    df_cog = load_cogscores_batch(COGNITIVE_SCORES_PATH)
    inventory = load_hdf5_inventory(
        query_path,
        SUBJECTS_DIR_BY_DATASET,
        args.model_type,
        analysis_step,
    )
    inventory = exclude_subject_results(inventory, list(EXCLUDED_SUBJIDS))

    validate_analysis_paths(output_dir)

    inventory.to_csv(output_dir / "selected_hdf5_inventory.csv", index=False)

    for dataset_root in SUBJECTS_DIR_BY_DATASET:
        count = inventory["dataset_root"].eq(dataset_root).sum()
        print(f"{dataset_root} {args.analysis_mode} HDF5 files: {count}")

    hdf5_paths = inventory["full_file_path"].tolist()
    subjects_dir_by_hdf5 = {
        row.full_file_path: SUBJECTS_DIR_BY_DATASET[row.dataset_root]
        for row in inventory.itertuples(index=False)
    }

    print(f"Total {args.analysis_mode} HDF5 files: {len(hdf5_paths)}")

    results = run_all_atlases_global_p95_weighted(
        hdf5_paths,
        df_cog,
        output_dir,
        subjects_dir_by_hdf5=subjects_dir_by_hdf5,
        subject_map=SUBJECT_MAP,
        include_surface_hcp=True,
        run_unweighted_sensitivity=True,
        global_percentile=95,
        brain_tags=(1, 2),
        field_name="magnE_mean",
        mesh_key="mesh_roi",
        min_volume_elements_per_roi=MIN_VOLUME_ELEMENTS_PER_ROI,
        freesurfer_lut=FREESURFER_LUT,
        exclude_volume_rois_by_atlas=EXCLUDE_VOLUME_ROIS_BY_ATLAS,
    )

    dk = results["volume_aparc_weighted"]

    print("\nGlobal brain P95 E:")
    print(dk.global_p95_e)

    print("\nGlobal-P95-weighted PCA input:")
    print(dk.pca.global_p95_weighted_parcels)

    print("\nComponents by explained-variance threshold:")
    print(dk.components_by_threshold)

    print("\nOutcome correlations:")
    print(dk.correlations)

    person_counts = (
        dk.outcomes.groupby("subjid_base", as_index=False)
        .agg(
            n_scans=("subjid", "nunique"),
            n_treatment_courses=("treatment_course_id", "nunique"),
            n_scan_course_observations=("observation_id", "size"),
            scan_ids=("subjid", lambda values: sorted(set(values))),
            treatment_course_ids=(
                "treatment_course_id",
                lambda values: sorted(set(values)),
            ),
        )
        .sort_values(
            ["n_scan_course_observations", "subjid_base"],
            ascending=[False, True],
        )
    )

    print(person_counts.to_string(index=False))
    print(f"People: {dk.outcomes['subjid_base'].nunique()}")
    print(f"Scans: {dk.outcomes['subjid'].nunique()}")
    print(
        f"Treatment courses: "
        f"{dk.outcomes['treatment_course_id'].nunique()}"
    )
    print(f"Scan-course observations: {len(dk.outcomes)}")


if __name__ == "__main__":
    main()
    
