from pathlib import Path

import pandas as pd

from example_usage_global_E_weighted import (
    run_all_atlases_global_p95_weighted,
)


def get_hdf5_paths(
    df: pd.DataFrame,
    column: str = "full_file_path",
) -> list[str]:
    """Return a sorted, unique list of HDF5 paths."""
    if column not in df.columns:
        raise KeyError(
            f"DataFrame does not contain column {column!r}"
        )

    paths = df.loc[
        df[column].str.endswith(".hdf5", na=False),
        column,
    ].tolist()

    if not paths:
        raise ValueError(
            f"No HDF5 paths were found in column {column!r}"
        )

    duplicate_paths = (
        pd.Index(paths)[
            pd.Index(paths).duplicated()
        ].unique().tolist()
    )

    if duplicate_paths:
        raise ValueError(
            f"Duplicate HDF5 paths were found: {duplicate_paths}"
        )

    return sorted(paths)


def add_base_subjid(
    df: pd.DataFrame,
    source_col: str = "subjid",
) -> pd.DataFrame:
    """Add the person-level base subject ID."""
    df = df.copy()

    if source_col not in df.columns:
        raise KeyError(
            f"Required subject-ID column {source_col!r} is missing"
        )

    df[source_col] = (
        df[source_col]
        .astype("string")
        .str.strip()
    )

    if df[source_col].isna().any():
        bad_rows = (
            df.index[
                df[source_col].isna()
            ].tolist()
        )
        raise ValueError(
            f"Missing subject IDs in column {source_col!r} "
            f"at rows: {bad_rows}"
        )

    if df[source_col].eq("").any():
        bad_rows = (
            df.index[
                df[source_col].eq("")
            ].tolist()
        )
        raise ValueError(
            f"Empty subject IDs in column {source_col!r} "
            f"at rows: {bad_rows}"
        )

    df["subjid_base"] = df[source_col].str.replace(
        r"-\d{3}$",
        "",
        regex=True,
    )

    return df


def load_cogscores_batch(
    filename: str | Path,
) -> pd.DataFrame:
    """Load clinical treatment courses with complete CGI values."""
    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(
            f"Cognitive-score file does not exist: {path}"
        )

    df = pd.read_excel(path)
    cleaned_columns = df.columns.astype(str).str.strip()

    if cleaned_columns.duplicated().any():
        duplicates = cleaned_columns[
            cleaned_columns.duplicated()
        ].tolist()
        raise ValueError(
            f"Cleaning whitespace creates duplicate columns: "
            f"{duplicates}"
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
            f"Cognitive-score table is missing required columns: "
            f"{missing_columns}"
        )

    complete_mask = (
        df[["cgi_start", "cgi_end"]]
        .notna()
        .all(axis=1)
    )
    excluded_rows = df.index[~complete_mask].tolist()

    if excluded_rows:
        print(
            f"Dropping {len(excluded_rows)} treatment-course rows "
            f"without complete CGI values: {excluded_rows}"
        )

    df = df.loc[complete_mask].copy()

    if df.empty:
        raise ValueError(
            "No treatment courses have both CGI start and CGI end"
        )

    df[["cgi_start", "cgi_end"]] = df[
        ["cgi_start", "cgi_end"]
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    df["subjid_org"] = df["subjid"].copy()
    df = add_base_subjid(
        df,
        source_col="subjid",
    )

    return df.drop(
        columns=["mrn"]
    ).reset_index(drop=True)


# Clinical treatment courses -------------------------------------------------

df_cog = load_cogscores_batch(
    "/projects/p32903/Alex2/datasets/STU00225089/"
    ".clincal_notes/cgi_batch_agg_2_clean.xlsx"
)


# Simulation-result inventory ------------------------------------------------

query_path = Path(
    "/projects/p32903/Alex2/tests/sql-table/query_result.csv"
)

if not query_path.is_file():
    raise FileNotFoundError(
        f"Query-result file does not exist: {query_path}"
    )

query = pd.read_csv(query_path)

required_query_columns = {
    "dataset_root",
    "model_type_dir",
    "step",
    "full_file_path",
}
missing_query_columns = sorted(
    required_query_columns.difference(query.columns)
)

if missing_query_columns:
    raise KeyError(
        f"Query table is missing required columns: "
        f"{missing_query_columns}"
    )

query = query.loc[
    (query["dataset_root"] == "STU00225089")
    & (query["model_type_dir"] == "skin_single")
].copy()

query_adaptive = query.loc[
    query["step"] == "step_2"
].copy()

hdf5_paths_adaptive = get_hdf5_paths(
    query_adaptive
)

print(
    f"Adaptive HDF5 files: "
    f"{len(hdf5_paths_adaptive)}"
)


# Analysis paths --------------------------------------------------------------

SUBJECTS_DIR = Path(
    "/projects/p32903/Alex2/datasets/"
    "STU00225089/subjectsdir"
)

OUTPUT_DIR = Path(
    "/projects/p32903/Alex2/results/"
    "STU00225089/stats/"
    "atlas_pca_global_p95_weighted/"
    "skin_single/adaptive"
)

if not SUBJECTS_DIR.is_dir():
    raise NotADirectoryError(
        f"SUBJECTS_DIR does not exist: {SUBJECTS_DIR}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# FreeSurfer matching overrides ----------------------------------------------

SUBJECT_MAP = {
    # Add entries only when automatic matching is ambiguous.
    # "subj-cat-001-002": "subj-cat-001-001",
}


# Run all atlas analyses ------------------------------------------------------

results = run_all_atlases_global_p95_weighted(
    hdf5_paths_adaptive,
    df_cog,
    SUBJECTS_DIR,
    OUTPUT_DIR,
    subject_map=SUBJECT_MAP,
    include_surface_hcp=True,
    run_unweighted_sensitivity=True,
    global_percentile=95,
    brain_tags=(1, 2),
    field_name="magnE_mean",
    mesh_key="mesh_roi",
    freesurfer_lut='/projects/p32903/Alex2/ecstim/resources/FreeSurferColorLUT.txt'
)

# Inspect primary DK results --------------------------------------------------

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
    dk.outcomes
    .groupby("subjid_base", as_index=False)
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
print(f"Treatment courses: {dk.outcomes['treatment_course_id'].nunique()}")
print(f"Scan-course observations: {len(dk.outcomes)}")