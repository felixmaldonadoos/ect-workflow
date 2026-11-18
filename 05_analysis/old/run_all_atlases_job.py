import pandas as pd
from pathlib import Path

def get_hdf5_paths(df: pd.DataFrame, column: str = "full_file_path") -> list[str]:
    if column not in df.columns:
        raise KeyError(f"DataFrame does not contain column {column!r}")

    return df.loc[df[column].str.endswith(".hdf5", na=False), column].tolist()

def load_cogscores_batch(fn: str) -> pd.DataFrame:
    path = Path(fn)

    if not path.is_file():
        raise FileNotFoundError(f"Cognitive-score file does not exist: {path}")

    df = pd.read_excel(path)

    cleaned_columns = df.columns.str.strip()

    if cleaned_columns.duplicated().any():
        duplicates = cleaned_columns[cleaned_columns.duplicated()].tolist()
        raise ValueError(f"Cleaning whitespace creates duplicate columns: {duplicates}")

    df.columns = cleaned_columns

    required_cols = ["subjid", "cgi_start", "cgi_end", "date_start", "date_end (acute)", "mrn"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Cognitive-score table is missing required columns: {missing_cols}")

    for column in ("cgi_start", "cgi_end"):
        if df[column].isna().any():
            idx_bad_rows = df.index[df[column].isna()].tolist()
            print(f"Dropping {len(idx_bad_rows)} rows due to missing values in '{column}': {idx_bad_rows}")
            df = df.drop(index=idx_bad_rows).copy()

    df["subjid_org"] = df["subjid"].copy()
    df = add_base_subjid(df, source_col="subjid")
    df = df.drop(columns=["mrn"])

    return df

def add_base_subjid(df: pd.DataFrame, source_col: str = "subjid") -> pd.DataFrame:
    
    df = df.copy()

    if source_col not in df.columns:
        raise KeyError(f"Required subject-ID column '{source_col}' is missing.")

    df[source_col] = df[source_col].astype("string").str.strip()

    if df[source_col].isna().any():
        bad_rows = df.index[df[source_col].isna()].tolist()
        raise ValueError(f"Missing subject IDs in column '{source_col}' at rows: {bad_rows}")

    if df[source_col].eq("").any():
        bad_rows = df.index[df[source_col].eq("")].tolist()
        raise ValueError(f"Empty subject IDs in column '{source_col}' at rows: {bad_rows}")

    df["subjid_base"] = df[source_col].str.replace(r"-\d{3}$", "", regex=True)

    return df

# CGI scores
df_cog = load_cogscores_batch(
    "/projects/p32903/Alex2/datasets/STU00225089/.clincal_notes/cgi_batch_agg_2_clean.xlsx"
)

# identify the paths
query = pd.read_csv('/projects/p32903/Alex2/tests/sql-table/query_result.csv')
query           = query[query.dataset_root == 'STU00225089']
query           = query[query.model_type_dir == 'skin_single']
query_static    = query[query.step == 'step_0']
query_adaptive  = query[query.step == 'step_2']

# collect the paths 
hdf5_paths_static   = get_hdf5_paths(query_static)
hdf5_paths_adaptive = get_hdf5_paths(query_adaptive)


from pathlib import Path
from example_usage import run_all_atlases
import os 

# Existing objects:
# hdf5_paths_adaptive: list of tdcs_uq_gpc.hdf5 paths
# df_cog: DataFrame containing subjid and either:
#         - cgi_change, or
#         - cgi_start and cgi_end

SUBJECTS_DIR = Path("/projects/p32903/Alex2/datasets/STU00225089/subjectsdir")
OUTPUT_DIR = Path("/projects/p32903/Alex2/ect-workflow/05_analysis/output")

# Usually this can remain empty.
# An explicit entry is only required if multiple FreeSurfer directories share
# the same base ID and the correct directory would otherwise be ambiguous.
SUBJECT_MAP = {
    # "subj-cat-001-002": "subj-cat-001-001",
}


results = run_all_atlases(
    hdf5_files=hdf5_paths_static,
    cognitive=df_cog,
    subjects_dir=SUBJECTS_DIR,
    output_dir=OUTPUT_DIR,
    subject_map=SUBJECT_MAP,
    include_surface_hcp=True,
    run_unweighted_sensitivity=True,
    change_direction="end_minus_start",
    freesurfer_lut='/projects/p32903/Alex2/ecstim/resources/FreeSurferColorLUT.txt'
)

# Primary volumetric DK analysis using tetrahedron-volume-weighted P95
dk = results["volume_aparc_weighted"]

print("PC counts:")
print(dk.components_by_threshold)

print("\nGlobal E correlations:")
print(
    dk.correlations.loc[
        ["global_mean_E_unweighted", "global_mean_E_weighted"]
    ]
)

print("\nAll global and PC correlations:")
print(dk.correlations)

print("\nPCA explained variance:")
print(dk.pca.variance)

print("\nPC loadings:")
print(dk.pca.loadings)

