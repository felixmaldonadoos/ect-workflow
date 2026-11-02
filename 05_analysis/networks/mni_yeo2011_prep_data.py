import pandas as pd 
import sys
from pathlib import Path
import re
import numpy as np
from nilearn import image, masking, plotting
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from scipy.stats import (
    mode,
)

analysis_dir = Path("/projects/p32903/Alex2/ect-workflow/05_analysis")
if str(analysis_dir) not in sys.path: sys.path.insert(0, str(analysis_dir))
from simnibs_parcel_analysis.utils.loader import (
    load_query_results,
    load_course_data,
    load_stimulation_data,
    prepare_session_data,
)
from simnibs_parcel_analysis.utils.prepare_session_file_table import (
    MissingMappingError,
)

def load_scalar_img(niimg):
    """Load a 3D image, accepting a singleton fourth dimension."""
    img = image.load_img(niimg)

    if len(img.shape) == 4 and img.shape[-1] == 1:
        img = img.slicer[..., 0]

    if len(img.shape) != 3:
        raise ValueError(f"Expected 3D or (X,Y,Z,1); got {img.shape}")

    return img


def load_label_atlas(atlas_path, lut_path):
    """Return the atlas and a Series mapping label IDs to ROI names."""
    atlas = load_scalar_img(atlas_path)
    ids = np.unique(image.get_data(atlas))

    if not np.isfinite(ids).all() or not np.equal(ids, np.rint(ids)).all():
        raise ValueError("The atlas must contain finite integer labels.")

    ids = ids[ids != 0].astype(int)

    lut = pd.read_csv(
        lut_path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1],
        names=["id", "name"],
    ).set_index("id")["name"]

    if not lut.index.is_unique:
        raise ValueError("Duplicate IDs in the LUT.")

    names = lut.reindex(ids)

    if names.empty or names.isna().any() or not names.is_unique:
        raise ValueError("Every atlas ROI needs a unique name in the LUT.")

    return atlas, names


def extract_region_stats(df,
    atlas_path,
    lut_path,
    path_column="full_file_path",
    *,
    mode_decimals=None,):
    """Return {statistic: DataFrame}, preserving df's index and row order.

    Column names use one underscore between statistic and ROI:
    E_mean_7Networks_LH_Vis.

    Zeros are included. Non-finite E-fields raise an error.
    Missing regions receive count 0 and NaN for other statistics.
    mode_decimals affects only the mode calculation.
    """
    atlas, names = load_label_atlas(atlas_path, lut_path)

    statistics = (
        "mean", "mode", "median", "min", "max",
        "95th_percentile", "99th_percentile", "n_voxels",
    )

    shape = (len(df), len(names))
    arrays = {s: np.full(shape, np.nan) for s in statistics[:-1]}
    arrays["n_voxels"] = np.zeros(shape, dtype=np.int64)

    cached_grid, masker = None, None

    for row, path in enumerate(df[path_column]):
        img = load_scalar_img(path)

        if not np.isfinite(img.get_fdata()).all():
            raise ValueError(f"Non-finite E-field values: {path}")

        grid = (img.shape, tuple(img.affine.ravel()))

        if grid != cached_grid:
            # Align categorical labels to the E-field grid.
            labels = image.resample_to_img(
                atlas,
                img,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            roi_mask = image.math_img("labels != 0", labels=labels)
            cached_grid, masker = grid, None

            if image.get_data(roi_mask).any():
                masker = NiftiMasker(
                    mask_img=roi_mask,
                    dtype=np.float64,
                    standardize=None,
                    detrend=False,
                    smoothing_fwhm=None,
                    reports=False,
                ).fit()

                # Same voxel ordering as masker.transform().
                ids = masking.apply_mask(
                    labels, masker.mask_img_
                ).astype(int)

        if masker is None:
            continue

        values = pd.Series(np.asarray(masker.transform(img)).ravel())
        grouped = values.groupby(ids)

        summary = grouped.agg(
            mean="mean",
            median="median",
            min="min",
            max="max",
            n_voxels="size",
        )

        quantiles = grouped.quantile(
            [0.95, 0.99], interpolation="linear"
        ).unstack()

        summary["95th_percentile"] = quantiles[0.95]
        summary["99th_percentile"] = quantiles[0.99]

        mode_values = (
            values if mode_decimals is None
            else values.round(mode_decimals)
        )

        summary["mode"] = mode_values.groupby(ids).agg(
            lambda v: float(mode(v, axis=None, keepdims=False).mode)
        )

        summary = summary.reindex(names.index)
        summary["n_voxels"] = (
            summary["n_voxels"].fillna(0).astype(np.int64)
        )

        for stat in statistics:
            arrays[stat][row] = summary[stat].to_numpy()

    return {
        stat: pd.DataFrame(
            arrays[stat],
            index=df.index.copy(),
            columns=[f"E_{stat}_{name}" for name in names],
        )
        for stat in statistics
    }
    

def merge_series_data(df_cgi: pd.DataFrame, df_stim_collapsed: pd.DataFrame,) -> pd.DataFrame:
    keys = ["person_id", "series_num"]
    problems = []

    try:
        for name, table in [
            ("df_stim_collapsed", df_stim_collapsed),
            ("df_cgi", df_cgi),
        ]:
            missing_columns = set(keys) - set(table.columns)
            if missing_columns:
                raise KeyError(
                    f"{name} is missing columns: {sorted(missing_columns)}"
                )

            missing_keys = table.loc[table[keys].isna().any(axis=1)]
            if not missing_keys.empty:
                problems.append(f"{name} contains missing merge-key values")
                print(f"\n{name}: rows with missing merge keys:")
                print(missing_keys.to_string(index=True))

            duplicates = table.loc[table.duplicated(keys, keep=False)]
            if not duplicates.empty:
                problems.append(f"{name} contains duplicate series keys")
                print(f"\n{name}: rows with duplicate merge keys:")
                print(duplicates.to_string(index=True))

        stim_keys = pd.MultiIndex.from_frame(df_stim_collapsed[keys])
        cgi_keys = pd.MultiIndex.from_frame(df_cgi[keys])

        stim_only = df_stim_collapsed.loc[~stim_keys.isin(cgi_keys)]
        cgi_only = df_cgi.loc[~cgi_keys.isin(stim_keys)]

        if not stim_only.empty:
            problems.append(
                f"{len(stim_only)} stimulation rows have no matching CGI series"
            )
            print("\nStimulation rows without matching CGI series:")
            print(stim_only.to_string(index=True))

        if not cgi_only.empty:
            problems.append(
                f"{len(cgi_only)} CGI rows have no matching stimulation series"
            )
            print("\nCGI rows without matching stimulation series:")
            print(cgi_only.to_string(index=True))

        if problems:
            raise pd.errors.MergeError(
                "Perfect series alignment required: " + "; ".join(problems)
            )

        shared_metadata = [
            col for col in ["age", "sex", "dob"]
            if col in df_stim_collapsed.columns and col in df_cgi.columns
        ]

        return df_stim_collapsed.drop(columns=shared_metadata).merge(
            df_cgi,
            on=keys,
            how="left",
            sort=False,
            validate="one_to_one",
            suffixes=("_stim", "_cgi"),
        )

    except (KeyError, ValueError) as error:
        # pandas MergeError is also a ValueError.
        print(
            f"\nMerge failed: {error}\n"
            f"Stimulation rows: {len(df_stim_collapsed)}; "
            f"CGI rows: {len(df_cgi)}"
        )
        raise


def load_series_data(
    subjects_exclude: list,
    fn_query,
    fn_cgi,
    query_filters: dict,
    fn_field: str,
    *,
    subject_scan_map=None,
    session_num=2,) -> pd.DataFrame:
    # Load and filter the simulation inventory.
    df_query = load_query_results(fn_query)

    for column, value in query_filters.items():
        df_query = df_query.loc[df_query[column].eq(value)]

    df_query = df_query.loc[
        ~df_query["subject_dir"].isin(subjects_exclude)
    ].copy()

    if df_query.empty:
        raise ValueError(
            f"No query rows remain after filtering: {query_filters}"
        )

    # Load and filter the per-series outcomes.
    df_cgi = load_course_data(fn_cgi)
    df_cgi = df_cgi.loc[
        ~df_cgi["subjid"].isin(subjects_exclude)
    ].copy()

    # Select the representative session from each series.
    df_stim = load_stimulation_data(fn_cgi)
    df_stim = df_stim.loc[
        ~df_stim["subjid"].isin(subjects_exclude)
        & df_stim["person_id"].isin(df_cgi["person_id"])
        & df_stim["session_num"].eq(session_num)
    ].copy()

    if df_stim.empty:
        raise ValueError(
            f"No stimulation rows remain for session_num={session_num}."
        )

    # Temporary indices make fallback assignment independent of original indices.
    sessions = df_stim.reset_index(drop=True)

    mapping_options = dict(
        df_query=df_query,
        df_cgi=df_cgi,
        filename=fn_field,
        subject_scan_map=subject_scan_map,
    )

    # prepare_session_data also computes ef_sf.
    try:
        df_stim_mapped = prepare_session_data(
            df_stim=sessions,
            **mapping_options,
        )

    except MissingMappingError as error:
        placements = (
            error.unmatched["electrode_placement"]
            .str.strip()
            .str.upper()
        )

        if not placements.isin(["BT", "BL"]).all():
            raise

        fallback_sessions = sessions.copy()
        fallback_sessions.loc[
            error.unmatched.index, "electrode_placement"
        ] = placements.replace({"BT": "BL", "BL": "BT"})

        # Unresolved or ambiguous mappings still raise.
        df_stim_mapped = prepare_session_data(
            df_stim=fallback_sessions,
            **mapping_options,
        )

    # Restore the original clinical labels and indices.
    df_stim_mapped["electrode_placement"] = sessions.loc[
        df_stim_mapped.index, "electrode_placement"
    ]
    df_stim_mapped.index = df_stim.index.take(df_stim_mapped.index)

    # Uses the strict alignment checks and diagnostics defined earlier.
    return merge_series_data(
        df_cgi=df_cgi,
        df_stim_collapsed=df_stim_mapped,
    )


def get_available_steps(df_query: pd.DataFrame, query_filters: dict, fn_field: str, subj_exclude: list | None = None) -> list:
    """Find available steps after field, configuration, and subject filtering."""
    excluded = [] if subj_exclude is None else subj_exclude
    selected = df_query

    for column, value in query_filters.items():
        selected = selected.loc[selected[column].eq(value)]

    selected = selected.loc[
        ~selected["subject_dir"].isin(excluded)
    ]

    selected = selected.loc[
        selected["full_file_path"].map(
            lambda path: Path(path).name == fn_field
        )
    ]

    available = selected["step"].unique().tolist()

    if any(not isinstance(step, str) or not re.fullmatch(r"step_\d+", step) for step in available):
        raise ValueError(
            f"Invalid step labels in matching query rows: {available}"
        )

    # Numeric ordering: step_0, step_2, step_9, step_10.
    return sorted(
        available,
        key=lambda step: (int(step.split("_", 1)[1]), step),
    )


def load_series_tables(fn_query, fn_cgi, subject_scan_map, field_name: str = "magnE_mean", steps: list | None = None, subj_exclude: list | None = None) -> dict:
    """Load requested steps, or discover available steps separately per model."""
    excluded = [] if subj_exclude is None else subj_exclude

    if steps is not None:
        if not isinstance(steps, list) or not steps:
            raise ValueError(
                "steps must be None or a nonempty list of step labels."
            )

        if any(not isinstance(step, str) or not re.fullmatch(r"step_\d+", step) for step in steps):
            raise ValueError(
                "Use step labels such as 'step_0' and 'step_9'."
            )

        if len(steps) != len(set(steps)):
            raise ValueError("steps contains duplicate labels.")

    fn_field = f"tdcs_uq_gpc_MNI_{field_name}.nii.gz"
    df_query = load_query_results(fn_query)

    common_filters = {
        "threshold": 95,
        "step_result": "mni_volumes",
        "asf": "asf_5",
    }

    # Validate the requested configurations before loading clinical data.
    configurations = []

    for model_type in ("skin_single", "skin_double"):
        filters = {
            **common_filters,
            "model_type_dir": model_type,
        }

        available = get_available_steps(
            df_query,
            filters,
            fn_field,
            subj_exclude=excluded,
        )

        if steps is None:
            selected_steps = available
        else:
            missing = set(steps) - set(available)

            if missing:
                raise ValueError(
                    f"{model_type}: requested steps have no matching field files: "
                    f"{sorted(missing)}. Available steps: {available}"
                )

            selected_steps = steps

        print(f"{model_type}: selected steps = {selected_steps}")

        configurations.extend(
            (model_type, step, filters)
            for step in selected_steps
        )

    if not configurations:
        raise ValueError(
            "No matching model/step configurations remain after filtering."
        )

    tables = {}

    for model_type, step, filters in configurations:
        table = load_series_data(
            subjects_exclude=excluded,
            fn_query=fn_query,
            fn_cgi=fn_cgi,
            query_filters={**filters, "step": step},
            fn_field=fn_field,
            subject_scan_map=subject_scan_map,
            session_num=2,
        )

        tables[(model_type, step)] = table.assign(
            model_type=model_type,
            step=step,
            field_name=field_name,
        )

    return tables


def build_roi_metadata(atlas_path, lut_path) -> pd.DataFrame:
    """Get ROI names and their hemisphere/network labels."""
    _, names = load_label_atlas(atlas_path, lut_path)

    metadata = pd.DataFrame({"roi_name": names.to_numpy()})
    metadata[["hemi", "network"]] = metadata["roi_name"].str.extract(
        r"^7Networks_(LH|RH)_([^_]+)(?:_.+)?$"
    )

    if metadata[["hemi", "network"]].isna().any().any():
        raise ValueError("Unrecognized ROI names in the atlas LUT.")

    return metadata


def extract_unique_field_stats(series_tables: dict, atlas_path, lut_path) -> tuple:
    """Extract statistics once per unique file across all configurations."""
    if not series_tables:
        raise ValueError("No series tables supplied.")

    unique_fields = (
        pd.concat(
            [table[["full_file_path"]] for table in series_tables.values()],
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if unique_fields.empty:
        raise ValueError("No simulation files to extract.")

    roi_stats = extract_region_stats(
        unique_fields,
        atlas_path=atlas_path,
        lut_path=lut_path,
        path_column="full_file_path",
        mode_decimals=None,
    )

    return unique_fields, roi_stats


def roi_statistics_to_rows(unique_fields: pd.DataFrame, roi_stats: dict, roi_metadata: pd.DataFrame) -> pd.DataFrame:
    """Convert metric-specific ROI columns to file/ROI rows."""
    metrics = {
        "mean": "mean",
        "mode": "mode",
        "median": "median",
        "min": "min",
        "max": "max",
        "95th_percentile": "p95",
        "99th_percentile": "p99",
        "n_voxels": "n_voxels",
    }

    rows = unique_fields.merge(
        roi_metadata[["roi_name"]],
        how="cross",
    )

    for source_metric, output_metric in metrics.items():
        table = roi_stats[source_metric]
        expected = [
            f"E_{source_metric}_{roi}"
            for roi in roi_metadata["roi_name"]
        ]

        if not table.index.equals(unique_fields.index):
            raise ValueError(
                f"Unexpected row alignment for {source_metric}."
            )

        if table.columns.has_duplicates or set(table.columns) != set(expected):
            raise ValueError(
                f"Unexpected ROI columns for {source_metric}."
            )

        rows[output_metric] = table[expected].to_numpy().ravel()

    return rows


def merge_series_with_rois(series_tables: dict, field_rows: pd.DataFrame, roi_metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach ROI statistics to every series, preserving clinical metadata."""
    if not series_tables:
        raise ValueError("No series tables supplied.")

    metric_columns = set(field_rows.columns) - {"full_file_path", "roi_name"}
    reserved = set(roi_metadata.columns) | metric_columns | {"_roi_join"}
    tables = []

    for configuration, series in series_tables.items():
        overlap = reserved.intersection(series.columns)

        if overlap:
            raise ValueError(
                f"{configuration}: clinical columns conflict with ROI output: "
                f"{sorted(overlap)}"
            )

        expanded = series.merge(roi_metadata, how="cross")

        table = expanded.merge(
            field_rows,
            on=["full_file_path", "roi_name"],
            how="left",
            sort=False,
            validate="many_to_one",
            indicator="_roi_join",
        )

        if table["_roi_join"].ne("both").any():
            raise ValueError(
                f"Missing ROI-statistic matches for {configuration}."
            )

        tables.append(table.drop(columns="_roi_join"))

    master = pd.concat(tables, ignore_index=True)

    keys = [
        "person_id", "series_num", "model_type",
        "step", "field_name", "roi_name",
    ]

    if master.duplicated(keys).any():
        raise ValueError(
            "Duplicate series/configuration/field/ROI rows."
        )

    first = [
        "person_id", "series_num", "model_type", "step", "field_name",
        "roi_name", "hemi", "network",
        "n_voxels", "mean", "mode", "median", "min", "max", "p95", "p99",
        "cgi_start", "cgi_end", "cgi_change",
    ]

    return master[
        first + [col for col in master.columns if col not in first]
    ]


def build_series_roi_master(fn_query, fn_cgi, atlas_path, lut_path, subject_scan_map, field_name: str = "magnE_mean", steps: list | None = None, subj_exclude: list | None = None) -> pd.DataFrame:
    """Coordinate loading, extraction, reshaping, and joining."""
    series_tables = load_series_tables(
        fn_query=fn_query,
        fn_cgi=fn_cgi,
        subject_scan_map=subject_scan_map,
        field_name=field_name,
        steps=steps,
        subj_exclude=subj_exclude,
    )

    roi_metadata = build_roi_metadata(atlas_path, lut_path)

    unique_fields, roi_stats = extract_unique_field_stats(
        series_tables, atlas_path, lut_path
    )

    field_rows = roi_statistics_to_rows(
        unique_fields, roi_stats, roi_metadata
    )

    return merge_series_with_rois(
        series_tables, field_rows, roi_metadata
    )


def parse_roi_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve the original atlas name and parse its component labels."""
    if df.columns.has_duplicates:
        raise ValueError("Dataframe contains duplicate column names.")

    if "roi_name" in df.columns and "roi_org" in df.columns:
        raise ValueError("Both roi_name and roi_org exist; resolve before parsing.")

    result = df.rename(columns={"roi_name": "roi_org"}).copy()

    if "roi_org" not in result.columns:
        raise KeyError("Expected a roi_name or roi_org column.")

    parsed = result["roi_org"].str.extract(
        r"^7Networks_(?P<hemi>LH|RH)_(?P<network>[^_]+)(?:_(?P<roi>.+))?$"
    )

    invalid = parsed[["hemi", "network"]].isna().any(axis=1)
    if invalid.any():
        names = result.loc[invalid, "roi_org"].unique().tolist()
        raise ValueError(f"Unrecognized ROI names: {names}")

    # Verify existing labels rather than silently replacing conflicting values.
    for column in ("hemi", "network"):
        if column in result.columns:
            conflict = (
                result[column].notna()
                & result[column].ne(parsed[column])
            )
            if conflict.any():
                raise ValueError(
                    f"Existing {column} values conflict with roi_org."
                )

        result[column] = parsed[column]

    result["roi"] = parsed["roi"].fillna(parsed["network"])

    # Place the short label immediately after the original name.
    columns = [column for column in result.columns if column != "roi"]
    columns.insert(columns.index("roi_org") + 1, "roi")

    return result[columns]


def save_roi_master(master: pd.DataFrame, filename: str | Path) -> Path:
    """Save an already-built master dataframe."""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    master.to_csv(output_path, index=False)

    print(
        f"Saved {len(master)} rows and "
        f"{len(master.columns)} columns to {output_path}"
    )

    return output_path

if __name__ == '__main__':
    field_name = "magnE_mean"

    fn_query = '/projects/p32903/Alex2/ect-results-tools/query_result_2.csv'
    fn_cgi   = '/projects/p32903/Alex2/datasets/STU00225089/.clincal_notes/cgi_batch_latest_20261029_1207.xlsx'

    # atlases
    fn_lut = '/projects/p32903/Alex2/ect-workflow/atlases/Yeo2011/7Networks_ColorLUT_freeview.txt'
    fn_atl = '/projects/p32903/Alex2/ect-workflow/atlases/Yeo2011/Yeo2011_7Networks_N1000.split_components.FSL_MNI152_1mm.nii.gz'

    steps = ["step_0", "step_9"]    # Or ["step_0", "step_9"].
    # steps = None
    subj_exclude = ['subj-cat-010-001']  # Add subject IDs to exclude here.
    
    print(f"Processing field:   {field_name} ")
    print(f"Steps:              {steps}")
    print(f"Excluding subjects: {subj_exclude}")
    
    df_series_roi_master = build_series_roi_master(
        fn_query=fn_query,
        fn_cgi=fn_cgi,
        atlas_path=fn_atl,
        lut_path=fn_lut,
        subject_scan_map=analysis_dir / "subject_scan_map.json",
        field_name=field_name,
        steps=steps,
        subj_exclude=subj_exclude,
    )

    df_series_roi_master = parse_roi_metadata(df_series_roi_master)

    saved_path = save_roi_master(
        df_series_roi_master,
        f"data/yeo2011_{field_name}_series_ROI_master.csv",
    )