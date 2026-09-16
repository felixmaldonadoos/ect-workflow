"""Preflight volumetric ROI sample counts before running the full PCA analysis.

This performs the same nearest-neighbor assignment of tetrahedral centers to
FreeSurfer atlas labels as ``build_volume_roi_summary``. It loads each HDF5 mesh
once and skips electric-field summaries, PCA, and outcome analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from run_pca_weighted_global_E_job_synthsr import (
    ANALYSIS_STEP,
    EXCLUDED_SUBJIDS,
    EXCLUDE_VOLUME_ROIS_BY_ATLAS,
    FREESURFER_LUT,
    MIN_VOLUME_ELEMENTS_PER_ROI,
    MODEL_TYPE_DIR,
    OUTPUT_DIR,
    QUERY_PATH,
    SUBJECT_MAP,
    SUBJECTS_DIR_BY_DATASET,
    exclude_subject_results,
    load_hdf5_inventory,
    validate_analysis_paths,
)
from simnibs_parcel_analysis.atlas_volume import (
    DEFAULT_SUBCORTICAL_STRUCTURES,
    load_label_volume,
    read_freesurfer_lut,
    resolve_volume_atlas_path,
    sample_volume_labels,
    select_volume_roi_labels,
    transform_points,
)
from simnibs_parcel_analysis.identifiers import resolve_freesurfer_subjects
from simnibs_parcel_analysis.mesh_io import build_hdf5_inventory, element_geometry, load_hdf5_mesh


DEFAULT_ATLASES = ("aparc", "a2009s")
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "volume_roi_sample_count_preflight"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minimum-samples",
        type=int,
        default=MIN_VOLUME_ELEMENTS_PER_ROI,
        help=f"Flag ROIs with fewer samples than this value (default: {MIN_VOLUME_ELEMENTS_PER_ROI}).",
    )
    parser.add_argument(
        "--atlases",
        nargs="+",
        choices=DEFAULT_ATLASES,
        default=list(DEFAULT_ATLASES),
        help="Volume atlases to inspect.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV reports.")
    args = parser.parse_args()
    if args.minimum_samples < 1:
        parser.error("--minimum-samples must be at least 1")
    if len(set(args.atlases)) != len(args.atlases):
        parser.error("--atlases contains duplicates")
    return args


def build_resolved_scan_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Resolve each selected HDF5 file to its FreeSurfer subject directory."""
    frames = []
    for dataset_root, subjects_dir in SUBJECTS_DIR_BY_DATASET.items():
        source = inventory.loc[inventory["dataset_root"].eq(dataset_root)]
        if source.empty:
            raise ValueError(f"No selected scans remain for dataset {dataset_root!r}")

        scans = build_hdf5_inventory(source["full_file_path"].tolist(), require_files=True)
        scans = resolve_freesurfer_subjects(scans, subjects_dir, subject_map=SUBJECT_MAP)
        scans.insert(0, "dataset_root", dataset_root)
        frames.append(scans)

    scans = pd.concat(frames, ignore_index=True)
    if scans["subjid"].duplicated().any():
        duplicates = scans.loc[scans["subjid"].duplicated(keep=False), "subjid"].unique().tolist()
        raise ValueError(f"Duplicate modeled-scan IDs across data sources: {duplicates}")
    return scans


def count_volume_roi_samples(
    scans: pd.DataFrame,
    atlas_names: Sequence[str],
    freesurfer_lut: str | Path,
    exclude_rois_by_atlas: Mapping[str, Sequence[str]],
    minimum_samples: int,
    *,
    mesh_key: str = "mesh_roi",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Count mesh tetrahedra assigned to every selected ROI for every scan."""
    lut = read_freesurfer_lut(freesurfer_lut)
    count_rows = []
    scan_qc_rows = []
    schema_rows = []
    expected_rois_by_atlas: dict[str, tuple[str, ...]] = {}

    for scan_number, row in enumerate(scans.itertuples(index=False), start=1):
        print(f"[{scan_number}/{len(scans)}] Loading mesh for {row.subjid}")
        mesh = load_hdf5_mesh(row.hdf5_file, mesh_key=mesh_key)
        centers, _, element_types = element_geometry(mesh)
        tetrahedra = element_types == 4
        if not tetrahedra.any():
            raise ValueError(f"No tetrahedra found for {row.subjid}")
        tetrahedron_centers = centers[tetrahedra]

        for atlas_name in atlas_names:
            atlas_file = resolve_volume_atlas_path(row.fs_subject_dir, atlas_name)
            atlas_data, voxel_to_ras = load_label_volume(atlas_file)
            points_ras = transform_points(tetrahedron_centers, None)
            sampled_labels, inside = sample_volume_labels(points_ras, atlas_data, voxel_to_ras)
            exclude_roi_names = tuple(exclude_rois_by_atlas.get(atlas_name, ()))
            label_names = select_volume_roi_labels(
                np.unique(atlas_data),
                lut,
                include_cortex=True,
                subcortical_structures=DEFAULT_SUBCORTICAL_STRUCTURES,
                include_brainstem=False,
                exclude_roi_names=exclude_roi_names,
            )

            roi_names_current = tuple(sorted(label_names.values()))
            if atlas_name not in expected_rois_by_atlas:
                expected_rois_by_atlas[atlas_name] = roi_names_current
            elif roi_names_current != expected_rois_by_atlas[atlas_name]:
                expected = set(expected_rois_by_atlas[atlas_name])
                current = set(roi_names_current)
                schema_rows.extend(
                    {
                        "dataset_root": row.dataset_root,
                        "subjid": row.subjid,
                        "atlas_name": atlas_name,
                        "issue": issue,
                        "roi_name": roi_name,
                    }
                    for issue, roi_names in (
                        ("missing", sorted(expected - current)),
                        ("extra", sorted(current - expected)),
                    )
                    for roi_name in roi_names
                )

            present_ids, present_counts = np.unique(sampled_labels, return_counts=True)
            samples_by_label = dict(zip(present_ids.astype(int), present_counts.astype(int), strict=True))
            selected_ids = np.fromiter(label_names, dtype=int)
            n_selected_samples = int(np.isin(sampled_labels, selected_ids).sum())

            for label_id, roi_name in label_names.items():
                n_samples = samples_by_label.get(label_id, 0)
                count_rows.append(
                    {
                        "dataset_root": row.dataset_root,
                        "subjid": row.subjid,
                        "atlas_name": atlas_name,
                        "roi_name": roi_name,
                        "label_id": label_id,
                        "n_samples": n_samples,
                        "minimum_samples": minimum_samples,
                        "below_minimum": n_samples < minimum_samples,
                        "hdf5_file": row.hdf5_file,
                        "atlas_file": str(atlas_file),
                    }
                )

            scan_qc_rows.append(
                {
                    "dataset_root": row.dataset_root,
                    "subjid": row.subjid,
                    "atlas_name": atlas_name,
                    "n_tetrahedra": int(tetrahedra.sum()),
                    "n_centers_inside_atlas": int(inside.sum()),
                    "n_nonzero_atlas_samples": int((sampled_labels != 0).sum()),
                    "n_selected_roi_samples": n_selected_samples,
                    "n_selected_rois": len(label_names),
                    "excluded_roi_names": ";".join(exclude_roi_names),
                }
            )

    counts = pd.DataFrame(count_rows)
    scan_qc = pd.DataFrame(scan_qc_rows)
    schema = pd.DataFrame(schema_rows, columns=["dataset_root", "subjid", "atlas_name", "issue", "roi_name"])
    return counts, scan_qc, schema


def write_reports(
    counts: pd.DataFrame,
    scan_qc: pd.DataFrame,
    schema: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Write complete and issue-only preflight reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    low_counts = counts.loc[counts["below_minimum"]].sort_values(["n_samples", "atlas_name", "subjid", "roi_name"])
    counts.sort_values(["atlas_name", "subjid", "roi_name"]).to_csv(
        output_dir / "volume_roi_sample_counts.csv", index=False
    )
    low_counts.to_csv(output_dir / "volume_roi_low_sample_rois.csv", index=False)
    scan_qc.sort_values(["atlas_name", "subjid"]).to_csv(output_dir / "volume_roi_scan_qc.csv", index=False)
    schema.sort_values(["atlas_name", "subjid", "issue", "roi_name"]).to_csv(
        output_dir / "volume_roi_schema_mismatches.csv", index=False
    )
    return low_counts


def main() -> None:
    """Run the exact volume-ROI sample-count preflight."""
    args = parse_args()
    validate_analysis_paths()
    inventory = load_hdf5_inventory(QUERY_PATH, SUBJECTS_DIR_BY_DATASET, MODEL_TYPE_DIR, ANALYSIS_STEP)
    if EXCLUDED_SUBJIDS:
        inventory = exclude_subject_results(inventory, list(EXCLUDED_SUBJIDS))
    scans = build_resolved_scan_inventory(inventory)

    print(f"Checking {len(scans)} scans; atlases={args.atlases}; minimum samples={args.minimum_samples}")
    for atlas_name in args.atlases:
        excluded = tuple(EXCLUDE_VOLUME_ROIS_BY_ATLAS.get(atlas_name, ()))
        if excluded:
            print(f"Excluding {atlas_name} ROIs from every scan: {list(excluded)}")

    counts, scan_qc, schema = count_volume_roi_samples(
        scans,
        args.atlases,
        FREESURFER_LUT,
        EXCLUDE_VOLUME_ROIS_BY_ATLAS,
        args.minimum_samples,
    )
    low_counts = write_reports(counts, scan_qc, schema, args.output_dir)

    print(f"\nChecked {len(scans)} scans across {len(args.atlases)} atlases.")
    print(f"ROIs below {args.minimum_samples} samples: {len(low_counts)}")
    print(f"ROI schema mismatches: {len(schema)}")
    print(f"Reports: {args.output_dir.resolve()}")

    if not low_counts.empty:
        print("\nLow-sample ROIs:")
        report_columns = ["dataset_root", "subjid", "atlas_name", "roi_name", "label_id", "n_samples"]
        print(low_counts[report_columns].to_string(index=False))
    if not schema.empty:
        print("\nROI schema mismatches:")
        print(schema.to_string(index=False))


if __name__ == "__main__":
    main()
