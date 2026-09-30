"""Parcel SimNIBS E-fields already mapped to the fsaverage cortical surface."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .mesh_io import build_hdf5_inventory, fsaverage_surface_path, load_surface_mesh, node_areas, scalar_mesh_field
from .summary import ParcelSummary, build_parcel_summary, mean_labeled_field, summarize_labeled_field


SURFACE_ATLASES = frozenset({"HCP_MMP1", "DK40", "a2009s"})


def load_simnibs_surface_atlas(atlas_name: str) -> Mapping[str, np.ndarray]:
    """Load one of the cortical fsaverage atlases distributed by SimNIBS."""
    if atlas_name not in SURFACE_ATLASES:
        raise ValueError(f"atlas_name must be one of {sorted(SURFACE_ATLASES)}")
    import simnibs

    return simnibs.get_atlas(atlas_name, hemi="both")


def atlas_masks_to_labels(atlas_masks: Mapping[str, np.ndarray], n_nodes: int) -> tuple[np.ndarray, dict[int, str]]:
    """Convert nonoverlapping named surface masks to one integer label array."""
    if not atlas_masks:
        raise ValueError("Surface atlas is empty")
    labels = np.zeros(n_nodes, dtype=np.int32)
    label_names = {}
    for label_id, name in enumerate(sorted(atlas_masks), start=1):
        mask = np.asarray(atlas_masks[name], dtype=bool).squeeze()
        if mask.shape != (n_nodes,):
            raise ValueError(f"Surface ROI {name!r} has shape {mask.shape}; expected ({n_nodes},)")
        if not mask.any():
            raise ValueError(f"Surface ROI {name!r} is empty")
        if (labels[mask] != 0).any():
            raise ValueError(f"Surface ROI {name!r} overlaps another atlas ROI")
        labels[mask] = label_id
        label_names[label_id] = str(name)
    return labels, label_names


def build_surface_roi_summary(
    hdf5_files: Sequence[str | Path],
    *,
    atlas_name: str = "HCP_MMP1",
    surface_files: Sequence[str | Path] | None = None,
    surface_relative_path: str | Path = "fsavg_overlays/tdcs_uq_gpc_fsavg.msh",
    subject_ids: Sequence[str] | None = None,
    simulation_ids: Sequence[str] | None = None,
    field_name: str = "magnE_mean",
    percentile: float = 95,
    min_nodes_per_roi: int = 20,
    mesh_loader: Callable | None = None,
    atlas_loader: Callable[[str], Mapping[str, np.ndarray]] | None = None,
) -> ParcelSummary:
    """Calculate cortical parcel P95 values from fsaverage E-field meshes.

    The HDF5 paths identify and order modeled scans. The actual surface field
    comes from each scan's fsaverage ``.msh`` overlay.
    """
    if atlas_name not in SURFACE_ATLASES:
        raise ValueError(f"atlas_name must be one of {sorted(SURFACE_ATLASES)}")
    scans = build_hdf5_inventory(hdf5_files, subject_ids=subject_ids, simulation_ids=simulation_ids, require_files=True)
    if surface_files is None:
        resolved_surface_files = [fsaverage_surface_path(path, relative_path=surface_relative_path) for path in scans["hdf5_file"]]
    else:
        if len(surface_files) != len(scans):
            raise ValueError(f"surface_files has {len(surface_files)} paths but hdf5_files has {len(scans)}")
        resolved_surface_files = [Path(path).resolve() for path in surface_files]
        missing = [str(path) for path in resolved_surface_files if not path.is_file()]
        if missing:
            raise FileNotFoundError("Surface mesh files do not exist:\n" + "\n".join(missing))
    scans["surface_file"] = [str(path) for path in resolved_surface_files]

    if atlas_loader is None:
        atlas_loader = load_simnibs_surface_atlas
    atlas_masks = atlas_loader(atlas_name)

    ordinary_rows, weighted_rows, size_rows, count_rows, qc_rows = [], [], [], [], []
    mean_rows = []
    expected_rois = None
    for row in scans.itertuples(index=False):
        mesh = load_surface_mesh(row.surface_file, mesh_loader=mesh_loader)
        field, _ = scalar_mesh_field(mesh, field_name, expected_location="node")
        areas = node_areas(mesh)
        labels, label_names = atlas_masks_to_labels(atlas_masks, field.size)
        roi_names_current = tuple(sorted(label_names.values()))
        if expected_rois is None:
            expected_rois = roi_names_current
        elif roi_names_current != expected_rois:
            raise RuntimeError("Surface atlas changed between modeled scans")

        ordinary, weighted, parcel_size, parcel_count = summarize_labeled_field(
            field,
            areas,
            labels,
            label_names,
            percentile=percentile,
            min_samples_per_roi=min_nodes_per_roi,
        )
        selected = labels != 0
        ordinary_rows.append(ordinary)
        weighted_rows.append(weighted)
        size_rows.append(parcel_size)
        count_rows.append(parcel_count)
        mean_rows.append(mean_labeled_field(field, labels, label_names))
        qc_rows.append(
            {
                "surface_file": row.surface_file,
                "n_nodes": field.size,
                "n_selected_roi_nodes": int(selected.sum()),
                "selected_roi_area_mm2": float(areas[selected].sum()),
                "n_rois": len(label_names),
            }
        )

    return build_parcel_summary(
        scans,
        ordinary_rows,
        weighted_rows,
        size_rows,
        count_rows,
        qc_rows,
        atlas_name=atlas_name,
        domain="surface",
        mean_rows=mean_rows,
    )
