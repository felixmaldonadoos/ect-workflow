"""Assign FreeSurfer volumetric atlas labels to SimNIBS tetrahedra."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .identifiers import resolve_freesurfer_subjects
from .mesh_io import build_hdf5_inventory, element_geometry, load_hdf5_mesh, scalar_mesh_field
from .summary import ParcelSummary, build_parcel_summary, mean_labeled_field, summarize_labeled_field


VOLUME_ATLAS_FILES = {
    "aparc": "aparc+aseg.mgz",
    "dk": "aparc+aseg.mgz",
    "dk40": "aparc+aseg.mgz",
    "a2009s": "aparc.a2009s+aseg.mgz",
    "destrieux": "aparc.a2009s+aseg.mgz",
}

DEFAULT_SUBCORTICAL_STRUCTURES = (
    "thalamus",
    "thalamus-proper",
    "caudate",
    "putamen",
    "pallidum",
    "hippocampus",
    "amygdala",
    "accumbens-area",
    "ventraldc",
    "cerebellum-cortex",
)


def canonical_volume_atlas_name(atlas_name: str) -> str:
    """Normalize supported volumetric atlas aliases."""
    key = str(atlas_name).strip().casefold()
    if key not in VOLUME_ATLAS_FILES:
        raise ValueError(f"Unsupported volume atlas {atlas_name!r}; choose from {sorted(VOLUME_ATLAS_FILES)}")
    return "a2009s" if key in {"a2009s", "destrieux"} else "aparc"


def resolve_volume_atlas_path(fs_subject_dir: str | Path, atlas_name: str) -> Path:
    """Locate a recon-all cortical-plus-subcortical segmentation volume."""
    canonical = canonical_volume_atlas_name(atlas_name)
    path = Path(fs_subject_dir) / "mri" / VOLUME_ATLAS_FILES[canonical]
    if not path.is_file():
        raise FileNotFoundError(f"FreeSurfer atlas volume does not exist: {path}")
    return path.resolve()


def resolve_freesurfer_lut(path: str | Path | None = None) -> Path:
    """Resolve ``FreeSurferColorLUT.txt`` without embedding installation paths."""
    if path is not None:
        resolved = Path(path).expanduser().resolve()
    else:
        freesurfer_home = os.getenv("FREESURFER_HOME")
        if not freesurfer_home:
            raise EnvironmentError("Set FREESURFER_HOME or pass freesurfer_lut explicitly")
        resolved = Path(freesurfer_home) / "FreeSurferColorLUT.txt"
    if not resolved.is_file():
        raise FileNotFoundError(f"FreeSurfer color lookup table does not exist: {resolved}")
    return resolved


def read_freesurfer_lut(path: str | Path) -> dict[int, str]:
    """Read label IDs and names from ``FreeSurferColorLUT.txt``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FreeSurfer color lookup table does not exist: {path}")
    labels = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        try:
            label_id = int(parts[0])
        except (IndexError, ValueError):
            continue
        if len(parts) < 2:
            raise ValueError(f"Malformed LUT row {line_number} in {path}")
        if label_id in labels:
            raise ValueError(f"Duplicate label ID {label_id} in {path}")
        labels[label_id] = parts[1]
    if not labels:
        raise ValueError(f"No labels were parsed from {path}")
    return labels


def load_label_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an integer-valued NIfTI/MGH/MGZ atlas and its voxel-to-RAS affine."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Atlas volume does not exist: {path}")
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Volumetric atlas analysis requires nibabel") from exc

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj).squeeze()
    affine = np.asarray(image.affine, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"Atlas volume must be 3D after squeezing singleton axes, got {data.shape}: {path}")
    if not np.isfinite(data).all() or not np.allclose(data, np.rint(data)):
        raise ValueError(f"Atlas volume must contain finite integer labels: {path}")
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or np.isclose(np.linalg.det(affine[:3, :3]), 0):
        raise ValueError(f"Atlas affine is invalid: {path}")
    return np.rint(data).astype(np.int32), affine


def transform_points(points: np.ndarray, affine: np.ndarray | None) -> np.ndarray:
    """Apply a 4x4 affine to N×3 points; ``None`` means identity."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"points must be a finite N x 3 array, got {points.shape}")
    if affine is None:
        return points.copy()
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("mesh_to_atlas_ras must be a finite 4 x 4 affine")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    transformed = homogeneous @ affine.T
    if np.isclose(transformed[:, 3], 0).any():
        raise ValueError("Affine transformation produced points at infinity")
    return transformed[:, :3] / transformed[:, 3, None]


def sample_volume_labels(points_ras: np.ndarray, label_data: np.ndarray, voxel_to_ras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor sample discrete atlas labels at RAS coordinates."""
    points_ras = np.asarray(points_ras, dtype=float)
    label_data = np.asarray(label_data)
    voxel_to_ras = np.asarray(voxel_to_ras, dtype=float)
    if points_ras.ndim != 2 or points_ras.shape[1] != 3:
        raise ValueError("points_ras must have shape (N, 3)")
    if label_data.ndim != 3:
        raise ValueError("label_data must be three-dimensional")
    if voxel_to_ras.shape != (4, 4) or np.isclose(np.linalg.det(voxel_to_ras[:3, :3]), 0):
        raise ValueError("voxel_to_ras must be an invertible 4 x 4 affine")

    ras_to_voxel = np.linalg.inv(voxel_to_ras)
    homogeneous = np.column_stack([points_ras, np.ones(len(points_ras))])
    voxel = np.rint((homogeneous @ ras_to_voxel.T)[:, :3]).astype(np.int64)
    shape = np.asarray(label_data.shape)
    inside = ((voxel >= 0) & (voxel < shape)).all(axis=1)
    labels = np.zeros(len(points_ras), dtype=np.int32)
    valid = voxel[inside]
    labels[inside] = label_data[valid[:, 0], valid[:, 1], valid[:, 2]].astype(np.int32)
    return labels, inside


def select_volume_roi_labels(
    labels_present: Sequence[int],
    lut: Mapping[int, str],
    *,
    include_cortex: bool = True,
    subcortical_structures: Sequence[str] = DEFAULT_SUBCORTICAL_STRUCTURES,
    include_brainstem: bool = False,
    roi_names: Sequence[str] | None = None,
    exclude_roi_names: Sequence[str] = (),
) -> dict[int, str]:
    """Select cortical parcels and named gray-matter structures from a LUT.

    Labels in ``exclude_roi_names`` are removed before ROI sets are compared
    across scans. Exclusion names must exist in the LUT, but they need not be
    present in every subject's atlas volume.
    """
    present = {int(value) for value in labels_present if int(value) != 0}
    unknown_ids = sorted(present.difference(lut))
    if unknown_ids:
        raise KeyError(f"Atlas contains label IDs absent from the FreeSurfer LUT: {unknown_ids[:20]}")

    excluded = {str(value) for value in exclude_roi_names}
    if len(excluded) != len(exclude_roi_names):
        raise ValueError("exclude_roi_names contains duplicates")

    lut_names = set(lut.values())
    missing_excluded = sorted(excluded.difference(lut_names))
    if missing_excluded:
        raise ValueError(f"Excluded ROIs are absent from the FreeSurfer LUT: {missing_excluded}")

    if roi_names is not None:
        requested = {str(value) for value in roi_names}
        if len(requested) != len(roi_names):
            raise ValueError("roi_names contains duplicates")
        overlap = sorted(requested.intersection(excluded))
        if overlap:
            raise ValueError(f"ROIs cannot be both requested and excluded: {overlap}")
        name_to_id = {name: label_id for label_id, name in lut.items()}
        missing_lut = sorted(requested.difference(name_to_id))
        missing_volume = sorted(name for name in requested if name in name_to_id and name_to_id[name] not in present)
        if missing_lut or missing_volume:
            raise ValueError(f"Requested ROI mismatch; absent from LUT={missing_lut}, absent from volume={missing_volume}")
        return {name_to_id[name]: name for name in sorted(requested)}

    structures = {str(value).casefold() for value in subcortical_structures}
    selected = {}
    for label_id in sorted(present):
        name = lut[label_id]
        if name in excluded:
            continue
        normalized = name.casefold()
        is_cortex = normalized.startswith(("ctx-lh-", "ctx-rh-", "ctx_lh_", "ctx_rh_")) and not normalized.endswith(
            ("-unknown", "_unknown", "-corpuscallosum", "_corpuscallosum", "-corpus_callosum", "_corpus_callosum")
        )
        without_side = normalized.removeprefix("left-").removeprefix("right-")
        is_subcortex = without_side in structures
        is_brainstem = normalized == "brain-stem"
        if (include_cortex and is_cortex) or is_subcortex or (include_brainstem and is_brainstem):
            selected[label_id] = name
    if len(selected) < 2:
        raise ValueError(f"ROI selection produced only {len(selected)} labels")
    return selected


def build_volume_roi_summary(
    hdf5_files: Sequence[str | Path],
    subjects_dir: str | Path,
    *,
    atlas_name: str = "aparc",
    subject_ids: Sequence[str] | None = None,
    simulation_ids: Sequence[str] | None = None,
    subject_map: Mapping[str, str] | None = None,
    field_name: str = "magnE_mean",
    percentile: float = 95,
    min_elements_per_roi: int = 20,
    freesurfer_lut: str | Path | None = None,
    roi_names: Sequence[str] | None = None,
    exclude_roi_names: Sequence[str] = (),
    include_cortex: bool = True,
    subcortical_structures: Sequence[str] = DEFAULT_SUBCORTICAL_STRUCTURES,
    include_brainstem: bool = False,
    mesh_key: str = "mesh_roi",
    mesh_to_atlas_ras: np.ndarray | Mapping[str, np.ndarray] | None = None,
    mesh_loader: Callable | None = None,
    atlas_loader: Callable[[str | Path], tuple[np.ndarray, np.ndarray]] | None = None,
    lut: Mapping[int, str] | None = None) -> ParcelSummary:

    """Calculate anatomical parcel P95 values from HDF5 volume meshes."""
    canonical = canonical_volume_atlas_name(atlas_name)
    scans = build_hdf5_inventory(hdf5_files, subject_ids=subject_ids, simulation_ids=simulation_ids,
                                 require_files=mesh_loader is None)
    scans = resolve_freesurfer_subjects(scans, subjects_dir, subject_map=subject_map)
    scans["atlas_file"] = [str(resolve_volume_atlas_path(path, canonical)) for path in scans["fs_subject_dir"]]
    if lut is None:
        lut = read_freesurfer_lut(resolve_freesurfer_lut(freesurfer_lut))
    if atlas_loader is None:
        atlas_loader = load_label_volume

    if isinstance(mesh_to_atlas_ras, Mapping):
        missing_transforms = sorted(set(scans["subjid"]).difference(mesh_to_atlas_ras))
        extra_transforms = sorted(set(mesh_to_atlas_ras).difference(scans["subjid"]))
        if missing_transforms or extra_transforms:
            raise ValueError(f"mesh_to_atlas_ras key mismatch; missing={missing_transforms}, extra={extra_transforms}")

    ordinary_rows, weighted_rows, size_rows, count_rows, qc_rows = [], [], [], [], []
    mean_rows = []
    expected_rois = None
    for row in scans.itertuples(index=False):
        mesh = load_hdf5_mesh(row.hdf5_file, mesh_key=mesh_key, mesh_loader=mesh_loader)
        field, _ = scalar_mesh_field(mesh, field_name, expected_location="element")
        centers, sizes, element_types = element_geometry(mesh)
        tetrahedra = element_types == 4
        if not tetrahedra.any():
            raise ValueError(f"No tetrahedra found for {row.subjid}")

        atlas_data, voxel_to_ras = atlas_loader(row.atlas_file)
        scan_affine = mesh_to_atlas_ras[row.subjid] if isinstance(mesh_to_atlas_ras, Mapping) else mesh_to_atlas_ras
        points_ras = transform_points(centers[tetrahedra], scan_affine)
        sampled_labels, inside = sample_volume_labels(points_ras, atlas_data, voxel_to_ras)
        label_names = select_volume_roi_labels(
            np.unique(atlas_data),
            lut,
            include_cortex=include_cortex,
            subcortical_structures=subcortical_structures,
            include_brainstem=include_brainstem,
            roi_names=roi_names,
            exclude_roi_names=exclude_roi_names,
        )
        roi_names_current = tuple(sorted(label_names.values()))
        if expected_rois is None:
            expected_rois = roi_names_current
        elif roi_names_current != expected_rois:
            missing = sorted(set(expected_rois).difference(roi_names_current))
            extra = sorted(set(roi_names_current).difference(expected_rois))
            raise ValueError(f"Atlas ROI mismatch for {row.subjid}; missing={missing}, extra={extra}")

        ordinary, weighted, parcel_size, parcel_count = summarize_labeled_field(
            field[tetrahedra],
            sizes[tetrahedra],
            sampled_labels,
            label_names,
            percentile=percentile,
            min_samples_per_roi=min_elements_per_roi,
        )
        selected_mask = np.isin(sampled_labels, list(label_names))
        ordinary_rows.append(ordinary)
        weighted_rows.append(weighted)
        size_rows.append(parcel_size)
        count_rows.append(parcel_count)
        mean_rows.append(mean_labeled_field(field[tetrahedra], sampled_labels, label_names))
        qc_rows.append(
            {
                "atlas_file": row.atlas_file,
                "n_tetrahedra": int(tetrahedra.sum()),
                "n_centers_inside_atlas": int(inside.sum()),
                "n_nonzero_atlas_elements": int((sampled_labels != 0).sum()),
                "n_selected_roi_elements": int(selected_mask.sum()),
                "selected_roi_volume_mm3": float(sizes[tetrahedra][selected_mask].sum()),
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
        atlas_name=canonical,
        domain="volume",
        mean_rows=mean_rows,
    )
