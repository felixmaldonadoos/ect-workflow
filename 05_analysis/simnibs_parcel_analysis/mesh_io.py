"""Discovery and loading of SimNIBS meshes stored in gPC HDF5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd

from .identifiers import add_subject_id_columns, subject_id_from_path


def discover_hdf5_files(
    results_root: str | Path,
    *,
    pattern: str = "**/tdcs_uq_gpc.hdf5",
) -> list[Path]:
    """Discover HDF5 files with a caller-configurable relative glob pattern."""
    results_root = Path(results_root)
    if not results_root.is_dir():
        raise NotADirectoryError(f"Results directory does not exist: {results_root}")
    paths = sorted(path.resolve() for path in results_root.glob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No HDF5 files matched {pattern!r} under {results_root}")
    return paths


def build_hdf5_inventory(
    hdf5_files: Sequence[str | Path],
    *,
    subject_ids: Sequence[str] | None = None,
    require_files: bool = True,
) -> pd.DataFrame:
    """Build a strict one-row-per-modeled-scan HDF5 inventory."""
    paths = [Path(path) for path in hdf5_files]
    if not paths:
        raise ValueError("hdf5_files is empty")
    if require_files:
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("HDF5 files do not exist:\n" + "\n".join(missing))
    bad_suffixes = [str(path) for path in paths if path.suffix.lower() not in {".h5", ".hdf5"}]
    if bad_suffixes:
        raise ValueError(f"Expected .h5 or .hdf5 files: {bad_suffixes}")

    if subject_ids is None:
        subject_ids = [subject_id_from_path(path) for path in paths]
    elif len(subject_ids) != len(paths):
        raise ValueError(f"subject_ids has {len(subject_ids)} values but hdf5_files has {len(paths)}")
    inventory = pd.DataFrame({"subjid": list(subject_ids), "hdf5_file": [str(path) for path in paths]})
    inventory = add_subject_id_columns(inventory, require_scan=True)

    duplicate_ids = inventory.loc[inventory["subjid"].duplicated(keep=False), "subjid"].tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate modeled-scan IDs: {duplicate_ids}")
    duplicate_paths = inventory.loc[inventory["hdf5_file"].duplicated(keep=False), "hdf5_file"].tolist()
    if duplicate_paths:
        raise ValueError(f"Duplicate HDF5 paths: {duplicate_paths}")
    return inventory[["subjid", "subjid_base", "scan_id", "hdf5_file"]]


def load_hdf5_mesh(
    hdf5_file: str | Path,
    *,
    mesh_key: str = "mesh_roi",
    mesh_loader: Callable | None = None,
):
    """Load a SimNIBS ``Msh`` from an HDF5 group such as ``mesh_roi``."""
    path = Path(hdf5_file)
    if mesh_loader is None:
        if not path.is_file():
            raise FileNotFoundError(f"HDF5 file does not exist: {path}")
        import simnibs

        mesh_loader = simnibs.Msh.read_hdf5
    try:
        return mesh_loader(str(path), mesh_key)
    except (KeyError, OSError, ValueError) as exc:
        raise type(exc)(f"Could not load HDF5 mesh {mesh_key!r} from {path}: {exc}") from exc


def fsaverage_surface_path(
    hdf5_file: str | Path,
    *,
    relative_path: str | Path = "fsavg_overlays/tdcs_uq_gpc_fsavg.msh",
) -> Path:
    """Return the expected fsaverage E-field mesh beside a gPC HDF5 file."""
    path = Path(hdf5_file).parent / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Surface E-field mesh does not exist: {path}. Surface atlas analysis requires a SimNIBS fsaverage overlay."
        )
    return path.resolve()


def load_surface_mesh(surface_file: str | Path, *, mesh_loader: Callable | None = None):
    """Load a SimNIBS surface ``.msh`` file."""
    path = Path(surface_file)
    if mesh_loader is None:
        if not path.is_file():
            raise FileNotFoundError(f"Surface mesh does not exist: {path}")
        import simnibs

        mesh_loader = simnibs.read_msh
    try:
        return mesh_loader(str(path))
    except (OSError, ValueError) as exc:
        raise type(exc)(f"Could not load surface mesh {path}: {exc}") from exc


def mesh_array(value) -> np.ndarray:
    """Convert a SimNIBS data object or array-like object to an ndarray."""
    raw = value.value if hasattr(value, "value") else value
    return np.asarray(raw)


def scalar_mesh_field(mesh, field_name: str, *, expected_location: Literal["element", "node"] | None = None) -> tuple[np.ndarray, str]:
    """Extract and strictly validate one scalar mesh field."""
    if field_name not in mesh.field:
        raise KeyError(f"Field {field_name!r} not found; available fields: {sorted(mesh.field.keys())}")
    element_fields = {field.field_name for field in mesh.elmdata}
    node_fields = {field.field_name for field in mesh.nodedata}
    if field_name in element_fields and field_name in node_fields:
        raise ValueError(f"Field {field_name!r} exists as both ElementData and NodeData")
    if field_name in element_fields:
        location, expected = "element", int(mesh.elm.nr)
    elif field_name in node_fields:
        location, expected = "node", int(mesh.nodes.nr)
    else:
        raise ValueError(f"Could not determine whether field {field_name!r} is element or node data")
    if expected_location is not None and location != expected_location:
        raise ValueError(f"Field {field_name!r} must be {expected_location} data, got {location} data")

    values = np.asarray(mesh_array(mesh.field[field_name]), dtype=float).squeeze()
    if values.ndim != 1 or values.size != expected:
        raise ValueError(f"Field {field_name!r} has shape {values.shape}; expected ({expected},)")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Field {field_name!r} must contain finite, nonnegative scalar values")
    return values, location


def element_geometry(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return element barycenters, element sizes, and element types."""
    centers = np.asarray(mesh_array(mesh.elements_baricenters()), dtype=float)
    sizes = np.asarray(mesh_array(mesh.elements_volumes_and_areas()), dtype=float).squeeze()
    element_types = np.asarray(mesh.elm.elm_type, dtype=int).squeeze()
    expected = int(mesh.elm.nr)
    if centers.shape != (expected, 3):
        raise ValueError(f"Element barycenters have shape {centers.shape}; expected ({expected}, 3)")
    if sizes.shape != (expected,) or element_types.shape != (expected,):
        raise ValueError("Element sizes/types do not match mesh.elm.nr")
    if not np.isfinite(centers).all() or not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError("Element geometry contains invalid coordinates or nonpositive sizes")
    return centers, sizes, element_types


def node_areas(mesh) -> np.ndarray:
    """Return finite, positive surface area represented by each node."""
    areas = np.asarray(mesh_array(mesh.nodes_areas()), dtype=float).squeeze()
    expected = int(mesh.nodes.nr)
    if areas.shape != (expected,):
        raise ValueError(f"Node areas have shape {areas.shape}; expected ({expected},)")
    if not np.isfinite(areas).all() or (areas <= 0).any():
        raise ValueError("Node areas must be finite and positive")
    return areas


def inspect_hdf5_mesh(
    hdf5_file: str | Path,
    *,
    mesh_key: str = "mesh_roi",
    mesh_loader: Callable | None = None,
) -> dict[str, object]:
    """Report mesh dimensions, element tags/types, and field locations."""
    mesh = load_hdf5_mesh(hdf5_file, mesh_key=mesh_key, mesh_loader=mesh_loader)
    element_fields = {field.field_name for field in mesh.elmdata}
    node_fields = {field.field_name for field in mesh.nodedata}
    fields = {}
    for name, field in mesh.field.items():
        location = "element" if name in element_fields else "node" if name in node_fields else "unknown"
        fields[name] = {"shape": tuple(np.asarray(field.value).shape), "type": type(field).__name__, "location": location}
    return {
        "hdf5_file": str(hdf5_file),
        "mesh_key": mesh_key,
        "n_nodes": int(mesh.nodes.nr),
        "n_elements": int(mesh.elm.nr),
        "element_types": np.unique(np.asarray(mesh.elm.elm_type)).astype(int).tolist(),
        "element_tags": np.unique(np.asarray(mesh.elm.tag1)).astype(int).tolist(),
        "fields": fields,
    }
