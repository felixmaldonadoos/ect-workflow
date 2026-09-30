"""Discovery and loading of SimNIBS meshes stored in gPC HDF5 files."""

from __future__ import annotations


from pathlib import Path
from typing import Callable, Literal, Sequence
import warnings
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
    simulation_ids: Sequence[str] | None = None,
    require_files: bool = True,
) -> pd.DataFrame:
    """Build an HDF5 inventory; explicit simulation IDs allow multiple fields per scan."""
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

    key = "subjid"
    if simulation_ids is not None:
        if len(simulation_ids) != len(paths):
            raise ValueError("simulation_ids and hdf5_files must have equal lengths")
        if any(not isinstance(value, str) for value in simulation_ids):
            raise TypeError("simulation_ids must contain strings")
        values = pd.Series(list(simulation_ids), dtype="string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError("simulation_ids must be nonempty strings")
        inventory["simulation_id"] = values
        key = "simulation_id"
    duplicate_ids = inventory.loc[inventory[key].duplicated(keep=False), key].tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate {key} values: {duplicate_ids}")
    duplicate_paths = inventory.loc[inventory["hdf5_file"].duplicated(keep=False), "hdf5_file"].tolist()
    if duplicate_paths:
        raise ValueError(f"Duplicate HDF5 paths: {duplicate_paths}")
    columns = ["subjid", "subjid_base", "scan_id", "hdf5_file"]
    return inventory[columns + (["simulation_id"] if simulation_ids is not None else [])]


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


def scalar_mesh_field(
    mesh,
    field_name: str,
    *,
    expected_location: Literal["element", "node"] | None = None,
    negative_atol: float = 1e-8,
) -> tuple[np.ndarray, str]:
    """Extract a finite, nonnegative scalar field without modifying the mesh.

    Validate the entire field. Warn and zero values in [-negative_atol, 0)
    in a copy; negative_atol uses the same units as the field.
    """
    if expected_location not in (None, "element", "node"):
        raise ValueError("expected_location must be 'element', 'node', or None")

    if isinstance(negative_atol, (bool, np.bool_)) or not isinstance(
        negative_atol, (int, float, np.integer, np.floating)
    ):
        raise TypeError("negative_atol must be numeric")
    if not np.isfinite(negative_atol) or negative_atol < 0:
        raise ValueError("negative_atol must be finite and nonnegative")

    element_fields = [item for item in mesh.elmdata if item.field_name == field_name]
    node_fields = [item for item in mesh.nodedata if item.field_name == field_name]

    if not element_fields and not node_fields:
        available = sorted({item.field_name for item in [*mesh.elmdata, *mesh.nodedata]})
        raise KeyError(f"Field {field_name!r} not found; available fields: {available}")
    if element_fields and node_fields:
        raise ValueError(f"Field {field_name!r} exists as both ElementData and NodeData")

    matches = element_fields or node_fields
    if len(matches) != 1:
        raise ValueError(f"Multiple fields named {field_name!r} at the same location")

    location, expected = ("element", int(mesh.elm.nr)) if element_fields else ("node", int(mesh.nodes.nr))
    if expected_location is not None and location != expected_location:
        raise ValueError(f"Field {field_name!r} must be {expected_location} data, got {location} data")

    raw = mesh_array(matches[0])
    if np.iscomplexobj(raw):
        raise ValueError(f"Field {field_name!r} must contain real values")

    values = np.asarray(raw, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]

    if values.shape != (expected,):
        raise ValueError(f"Field {field_name!r} has shape {values.shape}; expected ({expected},)")
    if not np.isfinite(values).all():
        raise ValueError(f"Field {field_name!r} contains nonfinite values")

    invalid = values < -negative_atol
    if invalid.any():
        raise ValueError(
            f"Field {field_name!r} contains {invalid.sum()} values below "
            f"-{negative_atol:g}; minimum={values.min():.8g}"
        )

    negative = values < 0
    if negative.any():
        warnings.warn(
            f"Field {field_name!r}: setting {negative.sum()} negligible negatives "
            f"to zero; minimum={values.min():.8g}; tolerance={negative_atol:g}",
            RuntimeWarning,
            stacklevel=2,
        )
        values = values.copy()
        values[negative] = 0.0

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
