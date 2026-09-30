import numpy as np
from simnibs import mesh_io


fn = '/projects/b32903/Alex/results/STU00225089/skin_single/subj-cat-051-001/BL/95/20260929_102127/step_0/tdcs_uq_gpc.hdf5'
m = mesh_io.Msh.read_hdf5(fn, "mesh_roi")

fields = [f for f in m.elmdata if f.field_name == "magnE_mean"]
if len(fields) != 1:
    raise ValueError(f"Expected one element magnE_mean field; found {len(fields)}")

values = np.asarray(fields[0].value).reshape(-1)
if values.size != m.elm.nr:
    raise ValueError("Field size does not match the number of mesh elements.")
if not np.isfinite(values).all():
    raise ValueError("magnE_mean contains NaN or infinite values.")

negative = values < 0
brain = (m.elm.elm_type == 4) & np.isin(m.elm.tag1, [1, 2])

print(f"All elements: min={values.min():.8g}, negative={negative.sum()}")
if brain.any():
    print(f"Brain tetrahedra: min={values[brain].min():.8g}, negative={(negative & brain).sum()}")

print("element_type tissue_tag n_negative minimum")
pairs = np.column_stack((m.elm.elm_type[negative], m.elm.tag1[negative]))
for element_type, tag in np.unique(pairs, axis=0):
    mask = negative & (m.elm.elm_type == element_type) & (m.elm.tag1 == tag)
    print(f"{element_type:12d} {tag:10d} {mask.sum():10d} {values[mask].min():.8g}")