from pathlib import Path
import csv

import numpy as np

cap_path = Path("/projects/p32903/Alex2/datasets/STU00225089/synthsr/m2ms/skin_double/m2m_subj-cat-005-001/eeg_positions/EEG10-10_UI_Jurak_2007.csv")
wanted = {"FT7", "FT8", "Fpz"}
positions = {}

with cap_path.open(newline="") as f:
    for row in csv.reader(f):
        name = row[-1].strip() if row else ""
        if len(row) >= 5 and name in wanted:
            positions[name] = np.asarray(row[1:4], dtype=float)

missing = wanted - positions.keys()
if missing:
    raise ValueError(f"Missing EEG positions: {sorted(missing)}")

for centre in ("FT7", "FT8"):
    distance = np.linalg.norm(positions[centre] - positions["Fpz"])
    print(f"{centre} -> Fpz: {distance:.6f} mm")