"""Run CLI checks with: python -m unittest test_build_subject_scan_map.py"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "build_subject_scan_map.py"
HEADER = ["subject_dir", "full_file_path", "model_type_dir", "step", "dataset_root"]


def record(subject: str, *, model: str = "skin_double", step: str = "step_2", extension: str = ".hdf5") -> list[str]:
    return [subject, f"/results/{model}/{subject}/BT/{step}/tdcs_uq_gpc{extension}", model, step, "org"]


class TestBuildSubjectScanMapCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv = self.root / "results_query.csv"
        self.output = self.root / "subject_scan_map.json"

    def write_query(self, rows: list[list[str]], header: list[str] | None = None) -> None:
        with self.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER if header is None else header)
            writer.writerows(rows)

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), str(self.csv), "-o", str(self.output), *args],
                              text=True, capture_output=True, check=False, cwd=self.root)

    def test_lowest_numeric_scan_single_scans_omitted_and_duplicates_ignored(self) -> None:
        self.write_query([record("subj-cat-001-010"), record("subj-cat-001-002"), record("subj-cat-001-001"),
                          record("subj-cat-001-001", model="skin_single", step="step_0"),
                          record("subj-cat-002-001"), record("subj-cat-002-001"), record("subj-cat-003-009"),
                          record("subj-cat-003-002"), record("subj-cat-001-000", extension=".msh")])
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.output.read_text()), {"subj-cat-001": "subj-cat-001-001",
                                                               "subj-cat-003": "subj-cat-003-002"})
        self.assertIn("Patients represented: 3", result.stdout)

    def test_model_and_simulation_filters_apply_before_scan_selection(self) -> None:
        self.write_query([record("subj-cat-001-001", model="skin_single"),
                          record("subj-cat-001-001", step="step_0"), record("subj-cat-001-003"),
                          record("subj-cat-001-002")])
        result = self.run_cli("--model-type", "skin_double", "--simulation-type", "adaptive")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.output.read_text()), {"subj-cat-001": "subj-cat-001-002"})

    def test_static_and_step_alias_are_equivalent(self) -> None:
        self.write_query([record("subj-cat-001-001", step="step_0"), record("subj-cat-001-002", step="step_0"),
                          record("subj-cat-001-000")])
        for mode in ["static", "step_0"]:
            result = self.run_cli("--simulation-type", mode, "--overwrite")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(self.output.read_text()), {"subj-cat-001": "subj-cat-001-001"})

    def test_metadata_and_derived_filename_filter(self) -> None:
        other = record("subj-cat-001-000")
        other[-1] = "synth"
        self.write_query([record("subj-cat-001-001"), record("subj-cat-001-002"), other])
        result = self.run_cli("--filter", "dataset_root=org", "--filter", "filename=tdcs_uq_gpc.hdf5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.output.read_text()), {"subj-cat-001": "subj-cat-001-001"})

    def test_existing_output_preserved_until_overwrite_is_explicit(self) -> None:
        self.write_query([record("subj-cat-001-001")])
        self.output.write_text('{"existing":"value"}\n')
        result = self.run_cli()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(self.output.read_text()), {"existing": "value"})
        self.assertIn("--overwrite", result.stderr)
        result = self.run_cli("--overwrite")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.output.read_text()), {})

    def test_no_matches_fails_without_creating_map(self) -> None:
        self.write_query([record("subj-cat-001-001", step="step_0")])
        result = self.run_cli("--simulation-type", "adaptive")
        self.assertEqual(result.returncode, 1)
        self.assertIn("No HDF5 records", result.stderr)
        self.assertFalse(self.output.exists())

    def test_invalid_id_or_csv_structure_never_silently_drops_rows(self) -> None:
        for rows, header in [([record("subj-cat-001")], HEADER),
                             ([record("subj-cat-001-001")[:-1]], HEADER),
                             ([record("subj-cat-001-001")], ["subject_dir"] * len(HEADER)),
                             ([record("subj-cat-001-001"), []], HEADER)]:
            with self.subTest(rows=rows, header=header):
                self.write_query(rows, header)
                result = self.run_cli()
                self.assertEqual(result.returncode, 1)
                self.assertFalse(self.output.exists())

    def test_duplicate_or_unknown_filter_is_rejected(self) -> None:
        self.write_query([record("subj-cat-001-001")])
        for args in [("--filter", "dataset_root=org", "--filter", "dataset_root=synth"),
                     ("--filter", "not_a_column=value"), ("--filter", "step=step_2")]:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 1)
                self.assertFalse(self.output.exists())

    def test_query_input_cannot_be_overwritten(self) -> None:
        self.write_query([record("subj-cat-001-001")])
        before = self.csv.read_bytes()
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.csv), "-o", str(self.csv), "--overwrite"],
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.csv.read_bytes(), before)

    def test_bom_and_whitespace_are_supported(self) -> None:
        self.write_query([record(" subj-cat-001-001 "), record("subj-cat-001-002")],
                         [f" {column} " for column in HEADER])
        self.csv.write_bytes(b"\xef\xbb\xbf" + self.csv.read_bytes())
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.output.read_text()), {"subj-cat-001": "subj-cat-001-001"})


if __name__ == "__main__":
    unittest.main()
