"""Explicit maintenance/invalid-series exclusions for acute-course selection."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_pca_weighted_global_E_job_synthsr as runner
from simnibs_parcel_analysis.clinical import _filter_acute_sessions, load_ect_sessions, modal_placements_by_course
from test_course_modal_selection import fixture, select


class TestAcuteSeriesFilter(unittest.TestCase):
    def test_only_finite_positive_integers_remain_without_changing_order(self) -> None:
        labels = [0, -1, 1.5, None, np.nan, np.inf, "unknown", "", True, "1", 2.0, "3.0"]
        frame = pd.DataFrame({"series_num": labels, "frequency_hz": ["age"] * len(labels)})
        before = frame.copy(deep=True)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = _filter_acute_sessions(frame)
        self.assertEqual(result.index.tolist(), [9, 10, 11])
        self.assertEqual(result.series_num.tolist(), ["1", 2.0, "3.0"])
        self.assertEqual(result.frequency_hz.tolist(), ["age"] * 3)
        self.assertIn("Excluding 9 stimulus rows", output.getvalue())
        pd.testing.assert_frame_equal(frame, before)

    def test_filter_runs_before_validation_of_excluded_sessions(self) -> None:
        sessions, _, _ = fixture(Path("/synthetic"))
        bad = sessions.iloc[[0]].assign(subjid=None, series_num=0, session_num=0, date="invalid", electrode_placement="bad")
        mixed = pd.concat([sessions, bad], ignore_index=True)
        with contextlib.redirect_stdout(io.StringIO()):
            actual = modal_placements_by_course(mixed)
        pd.testing.assert_frame_equal(actual, modal_placements_by_course(sessions))
        # Acute records still undergo strict session validation.
        bad = sessions.iloc[[0]].assign(session_num=0)
        with self.assertRaisesRegex(ValueError, "session_num must contain positive integers"):
            modal_placements_by_course(bad)

    def test_all_rows_excluded_is_an_explicit_error(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "No acute ECT sessions remain"):
            _filter_acute_sessions(pd.DataFrame({"series_num": [0, -1, 1.5, None]}))

    def test_workbook_acute_mode_filters_maintenance_but_default_remains_strict(self) -> None:
        sessions, _, _ = fixture(Path("/synthetic"))
        maintenance = sessions.iloc[[0]].assign(series_num=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.xlsx"
            pd.concat([sessions, maintenance]).to_excel(path, sheet_name="stimulus", index=False)
            with self.assertRaisesRegex(ValueError, "series_num must contain positive integers"):
                load_ect_sessions(path)
            with contextlib.redirect_stdout(io.StringIO()):
                loaded = load_ect_sessions(path, acute_only=True)
            self.assertEqual(len(loaded), len(sessions))
            self.assertTrue(loaded.series_num.ge(1).all())
            self.assertEqual(loaded.frequency_hz.unique().tolist(), ["age"])

    def test_direct_course_selection_excludes_maintenance_and_invalid_series(self) -> None:
        sessions, query, cognitive = fixture(Path("/synthetic"))
        expected_inventory, expected_mapping = select(sessions, query, cognitive)
        excluded = pd.concat([sessions.iloc[[0]].assign(series_num=value) for value in [0, -1, 1.5, "bad"]])
        with contextlib.redirect_stdout(io.StringIO()):
            inventory, mapping = select(pd.concat([sessions, excluded]), query, cognitive)
        pd.testing.assert_frame_equal(inventory, expected_inventory)
        pd.testing.assert_frame_equal(mapping, expected_mapping)

    def test_selection_only_runner_enables_acute_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions, query, cognitive = fixture(root)
            sessions = pd.concat([sessions, sessions.iloc[[0]].assign(series_num=0)], ignore_index=True)
            for path in query.full_file_path.map(Path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            workbook, csv = root / "cog.xlsx", root / "query.csv"
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                cognitive.to_excel(writer, sheet_name="scores", index=False)
                sessions.to_excel(writer, sheet_name="stimulus", index=False)
            query.to_csv(csv, index=False)
            args = ["runner", "--model-type", "skin_double", "--analysis-mode", "adaptive",
                    "--cognitive-scores", str(workbook), "--query-path", str(csv), "--selection-only"]
            output = io.StringIO()
            with patch("sys.argv", args), patch.object(runner, "OUTPUT_ROOT", root / "output"), \
                    patch.object(runner, "SUBJECTS_DIR_BY_DATASET", {"org": root, "synth": root}), \
                    patch.object(runner, "run_all_atlases_global_p95_weighted") as analyze, \
                    contextlib.redirect_stdout(output):
                runner.main()
            analyze.assert_not_called()
            self.assertIn("Excluding 1 stimulus rows", output.getvalue())
            exported = pd.read_csv(root / "output/skin_double/adaptive/selection_preview/course_hdf5_mapping.csv")
            self.assertEqual(len(exported), 6)
            self.assertTrue(exported.series_num.ge(1).all())


if __name__ == "__main__":
    unittest.main()
