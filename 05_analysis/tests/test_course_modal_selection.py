"""Course-specific placement selection and full weighted/demeaned pipeline regression."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

import example_usage_global_E_weighted as pipeline
import run_pca_weighted_global_E_job_synthsr as runner
from simnibs_parcel_analysis.atlas_surface import build_surface_roi_summary
from simnibs_parcel_analysis.atlas_volume import build_volume_roi_summary
from simnibs_parcel_analysis.clinical import (
    load_ect_sessions, modal_placements_by_course, prepare_scan_course_outcomes, select_modal_course_hdf5,
)
from simnibs_parcel_analysis.mesh_io import build_hdf5_inventory
from simnibs_parcel_analysis.pca import DEMEAN_REFERENCES, expand_predictors_to_courses


def fixture(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, courses, inventory = [], [], []
    configurations = {901: [(1, ["rul", "rul", "bt"]), (2, ["bt", "bt", "rul"]), (3, ["rul"])],
                      902: [(1, ["bl"])], 903: [(1, ["bf"])], 904: [(1, ["rul"])]}
    for patient, series_list in configurations.items():
        base = f"subj-cat-{patient:03d}"
        for series, placements in series_list:
            start = pd.Timestamp(2022, series, 1)
            for session, placement in enumerate(placements, start=1):
                rows.append({"subjid": base, "series_num": series, "session_num": session,
                             "date": start + pd.Timedelta(days=session - 1), "electrode_placement": placement,
                             "frequency_hz": "age", "percent_charge": 50})
            if patient != 904:
                courses.append({"subjid": base, "series_num": series, "date_start": start,
                                "date_end (acute)": start + pd.Timedelta(days=10),
                                "cgi_start": 7, "cgi_end": 7 - (patient - 900) - series, "mrn": "synthetic"})
        dataset = "org" if patient < 903 else "synth"
        scan = base + "-001"
        for placement in sorted({p.upper() for _, values in series_list for p in values} | {"BF"}):
            path = root / dataset / "skin_double" / scan / placement / "95" / "run1" / "step_2" / "tdcs_uq_gpc.hdf5"
            inventory.append({"dataset_root": dataset, "model_type_dir": "skin_double", "subject_dir": scan,
                              "electrode_configuration": placement, "step": "step_2", "full_file_path": str(path)})
    return pd.DataFrame(rows), pd.DataFrame(inventory), pd.DataFrame(courses)


def select(sessions: pd.DataFrame, inventory: pd.DataFrame, cognitive: pd.DataFrame, **kwargs):
    return select_modal_course_hdf5(sessions, inventory, cognitive, model_type="skin_double",
                                    simulation_type="adaptive", **kwargs)


class TestCourseSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions, self.inventory, self.cognitive = fixture(Path("/synthetic"))

    def test_modes_are_per_patient_and_series_and_ignore_missing_with_counts(self) -> None:
        df = self.sessions.iloc[:3].copy()
        for number, value in enumerate([None, "nan", " "], start=4):
            df = pd.concat([df, df.iloc[[0]].assign(session_num=number, electrode_placement=value)], ignore_index=True)
        with self.assertWarnsRegex(UserWarning, "Ignoring 3"):
            result = modal_placements_by_course(df).iloc[0]
        self.assertEqual(result.electrode_placement, "RUL")
        self.assertEqual((result.n_sessions, result.n_known_placements, result.n_modal_sessions), (6, 3, 2))
        self.assertAlmostEqual(result.modal_fraction, 2 / 3)
        self.assertEqual(json.loads(result.placement_counts), {"BT": 1, "RUL": 2})

    def test_ties_empty_modes_invalid_labels_and_duplicate_sessions_are_errors(self) -> None:
        for placements, message in [(["BT", "RUL"], "Tied modal"), ([None, "nan"], "No known"),
                                     (["RUL", "unknown"], "invalid electrode")]:
            with self.subTest(placements=placements), self.assertRaisesRegex(ValueError, message):
                modal_placements_by_course(self.sessions.iloc[:2].assign(electrode_placement=placements))
        with self.assertRaisesRegex(ValueError, "Duplicate patient/series/session"):
            modal_placements_by_course(pd.concat([self.sessions, self.sessions.iloc[[0]]]))

    def test_different_course_modes_and_reused_fields_select_unique_hdf5s(self) -> None:
        selected, mapping = select(self.sessions, self.inventory, self.cognitive)
        patient = mapping.loc[mapping.subjid_base.eq("subj-cat-901")]
        self.assertEqual(patient.electrode_placement.tolist(), ["RUL", "BT", "RUL"])
        self.assertEqual(patient.iloc[0].hdf5_fn, patient.iloc[2].hdf5_fn)
        self.assertNotEqual(patient.iloc[0].simulation_id, patient.iloc[1].simulation_id)
        self.assertEqual(len(selected), 5)
        self.assertEqual(len(mapping), 6)
        self.assertTrue(mapping.loc[mapping.subjid_base.eq("subj-cat-904"), "treatment_course_id"].isna().all())

    def test_acute_window_excludes_maintenance_and_dates_can_identify_series(self) -> None:
        maintenance = pd.concat([self.sessions.iloc[[0]].assign(session_num=n, date="2022-01-25", electrode_placement="BT")
                                 for n in range(4, 8)], ignore_index=True)
        sessions = pd.concat([self.sessions, maintenance], ignore_index=True)
        _, mapping = select(sessions, self.inventory, self.cognitive.drop(columns="series_num"))
        first = mapping.iloc[0]
        self.assertEqual(first.electrode_placement, "RUL")
        self.assertEqual(first.n_sessions, 3)
        self.assertEqual(first.mode_scope, "acute_date_window")

    def test_missing_or_ambiguous_series_and_reused_series_are_errors(self) -> None:
        bad = self.cognitive.drop(columns="series_num").copy()
        bad.loc[0, "date_end (acute)"] = pd.Timestamp("2022-02-15")
        with self.assertRaisesRegex(ValueError, "exactly one stimulus series"):
            select(self.sessions, self.inventory, bad)
        bad = self.cognitive.copy()
        bad.loc[0, "series_num"] = 99
        with self.assertRaisesRegex(ValueError, "conflicts with stimulus dates"):
            select(self.sessions, self.inventory, bad)
        extra = self.cognitive.iloc[[0]].assign(date_start=pd.Timestamp("2022-01-02"))
        with self.assertRaisesRegex(ValueError, "same stimulus patient/series"):
            select(self.sessions, self.inventory, pd.concat([self.cognitive, extra]))

    def test_missing_modal_simulation_never_uses_nonmodal_field(self) -> None:
        query = self.inventory.loc[~(self.inventory.subject_dir.eq("subj-cat-901-001")
                                    & self.inventory.electrode_configuration.eq("RUL"))]
        with self.assertRaisesRegex(ValueError, "No matching HDF5"):
            select(self.sessions, query, self.cognitive)

    def test_duplicate_runs_checked_after_nonmodal_selection(self) -> None:
        unused = self.inventory.loc[self.inventory.subject_dir.eq("subj-cat-902-001")
                                     & self.inventory.electrode_configuration.eq("BF")].copy()
        unused["full_file_path"] = unused.full_file_path.str.replace("/run1/", "/run2/")
        self.assertEqual(len(select(self.sessions, pd.concat([self.inventory, unused]), self.cognitive)[0]), 5)
        modal = self.inventory.loc[self.inventory.electrode_configuration.eq("RUL")].copy()
        modal["full_file_path"] = modal.full_file_path.str.replace("/run1/", "/run2/")
        with self.assertRaisesRegex(ValueError, "Multiple HDF5 files"):
            select(self.sessions, pd.concat([self.inventory, modal]), self.cognitive)

    def test_multiple_scans_require_map_and_alias_is_explicit(self) -> None:
        second = self.inventory.loc[self.inventory.subject_dir.eq("subj-cat-901-001")].copy()
        second["subject_dir"] = "subj-cat-901-002"
        second["full_file_path"] = second.full_file_path.str.replace("901-001", "901-002")
        query = pd.concat([self.inventory, second])
        with self.assertRaisesRegex(ValueError, "require subject_scan_map"):
            select(self.sessions, query, self.cognitive)
        selected, mapping = select(self.sessions, query, self.cognitive, subject_scan_map={"subj-cat-901": "subj-cat-901-002"})
        self.assertEqual(set(mapping.loc[mapping.subjid_base.eq("subj-cat-901"), "modeled_subjid"]), {"subj-cat-901-002"})
        query = self.inventory.copy()
        query["electrode_configuration"] = query.electrode_configuration.replace({"BT": "BL"})
        query["full_file_path"] = query.full_file_path.str.replace("/BT/", "/BL/")
        with self.assertRaisesRegex(ValueError, "No matching HDF5"):
            select(self.sessions, query, self.cognitive)
        _, mapping = select(self.sessions, query, self.cognitive, electrode_placement_map={"BT": "BL"})
        bt = mapping.loc[mapping.electrode_placement.eq("BT")].iloc[0]
        self.assertEqual(bt.simulated_placement, "BL")

    def test_explicit_outcome_join_keeps_course_specific_fields_and_actual_scan_counts(self) -> None:
        selected, mapping = select(self.sessions, self.inventory, self.cognitive)
        scans = build_hdf5_inventory(selected.full_file_path, simulation_ids=selected.simulation_id, require_files=False)
        cognitive = self.cognitive.assign(cgi_change=self.cognitive.cgi_end - self.cognitive.cgi_start)
        with self.assertWarnsRegex(UserWarning, "retained in ROI/PCA"):
            outcomes = prepare_scan_course_outcomes(scans, cognitive, course_hdf5_map=mapping)
        predictors = pd.DataFrame({"field": [100, 200, 300, 400, 500]},
                                  index=pd.Index(selected.simulation_id, name="simulation_id"))
        aligned = expand_predictors_to_courses(predictors, outcomes)
        patient = aligned.loc[aligned.subjid_base.eq("subj-cat-901")].sort_values("date_start")
        self.assertEqual(patient.field.iloc[0], patient.field.iloc[2])
        self.assertNotEqual(patient.field.iloc[0], patient.field.iloc[1])
        self.assertEqual(len(aligned), 5)
        self.assertEqual(aligned.subjid.nunique(), 3)
        with self.assertRaisesRegex(ValueError, "require explicit course_hdf5_map"):
            prepare_scan_course_outcomes(scans, cognitive)
        invalid = mapping.copy()
        invalid.loc[0, "hdf5_fn"] = "/wrong/file.hdf5"
        with self.assertRaisesRegex(ValueError, "conflicts with selected"):
            prepare_scan_course_outcomes(scans, cognitive, course_hdf5_map=invalid)

    def test_workbook_selection_only_runner_uses_stimulus_and_exports_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions, query, cognitive = fixture(root)
            for path in query.full_file_path.map(Path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            workbook, csv = root / "cog.xlsx", root / "query.csv"
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                cognitive.to_excel(writer, sheet_name="scores", index=False)
                sessions.to_excel(writer, sheet_name="stimulus", index=False)
            query.to_csv(csv, index=False)
            loaded = load_ect_sessions(workbook)
            self.assertEqual(loaded.frequency_hz.unique().tolist(), ["age"])
            args = ["runner", "--model-type", "skin_double", "--analysis-mode", "adaptive",
                    "--cognitive-scores", str(workbook), "--query-path", str(csv), "--selection-only"]
            with patch("sys.argv", args), patch.object(runner, "OUTPUT_ROOT", root / "output"), \
                    patch.object(runner, "SUBJECTS_DIR_BY_DATASET", {"org": root, "synth": root}), \
                    patch.object(runner, "run_all_atlases_global_p95_weighted") as analyze, \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            analyze.assert_not_called()
            destination = root / "output" / "skin_double" / "adaptive" / "selection_preview"
            self.assertEqual(len(pd.read_csv(destination / "selected_hdf5_inventory.csv")), 5)
            self.assertEqual(len(pd.read_csv(destination / "course_hdf5_mapping.csv")), 6)


class TestModalPipelineIntegration(unittest.TestCase):
    def test_all_atlases_and_preprocessing_modes_with_two_placements_for_one_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions, query, cognitive = fixture(root)
            selected, mapping = select(sessions, query, cognitive)
            paths = [Path(value) for value in selected.full_file_path]
            values = [[1, 2, 3, 5, 6, 8, 9, 10], [6, 9, 3, 4, 4, 10, 2, 3], [2, 5, 7, 8, 4, 6, 9, 15],
                      [10, 14, 6, 7, 3, 9, 8, 10], [5, 8, 6, 12, 15, 20, 4, 9]]
            by_path = dict(zip(map(str, paths), values, strict=True))
            sources = {path: root / ("fs_org" if "/org/" in str(path) else "fs_synth") for path in paths}
            for path, fs in sources.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                overlay = path.parent / "fsavg_overlays" / "tdcs_uq_gpc_fsavg.msh"
                overlay.parent.mkdir()
                overlay.touch()
                scan = pipeline.infer_subject_id(path)
                (fs / scan / "mri").mkdir(parents=True, exist_ok=True)
                for name in ["aparc+aseg.mgz", "aparc.a2009s+aseg.mgz"]:
                    (fs / scan / "mri" / name).touch()

            def mesh(path: str, *args, surface: bool = False):
                if surface:
                    path = str(Path(path).parent.parent / "tdcs_uq_gpc.hdf5")
                field = SimpleNamespace(value=np.array(by_path[str(path)], dtype=float), field_name="magnE_mean")
                return SimpleNamespace(field={"magnE_mean": field}, elmdata=[] if surface else [field],
                                       nodedata=[field] if surface else [], nodes=SimpleNamespace(nr=8),
                                       elm=SimpleNamespace(nr=8, elm_type=np.full(8, 4), tag1=np.full(8, 2)),
                                       elements_baricenters=lambda: np.column_stack([np.arange(8), np.zeros((8, 2))]),
                                       elements_volumes_and_areas=lambda: np.arange(1, 9), nodes_areas=lambda: np.arange(1, 9))

            lut = {10: "Left-Thalamus", 11: "Left-Caudate", 12: "Left-Putamen", 13: "Left-Pallidum"}
            def volume(files, fs, **kw):
                kw.update(mesh_loader=mesh, atlas_loader=lambda path: (np.repeat(list(lut), 2).reshape(8, 1, 1), np.eye(4)),
                          lut=lut, min_elements_per_roi=1)
                return build_volume_roi_summary(files, fs, **kw)

            def surface(files, **kw):
                kw.update(mesh_loader=lambda path: mesh(path, surface=True), min_nodes_per_roi=1,
                          atlas_loader=lambda name: {f"ROI{i}": np.repeat(np.arange(4) == i, 2) for i in range(4)})
                return build_surface_roi_summary(files, **kw)

            fake_simnibs = SimpleNamespace(Msh=SimpleNamespace(read_hdf5=mesh))
            with patch.dict("sys.modules", {"simnibs": fake_simnibs}), \
                    patch.object(pipeline, "inspect_hdf5_mesh", return_value={}), \
                    patch.object(pipeline, "build_volume_roi_summary", side_effect=volume), \
                    patch.object(pipeline, "build_surface_roi_summary", side_effect=surface), \
                    patch("matplotlib.figure.Figure.savefig"), contextlib.redirect_stdout(io.StringIO()), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                results = pipeline.run_all_atlases_global_p95_weighted(
                    paths, cognitive, root / "output", subjects_dir_by_hdf5=sources,
                    simulation_ids=selected.simulation_id.tolist(), course_hdf5_map=mapping,
                    demean_references=DEMEAN_REFERENCES)
            self.assertEqual(len(results), 24)
            for key, result in results.items():
                with self.subTest(branch=key):
                    self.assertEqual(len(result.summary.scans), 5)
                    self.assertEqual(result.pca.scores.index.name, "simulation_id")
                    self.assertEqual(len(result.outcomes), 5)
                    self.assertEqual(result.outcomes.subjid.nunique(), 3)
                    self.assertEqual(result.outcomes.simulation_id.nunique(), 4)
                    self.assertEqual(result.correlations.n_scans.unique().tolist(), [3])
                    self.assertEqual(result.correlations.n_simulations.unique().tolist(), [4])
                    self.assertEqual(result.correlations.n_people.unique().tolist(), [3])
                    self.assertEqual(result.correlations.n_observations.unique().tolist(), [5])
                    aligned = expand_predictors_to_courses(result.predictors, result.outcomes)
                    patient = aligned.loc[aligned.subjid_base.eq("subj-cat-901")].sort_values("date_start")
                    np.testing.assert_allclose(patient.iloc[0][result.predictors.columns].astype(float),
                                               patient.iloc[2][result.predictors.columns].astype(float))
            settings = json.loads((root / "output/volume_aparc/weighted_p95/analysis_settings.json").read_text())
            self.assertEqual(settings["pca_row_identity"], "simulation_id")
            self.assertIn("modal-placement", settings["predictor_matching"])


if __name__ == "__main__":
    unittest.main()
