"""Session/HDF5 matching checks; run from 05_analysis with unittest discovery."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from simnibs_parcel_analysis.clinical import (
    compute_cgi_change,
    compute_ef_scaling_factor,
    load_ect_sessions,
    load_ect_sessions_with_hdf5,
    load_hdf5_query,
    load_subject_scan_map,
    map_sessions_to_hdf5,
    validate_treatment_courses,
)


def sessions() -> pd.DataFrame:
    return pd.DataFrame({
        "subjid": ["subj-cat-901"] * 3,
        "series_num": [1, 1, 2],
        "session_num": [1, 2, 13],
        "date": ["3/4/22", "3/7/22", "11/8/22"],
        "electrode_placement": ["bt", "rul", "bt"],
        "frequency_hz": ["age"] * 3,
        "pulse_width_ms": [0.5] * 3,
        "percent_charge": [50] * 3,
        "comments": ["NA", None, "other"],
    }, index=[7, 7, 3])


def query(scan: str = "subj-cat-901-001", run: str = "20260929_102536",
          dataset: str = "STU00225089") -> pd.DataFrame:
    rows = []
    for model in ["skin_single", "skin_double"]:
        for step in ["step_0", "step_2"]:
            for placement in ["BT", "RUL"]:
                folder = f"/results/{dataset}/{model}/{scan}/{placement}/95/{run}/{step}"
                rows.append({"dataset_root": dataset, "model_type_dir": model, "subject_dir": scan,
                             "electrode_configuration": placement, "threshold": 95, "sim_result": run,
                             "step": step, "full_file_path": folder + "/tdcs_uq_gpc.hdf5"})
    return pd.DataFrame(rows)


def match(df: pd.DataFrame | None = None, catalog: pd.DataFrame | str | Path | None = None,
          **kwargs) -> pd.DataFrame:
    return map_sessions_to_hdf5(sessions() if df is None else df, query() if catalog is None else catalog,
                               model_type="skin_double", simulation_type="adaptive", **kwargs)


class TestSessionMapping(unittest.TestCase):
    def test_each_session_uses_its_placement_and_preserves_order_and_index(self) -> None:
        df, catalog = sessions(), query()
        original_df, original_catalog = df.copy(deep=True), catalog.copy(deep=True)
        result = match(df, catalog)
        self.assertEqual(result.index.tolist(), [7, 7, 3])
        self.assertEqual(result["session_num"].tolist(), [1, 2, 13])
        self.assertEqual(result["electrode_placement"].tolist(), ["BT", "RUL", "BT"])
        self.assertEqual(result["hdf5_fn"].str.contains("/BT/").tolist(), [True, False, True])
        self.assertEqual(result.iloc[0]["hdf5_fn"], result.iloc[2]["hdf5_fn"])
        self.assertEqual(result["frequency_hz"].tolist(), ["age"] * 3)
        self.assertEqual(result["comments"].tolist(), df["comments"].tolist())
        pd.testing.assert_frame_equal(df, original_df)
        pd.testing.assert_frame_equal(catalog, original_catalog)

    def test_model_and_static_adaptive_selectors(self) -> None:
        for model in ["skin_single", "skin_double"]:
            for mode, step in [("static", "step_0"), ("adaptive", "step_2"), ("step_0", "step_0"),
                               ("step_2", "step_2")]:
                with self.subTest(model=model, mode=mode):
                    result = map_sessions_to_hdf5(sessions(), query(), model_type=model, simulation_type=mode)
                    self.assertTrue(result["hdf5_fn"].str.contains(f"/{model}/").all())
                    self.assertTrue(result["hdf5_fn"].str.contains(f"/{step}/").all())

    def test_base_and_scan_labeled_clinical_ids_match_by_person(self) -> None:
        df = sessions()
        df["subjid"] = "subj-cat-901-099"
        result = match(df)
        self.assertEqual(result["subjid"].tolist(), ["subj-cat-901-099"] * 3)
        self.assertEqual(result["modeled_subjid"].tolist(), ["subj-cat-901-001"] * 3)

    def test_multiple_scans_require_explicit_preference(self) -> None:
        catalog = pd.concat([query(), query("subj-cat-901-002")], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "require subject_scan_map"):
            match(catalog=catalog)
        result = match(catalog=catalog, subject_scan_map={"subj-cat-901": "subj-cat-901-002"})
        self.assertEqual(result["modeled_subjid"].tolist(), ["subj-cat-901-002"] * 3)

    def test_multiple_scans_with_disjoint_placements_still_require_preference(self) -> None:
        first, second = query(), query("subj-cat-901-002")
        catalog = pd.concat([first[first["electrode_configuration"].eq("BT")],
                             second[second["electrode_configuration"].eq("RUL")]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "require subject_scan_map"):
            match(catalog=catalog)

    def test_unavailable_preferred_scan_never_falls_back(self) -> None:
        with self.assertRaisesRegex(ValueError, "Preferred scan.*unavailable"):
            match(subject_scan_map={"subj-cat-901": "subj-cat-901-009"})

    def test_preferred_scan_missing_one_placement_never_borrows_other_scan(self) -> None:
        first, second = query(), query("subj-cat-901-002")
        second = second[second["electrode_configuration"].eq("BT")]
        catalog = pd.concat([first, second], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "No matching HDF5"):
            match(catalog=catalog, subject_scan_map={"subj-cat-901": "subj-cat-901-002"})

    def test_unmatched_patient_or_placement_is_an_error(self) -> None:
        for column, value in [("subjid", "subj-cat-902"), ("electrode_placement", "bf")]:
            df = sessions()
            df[column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "No matching HDF5"):
                match(df)

    def test_multiple_runs_require_explicit_filter(self) -> None:
        catalog = pd.concat([query(), query(run="20260930_102536")], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Multiple HDF5 files"):
            match(catalog=catalog)
        result = match(catalog=catalog, query_filters={"sim_result": "20260929_102536"})
        self.assertTrue(result["hdf5_fn"].str.contains("20260929_102536").all())

    def test_multiple_datasets_require_explicit_filter(self) -> None:
        catalog = pd.concat([query(), query(dataset="STU00225089_synthsr")], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Multiple HDF5 files"):
            match(catalog=catalog)
        result = match(catalog=catalog, query_filters={"dataset_root": "STU00225089_synthsr", "threshold": 95})
        self.assertTrue(result["hdf5_fn"].str.contains("STU00225089_synthsr").all())

    def test_multiple_hdf5_basenames_can_be_filtered_without_filename_metadata(self) -> None:
        second = query()
        second["full_file_path"] = second["full_file_path"].str.replace("tdcs_uq_gpc", "auxiliary", regex=False)
        catalog = pd.concat([query(), second], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Multiple HDF5 files"):
            match(catalog=catalog)
        self.assertEqual(len(match(catalog=catalog, query_filters={"filename": "tdcs_uq_gpc.hdf5"})), 3)

    def test_repeated_identical_query_records_and_non_hdf5_rows(self) -> None:
        mesh = query()
        mesh["full_file_path"] = mesh["full_file_path"].str.replace(".hdf5", ".msh", regex=False)
        catalog = pd.concat([query(), query(), mesh], ignore_index=True)
        self.assertEqual(len(match(catalog=catalog)), 3)

    def test_bt_and_bl_are_distinct_unless_explicitly_aliased(self) -> None:
        catalog = query()
        catalog["electrode_configuration"] = catalog["electrode_configuration"].replace({"BT": "BL"})
        catalog["full_file_path"] = catalog["full_file_path"].str.replace("/BT/", "/BL/", regex=False)
        with self.assertRaisesRegex(ValueError, "No matching HDF5"):
            match(catalog=catalog)
        result = match(catalog=catalog, electrode_placement_map={"bt": "bl"})
        self.assertEqual(result["electrode_placement"].tolist(), ["BT", "RUL", "BT"])
        self.assertEqual(result["hdf5_fn"].str.contains("/BL/").tolist(), [True, False, True])

    def test_query_csv_and_duplicate_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "results_query.csv")
            query().to_csv(path, index=False)
            self.assertEqual(len(match(catalog=path)), 3)
            path.write_text("model_type_dir,model_type_dir,step\nskin_single,skin_double,step_2\n")
            with self.assertRaisesRegex(ValueError, "duplicate column"):
                match(catalog=path)

    def test_blank_or_invalid_required_values_and_duplicate_sessions(self) -> None:
        invalids = [("series_num", 1.5), ("session_num", 0), ("session_num", True), ("session_num", np.inf),
                    ("session_num", None), ("date", None), ("date", "NaT"), ("date", 45000),
                    ("electrode_placement", "nan"), ("electrode_placement", ""), ("subjid", None)]
        for column, value in invalids:
            df = sessions()
            df[column] = value
            with self.subTest(column=column, value=value), self.assertRaises((ValueError, TypeError)):
                match(df)
        df = pd.concat([sessions(), sessions().iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "Duplicate patient/series/session"):
            match(df)

    def test_reject_overwriting_existing_outputs_and_bad_selectors(self) -> None:
        df = sessions().assign(hdf5_fn="old")
        with self.assertRaisesRegex(ValueError, "already contains output"):
            match(df)
        for model, mode in [("unknown", "adaptive"), ("skin_double", "step_1")]:
            with self.subTest(model=model, mode=mode), self.assertRaises(ValueError):
                load_hdf5_query(query(), model_type=model, simulation_type=mode)
        with self.assertRaisesRegex(ValueError, "not query_filters"):
            match(query_filters={"step": "step_0"})

    def test_one_path_with_conflicting_placement_metadata_is_rejected(self) -> None:
        catalog = query()
        catalog["full_file_path"] = catalog["full_file_path"].str.replace("/RUL/", "/BT/", regex=False)
        with self.assertRaisesRegex(ValueError, "conflicting subject/placement metadata"):
            match(catalog=catalog)


class TestSubjectScanMap(unittest.TestCase):
    def test_valid_dictionary_and_reusable_global_preferences(self) -> None:
        preferred = {" subj-cat-901 ": "subj-cat-901-001", "subj-cat-902": "subj-cat-902-002"}
        normalized = load_subject_scan_map(preferred)
        self.assertEqual(normalized["subj-cat-901"], "subj-cat-901-001")
        self.assertEqual(len(match(subject_scan_map=normalized)), 3)

    def test_json_map_and_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "subject_scan_map.json")
            path.write_text('{"subj-cat-901": "subj-cat-901-001"}')
            self.assertEqual(len(match(subject_scan_map=path)), 3)
            path.write_text('{"subj-cat-901": "subj-cat-901-001", "subj-cat-901": "subj-cat-901-002"}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                load_subject_scan_map(path)

    def test_cross_patient_choices_and_invalid_keys_are_rejected(self) -> None:
        for preferred in [{"subj-cat-901": "subj-cat-902-001"}, {"subj-cat-901-001": "subj-cat-901-002"},
                          {"subj-cat-901": "subj-cat-901"}, {"subj-cat-901": 1},
                          {"subj-cat-901": "subj-cat-901-001", " subj-cat-901 ": "subj-cat-901-002"}]:
            with self.subTest(preferred=preferred), self.assertRaises((ValueError, TypeError)):
                load_subject_scan_map(preferred)


class TestExistingClinicalFunctions(unittest.TestCase):
    def test_existing_cgi_and_scaling_behavior(self) -> None:
        cgi = compute_cgi_change(pd.DataFrame({"cgi_start": [6], "cgi_end": [2]}))
        self.assertEqual(cgi.loc[0, "cgi_change"], -4)
        stim = compute_ef_scaling_factor(pd.DataFrame({"pulse_width_ms": [0.5], "frequency_hz": [50],
                                                       "percent_charge": [50]}))
        self.assertEqual(stim.loc[0, "ef_sf"], 1250)
        with self.assertRaises(ValueError):
            compute_ef_scaling_factor(sessions())

    def test_same_person_and_course_start_is_still_a_conflict(self) -> None:
        courses = pd.DataFrame({"subjid": ["subj-cat-901-001", "subj-cat-901-002"],
                                "date_start": ["2022-03-04"] * 2, "cgi_change": [-2, -3]})
        with self.assertRaisesRegex(ValueError, "same subject and treatment-course start date"):
            validate_treatment_courses(courses)


@unittest.skipUnless(os.environ.get("ECT_SESSION_TEST_XLSX"), "Set ECT_SESSION_TEST_XLSX for XLSX fixture checks")
class TestWorkbookInput(unittest.TestCase):
    def test_xlsx_loader_and_combined_entry_point(self) -> None:
        fn = Path(os.environ["ECT_SESSION_TEST_XLSX"])
        loaded = load_ect_sessions(fn, "stimulus")
        self.assertEqual(loaded["frequency_hz"].tolist(), ["age"] * 3)
        self.assertEqual(loaded.loc[0, "comments"], "NA")
        self.assertEqual(loaded["session_num"].tolist(), [1, 2, 13])
        self.assertEqual(loaded.loc[0, "date"], pd.Timestamp("2022-03-04"))
        mapped = load_ect_sessions_with_hdf5(fn, "stimulus", query(), model_type="skin_single",
                                            simulation_type="static")
        self.assertEqual(len(mapped), 3)
        self.assertTrue(mapped["hdf5_fn"].str.contains("/skin_single/").all())
        self.assertTrue(mapped["hdf5_fn"].str.contains("/step_0/").all())
        native = load_ect_sessions(fn, "native_dates")
        self.assertEqual(native["date"].tolist(), loaded["date"].tolist())

    def test_duplicate_headers_missing_ids_and_blank_rows_are_not_silently_dropped(self) -> None:
        fn = Path(os.environ["ECT_SESSION_TEST_XLSX"])
        for sheet, message in [("duplicate_header", "duplicate column"), ("missing_subject", "Missing subject"),
                               ("blank_session", "Missing subject")]:
            with self.subTest(sheet=sheet), self.assertRaisesRegex(ValueError, message):
                load_ect_sessions(fn, sheet)


if __name__ == "__main__":
    unittest.main()
