"""Scientific data-unit, session/ROI alignment, and MixedLM integration checks."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_lmm
from simnibs_parcel_analysis.clinical import map_sessions_to_hdf5
from simnibs_parcel_analysis.lmm import (
    aggregate_course_exposures, attach_course_covariates, build_fixed_effects, build_session_roi_matrices,
    fit_course_lmm, link_sessions_to_courses, prepare_session_dose, read_lmm_table, roi_table_from_summary,
)


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a, b = "subj-cat-901", "subj-cat-902"
    sessions = pd.DataFrame({"subjid": [a, a, a, b], "series_num": [1, 1, 2, 1],
                             "session_num": [1, 2, 13, 1],
                             "date": ["2022-01-01", "2022-01-03", "2023-01-01", "2022-02-01"],
                             "electrode_placement": ["rul", "bt", "bt", "rul"], "pulse_width_ms": [0.5] * 4,
                             "frequency_hz": [40, 50, 40, 40], "percent_charge": [20, 30, 50, 20]})
    courses = pd.DataFrame({"subjid": [a, a, b], "series_num": [1, 2, 1], "cgi_start": [7, 6, 5],
                            "cgi_end": [3, 4, 3], "date_start": ["2022-01-01", "2023-01-01", "2022-02-01"],
                            "date_end (acute)": ["2022-01-03", "2023-01-01", "2022-02-02"]})
    query_rows, roi_rows = [], []
    for subject, placement, field in [(a, "RUL", 10), (a, "BT", 100), (b, "RUL", 30)]:
        scan = subject + "-001"
        for step in ["step_0", "step_2"]:
            path = f"/synthetic/org/skin_double/{scan}/{placement}/{step}/tdcs.hdf5"
            query_rows.append({"subject_dir": scan, "electrode_configuration": placement,
                               "model_type_dir": "skin_double", "step": step,
                               "full_file_path": path, "dataset_root": "org"})
            roi_rows.append({"hdf5_fn": path, "ROI_a": field, "ROI_b": 2 * field})
    return sessions, courses, pd.DataFrame(query_rows), pd.DataFrame(roi_rows)


def prepared():
    sessions, outcomes, query, rois = fixture()
    linked = link_sessions_to_courses(sessions, outcomes)
    mapped = map_sessions_to_hdf5(linked.sessions, query, model_type="skin_double", simulation_type="adaptive")
    return prepare_session_dose(mapped), linked.courses, rois


def synthetic_courses() -> pd.DataFrame:
    rng = np.random.default_rng(348)
    rows = []
    for patient in range(80):
        intercept = rng.normal(0, 0.7)
        for series in range(1 if patient < 15 else 3):
            baseline, dose, number = rng.uniform(4, 7), rng.uniform(0.5, 3), int(rng.integers(4, 15))
            outcome = -1 + 0.12 * baseline - 0.4 * dose + 0.03 * number + intercept + rng.normal(0, 0.2)
            rows.append({"subjid_base": f"subj-cat-{patient:03d}", "series_num": series + 1,
                         "treatment_course_id": f"subj-cat-{patient:03d}_{series}", "cgi_start": baseline,
                         "dose": dose, "n_sessions": number, "cgi_change": outcome})
    return pd.DataFrame(rows)


class TestSessionCourseExposure(unittest.TestCase):
    def test_switching_placement_reused_field_and_exact_row_scaling(self) -> None:
        sessions, courses, rois = prepared()
        fields = build_session_roi_matrices(sessions, rois.iloc[::-1], atlas="DK")
        np.testing.assert_array_equal(fields.raw["ROI_a"], [10, 100, 100, 30])
        np.testing.assert_array_equal(sessions["ef_sf"], [400, 750, 1000, 400])
        np.testing.assert_array_equal(fields.scaled["ROI_a"], [4000, 75000, 100000, 12000])
        self.assertEqual(fields.raw.shape, (4, 2))
        self.assertEqual(courses["cgi_change"].tolist(), [-4, -2, -2])
        self.assertNotIn("cgi_change", sessions)
        self.assertTrue(sessions["hdf5_fn"].str.contains("/step_2/").all())
        self.assertEqual(sessions["session_num"].tolist(), [1, 2, 13, 1])

    def test_aggregate_session_products_not_product_of_course_sums(self) -> None:
        sessions, courses, rois = prepared()
        fields = build_session_roi_matrices(sessions, rois, atlas="DK")
        table, matrices = aggregate_course_exposures(sessions, courses, aggregation="sum", fields=fields)
        self.assertEqual(len(table), 3)
        self.assertEqual(table["n_sessions"].tolist(), [2, 1, 1])
        self.assertEqual(table["dose"].tolist(), [1150, 1000, 400])
        self.assertEqual(matrices["ef_raw"].iloc[0, 0], 110)
        self.assertEqual(matrices["ef_dose"].iloc[0, 0], 79000)
        self.assertNotEqual(matrices["ef_dose"].iloc[0, 0], 110 * 1150)
        mean, mean_matrices = aggregate_course_exposures(sessions, courses, aggregation="mean", fields=fields)
        self.assertEqual(mean["dose"].iloc[0], 575)
        self.assertEqual(mean_matrices["ef_dose"].iloc[0, 0], 39500)
        _, percent = aggregate_course_exposures(sessions, courses, aggregation="sum", fields=fields,
                                                dose_column="percent_charge")
        self.assertEqual(percent["ef_dose"].iloc[0, 0], 10 * 20 + 100 * 30)
        self.assertEqual(percent["ef_scaled"].iloc[0, 0], 79000)

    def test_dates_can_infer_series_and_unmatched_sessions_require_explicit_exclusion(self) -> None:
        sessions, courses, _, _ = fixture()
        inferred = link_sessions_to_courses(sessions, courses.drop(columns="series_num"))
        self.assertEqual(inferred.courses["series_num"].tolist(), [1, 2, 1])
        extra = sessions.iloc[[0]].assign(session_num=99, date="2022-03-01")
        with self.assertRaisesRegex(ValueError, "no outcome window"):
            link_sessions_to_courses(pd.concat([sessions, extra]), courses)
        result = link_sessions_to_courses(pd.concat([sessions, extra]), courses, allow_unmatched_sessions=True)
        self.assertEqual(len(result.sessions), 4)
        self.assertEqual(result.excluded_sessions["session_num"].tolist(), [99])

    def test_duplicate_missing_ambiguous_and_conflicting_outcomes_fail(self) -> None:
        sessions, courses, _, _ = fixture()
        bad = courses.copy()
        bad.loc[0, "cgi_end"] = np.nan
        with self.assertRaises(ValueError):
            link_sessions_to_courses(sessions, bad)
        with self.assertRaisesRegex(ValueError, "same subject and treatment-course"):
            link_sessions_to_courses(sessions, pd.concat([courses, courses.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "Duplicate patient/series/session"):
            link_sessions_to_courses(pd.concat([sessions, sessions.iloc[[0]]]), courses)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            link_sessions_to_courses(sessions, courses.assign(series_num=8))
        overlap = courses.copy()
        overlap.loc[0, "date_end (acute)"] = "2023-01-01"
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            link_sessions_to_courses(sessions, overlap)
        with self.assertRaisesRegex(ValueError, "cgi_change"):
            link_sessions_to_courses(sessions, courses.assign(cgi_change=4))

    def test_missing_or_invalid_stimulation_values_fail_instead_of_being_imputed(self) -> None:
        sessions, _, _, _ = fixture()
        for bad in ["age", np.nan, np.inf, -10, 0, True]:
            with self.subTest(value=bad), self.assertRaises((ValueError, TypeError)):
                prepare_session_dose(sessions.assign(frequency_hz=bad))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            prepare_session_dose(sessions.assign(ef_sf=1))

    def test_roi_contract_is_strict_but_explicit_parcel_exclusions_work(self) -> None:
        sessions, _, rois = prepared()
        with self.assertRaisesRegex(ValueError, "duplicate HDF5"):
            build_session_roi_matrices(sessions, pd.concat([rois, rois.iloc[[0]]]), atlas="DK")
        with self.assertRaisesRegex(ValueError, "No atlas fields"):
            build_session_roi_matrices(sessions, rois.iloc[:1], atlas="DK")
        bad = rois.assign(ROI_b=np.nan)
        with self.assertRaises(ValueError):
            build_session_roi_matrices(sessions, bad, atlas="DK")
        fields = build_session_roi_matrices(sessions, bad, atlas="DK", exclude_rois=["ROI_b"])
        self.assertEqual(list(fields.raw), ["ROI_a"])
        with self.assertRaisesRegex(ValueError, "not found"):
            build_session_roi_matrices(sessions, rois, atlas="DK", exclude_rois=["typo"])

    def test_reordered_session_matrices_cannot_silently_misalign(self) -> None:
        sessions, courses, rois = prepared()
        fields = build_session_roi_matrices(sessions, rois, atlas="DK")
        with self.assertRaisesRegex(ValueError, "ordered session IDs"):
            aggregate_course_exposures(sessions.iloc[::-1], courses, aggregation="sum", fields=fields)
        fields.scaled.iloc[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "does not equal"):
            aggregate_course_exposures(sessions, courses, aggregation="sum", fields=fields)

    def test_covariates_join_at_patient_level_without_averaging_age(self) -> None:
        _, courses, _ = prepared()
        info = pd.DataFrame({"subjid": ["subj-cat-901-001", "subj-cat-902"], "age": [30, 40], "sex": ["F", "M"]})
        result = attach_course_covariates(courses, info)
        self.assertEqual(result["age"].tolist(), [30, 30, 40])
        with self.assertRaisesRegex(ValueError, "one row per base"):
            attach_course_covariates(courses, pd.concat([info, info.iloc[[0]]]))

    def test_atlas_summary_bridge_uses_simulation_identity_not_position(self) -> None:
        frame = pd.DataFrame({"a": [20, 10]}, index=pd.Index(["sim2", "sim1"], name="simulation_id"))
        scans = pd.DataFrame({"simulation_id": ["sim1", "sim2"], "hdf5_file": ["/one.hdf5", "/two.hdf5"]})
        table = roi_table_from_summary(SimpleNamespace(scans=scans, p95=lambda method: frame))
        self.assertEqual(table["hdf5_fn"].tolist(), ["/two.hdf5", "/one.hdf5"])
        self.assertEqual(table["a"].tolist(), [20, 10])


class TestMixedModel(unittest.TestCase):
    def test_real_mixed_model_retains_singletons_and_recovers_known_effect(self) -> None:
        table = synthetic_courses()
        design, scaling = build_fixed_effects(table, standardize=False)
        result = fit_course_lmm(table, design, scaling=scaling)
        estimates = result.coefficients.set_index("term")["estimate"]
        self.assertAlmostEqual(estimates["dose"], -0.4, delta=0.08)
        self.assertEqual(result.diagnostics["n_patients"], 80)
        self.assertEqual(result.diagnostics["n_repeated_patients"], 65)
        self.assertGreater(result.diagnostics["patient_variance"], 0.2)
        self.assertTrue(result.diagnostics["converged"])
        self.assertEqual(len(result.predictions), len(table))

    def test_components_and_scaled_models_do_not_rescale_an_interaction_again(self) -> None:
        table = synthetic_courses()
        rng = np.random.default_rng(8)
        table["ef_raw"] = rng.uniform(5, 50, len(table))
        table["ef_dose"] = table["ef_raw"] * table["dose"]
        table["ef_scaled"] = 100 * table["ef_dose"]
        design, _ = build_fixed_effects(table, roi_model="components")
        self.assertIn("ef_raw", design)
        self.assertIn("ef_dose", design)
        self.assertNotIn("ef_scaled", design)
        scaled, _ = build_fixed_effects(table, roi_model="scaled")
        self.assertIn("ef_scaled", scaled)
        self.assertNotIn("ef_raw", scaled)
        self.assertNotIn("ef_dose", scaled)

    def test_no_repeats_collinearity_constant_predictors_and_missing_covariates_fail(self) -> None:
        table = synthetic_courses().drop_duplicates("subjid_base").reset_index(drop=True)
        design, _ = build_fixed_effects(table)
        with self.assertRaisesRegex(ValueError, "requires repeated courses"):
            fit_course_lmm(table, design)
        with self.assertRaisesRegex(ValueError, "Constant"):
            build_fixed_effects(table.assign(n_sessions=6))
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            build_fixed_effects(table.assign(age=table["dose"] * 2), numeric_covariates=["age"])
        with self.assertRaises(ValueError):
            build_fixed_effects(table.assign(sex=None), categorical_covariates=["sex"])
        with self.assertRaisesRegex(ValueError, "outcome-derived"):
            build_fixed_effects(table, numeric_covariates=["cgi_end"])

    def test_covariate_coding_and_age_sex_interaction_are_explicit(self) -> None:
        table = synthetic_courses()
        rng = np.random.default_rng(182)
        table["age"] = rng.uniform(20, 70, len(table))
        table["sex"] = rng.choice(["F", "M"], len(table))
        design, _ = build_fixed_effects(table, numeric_covariates=["age"], categorical_covariates=["sex"],
                                        age_sex_interaction=True)
        np.testing.assert_allclose(design["age:sex[M]"], design["age"] * design["sex[M]"])

    def test_optimizer_failure_is_not_silently_reported_as_success(self) -> None:
        table = synthetic_courses()
        design, _ = build_fixed_effects(table)
        target = "statsmodels.regression.mixed_linear_model.MixedLM.fit"
        with patch(target, return_value=SimpleNamespace(converged=False)):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                fit_course_lmm(table, design)


class TestLMMCLI(unittest.TestCase):
    def test_full_roi_fit_exports_one_course_row_and_correct_inference_family(self) -> None:
        rng = np.random.default_rng(301)
        sessions, courses, query, rois = [], [], [], []
        for patient in range(40):
            base = f"subj-cat-{patient:03d}"
            patient_effect = rng.normal(0, 0.6)
            values = {"RUL": rng.uniform(60, 120), "BT": rng.uniform(100, 180)}
            for placement, field in values.items():
                path = f"/synthetic/{base}-001/{placement}/field.hdf5"
                query.append({"model_type_dir": "skin_double", "step": "step_2", "subject_dir": base + "-001",
                              "electrode_configuration": placement, "full_file_path": path})
                rois.append({"hdf5_fn": path, "ROI_one": field, "ROI_two": 200 - 0.5 * field})
            for series in range(1, 2 if patient < 10 else 3):
                start = pd.Timestamp(2020 + series, 1, 1)
                count = int(rng.integers(3, 9))
                baseline = rng.uniform(4, 7)
                total_dose, total_e, total_product = 0, 0, 0
                for session in range(count):
                    placement = rng.choice(["RUL", "BT"])
                    frequency, percent = float(rng.choice([30, 40, 50])), float(rng.choice([20, 30, 50, 70]))
                    dose = 0.5 * frequency * percent
                    sessions.append({"subjid": base, "series_num": series, "session_num": session + 1,
                                     "date": start + pd.Timedelta(days=session), "electrode_placement": placement,
                                     "pulse_width_ms": 0.5, "frequency_hz": frequency, "percent_charge": percent})
                    total_dose += dose
                    total_e += values[placement]
                    total_product += values[placement] * dose
                change = (-1 + 0.1 * baseline - 0.0001 * total_dose - 0.0004 * total_e
                          - 0.0000004 * total_product + patient_effect + rng.normal(0, 0.2))
                courses.append({"subjid": base, "series_num": series, "cgi_start": baseline,
                                "cgi_end": baseline + change, "date_start": start,
                                "date_end (acute)": start + pd.Timedelta(days=count - 1)})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "clinical.xlsx"
            with pd.ExcelWriter(workbook) as writer:
                pd.DataFrame(sessions).to_excel(writer, sheet_name="stimulus", index=False)
                pd.DataFrame(courses).to_excel(writer, sheet_name="scores", index=False)
            pd.DataFrame(query).to_csv(root / "query.csv", index=False)
            pd.DataFrame(rois).to_csv(root / "rois.csv", index=False)
            argv = ["--sessions-xlsx", str(workbook), "--outcome-sheet", "scores",
                    "--query-path", str(root / "query.csv"),
                    "--model-type", "skin_double", "--simulation-type", "adaptive", "--aggregation", "sum",
                    "--roi-table", str(root / "rois.csv"), "--atlas-name", "synthetic", "--roi-statistic", "mean",
                    "--output-dir", str(root / "lmm")]
            with contextlib.redirect_stdout(io.StringIO()):
                run_lmm.main(argv)
            output = root / "lmm"
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["counts"]["courses"], 70)
            self.assertEqual(manifest["counts"]["patients"], 40)
            self.assertEqual(manifest["n_models"], 2)
            coeffs = pd.read_csv(output / "lmm_coefficients.csv")
            self.assertEqual(len(coeffs), 12)
            self.assertEqual(coeffs["q_value_bh"].notna().sum(), 4)
            model_data = pd.read_csv(output / "models/model_0001/model_data.csv")
            self.assertEqual(len(model_data), 70)
            self.assertFalse(model_data["treatment_course_id"].duplicated().any())
            failed = [str(root / "failed") if x == str(root / "lmm") else x for x in argv]
            with patch.object(run_lmm, "fit_course_lmm", side_effect=RuntimeError("synthetic failure")):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    run_lmm.main(failed)
            self.assertEqual(json.loads((root / "failed/manifest.json").read_text())["status"], "failed")
            self.assertFalse((root / "failed/lmm_coefficients.csv").exists())

    def test_prepare_only_exports_patient_matrices_and_never_fits_pca(self) -> None:
        sessions, courses, query, rois = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "clinical.xlsx"
            with pd.ExcelWriter(workbook) as writer:
                sessions.to_excel(writer, sheet_name="stimulus", index=False)
                courses.to_excel(writer, sheet_name="scores", index=False)
            query.to_csv(root / "query.csv", index=False)
            rois.to_csv(root / "roi.csv", index=False)
            argv = ["--sessions-xlsx", str(workbook), "--outcome-sheet", "scores",
                    "--query-path", str(root / "query.csv"),
                    "--model-type", "skin_double", "--simulation-type", "adaptive", "--aggregation", "sum",
                    "--roi-table", str(root / "roi.csv"), "--atlas-name", "DK", "--roi-statistic", "p95_unweighted",
                    "--prepare-only", "--output-dir", str(root / "lmm")]
            with contextlib.redirect_stdout(io.StringIO()):
                run_lmm.main(argv)
            output = root / "lmm"
            matrix = pd.read_csv(output / "session_matrices/DK/subj-cat-901/scaled_E.csv", index_col=0)
            self.assertEqual(matrix.shape, (3, 2))
            self.assertEqual(matrix["ROI_a"].tolist(), [4000, 75000, 100000])
            self.assertEqual(len(pd.read_csv(output / "course_exposures.csv")), 3)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "prepared")
            self.assertFalse((output / "models").exists())
            with self.assertRaises(FileExistsError):
                run_lmm.main(argv)

    def test_duplicate_csv_and_excel_headers_are_not_mangled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.csv").write_text("subjid,age,age\nsubj-cat-901,25,26\n")
            frame = pd.DataFrame([["subjid", "age", "age"], ["subj-cat-901", 25, 26]])
            frame.to_excel(root / "bad.xlsx", header=False, index=False)
            for name in ["bad.csv", "bad.xlsx"]:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "duplicate column"):
                    read_lmm_table(root / name)


if __name__ == "__main__":
    unittest.main()
