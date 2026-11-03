"""Saved-master identity and metadata contracts for CSV-only series modeling."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from series_models.adapters import read_and_prepare


def roi_master():
    rows = []
    for series in (1, 2, 3):
        for label, value in (("7Networks_LH_Vis", 2.0), ("7Networks_RH_Vis", 4.0)):
            rows.append({
                "person_id": "same_patient", "series_num": series,
                "model_type": "skin_single", "step": "step_0", "field_name": "magnE_mean",
                "roi_org": label, "roi": "Vis", "hemi": label.split("_")[1],
                "network": "Vis", "mean": value + series, "p95": value * 3,
                "n_voxels": 100 if "LH" in label else 200,
                "cgi_start": 7, "cgi_end": 7 - series, "cgi_change": series,
                "age": 35 if series != 3 else 40, "age_scan": 32,
                "sex": "Male", "ef_sf": 20 + series, "session_num": 2,
                "pulse_width_ms": 0.5, "frequency_hz": 40, "percent_charge": 1 + series / 20,
                "full_file_path": "same_simulation.nii.gz", "comments": "keep this metadata",
            })
    return pd.DataFrame(rows)


def pca_master():
    base = roi_master().drop_duplicates("series_num").drop(columns=[
        "roi_org", "roi", "hemi", "network", "mean", "p95", "n_voxels",
    ])
    rows = []
    for basis in ("basis_one", "basis_two"):
        for _, sample in base.iterrows():
            for index in (1, 2):
                row = sample.to_dict()
                row.update({
                    "basis_id": basis, "sample_id": f"sample_{sample['series_num']}",
                    "component": f"PC{index}", "component_index": index,
                    "score": float(sample["series_num"] * index * -2),
                    "explained_variance": 20 / index, "explained_variance_ratio": 0.6 / index,
                    "cumulative_variance_ratio": 0.6 if index == 1 else 0.9,
                    "selected_for_correlation": "True" if index == 1 else "False",
                    "component_map_path": f"{basis}_PC{index}.nii.gz",
                    "normalization": "brain_mean", "normalization_tag": "normed_brain_magnE_mean_mean",
                    "whole_brain_mean": 100 + sample["series_num"], "n_fit_samples": 3,
                    "pca_model_path": f"{basis}.joblib",
                })
                rows.append(row)
    return pd.DataFrame(rows)


class TestMasterAdapters(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def prepare(self, frame, feature_type="roi", name="master.csv", **kwargs):
        path = self.root / name
        frame.to_csv(path, index=False)
        return read_and_prepare(path, feature_type, "magnE_mean", **kwargs)

    def test_roi_preserves_shuffled_metadata_and_independent_series(self):
        original = roi_master().assign(comments_anesthesia="None", attending="NA").sample(frac=1, random_state=91)
        prepared = self.prepare(original)
        result = prepared.frame
        self.assertEqual(len(result), 6)
        self.assertEqual(result["person_id"].nunique(), 1)
        self.assertEqual(result["series_num"].nunique(), 3)
        self.assertEqual(result["feature_id"].nunique(), 2)
        self.assertEqual(result["full_file_path"].nunique(), 1)
        self.assertTrue(set(original).issubset(result))
        for _, row in result.iterrows():
            saved = original.loc[
                original["series_num"].eq(row["series_num"]) & original["roi_org"].eq(row["feature_id"])
            ].iloc[0]
            self.assertEqual(row["E"], saved["mean"])
            self.assertEqual(row["E_x_ef_sf"], saved["mean"] * saved["ef_sf"])
            self.assertEqual(row["cgi_change"], saved["cgi_start"] - saved["cgi_end"])
            self.assertEqual(row["age"], saved["age"])
        self.assertTrue(result["comments"].eq("keep this metadata").all())
        self.assertTrue(result["comments_anesthesia"].eq("None").all())
        self.assertTrue(result["attending"].eq("NA").all())
        self.assertTrue(result["basis_id"].eq("").all())
        self.assertEqual(prepared.metadata["statistic"], "mean")

    def test_normalized_roi_uses_saved_mean_and_preserves_other_statistics(self):
        frame = roi_master().assign(
            mean_raw=lambda x: x["mean"] * 100, whole_brain_mean=100,
            normalization="brain_mean", normalization_tag="normed_brain_magnE_mean_mean",
            normalization_applies_to="mean",
        )
        result = self.prepare(frame).frame
        np.testing.assert_array_equal(result["E"], frame["mean"])
        np.testing.assert_array_equal(result["p95"], frame["p95"])
        np.testing.assert_array_equal(result["mean_raw"], frame["mean_raw"])
        self.assertTrue(result["normalization"].eq("brain_mean").all())

    def test_legacy_filename_and_explicit_metadata_precedence(self):
        name = "master_normed_brain_magnE_mean_mean.csv"
        prepared = self.prepare(roi_master(), name=name)
        self.assertEqual(prepared.metadata["normalization"], "brain_mean")
        prepared = self.prepare(roi_master().assign(normalization="raw"), name=name)
        self.assertEqual(prepared.metadata["normalization"], "raw")
        prepared = self.prepare(roi_master().rename(columns={"roi_org": "roi_name"}))
        self.assertTrue(prepared.frame["feature_id"].equals(prepared.frame["roi_name"]))

    def test_pca_saved_booleans_negative_scores_and_basis_identity(self):
        original = pca_master().sample(frac=1, random_state=5)
        prepared = self.prepare(original, "pca")
        frame = prepared.frame
        self.assertEqual(len(frame), 6)
        self.assertTrue(frame["component"].eq("PC1").all())
        self.assertTrue(frame["E"].lt(0).all())
        self.assertEqual(frame["basis_id"].nunique(), 2)
        self.assertEqual(frame["person_id"].nunique(), 1)
        np.testing.assert_array_equal(frame["E"], frame["score"])
        self.assertTrue(frame["selected_for_correlation"].all())
        self.assertEqual(prepared.metadata["feature_selection"]["mode"], "saved_selected_for_correlation")
        self.assertTrue(set(original).issubset(frame))

    def test_explicit_pca_selection_overrides_saved_flag(self):
        result = self.prepare(pca_master(), "pca", features=["PC2"])
        self.assertEqual(len(result.frame), 6)
        self.assertTrue(result.frame["component"].eq("PC2").all())
        self.assertFalse(result.frame["selected_for_correlation"].any())
        self.assertEqual(result.metadata["feature_selection"]["mode"], "explicit")

    def test_all_and_explicit_pcs_do_not_require_correlation_flags(self):
        frame = pca_master().drop(columns="selected_for_correlation")
        self.assertEqual(len(self.prepare(frame, "pca", pca_scope="all").frame), 12)
        self.assertEqual(len(self.prepare(frame, "pca", features=["PC2"]).frame), 6)
        with self.assertRaisesRegex(ValueError, "selected_for_correlation"):
            self.prepare(frame, "pca")

    def test_boolean_zero_and_one_do_not_select_false_rows(self):
        frame = pca_master()
        frame["selected_for_correlation"] = frame["component_index"].eq(1).astype(int)
        self.assertTrue(self.prepare(frame, "pca").frame["component"].eq("PC1").all())
        frame["selected_for_correlation"] = "not_boolean"
        with self.assertRaisesRegex(ValueError, "Invalid Boolean"):
            self.prepare(frame, "pca")

    def test_duplicate_and_conflicting_series_records_fail(self):
        frame = roi_master()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare(pd.concat([frame, frame.iloc[[0]]]))
        for column, value in (("cgi_change", 99), ("comments", "conflict"), ("ef_sf", 99)):
            bad = frame.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "Conflicting.*metadata"):
                self.prepare(bad, features=["7Networks_RH_Vis"])

    def test_pca_sample_keys_and_component_metadata_fail_clearly(self):
        for column, value, message in (
            ("sample_id", "different_sample", "series/sample identity"),
            ("person_id", "different_person", "sample/series identity"),
            ("explained_variance", 999, "PCA component"),
            ("selected_for_correlation", "False", "PCA component"),
            ("component_index", 2, "component_index"),
            ("step", "step_9", "basis/configuration"),
        ):
            frame = pca_master()
            frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, message):
                self.prepare(frame, "pca")
        frame = pca_master()
        with self.assertRaisesRegex(ValueError, "Duplicate PCA"):
            self.prepare(pd.concat([frame, frame.iloc[[0]]]), "pca")

    def test_no_basis_is_silently_removed_by_saved_selection(self):
        frame = pca_master()
        frame.loc[frame["basis_id"].eq("basis_two"), "selected_for_correlation"] = "False"
        with self.assertRaisesRegex(ValueError, "No PCA components.*bases"):
            self.prepare(frame, "pca")

    def test_nonfinite_values_and_missing_sex_are_rejected(self):
        for column in ("cgi_change", "mean", "ef_sf", "age"):
            for value in (np.nan, np.inf):
                frame = roi_master()
                frame[column] = frame[column].astype(float)
                frame.loc[0, column] = value
                with self.subTest(column=column, value=value), self.assertRaisesRegex(ValueError, "finite numeric"):
                    self.prepare(frame)
        for value in (None, "", " Male", "unknown"):
            frame = roi_master()
            frame.loc[0, "sex"] = value
            with self.subTest(sex=value), self.assertRaisesRegex(ValueError, "sex"):
                self.prepare(frame)

    def test_conflicting_reserved_columns_and_ambiguous_roi_are_rejected(self):
        for column, value in (("E", 999), ("feature_id", "Vis"), ("feature_type", "pca"), ("E_x_ef_sf", 0)):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "Conflicting reserved"):
                self.prepare(roi_master().assign(**{column: value}))
        with self.assertRaisesRegex(ValueError, "Ambiguous ROI"):
            self.prepare(roi_master().assign(roi_name="different_label"))

    def test_header_and_required_column_validation_precedes_pandas_mangling(self):
        for header, message in ((["person_id", "person_id"], "duplicate"), (["person_id", ""], "empty")):
            path = self.root / "bad_header.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerow(["person", "person"])
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, message):
                read_and_prepare(path, "roi", "magnE_mean")
        with self.assertRaisesRegex(ValueError, "missing required columns.*age"):
            self.prepare(roi_master().drop(columns="age"))

    def test_selection_configuration_and_normalization_validation(self):
        second = roi_master().assign(model_type="skin_double", step="step_9", mean=lambda x: x["mean"] * 2)
        frame = self.prepare(pd.concat([roi_master(), second])).frame
        self.assertEqual(len(frame), 12)
        self.assertEqual(frame[["model_type", "step"]].drop_duplicates().shape[0], 2)
        with self.assertRaisesRegex(ValueError, "absent"):
            self.prepare(roi_master(), features=["Vis"])
        with self.assertRaisesRegex(ValueError, "Mixed normalization"):
            self.prepare(pd.concat([
                roi_master().assign(normalization="raw"),
                roi_master().assign(normalization="brain_mean"),
            ]))
        with self.assertRaisesRegex(ValueError, "Conflicting normalization/tag"):
            self.prepare(roi_master().assign(normalization="brain_mean", normalization_tag="raw"))

    def test_rank_and_variance_identifiability_are_left_to_fitter(self):
        frame = roi_master().assign(ef_sf=1, age=30, sex="Female")
        self.assertEqual(len(self.prepare(frame).frame), 6)


if __name__ == "__main__":
    unittest.main()
