"""Automatic per-fit plots, placeholder accounting, and protected exports."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from analysis_paths import SOURCE_DIR
from series_models.fit_plotting import (
    _draw_prediction, _prepare_matplotlib, plan_fit_plot_paths, render_fit_plots,
)
from series_models.prediction import FitPrediction


def identity(feature="LH_full_synthetic_ROI", *, kind="roi", step="step_0", basis=""):
    return {
        "feature_type": kind, "field_name": "magnE_mean", "normalization": "raw" if kind != "null" else "not_applicable",
        "model_type": "synthetic_model", "step": step, "basis_id": basis,
        "feature_id": feature, "roi_name": feature if kind == "roi" else "",
        "pc_name": feature if kind == "pca" else "", "status": "ok", "aic": 40.5,
    }


def prediction(*, null=False, provisional=False):
    x = np.linspace(25, 32, 8) if null else np.linspace(-2, 2, 8)
    sexes = np.array(["F"] * 4 + ["M"] * 4)
    slope = 0.04 if null else 0.6
    fixed = 1 + slope * x
    offsets = np.where(sexes == "F", -0.2, 0.2)
    adjusted = fixed + offsets + np.linspace(-0.1, 0.1, len(x))
    # Deliberately distinct y columns verify that feature plots adjust the
    # outcome, whereas null plots use the observed saved CGI change.
    observed = adjusted if null else adjusted + 3
    points = pd.DataFrame({
        "person_id": [f"person_{index // 2}" for index in range(8)],
        "series_num": np.tile([1, 2], 4), "sex": sexes, "age": np.linspace(25, 32, 8),
        "ef_sf": np.linspace(1, 2, 8), "observed_cgi_change": observed,
        "fixed_prediction": fixed, "conditional_prediction": fixed + offsets,
        "residual": observed - fixed - offsets,
        "adjusted_cgi_change": adjusted + 10 if null else adjusted,
    })
    if not null:
        points["E"] = x
    curves = []
    for sex in ("", "F", "M"):
        selected = np.ones(8, dtype=bool) if sex == "" else sexes == sex
        grid = np.linspace(x[selected].min(), x[selected].max(), 12)
        offset = 0 if sex == "" else -0.2 if sex == "F" else 0.2
        curves.extend({"curve_kind": "fixed" if sex == "" else "sex", "sex": sex,
                       "x": value, "predicted_cgi_change": 1 + slope * value + offset}
                      for value in grid)
    metadata = {
        "x_column": "age" if null else "E", "reference_age": 28.5,
        "reference_ef_sf": None if null else 1.5, "n_observations": 8,
        "provisional": provisional, "caveat": "Variance boundary; invalid Wald inference" if provisional else "",
        "status": "failed" if provisional else "ok", "converged": True, "aic": 40.5,
        "fit_method": "ML",
    }
    return FitPrediction(points, pd.DataFrame(curves), {"Intercept": 1, "age": 0.04}, {"F": -0.2, "M": 0.2}, metadata)


def record(feature="LH_full_synthetic_ROI", **kwargs):
    return {"identity": identity(feature, **kwargs), "prediction": prediction(null=kwargs.get("kind") == "null"), "error": ""}


def pdf_page_count(path):
    # PDF page dictionaries are uncompressed even when content streams are
    # compressed; matching /Page as a full token excludes the /Pages tree.
    return len(re.findall(rb"/Type\s*/Page\b", path.read_bytes()))


class TestFitPlotting(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.root = self.directory / "results"
        self.environment = patch.dict(os.environ, {"RESULTS_DIR": str(self.root)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def test_plan_matches_files_and_every_fit_gets_a_page_and_png(self):
        ordinary = record("ROI_2")
        provisional = record("ROI_1")
        provisional["prediction"] = prediction(provisional=True)
        unavailable = {"identity": identity("ROI_3", step="step_9"), "prediction": None,
                       "error": "Optimizer did not converge"}
        records = [ordinary, provisional, unavailable]
        plans = plan_fit_plot_paths(records, "fit_plots", "source_master")
        self.assertEqual(len(plans["reports"]), 2)
        self.assertEqual(len(plans["pngs"]), 3)
        self.assertFalse(self.root.exists())
        manifest = render_fit_plots(records, "fit_plots", "source_master", dpi=40)
        self.assertEqual(manifest.feature_id.tolist(), ["ROI_2", "ROI_1", "ROI_3"])
        self.assertEqual(manifest.pdf_page.tolist(), [2, 1, 1])
        self.assertEqual(manifest.plot_status.tolist(), ["ok", "provisional", "skipped"])
        self.assertEqual(manifest.provisional.tolist(), [False, True, False])
        self.assertEqual(manifest.plot_error.iloc[2], "Optimizer did not converge")
        self.assertEqual([Path(value) for value in manifest.png_path], plans["pngs"])
        self.assertEqual(set(map(Path, manifest.pdf_path)), set(plans["reports"]))
        self.assertEqual([pdf_page_count(path) for path in plans["reports"]], [2, 1])
        for path in plans["reports"] + plans["pngs"]:
            self.assertTrue(path.is_relative_to(self.root / "fit_plots"))
            content = path.read_bytes()
            self.assertGreater(len(content), 1000)
            self.assertTrue(content.startswith(b"%PDF") if path.suffix == ".pdf" else content.startswith(b"\x89PNG"))
        import matplotlib
        self.assertEqual(Path(matplotlib.get_cachedir()), plans["cache"])
        self.assertEqual(Path(os.environ["MPLCONFIGDIR"]), plans["cache"])
        self.assertEqual(os.environ["MPLBACKEND"], "Agg")
        self.assertFalse(plans["cache"].is_relative_to(SOURCE_DIR))
        from matplotlib import pyplot as plt
        self.assertEqual(plt.get_fignums(), [])

    def test_feature_uses_adjusted_points_and_only_supplied_sex_model_curves(self):
        current = prediction()
        plans = plan_fit_plot_paths([record()], "axes", "source")
        plt, _ = _prepare_matplotlib(plans["cache"], "axes")
        fig = plt.figure(figsize=(10.5, 7.6))
        try:
            self.assertFalse(_draw_prediction(fig, current, identity(), plt))
            ax = fig.axes[0]
            plotted_y = np.concatenate([collection.get_offsets()[:, 1] for collection in ax.collections])
            np.testing.assert_allclose(plotted_y, current.points.adjusted_cgi_change)
            fixed = [line for line in ax.lines if line.get_label().startswith("Fixed-effect")]
            self.assertEqual(len(fixed), 1)
            self.assertEqual(fixed[0].get_color(), "black")
            self.assertEqual(fixed[0].get_linestyle(), "--")
            for sex in ("F", "M"):
                line = [line for line in ax.lines if line.get_label() == f"{sex}: conditional model mean"][0]
                observed = current.points.loc[current.points.sex.eq(sex), "E"]
                self.assertEqual(min(line.get_xdata()), observed.min())
                self.assertEqual(max(line.get_xdata()), observed.max())
                np.testing.assert_allclose(np.diff(line.get_ydata()) / np.diff(line.get_xdata()), 0.6)
            self.assertIn("adjusted to reference age and ef_sf", ax.get_ylabel())
            captions = " ".join(text.get_text() for text in fig.texts)
            self.assertIn("reference ef_sf: 1.5", captions)
            self.assertIn("AIC = 40.5", captions)
            self.assertIn("No confidence intervals", captions)
        finally:
            plt.close(fig)

    def test_null_plots_observed_change_against_age(self):
        current = prediction(null=True)
        null_identity = identity("null", kind="null")
        plans = plan_fit_plot_paths([record("null", kind="null")], "null", "source")
        plt, _ = _prepare_matplotlib(plans["cache"], "null")
        fig = plt.figure(figsize=(10.5, 7.6))
        try:
            _draw_prediction(fig, current, null_identity, plt)
            ax = fig.axes[0]
            plotted_y = np.concatenate([collection.get_offsets()[:, 1] for collection in ax.collections])
            np.testing.assert_allclose(plotted_y, current.points.observed_cgi_change)
            self.assertEqual(ax.get_xlabel(), "Age (saved years)")
            self.assertEqual(ax.get_ylabel(), "Observed CGI change")
        finally:
            plt.close(fig)

    def test_different_pca_bases_keep_distinct_identity_pngs_and_pages(self):
        records = [record("PC1", kind="pca", basis="basis_A"), record("PC1", kind="pca", basis="basis_B")]
        plans = plan_fit_plot_paths(records, "pca", "pca_master")
        self.assertEqual(len(plans["reports"]), 1)
        self.assertEqual(len(set(plans["pngs"])), 2)
        manifest = render_fit_plots(records, "pca", "pca_master", dpi=30)
        self.assertEqual(manifest.basis_id.tolist(), ["basis_A", "basis_B"])
        self.assertEqual(manifest.feature_id.tolist(), ["PC1", "PC1"])
        self.assertEqual(manifest.pdf_page.tolist(), [1, 2])
        self.assertEqual(pdf_page_count(plans["reports"][0]), 2)

    def test_invalid_per_fit_points_or_curves_become_placeholders_and_keep_other_fits(self):
        nonfinite = prediction()
        points = nonfinite.points.copy()
        points.loc[0, "E"] = np.nan
        nonfinite = replace(nonfinite, points=points)
        extrapolation = prediction()
        curves = extrapolation.curves.copy()
        curves.loc[curves.curve_kind.eq("sex") & curves.sex.eq("F"), "x"] -= 10
        extrapolation = replace(extrapolation, curves=curves)
        nonconverged = prediction()
        nonconverged = replace(nonconverged, metadata={**nonconverged.metadata, "converged": False})
        records = [record(f"ROI_{index}") for index in range(4)]
        for item, current in zip(records, [nonfinite, extrapolation, nonconverged, prediction()]):
            item["prediction"] = current
        manifest = render_fit_plots(records, "partial", "source", dpi=30)
        self.assertEqual(manifest.plot_status.tolist(), ["skipped", "skipped", "skipped", "ok"])
        self.assertIn("nonfinite E", manifest.plot_error.iloc[0])
        self.assertIn("extrapolate", manifest.plot_error.iloc[1])
        self.assertIn("did not converge", manifest.plot_error.iloc[2])
        self.assertEqual(pdf_page_count(Path(manifest.pdf_path.iloc[0])), 4)

    def test_existing_outputs_are_never_overwritten(self):
        records = [record()]
        manifest = render_fit_plots(records, "protected", "source", dpi=30)
        paths = [Path(manifest.pdf_path.iloc[0]), Path(manifest.png_path.iloc[0])]
        original = {path: path.read_bytes() for path in paths}
        with self.assertRaisesRegex(FileExistsError, "already exist"):
            render_fit_plots(records, "protected", "source", dpi=30)
        for path in paths:
            self.assertEqual(path.read_bytes(), original[path])

    def test_safe_segments_and_full_identity_hashes_prevent_path_collisions(self):
        first = record("ROI/../A", step="step/../0")
        second = record("ROI_.._A", step="step/../0")
        first["identity"]["model_type"] = "model/../../other"
        second["identity"]["model_type"] = "model/../../other"
        # Missing inference values on a planned fit are not part of the model
        # identity hash and must not prevent a placeholder report.
        first["identity"]["aic"] = np.nan
        plans = plan_fit_plot_paths([first, second], "safe", "../../master")
        self.assertEqual(len(set(plans["pngs"])), 2)
        for path in plans["reports"] + plans["pngs"]:
            self.assertTrue(path.is_relative_to(self.root / "safe/plots/model_fits"))
            self.assertNotIn("..", path.relative_to(self.root / "safe/plots/model_fits").parts)
        self.assertFalse(self.root.exists())
        with self.assertRaisesRegex(ValueError, "Duplicate planned fit identities"):
            plan_fit_plot_paths([record(), record()], "duplicate", "source")

    def test_global_policy_and_symlink_escape_fail_before_any_output(self):
        records = [record()]
        for configured in ("", "relative/results", str(SOURCE_DIR / "series_models")):
            with self.subTest(configured=configured), patch.dict(os.environ, {"RESULTS_DIR": configured}):
                with self.assertRaises(ValueError):
                    plan_fit_plot_paths(records, "invalid", "source")
        for dpi in (0, -1, True, 2.5):
            with self.subTest(dpi=dpi), self.assertRaises(ValueError):
                render_fit_plots(records, "invalid", "source", dpi=dpi)
        with self.assertRaises(ValueError):
            plan_fit_plot_paths(records, "../escape", "source")
        with self.assertRaises(ValueError):
            plan_fit_plot_paths([], "invalid", "source")
        self.assertFalse(self.root.exists())
        outside = self.directory / "outside"
        analysis = self.root / "symlink"
        analysis.mkdir(parents=True)
        (analysis / "plots").symlink_to(outside)
        with self.assertRaises(ValueError):
            plan_fit_plot_paths(records, "symlink", "source")
        self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()
