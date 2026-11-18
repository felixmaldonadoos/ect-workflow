"""AIC plot selection, separate PCA axes, and results-only PNG/PDF exports."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from analysis_paths import SOURCE_DIR
from series_models.aic_plotting import (
    _heatmap_data, _ranked_rows, plan_plot_paths, render_comparison_plots,
)


def comparison_table():
    rows = []

    def add(feature, improvement, *, kind="roi", basis="", eligible=True,
            provisional=False, null=False, status="ok", normalization="raw"):
        rows.append({
            "feature_type": "null" if null else kind, "field_name": "magnE_mean",
            "normalization": "not_applicable" if null else normalization,
            "model_type": "shared_cohort" if null else "model_A", "step": "all" if null else "step_0",
            "basis_id": basis, "feature_id": feature, "roi_name": feature if kind == "roi" and not null else "",
            "pc_name": feature if kind == "pca" else "", "status": status,
            "is_null": null, "aic_eligible": eligible, "aic_provisional": provisional,
            "aic_caveat": "Boundary / invalid Wald inference" if provisional else "",
            "aic_improvement_over_null": improvement, "aic_rank_global": len(rows) + 1.0,
            "delta_aic_best_including_null": 3.0 - improvement if eligible else np.nan,
        })

    add("LH_full_visual_ROI", 3.0)
    add("RH_full_visual_ROI", 1.0, provisional=True, status="failed")
    # This failed/unconverged candidate must never win the ranked chart even
    # though its unaccepted numerical AIC would look particularly favorable.
    add("excluded_ROI", 20.0, eligible=False, status="failed")
    add("PC1", 2.0, kind="pca", basis="basis_A")
    add("PC1", -2.0, kind="pca", basis="basis_B", provisional=True, status="failed")
    add("PC10", 0.5, kind="pca", basis="basis_A")
    add("PC2", np.nan, kind="pca", basis="basis_A", eligible=False)
    add("null", 0.0, null=True, provisional=True, status="failed")
    return pd.DataFrame(rows)


class TestAicPlotSelection(unittest.TestCase):
    def test_top_n_uses_eligible_likelihoods_and_always_keeps_null(self):
        table = comparison_table()
        ranked = _ranked_rows(table, top_n=2)
        self.assertEqual(ranked.feature_id.tolist(), ["LH_full_visual_ROI", "PC1", "null"])
        self.assertEqual(ranked.basis_id.tolist(), ["", "basis_A", ""])
        self.assertEqual(ranked.aic_improvement_over_null.tolist(), [3.0, 2.0, 0.0])
        self.assertEqual(ranked.is_null.sum(), 1)
        self.assertNotIn("excluded_ROI", ranked.feature_id.tolist())
        larger = _ranked_rows(table, top_n=100)
        self.assertEqual(len(larger), 6)
        # Wald failure alone does not discard an eligible likelihood.
        accepted = larger.loc[larger.feature_id.eq("RH_full_visual_ROI")].iloc[0]
        self.assertTrue(accepted.aic_eligible)
        self.assertTrue(accepted.aic_provisional)
        self.assertEqual(accepted.status, "failed")

    def test_different_pca_bases_remain_distinct_cells_and_pc_names_sort_naturally(self):
        group = comparison_table().query("feature_type == 'pca'")
        mapped = _heatmap_data(group)
        self.assertEqual(mapped["row_labels"], ["PC1", "PC2", "PC10"])
        self.assertEqual(mapped["column_labels"], ["model_A/step_0\nbasis=basis_A", "model_A/step_0\nbasis=basis_B"])
        np.testing.assert_allclose(mapped["values"][0], [2.0, -2.0])
        self.assertEqual(mapped["values"][2, 0], 0.5)
        self.assertTrue(np.isnan(mapped["values"][1, 0]))
        self.assertTrue(mapped["present"][1, 0])
        self.assertFalse(mapped["present"][2, 1])
        self.assertTrue(mapped["provisional"][0, 1])
        self.assertFalse(mapped["provisional"][0, 0])

    def test_duplicate_candidate_cells_are_rejected_instead_of_averaged(self):
        group = comparison_table().query("feature_type == 'pca'")
        duplicate = pd.concat([group, group.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "cannot be pooled"):
            _heatmap_data(duplicate)


class TestAicPlotExports(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.root = self.directory / "results"
        self.environment = patch.dict(os.environ, {"RESULTS_DIR": str(self.root)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def test_planned_paths_match_png_pdf_exports_and_cache_stays_in_results(self):
        table = comparison_table()
        before = table.copy(deep=True)
        paths = plan_plot_paths(table, "plot_test")
        self.assertEqual(len(paths), 6)
        self.assertFalse(self.root.exists())
        manifest = render_comparison_plots(table, "plot_test", top_n=3, dpi=50)
        self.assertEqual([Path(value) for value in manifest.path], paths)
        self.assertEqual(set(manifest.kind), {"ranked_features", "heatmap"})
        self.assertEqual(manifest.query("kind == 'heatmap'").feature_type.tolist(), ["pca", "pca", "roi", "roi"])
        self.assertTrue(manifest.query("feature_type == 'pca'").notes.str.contains("different axes").all())
        for path in paths:
            self.assertTrue(path.is_relative_to(self.root / "plot_test"))
            content = path.read_bytes()
            self.assertGreater(len(content), 1000)
            self.assertTrue(content.startswith(b"\x89PNG") if path.suffix == ".png" else content.startswith(b"%PDF"))
        cache = self.root / "plot_test/logs/matplotlib_cache"
        self.assertEqual(Path(os.environ["MPLCONFIGDIR"]), cache)
        self.assertEqual(os.environ["MPLBACKEND"], "Agg")
        self.assertTrue(cache.is_dir())
        import matplotlib
        self.assertEqual(Path(matplotlib.get_cachedir()), cache)
        self.assertFalse(Path(matplotlib.get_cachedir()).is_relative_to(SOURCE_DIR))
        pd.testing.assert_frame_equal(table, before)
        saved = {path: path.read_bytes() for path in paths}
        with self.assertRaisesRegex(FileExistsError, "already exist"):
            render_comparison_plots(table, "plot_test", dpi=50)
        for path in paths:
            self.assertEqual(path.read_bytes(), saved[path])

    def test_all_excluded_features_still_produce_reference_and_gray_heatmap(self):
        table = comparison_table()
        table.loc[~table.is_null, "aic_eligible"] = False
        selected = _ranked_rows(table, top_n=2)
        self.assertEqual(selected.feature_id.tolist(), ["null"])
        manifest = render_comparison_plots(table, "excluded", dpi=30)
        self.assertEqual(len(manifest), 6)
        self.assertTrue(all(Path(path).is_file() for path in manifest.path))

    def test_null_only_table_produces_ranked_reference_without_heatmaps(self):
        table = comparison_table().query("is_null").copy()
        paths = plan_plot_paths(table, "null_only")
        self.assertEqual(len(paths), 2)
        manifest = render_comparison_plots(table, "null_only", dpi=30)
        self.assertEqual(manifest.kind.tolist(), ["ranked_features", "ranked_features"])
        self.assertEqual([Path(value) for value in manifest.path], paths)

    def test_unsafe_or_colliding_group_labels_get_distinct_safe_filenames(self):
        table = comparison_table()
        first = table.loc[table.feature_id.eq("LH_full_visual_ROI")].copy()
        first["field_name"] = "a/b"
        second = first.copy()
        second["field_name"] = "a_b"
        third = first.copy()
        third["field_name"] = "../" + "long field name" * 10
        planned = plan_plot_paths(pd.concat([table.loc[table.is_null], first, second, third]), "safe")
        self.assertEqual(len(planned), 8)
        self.assertEqual(len(set(planned)), 8)
        for path in planned:
            self.assertTrue(path.is_relative_to(self.root / "safe/plots"))
            self.assertLess(len(path.name), 240)
        self.assertFalse(self.root.exists())

    def test_invalid_input_or_policy_cannot_create_files(self):
        table = comparison_table()
        for value in (0, -1, True, 2.5):
            with self.subTest(top_n=value), self.assertRaises(ValueError):
                render_comparison_plots(table, "invalid", top_n=value)
        for value in (0, -1, True, 2.5):
            with self.subTest(dpi=value), self.assertRaises(ValueError):
                render_comparison_plots(table, "invalid", dpi=value)
        for root in ("", "relative/results", str(SOURCE_DIR / "series_models")):
            with self.subTest(root=root), patch.dict(os.environ, {"RESULTS_DIR": root}):
                with self.assertRaises(ValueError):
                    plan_plot_paths(table, "invalid")
        with self.assertRaises(ValueError):
            plan_plot_paths(table, "../escape")
        no_null = table.loc[~table.is_null]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            plan_plot_paths(no_null, "invalid")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            plan_plot_paths(table.drop(columns="aic_provisional"), "invalid")
        bad_flag = table.copy().astype({"aic_eligible": object})
        bad_flag.loc[0, "aic_eligible"] = "False"
        with self.assertRaisesRegex(ValueError, "Boolean"):
            plan_plot_paths(bad_flag, "invalid")
        self.assertFalse(self.root.exists())

    def test_symlink_escape_and_duplicate_cells_are_rejected_before_plot_writes(self):
        table = comparison_table()
        analysis = self.root / "escape"
        analysis.mkdir(parents=True)
        (analysis / "plots").symlink_to(self.directory / "outside")
        with self.assertRaises(ValueError):
            plan_plot_paths(table, "escape")
        self.assertFalse((self.directory / "outside").exists())
        duplicate = pd.concat([table, table.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "cannot be pooled"):
            render_comparison_plots(duplicate, "duplicate", dpi=30)
        self.assertFalse((self.root / "duplicate").exists())


if __name__ == "__main__":
    unittest.main()
