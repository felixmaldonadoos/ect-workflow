"""Grouped LMM namespaces and immutable model manifests (not run during editing)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from analysis_paths import lmm_analysis_name, lmm_run_dir, resolve_analysis_dir, validate_output_path
from run_series_mixed_effects import parse_args
from series_models.run_layout import ensure_run_manifest, run_configuration


class TestGroupedRunLayout(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "results"
        self.environment = patch.dict(os.environ, {"RESULTS_DIR": str(self.root)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_namespace_is_idempotent_and_preserves_results_root(self):
        name = lmm_analysis_name("sex_fixed_v1", "roi_raw")
        self.assertEqual(name, "lmm/sex_fixed_v1/roi_raw")
        self.assertEqual(lmm_analysis_name(name, "roi_raw"), name)
        self.assertEqual(resolve_analysis_dir(name), self.root / name)
        self.assertEqual(lmm_run_dir(name), self.root / "lmm/sex_fixed_v1")
        self.assertEqual(resolve_analysis_dir("legacy_analysis"), self.root / "legacy_analysis")
        self.assertFalse(self.root.exists())

    def test_traversal_conflicting_component_and_symlink_redirect_fail(self):
        for name in ("../escape", "lmm/run/../other", "lmm/run/roi/raw", "other/run", "lmm//run"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                lmm_analysis_name(name)
        with self.assertRaises(ValueError):
            lmm_analysis_name("lmm/run/null", "roi_raw")
        self.root.mkdir()
        other = self.root / "other"
        other.mkdir()
        (self.root / "lmm").symlink_to(other, target_is_directory=True)
        with self.assertRaises(ValueError):
            resolve_analysis_dir("lmm/run")

    def test_manifest_records_formula_and_rejects_changed_settings(self):
        config = run_configuration("fixed", {"reml": False, "optimizer": "powell", "maxiter": 2000})
        name = "lmm/fixed_run/roi_raw"
        source = str(Path(self.temporary.name) / "master.csv")
        self.assertEqual(ensure_run_manifest(name, config, inputs=[source], create=False),
                         self.root / "lmm/fixed_run/run.json")
        self.assertFalse(self.root.exists())
        path = ensure_run_manifest(name, config, inputs=[source])
        data = json.loads(path.read_text())
        self.assertEqual(data["analysis_type"], "series_lmm")
        self.assertIn("C(sex)", data["configuration"]["feature_formula"])
        self.assertEqual(data["configuration"]["variance_components"], {})
        original = path.read_bytes()
        self.assertEqual(ensure_run_manifest("lmm/fixed_run/null", config, inputs=[source]), path)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            ensure_run_manifest(name, run_configuration("random", {"reml": False}), inputs=[source])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(path.parent.glob(".run_manifest_*.json")), [])

    def test_child_output_cannot_escape_selected_component(self):
        name = "lmm/run/roi_raw"
        allowed = self.root / name / "plots/model.png"
        self.assertEqual(validate_output_path(allowed, name), allowed)
        with self.assertRaises(ValueError):
            validate_output_path(self.root / "lmm/run/pca_raw/output.csv", name)

    def test_cli_selects_sex_effect_and_normalizes_output_component(self):
        args = parse_args(["--master-path", "master.csv", "--feature-type", "roi",
                           "--analysis-name", "fixed_run", "--output-component", "roi_raw",
                           "--sex-effect", "fixed"])
        self.assertEqual(args.sex_effect, "fixed")
        self.assertEqual(args.analysis_name, "lmm/fixed_run/roi_raw")
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
