"""Shared-null/four-master preflight and dispatch with mocked Slurm jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]


class TestAllMasterSubmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.root = self.directory / "results"
        self.capture = self.directory / "sbatch_calls.jsonl"
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        stub = self.bin / "sbatch"
        stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['SERIES_TEST_SBATCH_CAPTURE'], 'a') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "print('Submitted batch job 1234')\n"
        )
        stub.chmod(0o755)
        self.environment = dict(
            os.environ, RESULTS_DIR=str(self.root), SERIES_VALIDATE_PYTHON=sys.executable,
            SERIES_TEST_SBATCH_CAPTURE=str(self.capture), PYTHONDONTWRITEBYTECODE="1",
            PATH=str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
        )
        for name in ("SERIES_ROI_RAW_MASTER", "SERIES_ROI_BRAIN_MASTER",
                     "SERIES_PCA_RAW_MASTER", "SERIES_PCA_BRAIN_MASTER"):
            path = self.directory / f"{name} with spaces.csv"
            path.write_text("placeholder: submission checks existence; worker validates the master\n")
            self.environment[name] = str(path)

    def tearDown(self):
        self.temp.cleanup()

    def run_launcher(self, *arguments):
        return subprocess.run(
            ["bash", str(SOURCE / "run_series_mixed_effects_all_submit.sh"), *arguments],
            cwd=self.directory, env=self.environment, capture_output=True, text=True,
        )

    def test_null_and_four_feature_jobs_keep_sources_and_models_separate(self):
        completed = self.run_launcher("batch_test", "--prepare-only", "--maxiter", "19")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        calls = [json.loads(line) for line in self.capture.read_text().splitlines()]
        self.assertEqual(len(calls), 5)
        expected = [
            ("null", "roi", "SERIES_ROI_RAW_MASTER"),
            ("roi_raw", "roi", "SERIES_ROI_RAW_MASTER"),
            ("roi_brain_mean", "roi", "SERIES_ROI_BRAIN_MASTER"),
            ("pca_raw", "pca", "SERIES_PCA_RAW_MASTER"),
            ("pca_brain_mean", "pca", "SERIES_PCA_BRAIN_MASTER"),
        ]
        for call, (suffix, kind, environment_name) in zip(calls, expected):
            self.assertEqual(call[call.index("--analysis-name") + 1], f"lmm/batch_test/{suffix}")
            self.assertEqual(call[call.index("--output-component") + 1], suffix)
            self.assertEqual(call[call.index("--feature-type") + 1], kind)
            self.assertEqual(call[call.index("--master-path") + 1], self.environment[environment_name])
            self.assertIn("--prepare-only", call)
            self.assertEqual("--null-model" in call, suffix == "null")
            self.assertEqual(call[call.index("--maxiter") + 1], "19")
            log_dir = self.root / "lmm" / "batch_test" / suffix / "logs"
            self.assertTrue(log_dir.is_dir())
            self.assertIn(f"--output={log_dir}/%x_%j.out", call)
        manifest = json.loads((self.root / "lmm/batch_test/run.json").read_text())
        self.assertEqual(manifest["configuration"]["sex_effect_choice"], "random")
        self.assertEqual(len(manifest["input_masters"]), 4)

    def test_dry_run_validates_without_submission_or_directories(self):
        completed = self.run_launcher("batch_test", "--dry-run")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("no jobs submitted", completed.stdout)
        self.assertFalse(self.capture.exists())
        self.assertFalse(self.root.exists())

    def test_missing_last_master_prevents_every_submission(self):
        self.environment["SERIES_PCA_BRAIN_MASTER"] = str(self.directory / "missing.csv")
        completed = self.run_launcher("batch_test")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("missing or unreadable", completed.stderr)
        self.assertFalse(self.capture.exists())
        self.assertFalse(self.root.exists())

    def test_invalid_shared_options_or_names_prevent_submission(self):
        for options in (("--optimizer", "bad"), ("--maxiter", "0"),
                        ("--features", "PC1"), ("--analysis-name", "override"),
                        ("--pca-scope", "all"), ("--null-model",), ("--output-component", "roi_raw")):
            with self.subTest(options=options):
                completed = self.run_launcher("batch_test", *options)
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(self.capture.exists())
                self.assertFalse(self.root.exists())
        completed = self.run_launcher("../escape")
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(self.capture.exists())
        self.assertFalse(self.root.exists())

    def test_fixed_sex_flag_is_forwarded_and_run_name_is_not_reused(self):
        completed = self.run_launcher("fixed_test", "--sex-effect", "fixed")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        calls = [json.loads(line) for line in self.capture.read_text().splitlines()]
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(call[call.index("--sex-effect") + 1] == "fixed" for call in calls))
        manifest = json.loads((self.root / "lmm/fixed_test/run.json").read_text())
        self.assertIn("C(sex)", manifest["configuration"]["feature_formula"])
        repeated = self.run_launcher("fixed_test", "--sex-effect", "random")
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(len(self.capture.read_text().splitlines()), 5)


if __name__ == "__main__":
    unittest.main()
