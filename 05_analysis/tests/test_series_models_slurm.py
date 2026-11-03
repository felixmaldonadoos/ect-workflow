"""Slurm submission preflight and argument forwarding without real jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]
SUBMIT = SOURCE / "run_series_mixed_effects_submit.sh"


class TestSeriesSlurmSubmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.work = self.root / "caller directory"
        self.work.mkdir()
        self.master = self.work / "master with spaces.csv"
        self.master.write_text("person_id,series_num\nperson_1,1\n")
        self.results = self.root / "results with spaces"
        self.capture = self.root / "submission.json"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        sbatch = self.bin / "sbatch"
        sbatch.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "pathlib.Path(os.environ['SBATCH_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
            "print('Submitted batch job 123')\n"
        )
        sbatch.chmod(0o755)
        self.environment = dict(
            os.environ, RESULTS_DIR=str(self.results), SERIES_VALIDATE_PYTHON=sys.executable,
            SBATCH_CAPTURE=str(self.capture), PYTHONDONTWRITEBYTECODE="1",
            PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
        )

    def tearDown(self):
        self.temp.cleanup()

    def run_wrapper(self, arguments=None, **environment_changes):
        args = arguments if arguments is not None else [
            "test_run", "--master-path", self.master.name, "--feature-type", "roi",
        ]
        return subprocess.run(
            ["bash", str(SUBMIT), *args], cwd=self.work,
            env=dict(self.environment, **environment_changes), capture_output=True, text=True,
        )

    def test_relative_master_is_canonicalized_before_job_changes_directory(self):
        completed = self.run_wrapper()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        args = json.loads(self.capture.read_text())
        self.assertEqual(args[args.index("--master-path") + 1], str(self.master.resolve()))
        self.assertIn(f"--chdir={SOURCE}", args)
        self.assertIn(f"--output={self.results}/test_run/logs/%j.out", args)
        self.assertTrue((self.results / "test_run/logs").is_dir())

    def test_equals_spelling_and_feature_names_with_spaces_are_preserved(self):
        completed = self.run_wrapper([
            "test_run", f"--master-path={self.master.name}", "--feature-type=roi",
            "--features", "ROI name with spaces", "--prepare-only",
        ])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        args = json.loads(self.capture.read_text())
        self.assertIn(f"--master-path={self.master.resolve()}", args)
        self.assertIn("ROI name with spaces", args)
        self.assertIn("--prepare-only", args)

    def test_dry_run_in_any_position_validates_without_writes_or_submission(self):
        base = ["test_run", "--master-path", self.master.name, "--feature-type", "roi"]
        for position in (0, 1, 2, 3, len(base)):
            with self.subTest(position=position):
                arguments = base.copy()
                arguments.insert(position, "--dry-run")
                completed = self.run_wrapper(arguments)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("Dry run; no job submitted", completed.stdout)
                command = shlex.split(completed.stdout.splitlines()[-1])
                self.assertEqual(command[0], "sbatch")
                self.assertIn(str(self.master.resolve()), command)
                self.assertNotIn("--dry-run", command)
                self.assertFalse(self.capture.exists())
                self.assertFalse(self.results.exists())

    def test_bad_arguments_and_missing_master_do_not_submit_or_create_logs(self):
        variants = [
            ["test_run"],
            ["test_run", "--master-path", self.master.name],
            ["test_run", "--master-path", self.master.name, "--feature-type", "bad"],
            ["test_run", "--master-path", self.master.name, "--feature-type", "roi", "--maxiter", "0"],
            ["test_run", "--master-path", "missing.csv", "--feature-type", "roi"],
            ["test_run", "--master-path", str(self.work), "--feature-type", "roi"],
            ["test_run", "--master-path", self.master.name, "--feature-type", "roi", "--analysis-name", "other"],
        ]
        for arguments in variants:
            with self.subTest(arguments=arguments):
                completed = self.run_wrapper(arguments)
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(self.capture.exists())
                self.assertFalse(self.results.exists())

    def test_invalid_results_roots_and_analysis_name_fail_before_submission(self):
        for results in ("", "relative/results", str(SOURCE / "results")):
            with self.subTest(results=results):
                completed = self.run_wrapper(RESULTS_DIR=results)
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(self.capture.exists())
                self.assertFalse(self.results.exists())
        completed = self.run_wrapper([
            "../escape", "--master-path", self.master.name, "--feature-type", "roi",
        ])
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(self.capture.exists())
        self.assertFalse(self.results.exists())

    def test_worker_dispatches_interpreter_arguments_threads_and_exit_code(self):
        worker_capture = self.root / "worker.json"
        srun = self.bin / "srun"
        srun.write_text("#!/usr/bin/env bash\nexec \"$@\"\n")
        srun.chmod(0o755)
        interpreter = self.bin / "analysis interpreter with spaces"
        interpreter.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "keys = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE']\n"
            "payload = {'args': sys.argv[1:], 'cwd': os.getcwd(), 'environment': {key: os.environ.get(key) for key in keys}}\n"
            "pathlib.Path(os.environ['SERIES_WORKER_CAPTURE']).write_text(json.dumps(payload))\n"
            "raise SystemExit(7)\n"
        )
        interpreter.chmod(0o755)
        completed = subprocess.run(
            ["bash", str(SOURCE / "run_series_mixed_effects.slurm"),
             "--analysis-name", "worker_test", "--master-path", str(self.master), "--feature-type", "roi"],
            cwd=self.work, env=dict(self.environment, SERIES_PYTHON=str(interpreter),
                                   SERIES_SCRIPT_ROOT=str(SOURCE), SLURM_CPUS_PER_TASK="3",
                                   SERIES_WORKER_CAPTURE=str(worker_capture)),
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 7, completed.stderr)
        dispatched = json.loads(worker_capture.read_text())
        self.assertEqual(dispatched["cwd"], str(SOURCE))
        self.assertEqual(dispatched["args"][:3], ["-B", "-u", str(SOURCE / "run_series_mixed_effects.py")])
        self.assertIn(str(self.master), dispatched["args"])
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(dispatched["environment"][key], "3")
        self.assertEqual(dispatched["environment"]["PYTHONDONTWRITEBYTECODE"], "1")


if __name__ == "__main__":
    unittest.main()
