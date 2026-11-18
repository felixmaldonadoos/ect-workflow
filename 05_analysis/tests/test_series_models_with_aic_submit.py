"""Mocked submission contracts for the separate model-plus-AIC runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest


SOURCE = Path(__file__).resolve().parents[1]
RUNNER = SOURCE / "run_series_mixed_effects_with_aic_submit.sh"
COMPONENTS = ["null", "roi_raw", "roi_brain_mean", "pca_raw", "pca_brain_mean"]


class TestSeriesModelsWithAICSubmit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="series dependent submit ")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.results = self.directory / "analysis outputs"
        self.calls_file = self.directory / "sbatch calls.jsonl"
        stub_bin = self.directory / "stub bin"
        stub_bin.mkdir()
        stub = stub_bin / "sbatch"
        stub.write_text(f"#!{sys.executable}\n" + textwrap.dedent(r'''
            import json
            import os
            from pathlib import Path
            import sys

            path = Path(os.environ["STUB_SBATCH_CALLS"])
            previous = path.read_text().splitlines() if path.exists() else []
            attempt = len(previous) + 1
            arguments = sys.argv[1:]
            record = {"attempt": attempt, "argv": arguments}
            # Model wrappers inherit their caller's stdin. Reading it would
            # consume unrelated input or block; only the AIC job uses stdin.
            if "--parsable" in arguments:
                record["stdin"] = sys.stdin.read()
            with path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if attempt == int(os.environ.get("STUB_FAIL_AT", "0")):
                print("simulated sbatch failure", file=sys.stderr)
                raise SystemExit(17)
            job_id = str(9000 + attempt)
            print(job_id + ";testcluster" if "--parsable" in arguments
                  else "Submitted batch job " + job_id)
        '''))
        stub.chmod(0o755)
        self.environment = {
            **os.environ, "PATH": str(stub_bin) + os.pathsep + os.environ.get("PATH", ""),
            "RESULTS_DIR": str(self.results), "SERIES_VALIDATE_PYTHON": sys.executable,
            "STUB_SBATCH_CALLS": str(self.calls_file), "STUB_FAIL_AT": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.masters = {}
        names = ["SERIES_ROI_RAW_MASTER", "SERIES_ROI_BRAIN_MASTER",
                 "SERIES_PCA_RAW_MASTER", "SERIES_PCA_BRAIN_MASTER"]
        for component, name in zip(COMPONENTS[1:], names, strict=True):
            path = self.directory / f"{component} input master with spaces.csv"
            path.write_text("placeholder for submission preflight\n")
            self.environment[name] = str(path)
            self.masters[component] = path

    def run_submit(self, *arguments, name="test_run", fail_at=0):
        return subprocess.run(
            ["bash", str(RUNNER), name, *arguments], cwd=self.directory,
            env={**self.environment, "STUB_FAIL_AT": str(fail_at)},
            capture_output=True, text=True,
        )

    def calls(self):
        if not self.calls_file.exists():
            return []
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()]

    def run_root(self, name="test_run"):
        return self.results / "lmm" / name.removeprefix("lmm/")

    def receipt(self, name="test_run"):
        return json.loads((self.run_root(name) / "submission_with_aic.json").read_text())

    @staticmethod
    def value(arguments, option):
        return arguments[arguments.index(option) + 1]

    @staticmethod
    def aic_arguments(call):
        line = next(line for line in call["stdin"].splitlines() if line.startswith("exec srun "))
        command = shlex.split(line)
        script_index = next(index for index, value in enumerate(command)
                            if value.endswith("run_series_aic_comparison.py"))
        return command[script_index + 1:]

    def test_submits_five_models_then_aic_with_all_dependencies_and_stdin_script(self):
        completed = self.run_submit()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 6)
        for component, call in zip(COMPONENTS, calls[:5], strict=True):
            self.assertNotIn("--parsable", call["argv"])
            self.assertNotIn("stdin", call)
            self.assertEqual(self.value(call["argv"], "--output-component"), component)
        aic = calls[5]
        self.assertIn("--parsable", aic["argv"])
        self.assertIn("--dependency=afterok:9001:9002:9003:9004:9005", aic["argv"])
        self.assertIn("--kill-on-invalid-dep=yes", aic["argv"])
        # No batch-script filename is passed: sbatch consumes the supplied text.
        self.assertTrue(all(argument.startswith("--") for argument in aic["argv"]))
        self.assertTrue(aic["stdin"].startswith("#!/usr/bin/env bash\n"))
        arguments = self.aic_arguments(aic)
        self.assertEqual(self.value(arguments, "--null-analysis-dir"), str(self.run_root() / "null"))
        index = arguments.index("--feature-analysis-dirs")
        self.assertEqual(arguments[index + 1:index + 5], [str(self.run_root() / part) for part in COMPONENTS[1:]])
        self.assertNotIn("--reml", arguments)
        self.assertNotIn("--prepare-only", arguments)
        receipt = self.receipt()
        self.assertEqual(receipt["status"], "submitted")
        self.assertEqual(receipt["aic_job_id"], "9006")
        self.assertEqual([job["job_id"] for job in receipt["model_jobs"]], [str(number) for number in range(9001, 9006)])

    def test_dry_run_has_no_writes_or_sbatch_calls_and_prints_placeholder_dependency(self):
        completed = self.run_submit("--dry-run")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse(self.results.exists())
        self.assertIn("afterok:<null-id>:<roi-raw-id>:<roi-brain-id>:<pca-raw-id>:<pca-brain-id>", completed.stdout)
        self.assertIn("run_series_aic_comparison.py", completed.stdout)
        self.assertIn("no files written or jobs submitted", completed.stdout)

    def test_missing_last_master_prevents_every_submission_and_output_write(self):
        self.masters["pca_brain_mean"].unlink()
        completed = self.run_submit()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("SERIES_PCA_BRAIN_MASTER", completed.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse(self.results.exists())

    def test_reml_and_prepare_only_are_rejected_before_any_submission(self):
        for option in ("--reml", "--prepare-only"):
            with self.subTest(option=option):
                completed = self.run_submit(option)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(option, completed.stderr)
                self.assertEqual(self.calls(), [])
                self.assertFalse(self.results.exists())

    def test_third_model_failure_keeps_two_accepted_jobs_without_submitting_aic(self):
        completed = self.run_submit(fail_at=3)
        self.assertEqual(completed.returncode, 17)
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        self.assertTrue(all("--parsable" not in call["argv"] for call in calls))
        receipt = self.receipt()
        self.assertEqual(receipt["status"], "model_submission_failed")
        self.assertEqual(receipt["failed_component"], "roi_brain_mean")
        self.assertEqual([job["job_id"] for job in receipt["model_jobs"]], ["9001", "9002"])
        self.assertNotIn("aic_job_id", receipt)
        self.assertNotIn("aic_submission_command", receipt)

    def test_sixth_submission_failure_keeps_all_model_jobs_and_failure_receipt(self):
        completed = self.run_submit(fail_at=6)
        self.assertEqual(completed.returncode, 17)
        self.assertEqual(len(self.calls()), 6)
        self.assertIn("--parsable", self.calls()[-1]["argv"])
        receipt = self.receipt()
        self.assertEqual(receipt["status"], "aic_submission_failed")
        self.assertEqual(len(receipt["model_jobs"]), 5)
        self.assertEqual(receipt["dependency"], "afterok:9001:9002:9003:9004:9005")
        self.assertIn("simulated sbatch failure", receipt["submission_stderr"])
        self.assertNotIn("aic_job_id", receipt)

    def test_model_and_aic_plot_options_are_routed_to_their_own_jobs(self):
        completed = self.run_submit("--sex-effect", "fixed", "--no-fit-plots", "--plot-dpi", "81",
                                    "--aic-no-plots", "--aic-top-n", "7", "--aic-dpi", "99")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for call in self.calls()[:5]:
            arguments = call["argv"]
            self.assertEqual(self.value(arguments, "--sex-effect"), "fixed")
            self.assertIn("--no-fit-plots", arguments)
            self.assertEqual(self.value(arguments, "--plot-dpi"), "81")
            self.assertFalse(any(argument.startswith("--aic-") for argument in arguments))
            self.assertNotIn("--no-plots", arguments)
        aic = self.aic_arguments(self.calls()[-1])
        self.assertIn("--no-plots", aic)
        self.assertEqual(self.value(aic, "--top-n"), "7")
        self.assertEqual(self.value(aic, "--dpi"), "99")
        for option in ("--sex-effect", "--no-fit-plots", "--plot-dpi", "--aic-no-plots"):
            self.assertNotIn(option, aic)

    def test_normalized_namespace_and_spaced_paths_remain_single_arguments(self):
        name = "lmm/namespaced_run"
        completed = self.run_submit(name=name)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        expected_masters = [self.masters["roi_raw"], *[self.masters[part] for part in COMPONENTS[1:]]]
        for component, master, call in zip(COMPONENTS, expected_masters, self.calls()[:5], strict=True):
            self.assertEqual(self.value(call["argv"], "--analysis-name"), f"lmm/namespaced_run/{component}")
            self.assertEqual(self.value(call["argv"], "--master-path"), str(master))
        arguments = self.aic_arguments(self.calls()[-1])
        self.assertEqual(self.value(arguments, "--analysis-name"), name)
        self.assertEqual(self.value(arguments, "--null-analysis-dir"), str(self.run_root(name) / "null"))
        self.assertIn(f"--output={self.run_root(name)}/aic/logs/%x_%j.out", self.calls()[-1]["argv"])
        self.assertEqual(self.receipt(name)["run_name"], "namespaced_run")

    def test_reusing_a_run_refuses_before_additional_sbatch_calls(self):
        first = self.run_submit()
        self.assertEqual(first.returncode, 0, first.stderr)
        receipt_path = self.run_root() / "submission_with_aic.json"
        original = receipt_path.read_bytes()
        second = self.run_submit()
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("Run directory already exists", second.stderr)
        self.assertEqual(len(self.calls()), 6)
        self.assertEqual(receipt_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
