"""Exercise correlation entry points without running the real dataset or Slurm."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from scipy.stats import spearmanr

from networks import mni_yeo2011_correlations as regular
from networks import mni_yeo2011_correlations_perm as permutation


FIELD = "magnE_mean"
NORM_TAG = f"normed_brain_{FIELD}_mean"
MODULES = (regular, permutation)
ANALYSIS_SOURCE = Path(__file__).resolve().parents[1]


def small_master() -> pd.DataFrame:
    rows = []
    # Ties and opposing ROI patterns make the numerical assertions useful.
    for series, (outcome, left, right) in enumerate(
        ((1, 2, 8), (1, 4, 5), (3, 4, 3), (2, 9, 1)), start=1
    ):
        for roi, value in (("7Networks_LH_Vis_1", left), ("7Networks_RH_Vis_1", right)):
            rows.append({
                "person_id": f"person-{series}", "series_num": 1,
                "model_type": "skin", "step": "step_0", "field_name": FIELD,
                "roi_org": roi, "mean": value, "cgi_change": outcome,
            })
    return pd.DataFrame(rows)


def cli_args(module, *args: str) -> list[str]:
    extra = [] if module is regular else ["--n-permutations", "24", "--batch-size", "7"]
    return ["--no-network-means", *args, *extra]


class TestYeoCorrelationPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="yeo-correlations-")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.results_root = self.workspace / "results"
        self.environment = patch.dict(os.environ, {"RESULTS_DIR": str(self.results_root)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def run_main(self, module, args: list[str]) -> None:
        with redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(args), 0)

    def output_name(self, module, *, normalized: bool = False) -> str:
        suffix = "_permutation" if module is permutation else ""
        norm = f"_{NORM_TAG}" if normalized else ""
        return f"spearman_{FIELD}_ROImean_only{norm}{suffix}_annotated.csv"

    def assert_statistics(self, table: pd.DataFrame) -> None:
        master = small_master()
        for row in table.itertuples():
            selected = master.loc[master.roi_org.eq(row.roi_org)]
            reference = spearmanr(selected["mean"], selected["cgi_change"])
            self.assertEqual(row.n, 4)
            self.assertAlmostEqual(row.rho, reference.statistic)
            self.assertAlmostEqual(row.p_value, reference.pvalue)
            self.assertEqual(row.field_name, FIELD)
            self.assertTrue(row.x.startswith(FIELD + "_"))

    def test_default_master_and_output_follow_selected_analysis(self) -> None:
        data = self.results_root / "example" / "data"
        data.mkdir(parents=True)
        master_path = data / f"yeo2011_{FIELD}_series_ROI_master.csv"
        small_master().to_csv(master_path, index=False)
        for module in MODULES:
            with self.subTest(module=module.__name__):
                self.run_main(module, cli_args(module, "--analysis-name", "example"))
                saved = pd.read_csv(data / self.output_name(module))
                self.assert_statistics(saved)
                self.assertEqual(saved.normalization.unique().tolist(), ["raw"])
                self.assertEqual(saved.normalization_tag.unique().tolist(), ["raw"])
                self.assertEqual(saved.master_path.unique().tolist(), [str(master_path.resolve())])
                if module is permutation:
                    self.assertTrue(saved.permutation_exact.all())
                    self.assertEqual(saved.permutation_n_resamples_actual.tolist(), [24, 24])
                    self.assertTrue(all(len(json.loads(value)) == 24 for value in saved.permutation_null_distribution))

    def test_legacy_input_and_metadata_choose_normalized_result_filename(self) -> None:
        legacy = self.workspace / "legacy"
        legacy.mkdir()
        # No filename clue: normalization must be read from the table itself.
        master_path = legacy / "arbitrary.csv"
        small_master().assign(
            normalization="brain_mean", normalization_tag=NORM_TAG,
        ).to_csv(master_path, index=False)
        for module in MODULES:
            with self.subTest(module=module.__name__):
                self.run_main(module, cli_args(module, "--master-path", str(master_path)))
                output = self.results_root / "mni_yeo2011" / "data" / self.output_name(module, normalized=True)
                saved = pd.read_csv(output)
                self.assert_statistics(saved)
                self.assertEqual(saved.normalization.unique().tolist(), ["brain_mean"])
                self.assertEqual(saved.normalization_tag.unique().tolist(), [NORM_TAG])
                self.assertEqual(saved.master_path.unique().tolist(), [str(master_path.resolve())])
        self.assertEqual(list(legacy.iterdir()), [master_path])

    def test_explicit_output_cannot_escape_selected_analysis_or_overwrite_input(self) -> None:
        data = self.results_root / "selected" / "data"
        data.mkdir(parents=True)
        master_path = data / "input.csv"
        small_master().to_csv(master_path, index=False)
        for module in MODULES:
            for target in (self.results_root / "other" / "out.csv", master_path):
                with self.subTest(module=module.__name__, target=target):
                    runner = "run_master_spearman" if module is regular else "run_master_spearman_permutation"
                    with patch.object(module, runner, side_effect=AssertionError("Analysis should not run")):
                        with self.assertRaises(ValueError):
                            module.main(cli_args(module, "--analysis-name", "selected", "--master-path", str(master_path), "--output-path", str(target)))
        self.assertEqual(len(pd.read_csv(master_path)), 8)

    def test_library_savers_guard_output_and_preserve_extra_columns(self) -> None:
        for module in MODULES:
            with self.subTest(module=module.__name__):
                if module is regular:
                    results = module.run_master_spearman(small_master(), include_network_means=False)
                else:
                    results = module.run_master_spearman_permutation(small_master(), include_network_means=False, n_permutations=24, batch_size=7)
                results = results.assign(normalization="brain_mean", normalization_tag=NORM_TAG, master_path="legacy/input.csv", custom_provenance="retained")
                output = self.results_root / "library" / (module.__name__.split(".")[-1] + ".csv")
                annotated = module.save_spearman_table(results, output)
                saved = pd.read_csv(output)
                for column in results:
                    self.assertIn(column, saved)
                self.assertEqual(saved.custom_provenance.tolist(), ["retained", "retained"])
                pd.testing.assert_series_equal(annotated.rho, results.rho)
                with self.assertRaises(ValueError):
                    module.save_spearman_table(results, self.workspace / "forbidden.csv")
                self.assertFalse((self.workspace / "forbidden.csv").exists())

    def test_cli_requires_results_dir_before_loading_scientific_packages(self) -> None:
        environment = dict(os.environ)
        environment.pop("RESULTS_DIR", None)
        for module in MODULES:
            with self.subTest(module=module.__name__):
                # -S disables third-party packages: a clear RESULTS_DIR failure
                # must still happen before any attempted NumPy/SciPy import.
                completed = subprocess.run(
                    [sys.executable, "-B", "-S", module.__file__],
                    env=environment, text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("RESULTS_DIR", completed.stderr)
                self.assertNotIn("No module named 'numpy'", completed.stderr)

    def test_modules_are_importable_without_results_dir(self) -> None:
        environment = dict(os.environ)
        environment.pop("RESULTS_DIR", None)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", "from networks import mni_yeo2011_correlations, mni_yeo2011_correlations_perm"],
            cwd=ANALYSIS_SOURCE, env=environment, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
