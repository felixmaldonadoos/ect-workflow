from __future__ import annotations

import itertools
import json
import unittest

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nifti_pca.correlations import correlate_scores


class TestPCACorrelations(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = pd.DataFrame(
            {"PC1": [-3.0, -1.0, 2.0, 2.0], "PC2": [2.0, -4.0, 0.0, 3.0]},
            index=pd.Index(["series-a", "series-b", "series-c", "series-d"], name="sample_id"),
        )
        self.outcome = pd.Series(
            [1.0, 1.0, 3.0, 2.0], index=self.scores.index, name="cgi_change"
        )

    def test_exact_tied_rank_reference_and_saved_provenance(self) -> None:
        results = correlate_scores(
            self.scores, self.outcome, n_permutations=9999, seed=7, batch_size=7
        )
        self.assertEqual(results.component.tolist(), ["PC1", "PC2"])
        for component, row in results.set_index("component").iterrows():
            observed = spearmanr(self.scores[component], self.outcome)
            reference = np.array([
                spearmanr(self.scores[component], self.outcome.to_numpy()[list(order)]).statistic
                for order in itertools.permutations(range(len(self.outcome)))
            ])
            epsilon = abs(observed.statistic) * np.finfo(float).eps * 100
            less = np.count_nonzero(reference <= observed.statistic + epsilon) / len(reference)
            greater = np.count_nonzero(reference >= observed.statistic - epsilon) / len(reference)
            self.assertAlmostEqual(row.rho, observed.statistic)
            self.assertAlmostEqual(row.p_value, observed.pvalue)
            self.assertAlmostEqual(row.p_permutation, min(1.0, 2 * min(less, greater)))
            self.assertAlmostEqual(row.permutation_statistic, row.rho)
            self.assertTrue(row.permutation_exact)
            self.assertFalse(row.permutation_add_one_correction)
            self.assertEqual(row.permutation_n_resamples_actual, 24)
            null = np.array(json.loads(row.permutation_null_distribution))
            np.testing.assert_allclose(np.sort(null), np.sort(reference), atol=1e-15)
            self.assertEqual(json.loads(row.permutation_sample_keys), self.scores.index.tolist())
            self.assertEqual(json.loads(row.permutation_input_scores), self.scores[component].tolist())
            self.assertEqual(json.loads(row.permutation_input_outcome), self.outcome.tolist())
            self.assertEqual(row.permutation_type, "pairings")
            self.assertEqual(row.permutation_axis, 0)
            self.assertTrue(row.permutation_vectorized)
            self.assertTrue(row.samples_assumed_independent)

    def test_monte_carlo_seed_reproducibility_and_shared_shuffles(self) -> None:
        scores = pd.DataFrame({"PC1": [-2.0, 0, 1, 4, 6, 8], "PC2": [2.0, 0, -1, -4, -6, -8]})
        outcome = pd.Series([1.0, 4, 5, 2, 7, 3], index=scores.index)
        first = correlate_scores(scores, outcome, n_permutations=37, seed=13, batch_size=8)
        repeated = correlate_scores(scores, outcome, n_permutations=37, seed=13, batch_size=8)
        changed = correlate_scores(scores, outcome, n_permutations=37, seed=14, batch_size=8)
        pd.testing.assert_frame_equal(first, repeated)
        self.assertNotEqual(first.loc[0, "permutation_null_distribution"], changed.loc[0, "permutation_null_distribution"])
        self.assertFalse(first.permutation_exact.any())
        self.assertTrue(first.permutation_add_one_correction.all())
        self.assertEqual(first.permutation_n_resamples_actual.tolist(), [37, 37])
        first_null = np.array(json.loads(first.loc[0, "permutation_null_distribution"]))
        second_null = np.array(json.loads(first.loc[1, "permutation_null_distribution"]))
        np.testing.assert_allclose(second_null, -first_null, atol=1e-15)
        self.assertEqual(first.loc[0, "permutation_rng_initial_state"], first.loc[1, "permutation_rng_initial_state"])

    def test_alignment_uses_labels_without_changing_score_order(self) -> None:
        expected = correlate_scores(self.scores, self.outcome, n_permutations=24)
        actual = correlate_scores(self.scores, self.outcome.iloc[::-1], n_permutations=24)
        pd.testing.assert_frame_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "exactly the same sample labels"):
            correlate_scores(self.scores, self.outcome.rename(index={"series-d": "different"}))

    def test_analytic_only_signed_scores_do_not_create_a_null(self) -> None:
        result = correlate_scores(self.scores, self.outcome, n_permutations=0)
        self.assertTrue(np.isfinite(result.rho).all())
        self.assertTrue(np.isfinite(result.p_value).all())
        self.assertTrue(result.p_permutation.isna().all())
        self.assertFalse(result.permutation_enabled.any())
        self.assertFalse(result.permutation_exact.any())
        self.assertEqual(result.permutation_n_resamples_actual.tolist(), [0, 0])
        self.assertEqual(result.permutation_null_distribution.tolist(), ["[]", "[]"])
        self.assertTrue(result.permutation_rng_initial_state.isna().all())

    def test_invalid_inputs_are_not_silently_filtered(self) -> None:
        bad_score = self.scores.copy()
        bad_score.loc["series-a", "PC1"] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            correlate_scores(bad_score, self.outcome)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            correlate_scores(self.scores, self.outcome.replace(3.0, np.inf))
        with self.assertRaisesRegex(ValueError, "Constant component"):
            correlate_scores(self.scores.assign(PC1=0.0), self.outcome)
        with self.assertRaisesRegex(ValueError, "constant outcome"):
            correlate_scores(self.scores, self.outcome * 0)
        with self.assertRaisesRegex(ValueError, "At least three"):
            correlate_scores(self.scores.iloc[:2], self.outcome.iloc[:2])
        with self.assertRaisesRegex(ValueError, "unique"):
            correlate_scores(self.scores.set_axis(["same"] * 4), self.outcome)
        with self.assertRaisesRegex(ValueError, "unique"):
            correlate_scores(self.scores, self.outcome.set_axis(["same"] * 4))
        with self.assertRaisesRegex(ValueError, "unique"):
            correlate_scores(self.scores.set_axis(["PC1", "PC1"], axis=1), self.outcome)
        with self.assertRaisesRegex(ValueError, "numeric"):
            correlate_scores(self.scores.astype(str).assign(PC1="not a number"), self.outcome)

    def test_invalid_permutation_parameters_raise(self) -> None:
        for name, value in (("n_permutations", -1), ("n_permutations", 1.5),
                            ("n_permutations", True), ("seed", -1), ("batch_size", 0)):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    correlate_scores(self.scores, self.outcome, **{name: value})


if __name__ == "__main__":
    unittest.main()
