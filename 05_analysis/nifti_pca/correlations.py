"""Spearman and outcome permutation tests for one PCA basis at a time.

The caller supplies complete, independent sample rows from a single PCA fit.
This module neither groups observations nor writes files. Array-valued result
columns contain JSON strings so callers can save the complete audit record in a
CSV. Permutation p-values are two-sided and have no multiple-testing correction.
"""

from __future__ import annotations

import json
from numbers import Integral

import numpy as np
import pandas as pd
import scipy
from scipy.stats import permutation_test, rankdata, spearmanr


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _json(value: object) -> str:
    """Preserve ordinary index labels, including NumPy integer/string scalars."""
    def encode(item: object) -> object:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, pd.Timestamp):
            return item.isoformat()
        raise TypeError(f"Cannot serialize sample label of type {type(item).__name__}.")

    return json.dumps(value, default=encode, allow_nan=False, separators=(",", ":"))


def _exact_test_requested(n_samples: int, n_resamples: int) -> bool:
    """Compare n! with the request without constructing a huge factorial."""
    pairings = 1
    for factor in range(2, n_samples + 1):
        pairings *= factor
        if pairings > n_resamples:
            return False
    return pairings <= n_resamples


def _validate_inputs(
    scores: pd.DataFrame,
    outcome: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(scores, pd.DataFrame) or not isinstance(outcome, pd.Series):
        raise TypeError("scores must be a DataFrame and outcome must be a Series.")
    if len(scores) < 3:
        raise ValueError("At least three independent sample rows are required.")
    if scores.shape[1] == 0:
        raise ValueError("At least one component score column is required.")
    if scores.columns.has_duplicates:
        raise ValueError("Component labels must be unique.")
    if any(not isinstance(label, str) or not label for label in scores.columns):
        raise ValueError("Component labels must be nonempty strings.")

    for label, index in (("scores", scores.index), ("outcome", outcome.index)):
        if isinstance(index, pd.MultiIndex):
            raise ValueError(f"{label} must have a single stable sample_id index.")
        if index.has_duplicates or index.hasnans:
            raise ValueError(f"{label} sample labels must be unique and nonmissing.")

    missing = scores.index.difference(outcome.index, sort=False)
    extra = outcome.index.difference(scores.index, sort=False)
    if len(missing) or len(extra):
        raise ValueError(
            "scores and outcome must contain exactly the same sample labels; "
            f"missing outcomes={missing.tolist()!r}, extra outcomes={extra.tolist()!r}."
        )

    try:
        numeric_scores = scores.apply(pd.to_numeric, errors="raise")
        numeric_outcome = pd.to_numeric(outcome.reindex(scores.index), errors="raise")
        if np.iscomplexobj(numeric_scores.to_numpy()) or np.iscomplexobj(
            numeric_outcome.to_numpy()
        ):
            raise ValueError("Complex values cannot be correlated.")
        x = numeric_scores.to_numpy(dtype=np.float64, na_value=np.nan)
        y = numeric_outcome.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError("Component scores and outcome must be real numeric values.") from error

    invalid = scores.columns[~np.isfinite(x).all(axis=0)].tolist()
    if invalid or not np.isfinite(y).all():
        raise ValueError(
            "Missing or nonfinite values cannot be dropped implicitly; "
            f"invalid components={invalid!r}, invalid outcome={not np.isfinite(y).all()}."
        )
    constant = scores.columns[np.all(x == x[0], axis=0)].tolist()
    if constant:
        raise ValueError(f"Constant component scores cannot be correlated: {constant!r}.")
    if np.all(y == y[0]):
        raise ValueError("A constant outcome cannot be correlated.")
    return x, y


def correlate_scores(
    scores: pd.DataFrame,
    outcome: pd.Series,
    *,
    n_permutations: int = 9999,
    seed: int = 42,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Correlate each signed PC score with an outcome for one sample group.

    Outcome rows are aligned to score rows by their unique sample labels; their
    label sets must match exactly. Neither missing observations nor constant
    components are silently excluded. All supplied rows are independent samples.

    SciPy permutes the pairing of the outcome ranks with the fixed component
    ranks. Average ranks handle ties. Centering and normalizing each rank vector
    makes their dot product equal to Spearman's rho. Each component starts with
    the same RNG state, sharing outcome shuffles across components. A request of
    at least n! permutations uses SciPy's exact enumeration; Monte Carlo tests
    use SciPy's add-one correction. The two-sided p-value doubles the smaller
    one-sided tail, following scipy.stats.permutation_test.

    Set n_permutations=0 for asymptotic Spearman p-values only. The returned
    p_permutation is then NaN, the null distribution is [], and no RNG is used.
    All outputs belong in a caller-selected RESULTS_DIR/<analysis_name> folder.
    """
    n_permutations = _nonnegative_integer(n_permutations, "n_permutations")
    seed = _nonnegative_integer(seed, "seed")
    batch_size = _nonnegative_integer(batch_size, "batch_size")
    if batch_size == 0:
        raise ValueError("batch_size must be at least one.")
    x, y = _validate_inputs(scores, outcome)
    sample_keys = _json(scores.index.tolist())

    x_ranks = rankdata(x, method="average", axis=0)
    y_ranks = rankdata(y, method="average")
    x_ranks -= x_ranks.mean(axis=0)
    y_ranks -= y_ranks.mean()
    x_ranks /= np.linalg.norm(x_ranks, axis=0)
    y_ranks /= np.linalg.norm(y_ranks)

    enabled = n_permutations > 0
    exact = enabled and _exact_test_requested(len(scores), n_permutations)
    y_ranks_json = _json(y_ranks.tolist())
    outcome_json = _json(y.tolist())
    rows = []

    for component_index, component in enumerate(scores.columns):
        rho, p_value = spearmanr(x[:, component_index], y, nan_policy="raise")
        fixed_x_ranks = x_ranks[:, component_index]

        def statistic(permuted_y_ranks: np.ndarray, axis: int = -1) -> np.ndarray:
            samples = np.moveaxis(permuted_y_ranks, axis, -1)
            return np.sum(samples * fixed_x_ranks, axis=-1)

        row = {
            "component": component,
            "n": len(scores),
            "rho": float(rho),
            "p_value": float(p_value),
            "p_permutation": np.nan,
            "permutation_enabled": enabled,
            "permutation_statistic": np.nan,
            "permutation_null_distribution": "[]",
            "permutation_type": "pairings",
            "permutation_vectorized": True,
            "permutation_n_resamples_requested": n_permutations,
            "permutation_n_resamples_actual": 0,
            "permutation_batch_requested": batch_size,
            "permutation_batch_effective": 0,
            "permutation_alternative": "two-sided",
            "permutation_axis": 0,
            "permutation_seed": seed,
            "permutation_rng_bit_generator": None,
            "permutation_rng_initial_state": None,
            "permutation_exact": exact,
            "permutation_statistic_definition": (
                "dot product of centered, L2-normalized average ranks"
            ),
            "permutation_data_y_ranks": y_ranks_json,
            "permutation_fixed_x_ranks": _json(fixed_x_ranks.tolist()),
            "permutation_sample_keys": sample_keys,
            "permutation_input_outcome": outcome_json,
            "permutation_input_scores": _json(x[:, component_index].tolist()),
            "permutation_shared_shuffles": True,
            "permutation_two_sided_convention": "min(1, 2 * min(p_less, p_greater))",
            "permutation_add_one_correction": enabled and not exact,
            "permutation_multiple_testing_adjustment": "none",
            "samples_assumed_independent": True,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
        }

        if enabled:
            rng = np.random.default_rng(seed)
            row["permutation_rng_bit_generator"] = type(rng.bit_generator).__name__
            row["permutation_rng_initial_state"] = _json(rng.bit_generator.state)
            result = permutation_test(
                data=(y_ranks,),
                statistic=statistic,
                permutation_type="pairings",
                vectorized=True,
                n_resamples=n_permutations,
                batch=batch_size,
                alternative="two-sided",
                axis=0,
                rng=rng,
            )
            null = np.asarray(result.null_distribution, dtype=np.float64).reshape(-1)
            row.update(
                p_permutation=float(result.pvalue),
                permutation_statistic=float(result.statistic),
                permutation_null_distribution=_json(null.tolist()),
                permutation_n_resamples_actual=len(null),
                permutation_batch_effective=min(batch_size, len(null)),
            )
        rows.append(row)

    return pd.DataFrame(rows)
