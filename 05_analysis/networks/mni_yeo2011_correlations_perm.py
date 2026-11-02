#!/usr/bin/env python3
"""
mni_yeo2011_correlations_perm.py

Run Spearman correlations and permutation tests between cgi_change and ROI means,
separately within every observed model_type × step subset.

Each (person_id, series_num) is treated as an independent sample.

By default, features include:
    - Each regional ROI's mean.
    - Seven bilateral network means, calculated before correlation.

Saved results include:
    - Original Spearman rho and asymptotic two-sided p_value.
    - SciPy permutation statistic and two-sided p_permutation.
    - All permutation settings, seeds, RNG states, and software versions.
    - Exact/Monte Carlo status and actual permutation counts.
    - Complete null distributions, numeric permutation inputs, and sample keys.
      Array-valued fields are serialized as JSON inside the results CSV.

P-values are unadjusted. Missing/nonfinite values, constant features, duplicate
ROI observations, and incomplete ROI coverage raise errors.

Examples:
    python mni_yeo2011_correlations_perm.py

    python mni_yeo2011_correlations_perm.py \
        --n-permutations 9999 --seed 42 --batch-size 256

    python mni_yeo2011_correlations_perm.py \
        --no-network-means

    python mni_yeo2011_correlations_perm.py \
        --master-path /path/to/master.csv \
        --output-path /path/to/permutation_results.csv

Requires Python >= 3.10, numpy, pandas, and scipy >= 1.15.
"""

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import scipy
from scipy.stats import permutation_test, rankdata, spearmanr


DEFAULT_FIELD_NAME = "magnE_mean"
DEFAULT_MASTER_DIR = Path(
    "/projects/p32903/Alex2/ect-workflow/05_analysis/networks/data"
)
NETWORKS = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)


def load_roi_master(filename: str | Path) -> pd.DataFrame:
    """Read the existing master without filtering or altering its rows."""
    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(f"Master CSV does not exist: {path}")

    master = pd.read_csv(path)

    if master.empty:
        raise ValueError(f"Master CSV contains no observations: {path}")

    return master


def summarize_spearman_results(results: pd.DataFrame) -> pd.DataFrame:
    """Report observation counts and regional/bilateral tests by configuration."""
    annotated = results.assign(
        is_regional=results["hemi"].isin(["LH", "RH"]),
        is_bilateral=results["hemi"].eq("both"),
    )

    return annotated.groupby(
        ["field_name", "metric", "model_type", "step"],
        sort=False,
    ).agg(
        n_series=("n", "first"),
        regional_roi_tests=("is_regional", "sum"),
        bilateral_network_tests=("is_bilateral", "sum"),
        total_tests=("x", "size"),
    )


def add_network_means_both_hemis(
    df: pd.DataFrame,
    field_name: str = DEFAULT_FIELD_NAME,
) -> pd.DataFrame:
    """Give every regional mean equal weight in its bilateral network mean."""
    result = df.copy()

    for network in NETWORKS:
        columns = []

        for hemisphere in ("LH", "RH"):
            prefix = f"{field_name}_7Networks_{hemisphere}_{network}"

            region_columns = [
                column
                for column in df.columns
                if column == prefix or column.startswith(prefix + "_")
            ]

            if not region_columns:
                raise ValueError(
                    f"No regional mean columns for {hemisphere} {network}."
                )

            columns.extend(region_columns)

        result[f"{field_name}_{network}_both_hemis"] = df[columns].mean(
            axis=1,
            skipna=False,
        )

    return result


def run_spearman(
    df: pd.DataFrame,
    y: str,
    x: list[str] | str | None = None,
) -> pd.DataFrame:
    """Compute the original two-sided Spearman tests on complete observations."""
    if df.columns.has_duplicates:
        raise ValueError("Dataframe contains duplicate column names.")

    if y not in df.columns:
        raise KeyError(f"Outcome column {y!r} is missing.")

    if x is None:
        features = [column for column in df.columns if column != y]
    elif isinstance(x, str):
        features = [x]
    else:
        features = list(x)

    if not features:
        raise ValueError("No feature columns selected.")

    if y in features or len(features) != len(set(features)):
        raise ValueError("Features must be unique and must not include y.")

    data = df[[y, *features]].apply(pd.to_numeric, errors="raise")

    if len(data) < 3:
        raise ValueError("At least three observations are required.")

    invalid = data.columns[
        ~np.isfinite(data.to_numpy(dtype=float)).all(axis=0)
    ].tolist()

    if invalid:
        raise ValueError(f"Missing or non-finite values in: {invalid}")

    constant = data.columns[data.nunique().lt(2)].tolist()

    if constant:
        raise ValueError(f"Constant columns cannot be correlated: {constant}")

    results = []

    for feature in features:
        rho, p_value = spearmanr(
            data[feature],
            data[y],
            nan_policy="raise",
        )

        results.append({
            "x": feature,
            "y": y,
            "n": len(data),
            "rho": float(rho),
            "p_value": float(p_value),
        })

    return pd.DataFrame(results)


def annotate_correlation_features(results: pd.DataFrame) -> pd.DataFrame:
    """Parse feature labels using the exact source field."""
    required = {"x", "field_name", "metric"}
    missing = required - set(results.columns)

    if missing:
        raise KeyError(f"Missing feature identity columns: {sorted(missing)}")

    result = results.copy()
    parents = "|".join(NETWORKS)
    metadata = []

    for feature, source_field, metric_name in result[
        ["x", "field_name", "metric"]
    ].itertuples(index=False, name=None):
        if (
            not isinstance(source_field, str)
            or not source_field
            or source_field != source_field.strip()
        ):
            raise ValueError(f"Invalid source field_name: {source_field!r}")

        if not isinstance(metric_name, str) or not metric_name:
            raise ValueError(f"Invalid ROI summary metric: {metric_name!r}")

        prefix = f"{source_field}_"

        if not isinstance(feature, str) or not feature.startswith(prefix):
            raise ValueError(
                f"Feature {feature!r} does not preserve "
                f"field_name={source_field!r}."
            )

        label = feature[len(prefix):]

        regional = re.fullmatch(
            rf"7Networks_(LH|RH)_({parents})(?:_(.+))?",
            label,
        )
        bilateral = re.fullmatch(
            rf"({parents})_both_hemis",
            label,
        )

        if regional:
            hemi, network, component = regional.groups()
            metadata.append((label, component or network, hemi, network))
        elif bilateral:
            network = bilateral.group(1)
            metadata.append((None, network, "both", network))
        else:
            raise ValueError(
                f"Unrecognized ROI/network label in feature {feature!r}."
            )

    parsed = pd.DataFrame(
        metadata,
        columns=["roi_org", "roi", "hemi", "network"],
        index=result.index,
    )

    for column in parsed:
        if column in result.columns:
            agrees = result[column].fillna("").eq(
                parsed[column].fillna("")
            )

            if not agrees.all():
                raise ValueError(
                    f"Existing {column!r} metadata conflicts with "
                    "x and field_name."
                )

        result[column] = parsed[column]

    return result


def save_spearman_table(
    df: pd.DataFrame,
    filename: str | Path,
) -> pd.DataFrame:
    """Save annotations and preserve every permutation parameter/result column."""
    columns = [
        "x",
        "y",
        "n",
        "rho",
        "p_value",
        "field_name",
        "metric",
        "model_type",
        "step",
    ]

    missing = set(columns) - set(df.columns)

    if missing:
        raise KeyError(f"Missing correlation columns: {sorted(missing)}")

    # Annotate the complete dataframe instead of selecting only fixed columns.
    result = annotate_correlation_features(df)

    leading = [*columns, "roi_org", "roi", "hemi", "network"]
    extra = [
        column for column in result.columns
        if column not in leading
    ]
    result = result[[*leading, *extra]]

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return result


def roi_master_to_mean_tables(
    master: pd.DataFrame,
    field_name: str = DEFAULT_FIELD_NAME,
    y: str = "cgi_change",
) -> dict:
    """Pivot ROI rows into one mean-feature table per model_type × step."""
    series_keys = ["person_id", "series_num"]
    config_keys = ["model_type", "step"]

    required = [
        *series_keys,
        *config_keys,
        "field_name",
        "roi_org",
        "mean",
        y,
    ]

    if master.columns.has_duplicates:
        raise ValueError("Master contains duplicate column names.")

    missing = set(required) - set(master.columns)

    if missing:
        raise KeyError(f"Missing master columns: {sorted(missing)}")

    selected = master.loc[master["field_name"].eq(field_name)].copy()

    if selected.empty:
        raise ValueError(f"No master rows for field_name={field_name!r}.")

    identity = [*series_keys, *config_keys, "roi_org"]

    if selected[identity].isna().any().any():
        raise ValueError("Missing series, configuration, or ROI identifiers.")

    if selected.duplicated(identity).any():
        raise ValueError("Duplicate series/configuration/ROI rows.")

    # Every ROI/configuration belonging to a series must have the same outcome.
    outcome_counts = selected.groupby(series_keys)[y].nunique(
        dropna=False
    )

    if outcome_counts.ne(1).any():
        raise ValueError(
            f"Inconsistent {y!r} values within treatment series."
        )

    roi_names = selected["roi_org"].drop_duplicates().tolist()
    tables = {}

    for configuration, group in selected.groupby(
        config_keys,
        sort=False,
    ):
        roi_counts = group.groupby(series_keys)["roi_org"].nunique()

        if roi_counts.ne(len(roi_names)).any():
            raise ValueError(
                f"{configuration}: incomplete ROI coverage for a series."
            )

        # Duplicate observations are rejected above, never averaged.
        wide = group.pivot(
            index=series_keys,
            columns="roi_org",
            values="mean",
        ).reindex(columns=roi_names)

        outcome = (
            group[series_keys + [y]]
            .drop_duplicates(series_keys)
            .set_index(series_keys)[y]
        )

        wide = wide.rename(
            columns=lambda roi: f"{field_name}_{roi}"
        )
        wide.columns.name = None

        # Align outcomes using the series identifiers.
        wide.insert(0, y, outcome.reindex(wide.index))
        tables[configuration] = wide

    return tables


def run_master_spearman_permutation(
    master: pd.DataFrame,
    field_name: str = DEFAULT_FIELD_NAME,
    y: str = "cgi_change",
    include_network_means: bool = True,
    n_permutations: int = 9_999,
    seed: int = 42,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Run original Spearman tests plus SciPy permutation tests.

    Outcomes are shuffled freely across independent samples within each
    model_type × step subset.

    p_value:
        Original asymptotic two-sided Spearman p-value.

    p_permutation:
        SciPy's two-sided permutation p-value.
    """
    for name, value in (
        ("n_permutations", n_permutations),
        ("batch_size", batch_size),
    ):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    tables = roi_master_to_mean_tables(
        master,
        field_name=field_name,
        y=y,
    )

    rows = []

    for (model_type, step), table in tables.items():
        table = table.sort_index()

        if include_network_means:
            table = add_network_means_both_hemis(
                table,
                field_name=field_name,
            )

        features = [
            column for column in table.columns
            if column != y
        ]

        # Retain the original validation and asymptotic correlation results.
        observed_results = run_spearman(
            table,
            y=y,
            x=features,
        )

        # Spearman correlation equals Pearson correlation of average-tie ranks.
        # Rank once because permutations only change the outcome ordering.
        x_ranks = rankdata(
            table[features].to_numpy(dtype=float),
            axis=0,
            method="average",
        )
        y_ranks = rankdata(
            table[y].to_numpy(dtype=float),
            method="average",
        )

        x_ranks -= x_ranks.mean(axis=0)
        y_ranks -= y_ranks.mean()

        x_ranks /= np.linalg.norm(x_ranks, axis=0)
        y_ranks /= np.linalg.norm(y_ranks)

        # SciPy automatically enumerates all permutations when the requested
        # count reaches n! for this single-vector pairings test.
        exact_test = n_permutations >= math.factorial(len(table))

        sample_keys_json = table.index.to_frame(
            index=False
        ).to_json(orient="records")

        y_ranks_json = json.dumps(
            y_ranks.tolist(),
            allow_nan=False,
        )

        for feature_index, observed in enumerate(
            observed_results.to_dict(orient="records")
        ):
            fixed_x_ranks = x_ranks[:, feature_index]

            def statistic(permuted_y_ranks, axis=-1):
                samples = np.moveaxis(
                    permuted_y_ranks,
                    axis,
                    -1,
                )
                return np.sum(
                    samples * fixed_x_ranks,
                    axis=-1,
                )

            # Reset the seed so all features with the same sample ordering
            # use the same outcome permutations.
            rng = np.random.default_rng(seed)
            initial_rng_state = json.dumps(
                rng.bit_generator.state
            )

            test = permutation_test(
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

            null_distribution = np.asarray(
                test.null_distribution,
                dtype=float,
            ).reshape(-1)

            row = observed.copy()

            row.update({
                "field_name": field_name,
                "metric": "mean",
                "model_type": model_type,
                "step": step,
                "include_network_means": include_network_means,
                "sample_unit": "person_id,series_num",
                "samples_assumed_independent": True,

                # SciPy result fields.
                "permutation_statistic": float(test.statistic),
                "p_permutation": float(test.pvalue),
                "permutation_null_distribution": json.dumps(
                    null_distribution.tolist(),
                    allow_nan=False,
                ),

                # SciPy call parameters and effective settings.
                "permutation_type": "pairings",
                "permutation_vectorized": True,
                "permutation_n_resamples_requested": int(n_permutations),
                "permutation_n_resamples_actual": len(null_distribution),
                "permutation_batch_requested": int(batch_size),
                "permutation_batch_effective": min(
                    batch_size,
                    len(null_distribution),
                ),
                "permutation_alternative": "two-sided",
                "permutation_axis": 0,
                "permutation_seed": seed,
                "permutation_rng_bit_generator": (
                    type(rng.bit_generator).__name__
                ),
                "permutation_rng_initial_state": initial_rng_state,
                "permutation_exact": exact_test,

                # Statistic definition and actual numeric inputs.
                "permutation_statistic_definition": (
                    "dot product of centered, L2-normalized average ranks"
                ),
                "permutation_data_y_ranks": y_ranks_json,
                "permutation_fixed_x_ranks": json.dumps(
                    fixed_x_ranks.tolist(),
                    allow_nan=False,
                ),
                "permutation_sample_keys": sample_keys_json,

                # Interpretation and software provenance.
                "permutation_two_sided_convention": (
                    "min(1, 2 * min(p_less, p_greater))"
                ),
                "permutation_add_one_correction": not exact_test,
                "permutation_multiple_testing_adjustment": "none",
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
            })

            rows.append(row)

    return annotate_correlation_features(
        pd.DataFrame(rows)
    )


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Spearman correlations and permutation tests between "
            "cgi_change and ROI means, separately for every "
            "model_type × step subset."
        )
    )

    parser.add_argument(
        "--field-name",
        default=DEFAULT_FIELD_NAME,
        help="Source field to select (default: %(default)s).",
    )
    parser.add_argument(
        "--master-path",
        type=Path,
        help=(
            f"Master CSV (default: {DEFAULT_MASTER_DIR}/"
            "yeo2011_<field-name>_series_ROI_master.csv)."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Results CSV (default: field/scope-specific permutation "
            "filename beside the selected master)."
        ),
    )
    parser.add_argument(
        "--no-network-means",
        action="store_true",
        help="Test regional ROI means only.",
    )

    # Additional command-line options for permutation testing.
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=9_999,
        help="Requested SciPy resample count (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Permutations processed per batch (default: %(default)s).",
    )

    args = parser.parse_args(argv)

    if not args.field_name or args.field_name != args.field_name.strip():
        parser.error(
            "--field-name must be nonempty with no surrounding whitespace."
        )

    if args.n_permutations < 1:
        parser.error("--n-permutations must be positive.")

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    if args.seed < 0:
        parser.error("--seed must be nonnegative.")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    field_name = args.field_name

    master_path = args.master_path
    if master_path is None:
        master_path = DEFAULT_MASTER_DIR / (
            f"yeo2011_{field_name}_series_ROI_master.csv"
        )

    include_network_means = not args.no_network_means

    scope = (
        "ROImean_and_network"
        if include_network_means
        else "ROImean_only"
    )

    output_path = args.output_path
    if output_path is None:
        output_path = master_path.parent / (
            f"spearman_{field_name}_{scope}_permutation_annotated.csv"
        )

    if output_path.resolve() == master_path.resolve():
        raise ValueError(
            "Output path must differ from the input master CSV."
        )

    master = load_roi_master(master_path)

    print(
        f"Loaded {len(master):,} ROI rows from {master_path}",
        flush=True,
    )

    results = run_master_spearman_permutation(
        master,
        field_name=field_name,
        y="cgi_change",
        include_network_means=include_network_means,
        n_permutations=args.n_permutations,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    results["master_path"] = str(master_path.resolve())

    annotated = save_spearman_table(
        results,
        output_path,
    )

    print(
        summarize_spearman_results(annotated).to_string(),
        flush=True,
    )
    print(
        f"Saved {len(annotated):,} correlations to {output_path}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())