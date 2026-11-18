#!/usr/bin/env python3
"""Correlate CGI change with Yeo2011 ROI means within each model/step subset.

Standalone version of mni_yeo2011_correlations.ipynb. Each observation is one
(person_id, series_num) treatment series. All available model_type x step
configurations are analyzed separately; results are stacked in one CSV.

The default analysis uses regional means and seven bilateral network means.
Network means give each member ROI equal weight, before correlation. Use
--no-network-means for regional ROIs only. Other spatial metrics in the master
(median, max, p95, etc.) are not used by this notebook's mean-feature analysis.
Missing/nonfinite values, constant features, and incomplete/duplicate ROI rows
raise errors. No observations are silently dropped. P-values are unadjusted;
the analysis does not adjust for covariates or repeated persons across series.

Examples:
    export RESULTS_DIR=/path/to/results
    python mni_yeo2011_correlations.py
    python mni_yeo2011_correlations.py --no-network-means
    python mni_yeo2011_correlations.py --master-path /path/to/master.csv \
        --output-path "$RESULTS_DIR/mni_yeo2011/data/correlations.csv"

Requires Python >= 3.10, numpy, pandas, and scipy.
"""

import argparse
from pathlib import Path
import re

if __package__ in (None, ""):
    from _results import (
        add_analysis_argument, bootstrap_cli, correlation_output_path,
        default_master_path, normalization_metadata, validate_output_path,
    )
else:
    from ._results import (
        add_analysis_argument, bootstrap_cli, correlation_output_path,
        default_master_path, normalization_metadata, validate_output_path,
    )

if __name__ == "__main__":
    bootstrap_cli()

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_FIELD_NAME = "magnE_mean"
NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")


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
        ["field_name", "metric", "model_type", "step"], sort=False
    ).agg(
        n_series=("n", "first"),
        regional_roi_tests=("is_regional", "sum"),
        bilateral_network_tests=("is_bilateral", "sum"),
        total_tests=("x", "size"),
    )


def add_network_means_both_hemis(
    df: pd.DataFrame, field_name: str = DEFAULT_FIELD_NAME
) -> pd.DataFrame:
    """Average all regional means belonging to each bilateral parent network."""
    result = df.copy()

    for network in NETWORKS:
        columns = []
        for hemisphere in ("LH", "RH"):
            prefix = f"{field_name}_7Networks_{hemisphere}_{network}"
            region_columns = [
                col for col in df.columns
                if col == prefix or col.startswith(prefix + "_")
            ]
            if not region_columns:
                raise ValueError(
                    f"No regional mean columns for {hemisphere} {network}."
                )
            columns.extend(region_columns)

        result[f"{field_name}_{network}_both_hemis"] = df[columns].mean(
            axis=1, skipna=False
        )

    return result


def run_spearman(
    df: pd.DataFrame, y: str, x: list[str] | str | None = None
) -> pd.DataFrame:
    """Compute one Spearman test per feature, requiring complete observations."""
    if df.columns.has_duplicates:
        raise ValueError("Dataframe contains duplicate column names.")
    if y not in df.columns:
        raise KeyError(f"Outcome column {y!r} is missing.")

    if x is None:
        features = [col for col in df.columns if col != y]
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
        rho, p_value = spearmanr(data[feature], data[y], nan_policy="raise")
        results.append({
            "x": feature,
            "y": y,
            "n": len(data),
            "rho": float(rho),
            "p_value": float(p_value),
        })

    return pd.DataFrame(results)


def annotate_correlation_features(results: pd.DataFrame) -> pd.DataFrame:
    """Parse feature labels using the exact source field, never underscore guesses."""
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
                f"Feature {feature!r} does not preserve field_name={source_field!r}."
            )

        label = feature[len(prefix):]
        regional = re.fullmatch(rf"7Networks_(LH|RH)_({parents})(?:_(.+))?", label)
        bilateral = re.fullmatch(rf"({parents})_both_hemis", label)

        if regional:
            hemi, network, component = regional.groups()
            metadata.append((label, component or network, hemi, network))
        elif bilateral:
            network = bilateral.group(1)
            metadata.append((None, network, "both", network))
        else:
            raise ValueError(f"Unrecognized ROI/network label in feature {feature!r}.")

    parsed = pd.DataFrame(
        metadata, columns=["roi_org", "roi", "hemi", "network"], index=result.index
    )
    for column in parsed:
        if column in result.columns:
            agrees = result[column].fillna("").eq(parsed[column].fillna(""))
            if not agrees.all():
                raise ValueError(
                    f"Existing {column!r} metadata conflicts with x and field_name."
                )
        result[column] = parsed[column]

    return result


def save_spearman_table(df: pd.DataFrame, filename: str | Path) -> pd.DataFrame:
    """Save source-aware features and retain provenance and extra result columns."""
    output_path = validate_output_path(filename)
    columns = [
        "x", "y", "n", "rho", "p_value", "field_name", "metric", "model_type", "step"
    ]
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"Missing correlation columns: {sorted(missing)}")
    result = annotate_correlation_features(df)
    leading = [*columns, "roi_org", "roi", "hemi", "network"]
    extra = [column for column in result.columns if column not in leading]
    result = result[[*leading, *extra]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def roi_master_to_mean_tables(
    master: pd.DataFrame,
    field_name: str = DEFAULT_FIELD_NAME,
    y: str = "cgi_change",
) -> dict:
    """Pivot ROI rows into one mean-feature table per model_type x step subset."""
    series_keys = ["person_id", "series_num"]
    config_keys = ["model_type", "step"]
    required = [
        *series_keys, *config_keys, "field_name", "roi_org", "mean", y
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

    # CGI must agree across every ROI/configuration belonging to a series.
    outcome_counts = selected.groupby(series_keys)[y].nunique(dropna=False)
    if outcome_counts.ne(1).any():
        raise ValueError(f"Inconsistent {y!r} values within treatment series.")

    roi_names = selected["roi_org"].drop_duplicates().tolist()
    tables = {}

    for configuration, group in selected.groupby(config_keys, sort=False):
        roi_counts = group.groupby(series_keys)["roi_org"].nunique()
        if roi_counts.ne(len(roi_names)).any():
            raise ValueError(
                f"{configuration}: incomplete ROI coverage for a series."
            )

        # Pivot directly: duplicate observations are never averaged.
        wide = group.pivot(
            index=series_keys, columns="roi_org", values="mean"
        ).reindex(columns=roi_names)

        # Repeated outcomes were checked above before deduplication.
        outcome = (
            group[series_keys + [y]]
            .drop_duplicates(series_keys)
            .set_index(series_keys)[y]
        )
        wide = wide.rename(columns=lambda roi: f"{field_name}_{roi}")
        wide.columns.name = None

        # Align outcomes by person/series keys, not row position.
        wide.insert(0, y, outcome.reindex(wide.index))
        tables[configuration] = wide

    return tables


def run_master_spearman(
    master: pd.DataFrame,
    field_name: str = DEFAULT_FIELD_NAME,
    y: str = "cgi_change",
    include_network_means: bool = True,
) -> pd.DataFrame:
    """Run all ROI-mean Spearman tests separately within every model/step subset."""
    tables = roi_master_to_mean_tables(master, field_name=field_name, y=y)
    results = []

    for (model_type, step), table in tables.items():
        if include_network_means:
            table = add_network_means_both_hemis(table, field_name=field_name)
        features = [column for column in table.columns if column != y]

        try:
            correlations = run_spearman(table, y=y, x=features)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"model_type={model_type!r}, step={step!r}: {exc}"
            ) from exc

        results.append(correlations.assign(
            model_type=model_type,
            step=step,
            field_name=field_name,
            metric="mean",
        ))

    return pd.concat(results, ignore_index=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Run Spearman correlations between cgi_change and each ROI mean "
            "separately for every model_type x step subset in the master CSV."
        )
    )
    add_analysis_argument(parser)
    parser.add_argument(
        "--field-name", default=DEFAULT_FIELD_NAME,
        help="Source field to select from the master (default: %(default)s).",
    )
    parser.add_argument(
        "--master-path", type=Path,
        help=(
            "Master CSV (default: $RESULTS_DIR/<analysis-name>/data/"
            "yeo2011_<field-name>_series_ROI_master.csv)."
        ),
    )
    parser.add_argument(
        "--output-path", type=Path,
        help=(
            "Output CSV inside $RESULTS_DIR/<analysis-name>/ "
            "(default: normalization-aware filename in its data/ directory)."
        ),
    )
    parser.add_argument(
        "--no-network-means", action="store_true",
        help="Test regional ROI means only; omit the seven bilateral network means.",
    )
    args = parser.parse_args(argv)
    if not args.field_name or args.field_name != args.field_name.strip():
        parser.error("--field-name must be nonempty with no surrounding whitespace.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    field_name = args.field_name
    master_path = args.master_path
    if master_path is None:
        master_path = default_master_path(args.analysis_name, field_name)

    if args.output_path is not None:
        validate_output_path(args.output_path, analysis_name=args.analysis_name)

    master = load_roi_master(master_path)
    metadata = normalization_metadata(master, field_name, source_path=master_path)

    include_network_means = not args.no_network_means
    scope = "ROImean_and_network" if include_network_means else "ROImean_only"
    output_path = args.output_path
    if output_path is None:
        output_path = correlation_output_path(
            args.analysis_name, field_name, scope, metadata,
        )
    output_path = validate_output_path(output_path, analysis_name=args.analysis_name)
    if output_path.resolve() == master_path.resolve():
        raise ValueError("Output path must differ from the input master CSV.")

    print(f"Loaded {len(master):,} ROI rows from {master_path}", flush=True)
    results = run_master_spearman(
        master, field_name=field_name, y="cgi_change",
        include_network_means=include_network_means,
    )
    results["master_path"] = str(master_path.resolve())
    for column, value in metadata.items():
        results[column] = value
    annotated = save_spearman_table(results, output_path)
    print(summarize_spearman_results(annotated).to_string(), flush=True)
    print(f"Saved {len(annotated):,} correlations to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
