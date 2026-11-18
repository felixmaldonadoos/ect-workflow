#!/usr/bin/env python3
"""Fit one ROI/PC model with continuous age and fixed or random sex effects."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from analysis_paths import lmm_analysis_name, resolve_analysis_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--master-path", type=Path, required=True)
    parser.add_argument("--feature-type", choices=["roi", "pca"], required=True)
    parser.add_argument("--field-name", default="magnE_mean")
    parser.add_argument("--analysis-name", required=True)
    parser.add_argument("--output-component", choices=["null", "roi_raw", "roi_brain_mean", "pca_raw", "pca_brain_mean"],
                        help="Child folder within RESULTS_DIR/lmm/<analysis-name>; managed by the all-master launcher.")
    parser.add_argument("--sex-effect", choices=["random", "fixed"], default="random",
                        help="Sex random intercept (default) or categorical fixed effect; fixed uses Gaussian OLS.")
    parser.add_argument("--null-model", action="store_true", help="Fit one shared age/sex null, without E or E:ef_sf.")
    parser.add_argument("--features", nargs="+", help="Full ROI labels or PC names; overrides PCA selected flags.")
    parser.add_argument("--pca-scope", choices=["selected", "all"], default="selected")
    parser.add_argument("--prepare-only", action="store_true", help="Export inputs and design/group validation without fitting.")
    parser.add_argument("--no-fit-plots", action="store_true", help="Skip automatic fitted-model PDFs, PNGs, and plotting data.")
    parser.add_argument("--plot-dpi", type=int, default=120, help="Resolution of automatic fit PNGs (default: 120).")
    parser.add_argument("--reml", action="store_true", help="Use REML; default is maximum likelihood (ML).")
    parser.add_argument("--optimizer", choices=["powell", "lbfgs", "bfgs", "cg"], default="powell")
    parser.add_argument("--maxiter", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.maxiter < 1:
        parser.error("--maxiter must be positive")
    if args.plot_dpi < 1:
        parser.error("--plot-dpi must be positive")
    if args.sex_effect == "fixed" and args.reml:
        parser.error("--sex-effect fixed uses ML Gaussian OLS and cannot be combined with --reml")
    if args.output_component == "null" and not args.null_model:
        parser.error("--output-component null requires --null-model")
    if args.null_model and args.output_component not in (None, "null"):
        parser.error("A null model must use --output-component null")
    if args.null_model and args.features:
        parser.error("--features does not apply to --null-model")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.field_name):
        parser.error("--field-name must contain only letters, digits, underscores, hyphens, or periods")
    if args.features and (len(set(args.features)) != len(args.features) or
                          any(not value.strip() or value != value.strip() for value in args.features)):
        parser.error("--features must be unique, nonempty, unpadded ROI/PC names")
    if args.feature_type == "roi" and args.pca_scope != "selected":
        parser.error("--pca-scope applies only to PCA masters")
    try:
        args.analysis_name = lmm_analysis_name(args.analysis_name, args.output_component or ("null" if args.null_model else None))
        resolve_analysis_dir(args.analysis_name)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.null_model:
            from series_models.null_workflow import run_null_analysis as run_analysis
        else:
            from series_models.workflow import run_analysis
        result = run_analysis(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Series mixed effects: {error}", file=sys.stderr)
        return 2
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
