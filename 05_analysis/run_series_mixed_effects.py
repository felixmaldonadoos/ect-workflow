#!/usr/bin/env python3
"""Fit one crossed age/sex mixed model per ROI or PC from a saved series master."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from analysis_paths import resolve_analysis_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--master-path", type=Path, required=True)
    parser.add_argument("--feature-type", choices=["roi", "pca"], required=True)
    parser.add_argument("--field-name", default="magnE_mean")
    parser.add_argument("--analysis-name", required=True)
    parser.add_argument("--features", nargs="+", help="Full ROI labels or PC names; overrides PCA selected flags.")
    parser.add_argument("--pca-scope", choices=["selected", "all"], default="selected")
    parser.add_argument("--prepare-only", action="store_true", help="Export inputs and design/group validation without fitting.")
    parser.add_argument("--reml", action="store_true", help="Use REML; default is maximum likelihood (ML).")
    parser.add_argument("--optimizer", choices=["powell", "lbfgs", "bfgs", "cg"], default="powell")
    parser.add_argument("--maxiter", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.maxiter < 1:
        parser.error("--maxiter must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.field_name):
        parser.error("--field-name must contain only letters, digits, underscores, hyphens, or periods")
    if args.features and (len(set(args.features)) != len(args.features) or
                          any(not value.strip() or value != value.strip() for value in args.features)):
        parser.error("--features must be unique, nonempty, unpadded ROI/PC names")
    if args.feature_type == "roi" and args.pca_scope != "selected":
        parser.error("--pca-scope applies only to PCA masters")
    try:
        resolve_analysis_dir(args.analysis_name)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    from series_models.workflow import run_analysis

    try:
        result = run_analysis(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Series mixed effects: {error}", file=sys.stderr)
        return 2
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
