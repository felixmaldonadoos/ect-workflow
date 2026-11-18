#!/usr/bin/env python3
"""Compare saved ML feature models with their shared age/sex null model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from analysis_paths import lmm_analysis_name, resolve_analysis_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--null-analysis-dir", type=Path, required=True)
    parser.add_argument("--feature-analysis-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--analysis-name", required=True)
    parser.add_argument("--no-plots", action="store_true", help="Export comparison tables without figures.")
    parser.add_argument("--top-n", type=int, default=20, help="Feature models shown beside the null in the ranking figure.")
    parser.add_argument("--dpi", type=int, default=800)
    args = parser.parse_args(argv)
    if args.top_n < 1 or args.dpi < 1:
        parser.error("--top-n and --dpi must be positive")
    try:
        args.analysis_name = lmm_analysis_name(args.analysis_name, "aic")
        resolve_analysis_dir(args.analysis_name)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        from series_models.aic_workflow import run_aic_comparison
        result = run_aic_comparison(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Series AIC comparison: {error}", file=sys.stderr)
        return 2
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
