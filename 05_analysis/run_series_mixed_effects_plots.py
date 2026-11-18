#!/usr/bin/env python3
"""Generate fitted-model plots from completed series LMM analyses without refitting."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from analysis_paths import lmm_analysis_name, resolve_analysis_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--analysis-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--analysis-name", required=True)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args(argv)
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    try:
        args.analysis_name = lmm_analysis_name(args.analysis_name, "plots")
        resolve_analysis_dir(args.analysis_name)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        from series_models.fit_plot_workflow import run_saved_fit_plots
        result = run_saved_fit_plots(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Series fit plots: {error}", file=sys.stderr)
        return 2
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
