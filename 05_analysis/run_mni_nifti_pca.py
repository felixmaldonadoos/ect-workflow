#!/usr/bin/env python3
"""Run NIfTI PCA and series-level correlations under RESULTS_DIR/analysis_name."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from nifti_pca.workflow import run_analysis


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_BRAIN_MASK = Path(
    "/projects/p32903/Alex2/ecstim/resources/templates/MNI152_T1_1mm_brain.nii.gz"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage", choices=["all", "prepare", "correlate", "variance", "plot"], default="all")
    parser.add_argument("--analysis-name", default="mni_nifti_pca")
    parser.add_argument("--field-name", default="magnE_mean")
    parser.add_argument("--series-csv", type=Path, help="Series manifest or existing ROI master; raw NIfTI paths are used.")
    parser.add_argument("--query-path", type=Path, default=Path("/projects/p32903/Alex2/ect-results-tools/query_result_2.csv"))
    parser.add_argument("--clinical-path", type=Path, default=Path("/projects/p32903/Alex2/datasets/STU00225089/.clincal_notes/cgi_batch_latest_20261029_1207.xlsx"))
    parser.add_argument("--subject-scan-map", type=Path, default=ANALYSIS_DIR / "subject_scan_map.json")
    step_options = parser.add_mutually_exclusive_group()
    step_options.add_argument("--steps", nargs="+", default=["step_0", "step_9"])
    step_options.add_argument("--all-steps", action="store_true", help="Clinical/query route: discover steps separately per model.")
    parser.add_argument("--exclude-subjects", nargs="*", default=["subj-cat-010-001"], help="Exclusions for the clinical/query route.")
    parser.add_argument("--brain-mask", type=Path, default=DEFAULT_BRAIN_MASK)
    parser.add_argument("--normalizations", nargs="+", choices=["raw", "brain_mean"], default=["raw", "brain_mean"])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--variance-threshold", type=float, help="Cumulative variance for selecting correlation PCs (default: 0.96).")
    selection.add_argument("--n-components", type=int, help="Number of leading PCs to correlate; all nonzero PCs are exported.")
    parser.add_argument("--resample", action="store_true", help="Linearly resample mismatched fields onto the fixed brain-mask grid.")
    parser.add_argument("--n-permutations", type=int, default=9999, help="0 disables permutation testing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--plot", action="store_true", help="Also render component maps and correlation plots in the all stage; variance plots are always saved after preparation.")
    parser.add_argument("--background-path", type=Path, default=DEFAULT_BRAIN_MASK)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if args.variance_threshold is None:
        args.variance_threshold = 0.96
    if not 0 < args.variance_threshold <= 1:
        parser.error("--variance-threshold must be in (0, 1].")
    if args.n_components is not None and args.n_components < 1:
        parser.error("--n-components must be positive.")
    if args.n_permutations < 0 or args.seed < 0 or args.batch_size < 1 or args.dpi < 1:
        parser.error("Require nonnegative permutations/seed and positive batch-size/dpi.")
    if len(args.normalizations) != len(set(args.normalizations)):
        parser.error("--normalizations must not repeat a mode.")
    if not args.field_name or args.field_name != args.field_name.strip() or any(c in args.field_name for c in "/\\"):
        parser.error("--field-name must be nonempty and cannot contain path separators.")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = run_analysis(args)
    print(f"Analysis results: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
