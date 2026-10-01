#!/usr/bin/env python3
"""Prepare session exposures and fit separate course-level ECT LMMs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

from simnibs_parcel_analysis.clinical import _validate_ect_sessions, load_hdf5_query, map_sessions_to_hdf5
from simnibs_parcel_analysis.identifiers import base_subject_id
from simnibs_parcel_analysis.lmm import (
    aggregate_course_exposures, attach_course_covariates, build_fixed_effects, build_session_roi_matrices,
    fit_course_lmm, link_sessions_to_courses, prepare_session_dose, read_lmm_table,
)


# Edit before submitting jobs. Empty means no explicit subject exclusions.
# Use base IDs. A scan ID also excludes the whole patient.
# All sessions and courses for these subjects are removed before validation.
EXCLUDE_SUBJECTS: list[str] = ["subj-cat-010"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("--sessions-xlsx", type=Path, required=True)
    result.add_argument("--session-sheet", default="stimulus")
    result.add_argument("--outcomes", type=Path, help="CSV/XLSX; defaults to the sessions workbook")
    result.add_argument("--outcome-sheet", default="0", help="Sheet name or zero-based integer")
    result.add_argument("--query-path", type=Path, required=True)
    result.add_argument("--model-type", choices=["skin_single", "skin_double"], required=True)
    result.add_argument("--simulation-type", choices=["static", "adaptive", "step_0", "step_2"], required=True)
    result.add_argument("--subject-scan-map", type=Path)
    result.add_argument("--query-filter", action="append", default=[], metavar="KEY=VALUE")
    result.add_argument("--placement-alias", action="append", default=[], metavar="BT=BL")
    result.add_argument("--acute-only", action="store_true",
                        help="Explicitly exclude nonpositive/invalid series before validation")
    result.add_argument("--allow-unmatched-sessions", action="store_true",
                        help="Export and exclude sessions outside all outcome windows")
    result.add_argument("--subject-info", type=Path,
                        help="Optional patient covariates; prefer course-specific age in outcomes")
    result.add_argument("--dose-column", choices=["ef_sf", "percent_charge"], default="ef_sf")
    result.add_argument("--aggregation", choices=["sum", "mean"], required=True,
                        help="Explicit exposure-history assumption")
    result.add_argument("--roi-table", type=Path,
                        help="Wide CSV: hdf5_fn, ROI_1, ..., ROI_M; unscaled field magnitudes")
    result.add_argument("--atlas-name", help="Required with --roi-table")
    result.add_argument("--roi-statistic",
                        help="Required with --roi-table, e.g. p95_unweighted, p95_spatial_weighted, mean")
    result.add_argument("--exclude-rois", nargs="+", default=[])
    result.add_argument("--roi-model", choices=["components", "scaled"], default="components")
    result.add_argument("--numeric-covariates", nargs="+", default=[])
    result.add_argument("--categorical-covariates", nargs="+", default=[])
    result.add_argument("--age-sex-interaction", action="store_true")
    result.add_argument("--no-session-count", action="store_true")
    result.add_argument("--no-standardize", action="store_true")
    result.add_argument("--ml", action="store_true", help="Use ML rather than REML")
    result.add_argument("--optimizer", choices=["powell", "lbfgs", "bfgs"], default="powell")
    result.add_argument("--maxiter", type=int, default=2000)
    result.add_argument("--prepare-only", action="store_true", help="Export data/matrices without fitting any model")
    result.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty directory, separate from PCA outputs")
    return result


def _pairs(values: list[str]) -> dict[str, str]:
    parsed = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key.strip() or not value.strip() or key.strip() in parsed:
            raise ValueError(f"Expected a unique KEY=VALUE argument, got {item!r}")
        parsed[key.strip()] = value.strip()
    return parsed


def _sheet(value: str) -> str | int:
    return int(value) if value.isdecimal() else value


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def _fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def _subject_exclusion_mask(table: pd.DataFrame, subjects: set[str], *, label: str) -> pd.Series:
    """Match whole patients while leaving other rows unchanged for validation."""
    if "subjid" not in table:
        raise KeyError(f"{label} is missing subjid")

    def is_excluded(value: object) -> bool:
        try:
            return base_subject_id(value) in subjects
        except ValueError:
            # Do not drop malformed IDs or reject blank unfinished outcome rows here.
            return False

    return table["subjid"].map(is_excluded).astype(bool)


def run(args: argparse.Namespace) -> Path:
    if args.roi_table and (not args.atlas_name or not args.roi_statistic):
        raise ValueError("--roi-table requires --atlas-name and --roi-statistic")
    if not args.roi_table and (args.atlas_name or args.roi_statistic or args.exclude_rois):
        raise ValueError("Atlas metadata/ROI exclusions require --roi-table")

    destination = args.output_dir
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    manifest = {"status": "started", "arguments": vars(args).copy(),
                "observation_unit": "one pre/post CGI outcome per patient treatment series",
                "outcome": "cgi_end - cgi_start; negative is improvement for severity scores",
                "dose_definition": args.dose_column, "ef_sf": "pulse_width_ms * frequency_hz * percent_charge",
                "ef_scaled_definition": "unscaled ROI E * session ef_sf; composite index, not physical V/m",
                "window_definition": "inclusive date_start through date_end (acute)",
                "measurement_dates": "window boundaries are not assumed to be actual CGI assessment dates",
                "random_effect": "patient intercept", "aggregation": args.aggregation,
                "inference": "Gaussian LMM, asymptotic Wald; not causal or small-sample corrected",
                "versions": {"python": sys.version, "pandas": pd.__version__}}
    _json(destination / "manifest.json", manifest)

    try:
        if not isinstance(EXCLUDE_SUBJECTS, (list, tuple, set)):
            raise ValueError("EXCLUDE_SUBJECTS must be a list of subject IDs, not a single string")
        try:
            exclude_subjects = {base_subject_id(value) for value in EXCLUDE_SUBJECTS}
        except ValueError as error:
            raise ValueError(f"Invalid EXCLUDE_SUBJECTS entry: {error}") from error
        manifest["subject_exclusions"] = {"subjects": sorted(exclude_subjects)}

        outcomes_path = args.outcomes or args.sessions_xlsx
        files = [args.sessions_xlsx, outcomes_path, args.query_path,
                 args.subject_scan_map, args.subject_info, args.roi_table]
        manifest["input_files"] = [_fingerprint(path) for path in dict.fromkeys(p for p in files if p is not None)]

        from simnibs_parcel_analysis import lmm
        manifest["code_files"] = [_fingerprint(Path(__file__)), _fingerprint(Path(lmm.__file__))]

        # Load both clinical tables before validating individual sessions/courses.
        sessions = read_lmm_table(args.sessions_xlsx, sheet_name=_sheet(args.session_sheet))
        outcomes = read_lmm_table(outcomes_path, sheet_name=_sheet(args.outcome_sheet))
        outcomes = outcomes.drop(columns=["mrn"] if "mrn" in outcomes else [])

        # Apply EXCLUDE_SUBJECTS to both tables using person-level IDs.
        session_mask = _subject_exclusion_mask(sessions, exclude_subjects, label="ECT-session table")
        course_mask = _subject_exclusion_mask(outcomes, exclude_subjects, label="Outcome table")
        excluded_subject_sessions = sessions.loc[session_mask].copy()
        excluded_subject_courses = outcomes.loc[course_mask].copy()

        found = {base_subject_id(value) for value in pd.concat([
            excluded_subject_sessions["subjid"], excluded_subject_courses["subjid"],
        ])}
        not_found = sorted(exclude_subjects - found)
        manifest["subject_exclusions"].update(
            matched_subjects=sorted(found), not_found=not_found,
            excluded_sessions=len(excluded_subject_sessions), excluded_courses=len(excluded_subject_courses))

        # Preserve original input row indices in the exclusion audit files.
        for table, name, row_label in [
            (excluded_subject_sessions, "excluded_subject_sessions.csv", "session_row_index"),
            (excluded_subject_courses, "excluded_subject_courses.csv", "outcome_row_index"),
        ]:
            table.assign(exclusion_reason="EXCLUDE_SUBJECTS").to_csv(
                destination / name, index=True, index_label=row_label)

        sessions = sessions.loc[~session_mask].copy()
        outcomes = outcomes.loc[~course_mask].copy()

        if exclude_subjects:
            print(f"EXCLUDE_SUBJECTS: {sorted(exclude_subjects)}; excluded "
                  f"{len(excluded_subject_sessions)} sessions and {len(excluded_subject_courses)} courses")
            if not_found:
                print(f"EXCLUDE_SUBJECTS IDs absent from both clinical tables: {not_found}")
            if sessions.empty or outcomes.empty:
                raise ValueError("No sessions or no outcome courses remain after applying EXCLUDE_SUBJECTS")

        if "series_num" not in sessions:
            raise KeyError("ECT-session table is missing series_num")

        # Parse only a temporary mask; all other values stay unchanged for strict validation.
        # String conversion keeps False from being mistaken for numeric zero.
        series_numbers = pd.to_numeric(sessions["series_num"].astype("string"), errors="coerce")
        zero_series = series_numbers.eq(0).fillna(False)
        excluded_rows = sessions.index[zero_series].tolist()
        manifest["zero_series_filter"] = {"n_excluded": len(excluded_rows), "row_indices": excluded_rows}
        if excluded_rows:
            print(f"Excluding {len(excluded_rows)} stimulus rows with series_num == 0 at rows: {excluded_rows}")

        sessions = _validate_ect_sessions(sessions.loc[~zero_series].copy(), acute_only=args.acute_only)
        linked = link_sessions_to_courses(sessions, outcomes,
                                          allow_unmatched_sessions=args.allow_unmatched_sessions)
        courses = linked.courses

        linked.excluded_courses.to_csv(destination / "excluded_courses.csv", index=True, index_label="outcome_row_index")
        linked.excluded_sessions.to_csv(destination / "excluded_sessions.csv", index=False)
        manifest["cgi_eligibility"] = {"required": "both cgi_start and cgi_end",
                                       "excluded_courses": len(linked.excluded_courses),
                                       "excluded_sessions": int(linked.excluded_sessions["exclusion_reason"].eq("incomplete_cgi_course").sum())}
        if not linked.excluded_courses.empty:
            print(f"Excluded {len(linked.excluded_courses)} courses missing CGI scores; see excluded_courses.csv")
        if not linked.excluded_sessions.empty:
            print(f"Excluded {len(linked.excluded_sessions)} sessions; see excluded_sessions.csv")

        if args.subject_info:
            courses = attach_course_covariates(courses, read_lmm_table(args.subject_info))

        query = load_hdf5_query(args.query_path, model_type=args.model_type, simulation_type=args.simulation_type,
                                query_filters=_pairs(args.query_filter))
        mapped = map_sessions_to_hdf5(linked.sessions, query, model_type=args.model_type,
                                      simulation_type=args.simulation_type, subject_scan_map=args.subject_scan_map,
                                      electrode_placement_map=_pairs(args.placement_alias))
        mapped = prepare_session_dose(mapped)
        selected = query.loc[query["hdf5_fn"].isin(mapped["hdf5_fn"])].drop_duplicates()
        mapped.to_csv(destination / "session_mapping.csv", index=False)
        selected.to_csv(destination / "selected_hdf5_inventory.csv", index=False)

        fields = None
        if args.roi_table:
            fields = build_session_roi_matrices(mapped, read_lmm_table(args.roi_table), atlas=args.atlas_name,
                                                exclude_rois=args.exclude_rois)
            fields.raw.to_csv(destination / "session_roi_raw.csv")
            fields.scaled.to_csv(destination / "session_roi_scaled.csv")
            atlas_slug = re.sub(r"[^A-Za-z0-9_.-]", "_", args.atlas_name)
            if atlas_slug in {".", ".."}:
                raise ValueError("Invalid atlas directory name")
            for patient, rows in mapped.groupby("subjid_base", sort=True):
                folder = destination / "session_matrices" / atlas_slug / patient
                folder.mkdir(parents=True)
                ids = rows["session_id"]
                rows[["session_id", *[c for c in mapped if c != "session_id"]]].to_csv(folder / "rows.csv", index=False)
                fields.raw.loc[ids].to_csv(folder / "raw_E.csv")
                fields.scaled.loc[ids].to_csv(folder / "scaled_E.csv")

        model_courses, matrices = aggregate_course_exposures(mapped, courses, aggregation=args.aggregation,
                                                            dose_column=args.dose_column, fields=fields)
        model_courses.to_csv(destination / "course_exposures.csv", index=False)
        for name, matrix in matrices.items():
            matrix.to_csv(destination / f"course_{name}.csv")

        counts = model_courses.groupby("subjid_base").size().rename("n_courses")
        counts.to_csv(destination / "patient_course_counts.csv")
        manifest["counts"] = {"patients": len(counts), "courses": len(model_courses), "sessions": len(mapped),
                              "repeated_patients": int(counts.gt(1).sum()), "unique_hdf5": mapped["hdf5_fn"].nunique(),
                              "rois": 0 if fields is None else fields.raw.shape[1],
                              "excluded_courses": len(excluded_subject_courses) + len(linked.excluded_courses),
                              "excluded_sessions": len(excluded_subject_sessions) + len(linked.excluded_sessions)}
        print(json.dumps(manifest["counts"]))
        manifest["status"] = "prepared"
        _json(destination / "manifest.json", manifest)
        if args.prepare_only:
            return destination

        roi_names = [None] if fields is None else list(fields.raw.columns)
        results = []
        fits = destination / "models"
        fits.mkdir()
        roi_catalog = []

        for number, roi in enumerate(roi_names, start=1):
            table = model_courses.copy()
            if roi is not None:
                for name, matrix in matrices.items():
                    table[name] = matrix.loc[table["treatment_course_id"], roi].to_numpy()

            roi_model = "none" if roi is None else args.roi_model
            design, scaling = build_fixed_effects(table, roi_model=roi_model,
                                                  numeric_covariates=args.numeric_covariates,
                                                  categorical_covariates=args.categorical_covariates,
                                                  include_session_count=not args.no_session_count,
                                                  age_sex_interaction=args.age_sex_interaction,
                                                  standardize=not args.no_standardize)
            folder = fits / f"model_{number:04d}"
            folder.mkdir()
            table.to_csv(folder / "model_data.csv", index=False)
            design.to_csv(folder / "fixed_effects_design.csv")
            scaling.to_csv(folder / "predictor_scaling.csv", index=False)

            label = "clinical" if roi is None else roi
            roi_catalog.append({"model": folder.name, "atlas": args.atlas_name, "roi": label, "roi_model": roi_model})
            pd.DataFrame(roi_catalog).to_csv(destination / "model_index.csv", index=False)

            fit = fit_course_lmm(table, design, scaling=scaling, reml=not args.ml,
                                 optimizer=args.optimizer, maxiter=args.maxiter)
            fit.coefficients.to_csv(folder / "coefficients.csv", index=False)
            fit.predictions.to_csv(folder / "predictions.csv", index=False)
            _json(folder / "diagnostics.json", fit.diagnostics)
            results.append(fit.coefficients.assign(atlas=args.atlas_name, roi=label, model=folder.name))
            print(f"Fitted {folder.name}: {label}; ICC={fit.diagnostics['icc']:.4g}")

        combined = pd.concat(results, ignore_index=True)
        if fields is not None:
            from statsmodels.stats.multitest import multipletests
            combined["q_value_bh"] = float("nan")
            exposure_terms = ["ef_raw", "ef_dose"] if args.roi_model == "components" else ["ef_scaled"]
            for term in exposure_terms:
                mask = combined["term"].eq(term)
                combined.loc[mask, "q_value_bh"] = multipletests(combined.loc[mask, "p_value"], method="fdr_bh")[1]
            manifest["multiplicity"] = "BH across all retained ROIs separately for each E-related term in this run"

        combined.to_csv(destination / "lmm_coefficients.csv", index=False)
        manifest["status"] = "complete"
        manifest["n_models"] = len(results)
        _json(destination / "manifest.json", manifest)
        return destination

    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        _json(destination / "manifest.json", manifest)
        raise


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    print(f"LMM output: {run(args).resolve()}")


if __name__ == "__main__":
    main()