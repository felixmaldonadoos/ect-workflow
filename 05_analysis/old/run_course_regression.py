#!/usr/bin/env python3
"""Fit the assigned clinical null and per-ROI age/sex mixed models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from simnibs_parcel_analysis.clinical import _validate_ect_sessions, load_hdf5_query, map_sessions_to_hdf5
from simnibs_parcel_analysis.course_regression import NULL_FORMULA, RANDOM_EFFECTS, ROI_FORMULA, fit_course_regressions, prepare_course_predictors
from simnibs_parcel_analysis.identifiers import base_subject_id
from simnibs_parcel_analysis.lmm import attach_course_covariates, build_session_roi_matrices, link_sessions_to_courses, prepare_session_dose, read_lmm_table
from simnibs_parcel_analysis.session_table import build_session_table


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--clinical-file", type=Path, required=True)
    result.add_argument("--session-sheet", default="stimulus")
    result.add_argument("--outcome-sheet", default="Sheet1")
    result.add_argument("--subject-info", type=Path, help="Optional patient table supplying age and sex")
    result.add_argument("--query-path", type=Path, required=True)
    result.add_argument("--subject-scan-map", type=Path)
    result.add_argument("--model-type", choices=["skin_single", "skin_double"], required=True)
    result.add_argument("--simulation-type", choices=["static", "adaptive"], required=True)
    result.add_argument("--roi-table", type=Path, required=True, help="Unscaled E: hdf5_fn followed by ROI columns")
    result.add_argument("--atlas-name", default="HCP_MMP1")
    result.add_argument("--aggregation", choices=["mode", "mean", "sum"], default="mode")
    result.add_argument("--exclude-rois", nargs="*", default=[])
    result.add_argument("--exclude-subjects", nargs="*", default=[])
    result.add_argument("--placement-alias", action="append", default=[], metavar="BT=BL")
    result.add_argument("--query-filter", action="append", default=[], metavar="KEY=VALUE")
    result.add_argument("--allow-unmatched-sessions", action="store_true")
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def _pairs(items: list[str]) -> dict[str, str]:
    result = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key.strip() or not value.strip() or key.strip() in result:
            raise ValueError(f"Expected a unique KEY=VALUE, got {item!r}")
        result[key.strip()] = value.strip()
    return result


def run(args: argparse.Namespace) -> Path:
    destination = args.output_dir
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {"status": "started", "arguments": vars(args), "observation_unit": "treatment course",
                "null_formula": NULL_FORMULA, "roi_fixed_formula": ROI_FORMULA, "roi_random_effects": RANDOM_EFFECTS,
                "patient_random_effect": False, "roi_estimation": "ML; crossed exact-age and sex random intercepts",
                "EF_scaled": "session E * session ef_sf, then mode selection or mean/sum within course",
                "ef_sf": "pulse_width_ms * frequency_hz * percent_charge", "standardization": "none"}

    try:
        sessions = read_lmm_table(args.clinical_file, sheet_name=args.session_sheet)
        outcomes = read_lmm_table(args.clinical_file, sheet_name=args.outcome_sheet)
        excluded = {base_subject_id(value) for value in args.exclude_subjects}
        for table, label in [(sessions, "sessions"), (outcomes, "courses")]:
            def selected(value):
                try:
                    return base_subject_id(value) in excluded
                except ValueError:
                    return False  # Other malformed IDs remain subject to ordinary validation.
            mask = table["subjid"].map(selected) if excluded else pd.Series(False, index=table.index)
            table.loc[mask].drop(columns=["mrn"], errors="ignore").to_csv(destination / f"excluded_subject_{label}.csv", index=True)
            table.drop(index=table.index[mask], inplace=True)
        # Preserve the existing, explicitly requested series-zero exclusion.
        series = pd.to_numeric(sessions["series_num"].astype("string"), errors="coerce")
        zero = series.eq(0).fillna(False)
        sessions.loc[zero].to_csv(destination / "excluded_zero_series.csv", index=True)
        sessions = _validate_ect_sessions(sessions.loc[~zero].copy())
        outcomes = outcomes.drop(columns=["mrn"], errors="ignore")
        linked = link_sessions_to_courses(sessions, outcomes, allow_unmatched_sessions=args.allow_unmatched_sessions)
        linked.excluded_courses.to_csv(destination / "excluded_courses.csv", index=True)
        linked.excluded_sessions.to_csv(destination / "excluded_sessions.csv", index=False)
        courses = linked.courses
        query = load_hdf5_query(args.query_path, model_type=args.model_type, simulation_type=args.simulation_type, query_filters=_pairs(args.query_filter))
        mapped = map_sessions_to_hdf5(linked.sessions, query, model_type=args.model_type, simulation_type=args.simulation_type,
                                      subject_scan_map=args.subject_scan_map, electrode_placement_map=_pairs(args.placement_alias))
        mapped = prepare_session_dose(mapped)
        fields = build_session_roi_matrices(mapped, read_lmm_table(args.roi_table), atlas=args.atlas_name, exclude_rois=args.exclude_rois)
        build_session_table(mapped, fields, model_type=args.model_type, simulation_type=args.simulation_type).to_csv(destination / "session_table.csv", index=False)
        mapped.to_csv(destination / "session_mapping.csv", index=False)
        fields.raw.to_csv(destination / "session_EF_roi.csv")
        fields.scaled.to_csv(destination / "session_EF_scaled.csv")
        metadata.update(session_table_written=True, n_sessions=len(mapped), n_rois=len(fields.raw.columns))
        if args.subject_info:
            courses = attach_course_covariates(courses, read_lmm_table(args.subject_info))
        clinical, raw, scaled, audit = prepare_course_predictors(mapped, courses, fields, aggregation=args.aggregation)
        clinical.to_csv(destination / "course_data.csv")
        raw.to_csv(destination / "course_EF_roi.csv")
        scaled.to_csv(destination / "course_EF_scaled.csv")
        audit.to_csv(destination / "session_mapping.csv", index=False)
        metadata.update(n_courses=len(clinical), n_patients=clinical["subjid_base"].nunique(), n_sessions=len(mapped), n_rois=len(raw.columns))
        if args.prepare_only:
            metadata["status"] = "prepared"
            return destination
        coefficients, fits = fit_course_regressions(clinical, raw, scaled)
        coefficients.assign(atlas=args.atlas_name).to_csv(destination / "coefficients.csv", index=False)
        fits.assign(atlas=args.atlas_name).to_csv(destination / "fit_status.csv", index=False)
        metadata["multiplicity"] = "BH across requested ROIs separately for EF_roi and EF_scaled; failed fits count as p=1"
        failed = fits.loc[fits["status"].eq("failed")]
        if not failed.empty:
            raise RuntimeError(f"{len(failed)} model(s) failed; see fit_status.csv. Valid fits remain in coefficients.csv.\n" + failed[["model", "roi", "error"]].to_string(index=False))
        metadata["status"] = "complete"
        return destination
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        (destination / "analysis.json").write_text(json.dumps(metadata, indent=2, default=str, allow_nan=False) + "\n")


if __name__ == "__main__":
    print(f"Course regression output: {run(parser().parse_args()).resolve()}")
