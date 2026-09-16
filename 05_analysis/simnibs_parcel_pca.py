"""Backward-compatible entry point for the reorganized analysis package.

New code should import from :mod:`simnibs_parcel_analysis` directly.
"""

from simnibs_parcel_analysis import *  # noqa: F401,F403
from simnibs_parcel_analysis import subject_id_from_path as infer_subject_id
