#!/usr/bin/env python3
r"""Attach exactly one simulation file to each clinical session row.

Notebook API:
    filtered_query = query_df.loc[(query_df['asf'] == 'asf_5') & (query_df['step'] == 'step_2')].copy()
    mapped = map_session_files(sessions, query=filtered_query, filename='tdcs_uq_gpc.hdf5',
                              query_filters={'model_type_dir': 'skin_double'},
                              subject_scan_map='subject_scan_map.json')
    # Omitting query loads query_results.csv in the current working directory.

CLI:
    python prepare_session_file_table.py --clinical-file cgi_batch.xlsx --query-path query_results.csv \
        --filename tdcs_uq_gpc.hdf5 --model-type skin_double --step step_2 --asf asf_5 \
        --subject-scan-map subject_scan_map.json --output session_file_table.csv

Every original session column, row, and notebook index is preserved. Series zero
and missing CGI do not exclude a session. No ROI extraction, dose calculation,
implicit scan choice, or implicit latest-run selection is performed. Selection
must leave one file per subject/placement; distinct placements can use distinct
files. Inventory paths are not opened unless check_files=True.
By default, only full_file_path is added to the original session columns.
Use include_provenance=True (CLI: --include-provenance) for matching details.

Dependencies: pandas; openpyxl is additionally needed for XLSX/XLSM input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

SUBJECT = re.compile(r'(?P<base>subj-[A-Za-z0-9]+-\d{3})(?:-(?P<scan>\d{3}))?\Z')
PLACEMENTS = frozenset(('RUL', 'BL', 'BT', 'BF'))
SUBJECT_COLUMNS = ('subject_dir', 'modeled_subjid', 'inferred_subjid', 'subjid')
PLACEMENT_COLUMNS = ('electrode_configuration', 'electrode_placement', 'simulation_electrode_placement')
PATH_COLUMNS = ('full_file_path', 'file_path', 'hdf5_fn')
GENERATED_COLUMNS = ('subjid_base', 'modeled_subjid', 'simulation_electrode_placement', 'file_path',
                     'query_step_result', 'file_type', 'mapping_status', 'mapping_reason')
FILE_TYPES = {'.nii': 'nifti', '.nii.gz': 'nifti', '.msh': 'msh', '.hdf5': 'hdf5', '.h5': 'hdf5'}


class AmbiguousMappingError(ValueError):
    """More than one scan or file remains after explicit selection."""

    def __init__(self, message: str, candidates: pd.DataFrame) -> None:
        super().__init__(message)
        self.candidates = candidates


class MissingMappingError(ValueError):
    """One or more valid sessions lack a matching file."""

    def __init__(self, message: str, unmatched: pd.DataFrame) -> None:
        super().__init__(message)
        self.unmatched = unmatched


def _columns(columns: Any, source: str) -> list[str]:
    result = list(columns)
    if any(not isinstance(column, str) or not column or column != column.strip() for column in result):
        raise ValueError(f'{source} needs nonempty string column names without surrounding whitespace.')
    if len(set(result)) != len(result):
        raise ValueError(f'{source} contains duplicate column names.')
    return result


def read_table(source: pd.DataFrame | str | Path, *, sheet_name: str = 'stimulus') -> pd.DataFrame:
    """Copy a DataFrame or strictly read CSV/XLSX values without numeric coercion."""
    if isinstance(source, pd.DataFrame):
        _columns(source.columns, 'DataFrame')
        return source.copy(deep=True)
    path = Path(source).expanduser()
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle, strict=True)
            columns = _columns(next(reader, ()), str(path))
            if not columns:
                raise ValueError(f'Empty CSV: {path}')
            rows = []
            for row in reader:
                if len(row) != len(columns):
                    raise ValueError(f'{path}:{reader.line_num}: expected {len(columns)} cells, got {len(row)}.')
                rows.append(row)
        return pd.DataFrame(rows, columns=columns)
    if path.suffix.lower() not in ('.xlsx', '.xlsm'):
        raise ValueError(f'Expected CSV, XLSX, or XLSM input: {path}')
    raw = pd.read_excel(path, sheet_name=sheet_name, engine='openpyxl', header=None, dtype=object, keep_default_na=False)
    if raw.empty:
        raise ValueError(f'Empty sheet {sheet_name!r}: {path}')
    columns = _columns(raw.iloc[0].tolist(), f'{path}:{sheet_name}')
    result = raw.iloc[1:].copy().reset_index(drop=True)
    result.columns = columns
    return result


def _text(value: Any) -> str:
    if isinstance(value, os.PathLike):
        return str(value)
    if not pd.api.types.is_scalar(value):
        raise ValueError(f'Table cells must be scalar values, got {type(value).__name__}.')
    return '' if pd.isna(value) else str(value)


def _subject(value: Any) -> tuple[str, str, bool]:
    if not isinstance(value, str) or not (match := SUBJECT.fullmatch(value.strip())):
        raise ValueError(f'Invalid subject ID {value!r}; expected subj-cat-012 or subj-cat-012-001.')
    return value.strip(), match['base'], match['scan'] is not None


def _placement(value: Any) -> str:
    if not isinstance(value, str) or (result := value.strip().upper()) not in PLACEMENTS:
        raise ValueError(f'Invalid electrode placement {value!r}; expected {sorted(PLACEMENTS)}.')
    return result


def _path(value: Any) -> str:
    if not isinstance(value, (str, os.PathLike)) or not str(value) or not Path(value).is_absolute():
        raise ValueError(f'Inventory file paths must be nonempty absolute paths, got {value!r}.')
    return str(Path(value))


def _identity(row: pd.Series, columns: tuple[str, ...], normalizer: Any) -> Any:
    present = [key for key in columns if key in row.index and _text(row[key]) != '']
    if not present:
        raise ValueError(f'Query needs one nonempty column from {columns}.')
    values = [normalizer(row[key]) for key in present]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f'Conflicting query columns {present}: {[row[key] for key in present]}.')
    return values[0]


def _query_subject(row: pd.Series) -> tuple[str, str, bool]:
    present = [key for key in SUBJECT_COLUMNS if key in row.index and _text(row[key]) != '']
    if not present:
        raise ValueError(f'Query needs one nonempty column from {SUBJECT_COLUMNS}.')
    values = [_subject(row[key]) for key in present]
    bases = {base for _, base, _ in values}
    scans = {identifier for identifier, _, is_scan in values if is_scan}
    if len(bases) != 1 or len(scans) > 1:
        raise ValueError(f'Conflicting query subject columns {present}: {[row[key] for key in present]}.')
    # A base-ID alias and a modeled-ID alias are compatible; the modeled ID is more specific.
    return next((value for value in values if value[2]), values[0])


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _scan_map(source: Mapping[str, str] | str | Path | None) -> dict[str, str]:
    if source is None:
        return {}
    if not isinstance(source, Mapping):
        with Path(source).expanduser().open(encoding='utf-8') as handle:
            source = json.load(handle, object_pairs_hook=_json_pairs)
    if not isinstance(source, Mapping):
        raise ValueError('subject_scan_map must be a JSON object or mapping of base IDs to modeled scan IDs.')
    result = {}
    for key, value in source.items():
        identifier, base, is_scan = _subject(key)
        modeled, modeled_base, has_scan = _subject(value)
        if is_scan or not has_scan or modeled_base != base or identifier in result:
            raise ValueError(f'Invalid subject_scan_map entry {key!r}: {value!r}; use a unique base ID and its modeled scan.')
        result[identifier] = modeled
    return result


def _filter_values(filters: Mapping[str, Any] | None) -> dict[str, tuple[str, ...]]:
    result = {}
    for key, value in (filters or {}).items():
        if not isinstance(key, str) or not key:
            raise ValueError('Query filter keys must be nonempty column names.')
        values = tuple(value) if isinstance(value, (tuple, list)) else (value,)
        if not values or any(_text(item) == '' for item in values):
            raise ValueError(f'Query filter {key} needs nonempty values.')
        result[key] = tuple(_text(item) for item in values)
    return result


def _candidates(query: pd.DataFrame, *, query_filters: Mapping[str, Any] | None,
                filename: str | None, file_type: str | None) -> tuple[list[dict], tuple[str, ...]]:
    if file_type is not None and file_type not in set(FILE_TYPES.values()):
        raise ValueError(f'Unknown file_type {file_type!r}; use nifti, msh, or hdf5.')
    if filename is not None and (not filename or Path(filename).name != filename):
        raise ValueError('filename must be one exact filename, without directory components.')
    filters = _filter_values(query_filters)
    unknown = set(filters) - set(query.columns) - {'filename', 'extension'}
    if unknown:
        raise ValueError(f'Unknown query filter columns: {sorted(unknown)}.')
    for names in (SUBJECT_COLUMNS, PLACEMENT_COLUMNS, PATH_COLUMNS):
        if not set(names).intersection(query.columns):
            raise ValueError(f'Query needs one column from {names}.')
    # Filename and extension can be derived from legacy exports that lack these columns.
    selected = query.copy(deep=True)
    for key, values in filters.items():
        if key not in ('filename', 'extension'):
            selected = selected.loc[selected[key].map(_text).isin(values)]
    identities = set((*SUBJECT_COLUMNS, *PLACEMENT_COLUMNS, *PATH_COLUMNS))
    provenance_names = {key: 'query_step_result_dir' if key == 'step_result' else f'query_{key}'
                        for key in query.columns if key not in identities}
    provenance_columns = tuple(_columns(provenance_names.values(), 'Generated query metadata'))
    candidates, seen = [], {}
    for position, (_, row) in enumerate(selected.iterrows()):
        path = _identity(row, PATH_COLUMNS, _path)
        name = Path(path).name
        extension = next((suffix for suffix in FILE_TYPES if name.lower().endswith(suffix)), Path(path).suffix.lower())
        if filename is not None and name != filename:
            continue
        if 'filename' in filters and name not in filters['filename']:
            continue
        if 'extension' in filters and extension not in filters['extension']:
            continue
        kind = FILE_TYPES.get(extension)
        if kind is None or (file_type is not None and kind != file_type):
            continue  # The intentionally supported candidate set excludes logs and other artifacts.
        if 'filename' in row.index and _text(row['filename']) not in ('', name):
            raise ValueError(f'Query filename disagrees with file path: {path}')
        if 'extension' in row.index and _text(row['extension']).lower() not in ('', extension):
            raise ValueError(f'Query extension disagrees with file path: {path}')
        try:
            identifier, base, _ = _query_subject(row)
            placement = _identity(row, PLACEMENT_COLUMNS, _placement)
        except ValueError as error:
            raise ValueError(f'Selected query row {position}: {error}') from error
        provenance = {output: row[key] for key, output in provenance_names.items()}
        signature = (identifier, placement, tuple((key, _text(value)) for key, value in provenance.items()))
        if path in seen:
            if seen[path] != signature:
                raise ValueError(f'One file path has conflicting subject, placement, or query metadata: {path}')
            continue
        seen[path] = signature
        candidates.append(dict(modeled_subjid=identifier, subjid_base=base, simulation_electrode_placement=placement,
                               file_path=path, query_step_result=path, file_type=kind, **provenance))
    return candidates, provenance_columns


def map_session_files(sessions: pd.DataFrame | str | Path, query: pd.DataFrame | str | Path = 'query_results.csv', *,
                      session_sheet: str = 'stimulus', query_filters: Mapping[str, Any] | None = None,
                      filename: str | None = None, file_type: str | None = None,
                      subject_scan_map: Mapping[str, str] | str | Path | None = None,
                      placement_aliases: Mapping[str, str] | None = None,
                      allow_unmatched: bool = False, check_files: bool = False,
                      include_provenance: bool = False) -> pd.DataFrame:
    """Map one selected file per subject/placement while preserving all session rows.

    query accepts a prefiltered DataFrame or file path; it defaults to
    query_results.csv in the current working directory. DataFrames are copied,
    so the supplied sessions and query are not modified.
    By default, the only added column is full_file_path, containing the matched
    absolute path as a string, including when query path cells are Path objects.
    include_provenance=True also adds scan, placement, status, and query metadata.
    Its file_path and query_step_result aliases contain the same matched path;
    the inventory's step_result directory label becomes query_step_result_dir.

    Exact query filters combine with AND; a tuple/list of values uses OR within
    a column. Multiple files are always an error, even with allow_unmatched.
    A base subject with multiple candidate scans for its required placements
    needs subject_scan_map. A subject_scan_map entry selects the modeled scan
    for every session of that base subject, even when clinical subjid contains
    a different scan suffix. The original clinical subjid remains unchanged.
    Without a map entry, an explicit modeled ID selects that exact scan and
    remains a distinct modeled subject. BT and BL are distinct unless explicitly
    aliased, e.g. placement_aliases={'BT': 'BL'}.

    Unmatched rows raise MissingMappingError, whose .unmatched contains those
    original rows with blank full_file_path values. allow_unmatched=True instead
    returns all rows, with blank full_file_path values for missing matches.
    include_provenance=True adds mapping_status and mapping_reason. Ambiguity
    raises AmbiguousMappingError with .candidates for inspection.
    """
    sessions = read_table(sessions, sheet_name=session_sheet)
    query = read_table(query)
    if missing := {'subjid', 'electrode_placement'} - set(sessions.columns):
        raise ValueError(f'Sessions are missing required columns: {sorted(missing)}.')
    scans = _scan_map(subject_scan_map)
    aliases = {}
    for key, value in (placement_aliases or {}).items():
        normalized = _placement(key)
        if normalized in aliases:
            raise ValueError(f'Duplicate placement alias: {normalized}')
        aliases[normalized] = _placement(value)
    candidates, provenance_columns = _candidates(query, query_filters=query_filters, filename=filename, file_type=file_type)
    output_columns = ('full_file_path', *GENERATED_COLUMNS, *provenance_columns) if include_provenance else ('full_file_path',)
    if overlap := set(output_columns).intersection(sessions.columns):
        raise ValueError(f'Remove generated output columns from sessions before mapping: {sorted(overlap)}.')
    information, required, bases = [], {}, {}
    for position, (_, row) in enumerate(sessions.iterrows()):
        try:
            identifier, base, is_scan = _subject(row['subjid'])
            placement = _placement(row['electrode_placement'])
        except ValueError as error:
            raise ValueError(f'Session row position {position} (index {sessions.index[position]!r}): {error}') from error
        target = aliases.get(placement, placement)
        information.append((identifier, base, is_scan, target))
        required.setdefault(identifier, set()).add(target)
        bases[identifier] = (base, is_scan)
    selected_scans = {}
    for identifier, placements in required.items():
        base, is_scan = bases[identifier]
        if base in scans:
            selected_scans[identifier] = scans[base]
            continue
        if is_scan:
            selected_scans[identifier] = identifier
            continue
        options = [item for item in candidates if item['subjid_base'] == base
                   and item['simulation_electrode_placement'] in placements]
        scan_ids = sorted({item['modeled_subjid'] for item in options})
        if len(scan_ids) > 1:
            raise AmbiguousMappingError(f'{identifier} has multiple candidate scans {scan_ids}; provide subject_scan_map. '
                                        'One scan must serve all placements for a base subject.', pd.DataFrame(options))
        selected_scans[identifier] = scan_ids[0] if scan_ids else None
    pair_mapping = {}
    for identifier, placements in required.items():
        for placement in sorted(placements):
            scan = selected_scans[identifier]
            choices = [item for item in candidates if item['modeled_subjid'] == scan
                       and item['simulation_electrode_placement'] == placement]
            if len(choices) > 1:
                paths = '\n'.join(item['file_path'] for item in choices[:8])
                raise AmbiguousMappingError(f'{identifier}, {placement}: {len(choices)} files remain after filtering. '
                                            f'Select one model/step/ASF/run/artifact. Candidates:\n{paths}', pd.DataFrame(choices))
            pair_mapping[(identifier, placement)] = choices[0] if choices else None
    records = []
    for identifier, base, _, placement in information:
        matched = pair_mapping[(identifier, placement)]
        scan = selected_scans[identifier]
        if matched is not None:
            records.append(dict(matched, mapping_status='matched', mapping_reason=''))
        else:
            reason = (f'No selected file for scan {scan}, placement {placement}.' if scan is not None
                      else f'No selected files for subject {base} and its required placements.')
            records.append(dict.fromkeys(provenance_columns, pd.NA) | dict(subjid_base=base, modeled_subjid=scan,
                           simulation_electrode_placement=placement, file_path=pd.NA, file_type=pd.NA,
                           mapping_status='unmatched', mapping_reason=reason))
    result = sessions.copy(deep=True)
    for column in output_columns:
        key = 'file_path' if column == 'full_file_path' else column
        result[column] = pd.array([record.get(key, pd.NA) for record in records], dtype=object)
    missing_mask = result['full_file_path'].isna().to_numpy()
    if missing_mask.any() and not allow_unmatched:
        positions = missing_mask.nonzero()[0]
        unmatched = result.iloc[positions].copy()
        pairs = pd.DataFrame({'subjid': unmatched['subjid'].to_numpy(),
                              'simulation_electrode_placement': [information[position][3] for position in positions]})
        pairs = pairs.drop_duplicates().to_dict('records')
        raise MissingMappingError(f'{len(unmatched)} session rows lack files for {len(pairs)} subject/placement pairs: '
                                  f'{pairs[:12]}. Use allow_unmatched=True or --allow-unmatched to retain blank mappings.', unmatched)
    if check_files:
        missing_files = sorted({record['file_path'] for record in records if record['mapping_status'] == 'matched'
                                and not Path(record['file_path']).is_file()})
        if missing_files:
            raise FileNotFoundError(f'Matched files are missing or are not regular files: {missing_files[:12]}')
    return result


def _pairs(items: list[str], *, repeated: bool = False) -> dict:
    result = {}
    for item in items:
        key, separator, value = item.partition('=')
        key, value = key.strip(), value.strip()
        if not separator or not key or not value or (not repeated and key in result):
            raise ValueError(f'Expected {"a unique " if not repeated else ""}KEY=VALUE, got {item!r}.')
        result[key] = (*result.get(key, ()), value) if repeated else value
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument('--clinical-file', type=Path, required=True)
    result.add_argument('--session-sheet', default='stimulus')
    result.add_argument('--query-path', '--query', type=Path, default=Path('query_results.csv'),
                        help='Query CSV path; default query_results.csv in the current working directory.')
    result.add_argument('--subject-scan-map', type=Path)
    result.add_argument('--filename', help='Exact basename; works even when the query lacks a filename column.')
    result.add_argument('--file-type', choices=('nifti', 'msh', 'hdf5'))
    result.add_argument('--query-filter', action='append', default=[], metavar='COLUMN=VALUE', help='Exact filter; repeat a column for OR.')
    result.add_argument('--model-type', help='Optional shorthand for query-filter model_type_dir=VALUE.')
    result.add_argument('--simulation-type', choices=('static', 'adaptive'), help='Optional shorthand for step_0 or step_2.')
    result.add_argument('--step', help='Exact step label; arbitrary step_N values supported.')
    result.add_argument('--asf', help='Optional shorthand for query-filter asf=VALUE, e.g. asf_5.')
    result.add_argument('--placement-alias', action='append', default=[], metavar='BT=BL')
    result.add_argument('--allow-unmatched', action='store_true', help='Keep missing matches as blank paths; ambiguity still fails.')
    result.add_argument('--check-files', action='store_true', help='Check matched paths on this computer; default uses cached paths only.')
    result.add_argument('--include-provenance', action='store_true', help='Add matching details; default adds only full_file_path to the session columns.')
    result.add_argument('--output', required=True, help='CSV output path or - for stdout.')
    result.add_argument('--overwrite', action='store_true', help='Atomically replace an existing CSV output.')
    return result


def _write_csv(table: pd.DataFrame, output: str, *, overwrite: bool, protected: set[Path]) -> str:
    if output == '-':
        table.to_csv(sys.stdout, index=False)
        return 'stdout'
    target = Path(output).expanduser().absolute()
    if target.suffix.lower() != '.csv':
        raise ValueError('Output must have a .csv extension, or use - for stdout.')
    if target.resolve() in protected:
        raise ValueError(f'Output would replace an input file: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.session-files-', dir=target.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as handle:
            table.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
        return str(target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(args: argparse.Namespace) -> Path | str:
    filters = _pairs(args.query_filter, repeated=True)
    step = args.step
    if args.simulation_type:
        selected_step = {'static': 'step_0', 'adaptive': 'step_2'}[args.simulation_type]
        if step is not None and step != selected_step:
            raise ValueError('--step conflicts with --simulation-type.')
        step = selected_step
    for column, value in (('model_type_dir', args.model_type), ('step', step), ('asf', args.asf)):
        if value is not None:
            if column in filters and filters[column] != (value,):
                raise ValueError(f'Shorthand conflicts with --query-filter {column}.')
            filters[column] = (value,)
    mapped = map_session_files(args.clinical_file, args.query_path, session_sheet=args.session_sheet,
                               query_filters=filters, filename=args.filename, file_type=args.file_type,
                               subject_scan_map=args.subject_scan_map, placement_aliases=_pairs(args.placement_alias),
                               allow_unmatched=args.allow_unmatched, check_files=args.check_files,
                               include_provenance=args.include_provenance)
    inputs = [args.clinical_file, args.query_path, *(item for item in (args.subject_scan_map,) if item is not None)]
    protected = {Path(item).expanduser().resolve() for item in inputs}
    protected.update(Path(value).resolve() for value in mapped['full_file_path'].dropna())
    destination = _write_csv(mapped, args.output, overwrite=args.overwrite, protected=protected)
    counts = {'matched': int(mapped['full_file_path'].notna().sum()), 'unmatched': int(mapped['full_file_path'].isna().sum())}
    print(f'Session file table: {destination}; {len(mapped)} rows, {counts}.', file=sys.stderr)
    return destination if destination == 'stdout' else Path(destination)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parser().parse_args(argv))
        return 0
    except (ValueError, OSError, ImportError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
