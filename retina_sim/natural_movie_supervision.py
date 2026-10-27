"""Data-only reader for the frozen Schottdorf/Lee GIN public source.

This module does not import or execute the source notebooks/model, construct a
retinal graph, change input calibration, or infer an anatomical correspondence.
Raw 0.1ms spike times are retained, including pre/post-movie events.
"""
from pathlib import Path
import ast
import hashlib
import json
import re
import numpy as np

ARCHIVE_SHA256 = '3096eafe97ead9c589096e454bf3f11e9bc07124e1ecb30255da194452af3fb3'
TICKS_PER_SECOND = 10000
PLAYBACK_HZ = 150
DOCUMENTED_LIVE_FRAME_ZERO_BASED = 751


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def source_roster(readme):
    rows = {}
    eccentricities = {}
    for line_no, line in enumerate(Path(readme).read_text(encoding='utf8').splitlines(), 1):
        if not re.match(r'^\|\s*Lss\d+\.txt', line, re.I):
            continue
        fields = [x.strip() for x in line.split('|')[1:-1]]
        name, cell_type, duration, cell_key, eccentricity = fields
        if eccentricity.lower() != 'same':
            eccentricities[cell_key] = float(eccentricity)
        if cell_key not in eccentricities:
            raise ValueError('Source same-eccentricity row lacks its cell predecessor')
        rows[name.lower()] = dict(source_file_name=name, source_cell_key=cell_key,
            source_experiment_key_prefix=cell_key.split('#')[0],
            animal_id=None, session_id=None, cell_type_source=cell_type,
            eccentricity_deg=eccentricities[cell_key], duration_source=duration,
            live_seconds=600 if '10 min' in duration else 60,
            readme_line=line_no)
    if not rows:
        raise ValueError('Missing primary cell roster')
    return rows


def _ints(line):
    return [int(s.strip()) for s in line.split('\t') if s.strip()]


def read_raw_spike_tables(path, metadata):
    lines = Path(path).read_text(encoding='ascii').splitlines()
    headers = [i for i, line in enumerate(lines) if re.fullmatch(r'\s*(?:Spikes?\s+times?|No\s+Time)\s*', line, re.I)]
    if not headers:
        raise ValueError('No declared raw spike-time table: ' + str(path))
    repeats = 1 if metadata['live_seconds'] == 600 else 7
    result = []
    for segment_no, header in enumerate(headers):
        before = 0 if segment_no == 0 else headers[segment_no-1]+1
        totals = [(i, _ints(line[len(line.split('\t')[0]):])[0])
                  for i, line in enumerate(lines[before:header], before)
                  if re.match(r'^Total spikes\t\d+', line, re.I)]
        rpt_headers = [i for i in range(before, header) if lines[i].strip().lower() == 'spikes per rpt']
        declared_repeat = _ints(lines[rpt_headers[-1]+1]) if rpt_headers else None
        video_starts = [int(line.split('\t')[1]) for line in lines[before:header]
                        if re.match(r'^Video Starts?\t\d+', line, re.I)]
        columns = [[] for _ in range(repeats)]
        row_indices = []
        for line in lines[header+1:]:
            if not line.strip():
                if row_indices:
                    break
                continue
            cells = line.split('\t')
            if not re.fullmatch(r'\d+', cells[0].strip()):
                break
            index = int(cells[0])
            if index != len(row_indices):
                raise ValueError(f'Raw row index discontinuity in {path.name} segment{segment_no}')
            values = cells[1:]
            if any(x.strip() for x in values[repeats:]):
                extras=[x.strip() for x in values[repeats:] if x.strip()]
                # The published1159 file concatenates its duplicate second
                # table's Total-spikes heading onto the empty final data row.
                if not any(x.strip() for x in values[:repeats]) and len(extras)==2 and extras[0].lower()=='total spikes' and extras[1].isdigit():
                    break
                raise ValueError('Unexpected nonempty spike columns')
            row_indices.append(index)
            for k, value in enumerate(values[:repeats]):
                if value.strip():
                    columns[k].append(int(value.strip()))
        ticks = tuple(np.asarray(a, dtype=np.int64) for a in columns)
        if not any(len(a) for a in ticks):
            raise ValueError('Empty raw source table')
        if any(np.any(np.diff(a) < 0) for a in ticks):
            raise ValueError('Descending source event times; no silent sorting')
        if declared_repeat is not None and list(map(len, ticks)) != declared_repeat:
            raise ValueError('Raw column counts differ from source repeat totals')
        if totals and sum(map(len, ticks)) != totals[-1][1]:
            raise ValueError('Raw event count differs from Total spikes')
        exact_hash = hashlib.sha256(b''.join(a.tobytes()+b'|' for a in ticks)).hexdigest()
        duplicate_of = next((x['segment_index'] for x in result if x['raw_columns_sha256']==exact_hash), None)
        result.append(dict(segment_index=segment_no, source_header_line=header+1,
            declared_total_spikes=totals[-1][1] if totals else None,
            declared_repeat_counts=declared_repeat, video_start_acquisition_ticks=video_starts[-1] if video_starts else None,
            ticks=ticks, raw_columns_sha256=exact_hash, exact_duplicate_of_segment=duplicate_of,
            trials_in_table=repeats, live_trials=1 if repeats==1 else 6,
            final_seventh_column_is_maintained_activity=repeats==7))
    return result


def bin_live_trial(ticks, live_seconds):
    """Exact integer-edge native1ms bins; retain counts>1 rather than clipping."""
    a=np.asarray(ticks, dtype=np.int64)
    live=a[(a>=0)&(a<int(live_seconds)*TICKS_PER_SECOND)]
    counts=np.bincount(live//10, minlength=int(live_seconds)*1000).astype(np.int16)
    if len(counts)!=int(live_seconds)*1000:
        raise ValueError('Source event outside declared live support')
    return counts


def frame_index_for_ticks(ticks, *, live_frame_zero_based=DOCUMENTED_LIVE_FRAME_ZERO_BASED):
    """Declared150Hz mapping. It does not resolve the observed one-frame issue."""
    t=np.asarray(ticks,dtype=np.int64)
    return live_frame_zero_based + (t*PLAYBACK_HZ)//TICKS_PER_SECOND


def read_author_fixed_rf_parameters(directory):
    """Read literal notebook dictionaries without importing or executing code.

    These are author model fits in video coordinates, not independent measured
    RFs or a binding to this project's canonical retinal graph.
    """
    result={}
    for path in sorted(Path(directory).glob('*.ipynb')):
        data=json.loads(path.read_bytes())
        active=None
        for cell_index, cell in enumerate(data['cells']):
            if cell['cell_type']!='code':
                continue
            text=''.join(cell['source'])
            for match in re.finditer(r"fitparameters\s*=\s*(\{.*?\})|compare_to_file\(\s*['\"]([^'\"]+)['\"]",text,re.S):
                if match.group(1):
                    node=ast.parse(match.group(1),mode='eval').body
                    if not isinstance(node,ast.Dict):
                        raise ValueError('RF parameters must be a literal source dictionary')
                    active={ast.literal_eval(k):ast.literal_eval(v) for k,v in zip(node.keys,node.values)
                            if ast.literal_eval(k) in ['xoff','yoff','rcenter','rsurr','rcenterM','rcenterL','rcenterB','rcenterY']}
                elif active is not None:
                    number=re.search(r'File(\d+)',match.group(2))
                    if number:
                        rf={k:v for k,v in active.items() if k in ['xoff','yoff','rcenter','rsurr','rcenterM','rcenterL','rcenterB','rcenterY']}
                        result[int(number.group(1))]=dict(parameters=rf, notebook=path.name,
                            notebook_cell_zero_based=cell_index, source_file_argument=match.group(2),
                            status='AUTHOR_FITTED_VIDEO_COORDINATES_NOT_PROJECT_GRAPH_MAPPING',
                            coordinate_definition='np.linspace(-128,128,256) for x and image-row y',
                            not_independent_measured_RF=True, project_cell_binding=None)
    return result


def load_prepared_real_trial(directory, record_id, trial_zero_based):
    """Return real1ms counts and fixed150Hz frame indices, without an RF guess.

    Negative/post-movie raw events are returned as context, not reassigned to
    live movie frames. The original source RF/project binding status is kept.
    """
    directory=Path(directory)
    index=json.loads((directory/'source_spike_index.json').read_bytes())
    record=next((r for r in index['records'] if r['record_id']==record_id),None)
    if record is None or record['exact_duplicate_of_record'] is not None:
        raise ValueError('A unique actual source record is required')
    if isinstance(trial_zero_based,bool) or int(trial_zero_based)!=trial_zero_based:
        raise ValueError('Integer source trial index required')
    trials=[r for r in record['trials'] if r['is_live_movie_repeat']]
    trial=next((r for r in trials if r['column_zero_based']==trial_zero_based),None)
    if trial is None:
        raise ValueError('The maintained-activity column is not a movie repeat')
    path=directory/index['arrays']['path']
    if file_sha(path)!=index['arrays']['sha256']:
        raise ValueError('Prepared real-source arrays changed')
    with np.load(path,allow_pickle=False) as data:
        counts=np.array(data[trial['count_key']],copy=True)
        ticks=np.array(data[trial['raw_ticks_key']],copy=True)
    if int(counts.sum())!=trial['live_events']:
        raise ValueError('Prepared counts lost actual source events')
    live_ticks=np.arange(len(counts),dtype=np.int64)*10
    return dict(record_id=record_id,source_cell_key=record['source_cell_key'],
        model_family_candidate=record['model_family_candidate'],
        counts_1ms=counts,raw_context_ticks_100us=ticks,
        bin_start_source_ticks_100us=live_ticks,
        movie_frame_zero_based=frame_index_for_ticks(live_ticks),
        project_RF_binding=record['project_RF_binding'],RF_source=record['RF_source'],
        timing='Fixed primary-source150Hz/live frame751; unresolved physical one-frame discrepancy retained',
        underlying_movie_id=record['underlying_movie_id'],
        conditional_source_pairing=True,full_model_training_admitted=False)
