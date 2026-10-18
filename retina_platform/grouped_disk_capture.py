"""Pack independent 100-us exposures into the unchanged multi-frame exporter.

The capture index axis is storage only. Original video times and the full
100-us neural-input mapping remain outside this stateless optical calculation.
"""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.disk_graph_storage import file_sha
from .disk_frontend import _capture_request, _finish_capture, read_disk_capture

SCHEMA = 'grouped_memoryless_tile_captures_v1'
MAX_RGB_BYTES = 64 * 1024**2
_VARIABLES = ('rgb_frames', 'window_edges_seconds', 'frame_indices', 'reference_rgb',
              'is_mm', 'types', 'frozen_apertures_um', 'image_center_deg')


def _source(job, labels, root, scope):
    config_path = Path(job['configuration']).resolve()
    if not config_path.is_relative_to(root):
        raise ContractError('Grouped source configuration outside project')
    config = json.loads(config_path.read_bytes())
    if config['global_reference']['definition_sha256'] != scope or config['background_signature'] != scope:
        raise ContractError('Grouped request changed fixed reference scope')
    source = Path(config['input_mat']).resolve()
    if not source.is_relative_to(root):
        raise ContractError('Grouped source arrays outside project')
    data = {key: value for key, value in loadmat(source, variable_names=_VARIABLES).items() if key in _VARIABLES}
    if (set(data) != set(_VARIABLES) or data['rgb_frames'].ndim != 4 or
            data['rgb_frames'].shape[-1] != 1 or data['rgb_frames'].dtype != np.float32 or
            not np.array_equal(data['window_edges_seconds'].reshape(-1), [0., .0001]) or
            not np.array_equal(data['frame_indices'].reshape(-1), [1.])):
        raise ContractError('Grouping requires unchanged single-frame float32/100-us source requests')
    static = {k: v for k, v in config.items() if k not in ('input_mat', 'request_signature')}
    record = dict(labels[str(config_path)], single_request_signature=config['request_signature'],
                  configuration_sha256=file_sha(config_path), input_mat_sha256=file_sha(source),
                  single_configuration=str(config_path))
    return config, static, data, record


def group_missing_jobs(root, profile, jobs, labels, scope, *, on_status=None, on_boundary=None,
                       max_rgb_bytes=MAX_RGB_BYTES):
    """Reuse completed legacy captures upstream; group only missing requests."""
    root = Path(root).resolve()
    by_tile = {}
    for job in jobs:
        label = labels[str(Path(job['configuration']).resolve())]
        by_tile.setdefault(canonical_hash(label['tile']), []).append(job)
    grouped_jobs, views, groups = [], {}, []
    for tile_number, members in enumerate(by_tile.values()):
        start = 0
        while start < len(members):
            if on_boundary is not None and on_boundary() is False:
                raise ContractError('Grouped input planning stopped; all existing captures retained')
            config, static, first, first_record = _source(members[start], labels, root, scope)
            frame_bytes = first['rgb_frames'].nbytes
            limit = max_rgb_bytes // frame_bytes
            if limit < 1:
                raise ContractError('One RGB tile exceeds the explicit grouping memory budget')
            selected = members[start:start+limit]
            frames, records = [first['rgb_frames']], [first_record]
            for job in selected[1:]:
                _, other_static, data, record = _source(job, labels, root, scope)
                if other_static != static or any(
                        data[key].dtype != first[key].dtype or not np.array_equal(data[key], first[key])
                        for key in _VARIABLES if key != 'rgb_frames'):
                    raise ContractError('Grouped frames differ in frozen geometry, optics or background')
                if data['rgb_frames'].shape != first['rgb_frames'].shape:
                    raise ContractError('Grouped RGB patch dimensions differ')
                frames.append(data['rgb_frames']);records.append(record)
            data = dict(first, rgb_frames=np.concatenate(frames, axis=3),
                        window_edges_seconds=np.arange(len(records)+1, dtype=float)*.0001,
                        frame_indices=np.arange(1, len(records)+1, dtype=float))
            description = {'schema': SCHEMA, 'reference_scope': scope, 'frames': records,
                           'capture_axis': 'Independent 100-us captures, not video timestamps',
                           'maximum_rgb_bytes': max_rgb_bytes}
            signature = canonical_hash(description)
            cache = root/'reports/iset_disk_input_cache'/scope/'grouped_tiles_v1'/signature
            export = dict(static, request_signature=signature, grouped_capture=description)
            (raw, meta), called = _capture_request(root, profile, cache, data, export,
                on_status=on_status, on_boundary=on_boundary, defer_jobs=grouped_jobs)
            del raw, data, frames, first
            groups.append(dict(description, directory=str(cache), request_signature=signature,
                               export_job_required=called))
            for index, record in enumerate(records):
                single_cache = str(Path(record['single_configuration']).parent.resolve())
                views[single_cache] = {'directory': str(cache), 'request_signature': signature,
                    'pending_path': meta.get('pending_path'), 'capture_frame_index': index,
                    'capture_frame_count': len(records), 'frame_source': record}
            start += len(selected)
        if on_status:
            on_status('ISET grouped tile requests prepared %d/%d'%(tile_number+1, len(by_tile)))
    return grouped_jobs, views, {'schema': SCHEMA, 'groups': groups,
        'missing_legacy_frame_requests': len(jobs), 'group_capture_requests': len(groups),
        'new_export_jobs': len(grouped_jobs), 'maximum_rgb_bytes': max_rgb_bytes}


def read_group_frame(view):
    """A provenance-labelled one-row view; never masquerades as a legacy cache."""
    cache = Path(view['directory']);signature = view['request_signature']
    raw, meta = (read_disk_capture(cache/'capture.mat', signature) if (cache/'capture.mat').exists()
                 else _finish_capture(cache, view['pending_path'], signature))
    index = view['capture_frame_index'];count = view['capture_frame_count']
    if (raw['expected_counts'].shape[0] != count or not 0 <= index < count or
            not np.array_equal(raw['window_edges_seconds'].reshape(-1), np.arange(count+1)*.0001)):
        raise ContractError('Grouped capture frame count/index or independent exposure axis changed')
    selected = dict(raw, expected_counts=raw['expected_counts'][index:index+1].copy(),
                    window_edges_seconds=np.array([[0., .0001]]))
    return selected, dict(meta, grouped_capture_schema=SCHEMA,
                         capture_frame_index=index, frame_source=view['frame_source'])
