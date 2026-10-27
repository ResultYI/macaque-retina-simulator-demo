"""Join only complete, original optical captures for the centered demo."""
from pathlib import Path
import json
import time
import numpy as np

from tools.prepare_foveal_demo_input import ROOT, MAT_LIMIT, checked, load_scope, sha, check_reference


def validate_packet(raw, packet, policy_sha, reference, rows):
    meta = json.loads(raw['manifest_json'].item())
    start, stop = packet['scene_range']
    if (meta['schema'] != 'approved_vr_linear_capture_v1'
            or meta['request_signature'] != packet['request_signature']
            or meta['input_policy_sha256'] != policy_sha
            or meta['background_signature'] != reference['global_reference_definition_sha256']
            or not np.array_equal(raw['q_ref_LMS'].reshape(-1), reference['q_ref_LMS_rstar_s'])
            or raw['expected_counts'].shape != (stop-start, len(rows))
            or raw['background_counts'].size != len(rows)
            or not np.array_equal(raw['window_edges_seconds'].reshape(-1), np.arange(stop-start+1)/150.)):
        raise ValueError('Actual optical packet identity, reference, layout or time changed')
    for name in ('expected_counts', 'background_counts'):
        if not np.isfinite(raw[name]).all() or np.any(raw[name] < 0):
            raise ValueError('Invalid actual optical counts')
    return meta


def finish(run, *, out, guard):
    from scipy.io import loadmat
    from retina_sim.contracts import canonical_hash
    from tools.live_status_io import write_json
    started = time.perf_counter()
    run = Path(run).resolve()
    config, policy = load_scope(run)
    out = Path(out).resolve()
    if not out.is_relative_to(run):
        raise ValueError('Input must belong to this run')
    plan = json.loads((out/'capture_plan.json').read_bytes())
    if (plan['configuration_sha256'] != sha(run/'configuration.json')
            or plan['policy_sha256'] != config['input_policy_proposal']['sha256']
            or plan['source_inventory_sha256'] != sha(run/'raw_frame_inventory.json')):
        raise ValueError('Input preparation bindings changed before join')
    ref_receipt = json.loads((out/'reference_receipt.json').read_bytes())
    ref = check_reference(out/'actual_global_reference.json', ref_receipt)
    optical = json.loads(checked(ROOT, config['input_plan']).read_bytes())
    table = json.loads(checked(ROOT, optical['cone_table']).read_bytes())
    array = lambda binding: np.load(checked(ROOT, binding), allow_pickle=False, mmap_mode='r')
    types = array(table['extras']['subtype'])
    if plan['source_shape'] != [781, len(types)] or len(plan['tiles']) != len(optical['tiles']):
        raise ValueError('Incomplete optical geometry or temporal coverage')
    targets = ('cone_unique_counts.npy', 'cone_background_counts.npy', 'cone_background_native_td.npy', 'frontend_receipt.json')
    if any((out/name).exists() for name in targets):
        raise FileExistsError('Preserve existing or partial joined input')
    guard()
    counts = np.lib.format.open_memmap(out/targets[0], mode='w+', dtype=np.float64,
                                     shape=(plan['unique_source_frames'], len(types)))
    background = np.lib.format.open_memmap(out/targets[1], mode='w+', dtype=np.float64, shape=(len(types),))
    coverage = np.zeros(len(types), dtype=np.uint8)
    captures = {}
    for tile, original in zip(plan['tiles'], optical['tiles']):
        guard()
        if tile['global_rows'] != original['global_rows']:
            raise ValueError('Original tile ownership changed')
        rows = array(tile['global_rows'])
        if np.any(coverage[rows]) or len(np.unique(rows)) != len(rows):
            raise ValueError('Overlapping optical ownership')
        mapping = np.asarray(tile['scene_mapping'], dtype=np.int64)
        if mapping.shape != (plan['unique_source_frames'],) or np.any(mapping < 0):
            raise ValueError('Missing original scene mapping')
        seen = np.zeros(len(mapping), dtype=np.uint8)
        tile_bg = None
        for packet in tile['packets']:
            guard()
            path = Path(packet['output']).resolve()
            if (not path.is_relative_to(out) or not Path(str(path)+'.ready').is_file()
                    or path.stat().st_size > MAT_LIMIT):
                raise ValueError('Missing completed bounded optical packet')
            export = json.loads(Path(packet['configuration']).read_bytes())
            if sha(export['input_mat']) != packet['input_mat_sha256']:
                raise ValueError('Optical input packet changed')
            raw = loadmat(path)
            validate_packet(raw, packet, plan['policy_sha256'], ref, rows)
            bg = raw['background_counts'].reshape(-1)
            if tile_bg is None:
                tile_bg = bg.copy()
            elif not np.array_equal(tile_bg, bg):
                raise ValueError('Identical fixed background capture differs across packets')
            start, stop = packet['scene_range']
            selected = np.flatnonzero((mapping >= start) & (mapping < stop))
            if np.any(seen[selected]):
                raise ValueError('Overlapping packet scene ranges')
            counts[np.ix_(selected, rows)] = raw['expected_counts'][mapping[selected]-start]
            seen[selected] = 1
            captures[str(path.relative_to(out))] = sha(path)
        if not np.all(seen == 1) or tile_bg is None or np.any(tile_bg <= 0):
            raise ValueError('Incomplete scene capture or invalid background')
        background[rows] = tile_bg
        coverage[rows] = 1
    if not np.all(coverage == 1):
        raise ValueError('Missing original cone capture coverage')
    counts.flush()
    background.flush()
    indices = np.array([{'L':0, 'M':1, 'S':2}[x.decode('ascii')] for x in types])
    native = np.asarray(background)/(1/150.)/(np.asarray(ref['q_ref_LMS_rstar_s'])[indices]/100.)
    if not np.isfinite(native).all() or np.any(native <= 0):
        raise ValueError('Invalid captured native background')
    np.save(out/targets[2], native, allow_pickle=False)
    del counts, background
    files = {name: sha(out/name) for name in targets[:-1] + ('source_unique_gray8.npy', 'source_frame_to_unique.npy')}
    implementation = {name: sha(ROOT/name) for name in (
        'tools/prepare_foveal_demo_input.py', 'tools/finalize_foveal_demo_input.py',
        'tools/export_vr_linear_disk_capture.m', 'tools/export_vr_linear_disk_batch.m',
        'tools/vr_teacher_handoff_v1/fit_integration_v1/capture_preparation_v2/linear_gray.py',
        'tools/vr_teacher_handoff_v1/fit_integration_v1/capture_preparation_v2/make_gray_reference_scene.m')}
    receipt = dict(schema='foveal-centered-actual-optical-input-1', files=files, captures=captures,
        policy_sha256=plan['policy_sha256'], configuration_sha256=plan['configuration_sha256'],
        capture_plan_sha256=sha(out/'capture_plan.json'), implementation_sources=implementation,
        original_dependencies=ref_receipt['producer_dependencies'],
        global_reference_numeric_sha256=ref_receipt['actual_reference_sha256'],
        capture_geometry_sha256=canonical_hash(dict(cone_table=optical['cone_table'], tiles=optical['tiles'])),
        actual_background_input_sha256=canonical_hash(dict(native_values=native, native_unit='td',
            rod_rate_Rstar_per_s=policy['rod']['outside_rate_Rstar_per_s'], cone_order=optical['cone_table'])),
        source_frames=[0, 781], original_teacher_background_frames_retained=750,
        full_background_captured=True, actual_capture_window_seconds=1/150.,
        same_global_reference_for_all_rois=True, per_frame_or_cell_normalization=False,
        physical_calibration_claim=False, fit_eligible=False, background_qualified=False,
        join_seconds=time.perf_counter()-started)
    write_json(out/'frontend_receipt.json', receipt)
    return receipt
