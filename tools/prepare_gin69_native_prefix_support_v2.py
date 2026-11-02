"""Frozen source captures to the existing native LMS bridge, without refitting."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from scipy.io import loadmat
from retina_sim.contracts import canonical_hash
from retina_sim.isetbio_input import reference_equivalent_drive
from retina_sim.spatial_group_graph import digest
from tools.durable_research_runtime_v1 import atomic_json
from tools.gin_hc2_bc1_context_support_v2 import OUT, read, bundle, background_directory


def build(output, guard):
    began = time.perf_counter()
    optical = OUT/'formal_optics'
    plan = read(optical/'MANIFEST.json')
    if plan['split_sha256'] != digest(OUT/'SPLIT_CONTRACT.json'):
        raise ValueError('Frozen split changed')
    geometry = bundle(OUT/'optical/APERTURE_GEOMETRY.npz')
    types = geometry['cone_types'].astype('U1')
    cuts = np.r_[0, np.cumsum(geometry['global_group_counts'])]
    local_rows = geometry['local_cone_rows']
    reference = read(background_directory()/'FIXED_LMS_BRIDGE.json')['profile']
    bridge = deepcopy(reference)
    bridge['capture_scope_sha256'] = digest(optical/'MANIFEST.json')
    exposure = read(OUT/'optical/CONFIG.json')['exposure_seconds']
    short = loadmat(OUT/'optical/CAPTURE.mat', variable_names=['expected_counts'])['expected_counts']
    captures = {-1: short}
    receipts = []
    drive = np.empty((6750, len(local_rows)), dtype=np.float64)

    def convert(q):
        values, _ = reference_equivalent_drive(q/exposure, types, bridge,
            capture_scope_sha256=bridge['capture_scope_sha256'])
        return np.stack([np.mean(values[:, cuts[i]:cuts[i+1]], axis=1)
            for i in local_rows], axis=1)

    converted = {-1: convert(short)}
    for job in plan['jobs']:
        guard()
        folder = optical/f"job_{job['job']:04d}"
        receipt = read(folder/'COMPLETE.json')
        if receipt['config_sha256'] != job['config_sha256'] or digest(folder/'CONFIG.json') != job['config_sha256']:
            raise ValueError('Committed optical job identity changed')
        data = loadmat(folder/'CAPTURE.mat', variable_names=['expected_counts', 'manifest_json'])
        import json
        metadata = json.loads(str(data['manifest_json'].ravel()[0]))
        config = read(folder/'CONFIG.json')
        bank = folder/'SOURCE_BANK.mat'
        if Path(config['source_bank']).resolve() != bank.resolve() or digest(bank) != job['bank_sha256']:
            raise ValueError('Actual committed source RGB bank differs from the frozen movie plan')
        if metadata['request_signature'] != config['request_signature'] or metadata['exposure_seconds'] != exposure:
            raise ValueError('Actual capture request/exposure mismatch')
        q = data['expected_counts']
        if q.shape != (job['count'], len(types)) or not np.isfinite(q).all() or np.any(q < 0):
            raise ValueError('Actual fine-cone capture invalid')
        converted[job['job']] = convert(q)
        receipts.append(dict(job=job['job'],verified_source_bank_sha256=job['bank_sha256'],capture_sha256=digest(folder/'CAPTURE.mat'),
            complete_sha256=digest(folder/'COMPLETE.json')))
    for index, frame in enumerate(plan['frames']):
        if 'short_capture_row' in frame:
            drive[index] = converted[-1][frame['short_capture_row']]
        else:
            drive[index] = converted[frame['job']][frame['capture_row']]
    with np.load(background_directory()/'MOVIE_NATIVE_INPUTS.npz', allow_pickle=False) as data:
        if not np.array_equal(drive[:8], data['live_native_drive']):
            raise ValueError('Existing short prefix native conversion changed')
    if not np.isfinite(drive).all() or np.any(drive < 0):
        raise ValueError('Full input is not finite/nonnegative')
    np.savez_compressed(output/'NATIVE_INPUTS.npz', drive=drive,
        frame_start_seconds=np.arange(6750)/150, local_cone_rows=local_rows)
    result = dict(status='COMPLETE_ACTUAL_NATIVE_PREFIX',duration_seconds=45,frames=6750,
        local_cone_groups=len(local_rows),native_payload_sha256=canonical_hash(drive),
        array_sha256=digest(output/'NATIVE_INPUTS.npz'),source_plan_sha256=digest(optical/'MANIFEST.json'),
        split_sha256=plan['split_sha256'],fixed_reference_profile=reference,actual_capture_profile=bridge,
        capture_receipts=receipts,first_eight_frames_bitwise_equal=True,
        converter_code_sha256=digest(Path(__file__)),
        future_spikes_or_sealed_segment_used=False,wall_seconds=time.perf_counter()-began,
        FIT_READY=False,background_qualified=False,production_adopted=False)
    atomic_json(output/'RESULT.json',result)
    atomic_json(OUT/'FORMAL_INPUT_LOCATION.json',dict(path=str(output),result_sha256=digest(output/'RESULT.json')))
