"""Capture the authorized centered demo through the original optical producer.

Exact repeated scenes share expected captures only; all 781 source frame
positions remain in the input mapping. No neural state is constructed here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIMIT = 12 * 1024**3
MAT_LIMIT = 128 * 1024**2


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(root, binding):
    path = (root / binding['path']).resolve()
    if not path.is_relative_to(root) or sha(path) != binding['sha256']:
        raise ValueError('Changed bound input asset: ' + str(path))
    return path


def load_scope(run):
    run = Path(run).resolve()
    if not run.is_relative_to(ROOT):
        raise ValueError('Foveal run must remain in this project')
    config = json.loads((run/'configuration.json').read_bytes())
    policy = json.loads(checked(ROOT, config['input_policy_proposal']).read_bytes())
    if (config.get('execution_enabled') is not True or config.get('default_adopted') is not False
            or config.get('fit_eligible') is not False
            or config.get('source_graph_scope') != 'all_original_outputs_in_foveal_disk_v1'
            or config.get('output_owner_radius_mm') != 6*.223
            or config.get('crop_recurrent_dependencies') is not False
            or policy.get('approval_status') != 'USER_AUTHORIZED_CENTERED_ENGINEERING_DEMO_20261024'
            or policy.get('execution_enabled') is not True or policy.get('fit_eligible') is not False
            or policy['execution_scope']['active_seconds'] != .2
            or policy['execution_scope']['raw_recorded_background_frames'] != 750):
        raise ValueError('Explicit original full-disk 200ms engineering scope required')
    reg = policy['spatial_registration']
    if (reg['pixels_per_degree'] != 256/12 or reg['teacher_rf_deg'] != [0., 0.]
            or reg['patch_center_xy'] != [127.5, 127.5] or reg['project_mm_per_degree'] != .223):
        raise ValueError('Centered input registration changed')
    return config, policy


def resource_budget(run):
    config, _ = load_scope(run)
    return dict(deadline_unix=None, max_rss_bytes=LIMIT, max_total_gpu_bytes=LIMIT,
                minimum_available_ram_bytes=config['budgets']['minimum_available_bytes'])


def check_reference(path, receipt):
    from retina_sim.contracts import canonical_hash
    value = json.loads(Path(path).read_bytes())
    # The original reference producer hashes canonical JSON values, not the
    # whitespace-bearing JSON file. Preserve its established identity rule.
    if canonical_hash(value) != receipt['actual_reference_sha256']:
        raise ValueError('Actual fixed reference changed')
    return value


def scene_multiplier(frame, xy, policy):
    from tools.vr_teacher_handoff_v1.fit_integration_v1.capture_preparation_v2.linear_gray import _sample_codes
    return _sample_codes(frame, xy, policy['spatial_registration'],
                         policy['outside']['proposed_gray8_code']) / 255. / policy['spectrum']['reference_linear_value']


def unique_scenes(frames, xy, policy):
    """Exact equality only, including after spatial sampling on this tile."""
    import numpy as np
    reg = policy['spatial_registration']
    pixels = (np.asarray(xy)/reg['project_mm_per_degree']-np.asarray(reg['teacher_rf_deg']))
    pixels = pixels*reg['pixels_per_degree']+np.asarray(reg['patch_center_xy'])
    if not np.any(np.all((pixels >= 0) & (pixels <= 255), axis=-1)):
        # _sample_codes assigns the same exterior value at every such point.
        # This skips only repeated scene construction, never neural history.
        return [scene_multiplier(frames[0], xy, policy)], [0]*len(frames)
    scenes, mapping, by_hash = [], [], {}
    for frame in frames:
        value = scene_multiplier(frame, xy, policy)
        digest = hashlib.sha256(value.tobytes()).hexdigest()
        index = by_hash.get(digest)
        if index is None:
            index = len(scenes)
            by_hash[digest] = index
            scenes.append(value)
        elif not np.array_equal(value, scenes[index]):
            raise ValueError('Scene hash collision')
        mapping.append(index)
    return scenes, mapping


def capture_prepared(run, out, *, attempt):
    """Explicit retry of an intact packet set, with separate failure evidence."""
    from tools.live_status_io import write_json
    from retina_sim.useful_runtime_budget import foveal_runtime_limits
    from retina_sim.split_execution_guard import SplitExecutionGuard
    run, out = Path(run).resolve(), Path(out).resolve()
    config, policy = load_scope(run)
    if not out.is_relative_to(run) or attempt < 1:
        raise ValueError('Existing owned input and positive capture attempt required')
    plan = json.loads((out/'capture_plan.json').read_bytes())
    if (plan['configuration_sha256'] != sha(run/'configuration.json')
            or plan['policy_sha256'] != config['input_policy_proposal']['sha256']
            or plan['source_inventory_sha256'] != sha(run/'raw_frame_inventory.json')):
        raise ValueError('Prepared input identity changed')
    all_jobs = json.loads((out/'batch.json').read_bytes())
    expected = [dict(configuration=p['configuration'], output=p['output']) for t in plan['tiles'] for p in t['packets']]
    if all_jobs != expected:
        raise ValueError('Prepared batch and capture plan differ')
    jobs = []
    for tile in plan['tiles']:
        for packet in tile['packets']:
            source = Path(packet['configuration']).resolve()
            target = Path(packet['output']).resolve()
            if not source.is_relative_to(out) or not target.is_relative_to(out):
                raise ValueError('Packet escaped owned input directory')
            export = json.loads(source.read_bytes())
            if (export['request_signature'] != packet['request_signature']
                    or export['input_policy_sha256'] != plan['policy_sha256']
                    or sha(export['input_mat']) != packet['input_mat_sha256']):
                raise ValueError('Prepared packet changed')
            if target.exists() or Path(str(target)+'.ready').exists():
                raise ValueError('This retry requires zero prior captures; partial results need explicit recovery')
            jobs.append(dict(configuration=str(source), output=str(target)))
    profile = json.loads(checked(ROOT, policy['provenance']['approved_display_profile']).read_bytes())
    receipt = json.loads((out/'reference_receipt.json').read_bytes())
    check_reference(out/'actual_global_reference.json', receipt)
    for path, digest in receipt['producer_dependencies'].items():
        if sha(path) != digest:
            raise ValueError('Original optical dependency changed')
    status_path = out/f'capture_attempt_{attempt}.json'
    started = time.perf_counter()
    status = dict(stage='CAPTURING_PREPARED_CENTERED_INPUT', pid=os.getpid(), total_packets=len(jobs),
                  completed_packets=0, model_forward_started=False, previous_failures_retained=True)
    write_json(status_path, status)
    guard = SplitExecutionGuard(dict(foveal_runtime_limits(run), round_started_unix=time.time()),
        min_free_ram=config['budgets']['minimum_available_bytes'], stop_file=run/'REQUEST_STOP', output_directory=out)
    with (out/f'capture_watchdog_{attempt}.log').open('xb') as log:
        watcher = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', str(ROOT/'tools/watch_useful_capture_process.py'),
            '--run-dir', str(run), '--root-pid', str(os.getpid()), '--status-file', str(status_path), '--foveal'],
            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    def check():
        if watcher.poll() is not None:
            raise RuntimeError('Independent capture resource watcher exited')
        guard.require()
        status.update(resources=dict(guard.last), elapsed_seconds=time.perf_counter()-started)
        write_json(status_path, status, replace=True)
    try:
        check()
        quote = lambda path: str(Path(path).resolve()).replace("'", "''")
        expression = "addpath('"+quote(ROOT/'tools')+"');export_vr_linear_disk_batch('"+quote(out/'batch.json')+"');"
        pref = out/f'matlab_preferences_capture_{attempt}'
        pref.mkdir(exist_ok=False)
        with (out/f'matlab_capture_{attempt}.log').open('xb') as log:
            child = subprocess.Popen([profile['matlab_executable'], '-batch', expression], stdout=log,
                stderr=subprocess.STDOUT, env=dict(os.environ, MATLAB_PREFDIR=str(pref)),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            status['matlab_pid'] = child.pid
            count, progress = 0, time.monotonic()
            while child.poll() is None:
                done = sum(Path(job['output']+'.ready').is_file() for job in jobs)
                if done > count:
                    count, progress = done, time.monotonic()
                status['completed_packets'] = done
                check()
                if time.monotonic()-progress > profile['maximum_export_wall_seconds']:
                    raise RuntimeError('Original per-export progress limit reached')
                time.sleep(2.)
            status['matlab_returncode'] = child.returncode
            if child.returncode:
                raise RuntimeError('MATLAB producer failed; preserved log contains the actual error')
        status['optical_capture_seconds'] = time.perf_counter()-started
        from tools.finalize_foveal_demo_input import finish
        finish(run, out=out, guard=check)
        status.update(stage='CENTERED_INPUT_READY_NEURAL_INITIAL_STATE_PENDING', completed_packets=len(jobs))
    except Exception as exc:
        status.update(stage='CENTERED_CAPTURE_FAILED', error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        status['elapsed_seconds'] = time.perf_counter()-started
        write_json(status_path, status, replace=True)
    print(json.dumps(status), flush=True)


def prepare(run, *, attempt=1):
    import numpy as np
    from scipy.io import savemat
    from retina_sim.contracts import canonical_hash
    from tools.live_status_io import write_json
    from tools.supervise_resident_validation import memory
    from tools.vr_teacher_handoff_v1.fit_integration_v1.pilot import Gray8Chunks
    from ctypes import WinDLL, wintypes

    run = Path(run).resolve()
    config, policy = load_scope(run)
    budget = resource_budget(run)
    if attempt < 1:
        raise ValueError('Positive input attempt required')
    out = run/f'centered_inputs_v{attempt}'
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    status = dict(stage='PREPARING_CENTERED_OPTICAL_INPUT', pid=os.getpid(),
                  neural_forward_started=False, fit_eligible=False, completed_packets=0)
    status_path = out/'status.json'
    write_json(status_path, status)
    kernel = WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.GetCurrentProcess()
    child = watcher = None

    def guard():
        if watcher is not None and watcher.poll() is not None:
            raise RuntimeError('Independent resource watcher exited')
        host = memory(handle)
        if (host['peak_rss_bytes'] > LIMIT or host['available_physical_bytes'] < budget['minimum_available_ram_bytes']
                or shutil.disk_usage(out).free < 8*1024**3 or (run/'REQUEST_STOP').exists()):
            raise RuntimeError('Foveal input resource or explicit stop')
        status.update(resources=host, elapsed_seconds=time.perf_counter()-started, updated_unix=time.time())
        write_json(status_path, status, replace=True)

    try:
        guard()
        with (out/'watchdog.log').open('xb') as log:
            watcher = subprocess.Popen([sys.executable, '-B', '-X', 'utf8',
                str(ROOT/'tools/watch_useful_capture_process.py'), '--run-dir', str(run),
                '--root-pid', str(os.getpid()), '--status-file', str(status_path), '--foveal'],
                stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        optical = json.loads(checked(ROOT, policy['provenance']['approved_global_input_plan']).read_bytes())
        profile = json.loads(checked(ROOT, policy['provenance']['approved_display_profile']).read_bytes())
        table = json.loads(checked(ROOT, optical['cone_table']).read_bytes())
        array = lambda binding: np.load(checked(ROOT, binding), allow_pickle=False, mmap_mode='r')
        rf, types = array(table['coordinates']['rf']), array(table['extras']['subtype'])
        diameters = array(table['extras']['aperture_diameter_um'])
        previous = ROOT/'reports/first_useful_split_capture_20261023_v1'
        reference_receipt = json.loads((previous/'reference_receipt.json').read_bytes())
        ref_path = previous/'actual_global_reference.json'
        ref = check_reference(ref_path, reference_receipt)
        for path, digest in reference_receipt['producer_dependencies'].items():
            if sha(path) != digest:
                raise ValueError('Fixed-reference producer dependency changed: ' + path)
        shutil.copyfile(ref_path, out/'actual_global_reference.json')
        shutil.copyfile(previous/'reference_receipt.json', out/'reference_receipt.json')
        inventory = json.loads((run/'raw_frame_inventory.json').read_bytes())
        intake = Path(inventory['intake_root']).resolve()
        index = checked(intake, inventory['index_binding'])
        reader = Gray8Chunks(intake, json.loads(index.read_bytes()))
        frames = [reader.frame(i) for i in inventory['unique_representatives']]
        frame_map = inventory['frame_to_unique']
        if (len(frame_map) != 781 or len(frames) != 51 or len(inventory['frame_sha256']) != 781
                or any(hashlib.sha256(frames[frame_map[i]].tobytes()).hexdigest() != inventory['frame_sha256'][i]
                       for i in range(781))):
            raise ValueError('Original 750 background frames, 30 active frames and endpoint required')
        np.save(out/'source_unique_gray8.npy', np.stack(frames), allow_pickle=False)
        np.save(out/'source_frame_to_unique.npy', np.asarray(frame_map, dtype=np.int64), allow_pickle=False)
        reference = dict(optical['reference'], **ref)
        reference['definition_sha256'] = ref['global_reference_definition_sha256']
        reference['plan_definition_sha256'] = optical['reference']['definition_sha256']
        policy_hash = config['input_policy_proposal']['sha256']
        jobs, tiles = [], []
        focal = .223e-3 / np.tan(np.pi/180)
        for i, tile in enumerate(optical['tiles']):
            guard()
            rows = array(tile['global_rows'])
            samples = tile['scene_samples']
            center = np.asarray(tile['center_mm'])
            axis = np.arange(samples)-(samples-1)/2
            xx, yy = np.meshgrid(axis*.001+center[0], -axis*.001+center[1])
            scenes, mapping = unique_scenes(frames, np.stack([xx, yy], axis=-1), policy)
            width = MAT_LIMIT // (samples*samples*8)
            if width < 1:
                raise ValueError('Original scene tensor limit exceeded')
            packets = []
            folder = out/f'tile-{i:04d}'
            folder.mkdir()
            for offset in range(0, len(scenes), width):
                stop = min(offset+width, len(scenes))
                packet = folder/f'packet-{offset:03d}'
                packet.mkdir()
                source = packet/'input.mat'
                savemat(source, dict(scene_multipliers=np.stack(scenes[offset:stop], axis=2),
                    window_edges_seconds=np.arange(stop-offset+1, dtype=np.float64)/150.,
                    frame_indices=np.arange(1, stop-offset+1, dtype=float),
                    reference_rgb=np.asarray(ref['reference_rgb']), is_mm=np.asarray(rf[rows]),
                    types=np.array([{'L':1., 'M':2., 'S':3.}[x.decode('ascii')] for x in types[rows]]),
                    frozen_apertures_um=np.asarray(diameters[rows]), image_center_deg=center/.223), do_compression=True)
                signature = canonical_hash(dict(policy=policy_hash, tile=tile, reference=ref,
                    source_frame_sha256=inventory['frame_sha256'], tile_scene_mapping=mapping,
                    scene_range=[offset, stop], input_mat_sha256=sha(source)))
                export = dict(profile=profile, global_reference=reference,
                    patch_fov_deg=float(np.rad2deg(2*np.arctan(samples*.001e-3/(2*focal)))),
                    request_signature=signature, background_signature=ref['global_reference_definition_sha256'],
                    input_mat=str(source), input_policy_sha256=policy_hash, approved_policy_sha256=policy_hash,
                    outside_reference_multiplier=policy['outside']['reference_fraction'],
                    linear_gray_adapter_directory=str(ROOT/'tools/vr_teacher_handoff_v1/fit_integration_v1/capture_preparation_v2'))
                write_json(packet/'export.json', export)
                job = dict(configuration=str(packet/'export.json'), output=str(packet/'capture.mat'))
                jobs.append(job)
                packets.append(dict(**job, scene_range=[offset, stop], request_signature=signature,
                                    input_mat_sha256=sha(source)))
            tiles.append(dict(global_rows=tile['global_rows'], scene_mapping=mapping, packets=packets))
            status.update(prepared_tiles=i+1, total_tiles=len(optical['tiles']), prepared_packets=len(jobs))
            del scenes
        plan = dict(schema='foveal-centered-capture-plan-1', tiles=tiles,
                    source_shape=[781, len(rf)], unique_source_frames=len(frames),
                    source_inventory_sha256=sha(run/'raw_frame_inventory.json'), policy_sha256=policy_hash,
                    configuration_sha256=sha(run/'configuration.json'), temporal_frame_dedup_is_neural_skip=False)
        write_json(out/'capture_plan.json', plan)
        write_json(out/'batch.json', jobs)
        status['input_packet_preparation_seconds'] = time.perf_counter()-started
        quote = lambda path: str(Path(path).resolve()).replace("'", "''")
        expression = "addpath('"+quote(ROOT/'tools')+"');export_vr_linear_disk_batch('"+quote(out/'batch.json')+"');"
        pref = out/'matlab_preferences'
        pref.mkdir()
        capture_started = time.perf_counter()
        with (out/'matlab.log').open('xb') as log:
            guard()
            child = subprocess.Popen([profile['matlab_executable'], '-batch', expression], stdout=log,
                stderr=subprocess.STDOUT, env=dict(os.environ, MATLAB_PREFDIR=str(pref)),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            status.update(stage='CAPTURING_CENTERED_OPTICAL_INPUT', matlab_pid=child.pid, total_packets=len(jobs))
            last_count, last_progress = 0, time.monotonic()
            while child.poll() is None:
                count = sum(Path(job['output']+'.ready').is_file() for job in jobs)
                if count > last_count:
                    last_count, last_progress = count, time.monotonic()
                status['completed_packets'] = count
                guard()
                if time.monotonic()-last_progress > profile['maximum_export_wall_seconds']:
                    raise RuntimeError('Original per-export progress limit reached; partial files retained')
                time.sleep(2.)
            if child.returncode:
                raise RuntimeError('Original optical producer failed; no substituted native input')
        status['optical_capture_seconds'] = time.perf_counter()-capture_started
        guard()
        from tools.finalize_foveal_demo_input import finish
        receipt = finish(run, out=out, guard=guard)
        status.update(stage='CENTERED_INPUT_READY_NEURAL_INITIAL_STATE_PENDING',
                      frontend_receipt_sha256=canonical_hash(receipt), completed_packets=len(jobs))
    except Exception as exc:
        status.update(stage='CENTERED_INPUT_FAILED_EVIDENCE_RETAINED', error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        status['elapsed_seconds'] = time.perf_counter()-started
        write_json(status_path, status, replace=True)
        # The independent watcher owns descendant cleanup when this root exits.
        # Never kill only MATLAB's launcher and strand its calculation process.
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--capture-prepared', type=Path)
    args = parser.parse_args()
    if args.capture_prepared:
        capture_prepared(args.run_dir, args.capture_prepared, attempt=args.attempt)
    else:
        prepare(args.run_dir, attempt=args.attempt)
