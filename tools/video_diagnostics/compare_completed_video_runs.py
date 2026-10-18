"""Compare both complete actual-video runs, preserving all existing artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import json
import math
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from retina_sim.contracts import canonical_hash
from retina_sim.disk_graph_storage import file_sha

FOLDER = ROOT / 'reports/video_lossless_optimization_20261018T1723Z'
DIAGNOSTIC_TIMERS = ('compile_or_cache_load_seconds', 'plan_seconds')


def read(path):
    return json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def phase_summary(trial):
    p = trial['phase_seconds']
    checkpoints = sum(p.get(key, 0.) for key in ('start_checkpoint_write', 'initial_latest_checkpoint_write',
        'rolling_checkpoint_write', 'end_checkpoint_write', 'final_latest_checkpoint_write'))
    processing = p['advance_and_record'] + p['record_files_write'] + p['rolling_checkpoint_write']
    return {'end_to_end_seconds': trial['wall_seconds'],
            'input_preparation_seconds': p['ISET_input_or_strict_input_cache_load'] + p.get('prepared_input_admission', 0.),
            'pure_solver_seconds_included_in_advance': trial['integration_metrics']['integration_wall_seconds'],
            'advance_and_memory_record_seconds': p['advance_and_record'],
            'record_files_write_seconds': p['record_files_write'],
            'all_checkpoint_write_seconds': checkpoints,
            'rolling_checkpoint_write_seconds': p['rolling_checkpoint_write'],
            'video_processing_seconds': processing,
            'video_processing_and_all_checkpoints_seconds': p['advance_and_record'] + p['record_files_write'] + checkpoints,
            'solver_calls': trial['integration_metrics']['solver_calls'], 'nfev': trial['integration_metrics']['nfev']}


def differences(a, b, path='', found=None):
    found = [] if found is None else found
    if a == b:
        return found
    if len(found) >= 30:
        return found
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            differences(a.get(key), b.get(key), path + '.' + str(key), found)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for index, (left, right) in enumerate(zip(a, b)):
            differences(left, right, path + '[' + str(index) + ']', found)
            if len(found) >= 30:
                break
    else:
        found.append({'path': path, 'baseline': repr(a)[:250], 'optimized': repr(b)[:250]})
    return found


def checkpoint_events(path):
    events, previous = [], 0.
    for line in Path(path).read_text(encoding='utf8').splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        current = row.get('phase_seconds', {}).get('rolling_checkpoint_write', 0.)
        if current > previous:
            events.append({'completion_wall_seconds': row['wall_seconds'],
                           'stimulus_seconds': row['stimulus_seconds'],
                           'write_seconds': current - previous})
            previous = current
    return events


def checkpoint_policy(source):
    tree = ast.parse(source)
    triggers, resets = [], 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and 'last_checkpoint' in ast.unparse(node.test):
            triggers.append(ast.dump(node.test))
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'last_checkpoint' for target in node.targets):
            if ast.unparse(node.value) == 'time.perf_counter()':
                resets += 1
    return {'trigger_AST': triggers, 'timer_resets': resets}


def scientific_metadata(value):
    """Separate only the original per-process perf_counter diagnostics."""
    result = dict(value)
    backend = dict(result['compute_backend'])
    for key in DIAGNOSTIC_TIMERS:
        timing = backend[key]
        require(isinstance(timing, (float, int)) and math.isfinite(timing) and timing >= 0,
                'Malformed diagnostic timing: ' + key)
        del backend[key]
    result['compute_backend'] = backend
    return result


def main():
    began = time.perf_counter()
    name = sys.argv[1] if len(sys.argv) > 1 else 'complete_comparison.json'
    require(Path(name).name == name, 'Comparison report name must remain inside its reports folder')
    report = FOLDER / name
    require(not report.exists(), 'Comparison report already exists; preserve it')
    receipt = read(ROOT / 'reports/clean_runtime_20261018/active_video_run.json')
    require(receipt.get('fast_lossless_storage') is True, 'Active receipt is not the optimized run')
    original = ROOT / receipt['baseline_output']
    optimized = ROOT / receipt['output_relative']
    control = ROOT / receipt['control_relative']
    a, b = read(original / 'trial.json'), read(optimized / 'trial.json')
    av, bv = read(FOLDER / 'baseline_video_verified.json'), read(control / 'optimized_video_verified.json')
    for value in (av, bv):
        require(value['status'] == 'ACTUAL_VIDEO_05S_OUTPUT_VERIFIED', 'A complete artifact verification did not pass')
    am, bm = read(original / 'manifest.json'), read(optimized / 'manifest.json')
    checks = {'same_configuration': canonical_hash(am['config']) == canonical_hash(bm['config']),
        'same_model': am['runtime']['model_hash'] == bm['runtime']['model_hash'],
        'same_assembly': am['assembly_hash'] == bm['assembly_hash'],
        'same_exact_input_manifest': am['input_assumptions']['source_sha256'] == bm['input_assumptions']['source_sha256'],
        'same_frozen_background': am['identity']['background_start']['checkpoint_sha256'] == bm['identity']['background_start']['checkpoint_sha256'],
        'same_initial_checkpoint_bytes': av['start_checkpoint']['sha256'] == bv['start_checkpoint']['sha256'],
        'same_final_checkpoint_bytes': av['end_checkpoint']['sha256'] == bv['end_checkpoint']['sha256'],
        'same_solver_calls': a['integration_metrics']['solver_calls'] == b['integration_metrics']['solver_calls'],
        'same_nfev': a['integration_metrics']['nfev'] == b['integration_metrics']['nfev']}
    numeric_equal, unit_equal, metadata_equal, raw_metadata_equal = True, True, True, True
    max_absolute = 0.
    metadata_differences, raw_metadata_differences, bytes_before, bytes_after = [], [], 0, 0
    diagnostic_timing = {key: {'baseline_values': set(), 'optimized_values': set(), 'different_windows': 0}
                         for key in DIAGNOSTIC_TIMERS}
    for index in range(250):
        left = original / ('windows/window_%08d.npz' % index)
        right = optimized / ('windows/window_%08d.npz' % index)
        with np.load(left, allow_pickle=False) as la, np.load(right, allow_pickle=False) as lb:
            require(set(la.files) == set(lb.files), 'Recording member set changed')
            for key in set(la.files) - {'manifest_json'}:
                x, y = la[key], lb[key]
                require(x.shape == y.shape and x.dtype == y.dtype, 'Recording layout/dtype changed: ' + key)
                if x.tobytes() != y.tobytes():
                    numeric_equal = False
                    max_absolute = max(max_absolute, float(np.max(np.abs(x - y))))
            old, new = json.loads(str(la['manifest_json'])), json.loads(str(lb['manifest_json']))
            unit_equal &= old['ports'] == new['ports']
            if old['metadata'] != new['metadata']:
                raw_metadata_equal = False
                if not raw_metadata_differences:
                    raw_metadata_differences = differences(old['metadata'], new['metadata'])
            for key in DIAGNOSTIC_TIMERS:
                values = diagnostic_timing[key]
                left_time, right_time = old['metadata']['compute_backend'][key], new['metadata']['compute_backend'][key]
                values['baseline_values'].add(left_time)
                values['optimized_values'].add(right_time)
                values['different_windows'] += int(left_time != right_time)
            old_content, new_content = scientific_metadata(old['metadata']), scientific_metadata(new['metadata'])
            if old_content != new_content:
                metadata_equal = False
                if not metadata_differences:
                    metadata_differences = differences(old_content, new_content)
        bytes_before += left.stat().st_size
        bytes_after += right.stat().st_size
        if index % 50 == 0:
            print('WINDOWS_COMPARED', index + 1, '/250', flush=True)
    checks.update(recorded_numeric_bits_equal=numeric_equal, recorded_units_meanings_equal=unit_equal,
                  all_scientific_and_event_metadata_equal=metadata_equal)
    (FOLDER / (report.stem + '_observations.json')).write_text(json.dumps({
        'baseline': str(original), 'optimized': str(optimized), 'windows': 250, 'samples': 501,
        'checks': checks, 'maximum_absolute_difference': max_absolute,
        'metadata_differences': metadata_differences, 'baseline_record_bytes': bytes_before,
        'optimized_record_bytes': bytes_after}, indent=2, allow_nan=False) + '\n', encoding='utf8')
    with zipfile.ZipFile(FOLDER / 'source_rollback.zip') as archive:
        policy_before = checkpoint_policy(archive.read('tools/run_video_05s.py').decode('utf8'))
    policy_after = checkpoint_policy((ROOT / 'tools/run_video_05s.py').read_text(encoding='utf8'))
    checks['same_checkpoint_trigger_and_timer_reset_rules'] = policy_before == policy_after
    baseline_receipt = read(FOLDER / 'baseline_run_receipt.json')
    baseline_events, optimized_events = checkpoint_events(baseline_receipt['stdout']), checkpoint_events(receipt['stdout'])
    physical_sources = read(FOLDER / 'frozen_runtime_unchanged.json')
    checks['frozen_numerical_sources_unchanged'] = all(file_sha(ROOT / item['path']) == item['sha256'] for item in physical_sources)
    cold = read(ROOT / 'reports/clean_runtime_20261018/video_stage_20261018T140425Z/latest_observed_progress.json')
    failed = ROOT / 'reports/notebook_clean_grouped_transient_video05s_20261018T141627Z'
    failed_trial = read(failed / 'failure.json')
    failure_time = datetime.fromtimestamp((failed / 'failure.json').stat().st_mtime, timezone.utc)
    optical_start = datetime.fromisoformat('2026-10-18T14:24:58.093528+00:00')
    optical_end = datetime.fromisoformat(cold['last_new_ready_utc'].replace('Z', '+00:00'))
    creation = cold.get('worker_creation_utc')
    if creation is None:
        history = ROOT / 'reports/clean_runtime_20261018/video_stage_20261018T140425Z/progress_telemetry.jsonl'
        for line in history.read_text(encoding='utf-8-sig').splitlines():
            row = json.loads(line)
            if row.get('worker_pid') == 19348 and row.get('worker_creation_utc'):
                creation = row['worker_creation_utc']
                break
    require(creation is not None, 'Actual first-worker creation time missing from retained history')
    old_creation = datetime.fromisoformat(creation.replace('Z', '+00:00'))
    recovery_creation = datetime.fromisoformat(read(FOLDER / 'baseline_run_receipt.json')['worker_created_utc'].replace('Z', '+00:00'))
    old_input = read(failed / 'input_stream_shards/manifest.json')
    actual_input = read(original / 'input_stream_shards/manifest.json')
    productive_shards = old_input['channels']['native_light']['shards'] == actual_input['channels']['native_light']['shards']
    before, after = phase_summary(a), phase_summary(b)
    speedups = {key: before[key] / after[key] for key in before if key.endswith('_seconds') and after[key] > 0}
    result = {'status': 'MATCHED_ACTUAL_VIDEO_05S_EQUIVALENCE_AND_PERFORMANCE_PASS' if all(checks.values()) else 'COMPARISON_REQUIRES_REVIEW',
        'baseline_directory': str(original), 'optimized_directory': str(optimized), 'checks': checks,
        'duration_seconds_each': .5, 'samples_each': 501, 'windows_each': 250,
        'full_state_scalars': 4967023, 'recording': am['recording'],
        'maximum_recorded_absolute_difference': max_absolute,
        'full_state_clock_RNG_history_identity_proof': {'initial_sha256': av['start_checkpoint']['sha256'],
            'baseline_final_sha256': av['end_checkpoint']['sha256'], 'optimized_final_sha256': bv['end_checkpoint']['sha256'],
            'meaning': 'Identical original-protocol whole files include every registered state and every auxiliary clock/RNG/history/input field'},
        'record_metadata_differences': metadata_differences,
        'raw_metadata_including_process_timers_equal': raw_metadata_equal,
        'raw_metadata_difference_example': raw_metadata_differences,
        'diagnostic_timing_differences': {key: {**value, 'baseline_values': sorted(value['baseline_values']),
            'optimized_values': sorted(value['optimized_values'])} for key, value in diagnostic_timing.items()},
        'diagnostic_timing_classification': 'Only compute_backend.compile_or_cache_load_seconds and compute_backend.plan_seconds are separated. Both are assigned perf_counter durations in unchanged compiled_kernel.py and excluded from persistent identity by unchanged network.py. All original values remain in both recordings; every other metadata field is compared exactly.',
        'baseline': before, 'optimized': after,
        'baseline_phase_seconds': a['phase_seconds'], 'optimized_phase_seconds': b['phase_seconds'],
        'baseline_integration_metrics': a['integration_metrics'], 'optimized_integration_metrics': b['integration_metrics'],
        'speedup_ratios': speedups, 'baseline_memory': a['memory_statistics'], 'optimized_memory': b['memory_statistics'],
        'checkpoint_policy': {'rule': 'At each unchanged 2ms window boundary: if at least60 wall seconds elapsed after the previous checkpoint write finished, save a complete checkpoint; then reset the timer',
            'policy_AST_equal': policy_before == policy_after, 'baseline_rolling_count': len(baseline_events),
            'optimized_rolling_count': len(optimized_events), 'baseline_events': baseline_events,
            'optimized_events': optimized_events,
            'interpretation': 'Trigger rule unchanged. Faster writes/processing can change actual simulated-time positions and observed counts; no checkpoint was intentionally downsampled.'},
        'record_files_bytes': {'baseline': bytes_before, 'optimized': bytes_after},
        'optics': {'scope': '142 newly computed groups/1730 frame-tiles; 490 frame-tiles already cached; not a fully cold 2220-frame-tile run',
            'MATLAB_worker_creation_utc': optical_start.isoformat(), 'last_new_ready_utc': optical_end.isoformat(),
            'worker_start_to_last_new_ready_seconds': (optical_end - optical_start).total_seconds(),
            'all_15_native_count_shards_unchanged_after_recovery': productive_shards,
            'effective_baseline_input_preparation_seconds': a['phase_seconds']['ISET_input_or_strict_input_cache_load'],
            'optimized_reuses_identical_validated_input_manifest': bm['input_assumptions']['source_sha256']},
        'first_geometry_failure': {'directory': str(failed), 'error': failed_trial['error'], 'neural_seconds_lost': 0.,
            'first_worker_creation_utc': old_creation.isoformat(), 'failure_file_utc': failure_time.isoformat(),
            'first_worker_to_failure_seconds': (failure_time - old_creation).total_seconds(),
            'failure_to_recovery_launch_seconds': (recovery_creation - failure_time).total_seconds(),
            'bounded_same_geometry_probe_seconds': read(ROOT / 'reports/clean_runtime_20261018/geometry_failure_20261018T1614Z/result.json')['wall_seconds'],
            'root_cause': 'UNRESOLVED; isolated probe and later full original gates passed; no check bypass or scientific-source edit',
            'productive_optics_retained_and_reused': productive_shards},
        'background_initialization': am['background_initialization'],
        'comparison_wall_seconds': time.perf_counter() - began}
    report.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({'status': result['status'], 'checks': checks, 'speedups': speedups}), flush=True)
    require(all(checks.values()), 'Complete comparison contains differences; inspect saved report')


if __name__ == '__main__':
    main()
