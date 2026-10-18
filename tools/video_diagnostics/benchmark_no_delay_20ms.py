"""Actual20-40ms paired benchmark of only the empty-publication guard."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if os.environ.get('OPENBLAS_NUM_THREADS') != '1':
    raise RuntimeError('Original BLAS environment required before NumPy import')
from tools.benchmark_compiler_snapshot import use_v7_compiler
use_v7_compiler()
import numpy as np
from retina_platform.config import RunConfig
from retina_sim.atlas_population_network import atlas_population_network
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.checkpoint_loading import restore_retina_payload
from retina_sim.contracts import SolverConfig
from retina_sim.disk_graph_storage import file_sha
from retina_sim.execution_backends import make_runner, verify_cuda_execution
from retina_sim.lossless_video_storage import checkpoint_file_spooled
from retina_sim.offline_input import rebase_offline_inputs
from retina_sim.recording import RecordingPlan
from retina_sim.recording_indices import IndexedRecordingPlan
from retina_sim.streamed_input import load_streamed_input
from tools.run_video_05s import recording_cohort
from retina_sim.no_delay_publication import no_empty_link_advance

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
output = (ROOT / args.output).resolve()
if not output.is_relative_to(ROOT):
    raise ValueError('Project output required')
output.mkdir(exist_ok=False)
source = ROOT / 'reports/notebook_clean_geometry_observed_video05s_20261018T162158Z'
guard = MemoryGuard(14336 * 1024**2, 8192 * 1024**2)
from first_json_fault_witness import install
cleanup, terminal = install(output / 'first_error')


def save(name, value):
    path = output / (name + '.json')
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    os.replace(temporary, path)


def boundary():
    if not guard.check():
        raise MemoryError('Original resource reserve reached')


def progress(stage, **fields):
    value = {'stage': stage, 'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat(), **fields}
    save('progress', value)
    print(json.dumps(value), flush=True)


def restore(runner, path, source_sha):
    document = json.loads(path.read_bytes())
    state, receipt = restore_retina_payload(runner, document, expected_input_context={
        'phase': 'media_stimulus', 'input_source_sha256': source_sha})
    del document
    gc.collect()
    boundary()
    return state, receipt


def scientific_metadata(value):
    # Compare the persisted NPZ protocol, which normalizes tuples such as
    # torch device_capability to JSON arrays. No scientific fields are dropped.
    value = json.loads(json.dumps(value, allow_nan=False))
    compute = dict(value['compute_backend'])
    for key in ('plan_seconds', 'compile_or_cache_load_seconds'):
        compute.pop(key, None)
    value['compute_backend'] = compute
    return value


def verify_window(result, index):
    with np.load(source / ('windows/window_%08d.npz' % index), allow_pickle=False) as original:
        if set(original.files) - {'manifest_json'} != set(result.outputs_with_units):
            raise AssertionError('Observation scope changed')
        for key, quantity in result.outputs_with_units.items():
            before = original[key]
            if before.dtype != quantity.values.dtype or before.shape != quantity.values.shape or before.tobytes() != quantity.values.tobytes():
                raise AssertionError('Numeric bytes changed: ' + key)
        document = json.loads(str(original['manifest_json']))
        if scientific_metadata(document['metadata']) != scientific_metadata(result.metadata):
            raise AssertionError('Scientific/event metadata changed')
        for key, quantity in result.outputs_with_units.items():
            if any(document['ports'][key][field] != getattr(quantity, field) for field in ('unit', 'meaning', 'evidence_tag')):
                raise AssertionError('Observed unit/meaning changed')


def run_windows(runner, state, channels, plan, first, stop, label, advance=None):
    rows = []
    operation = runner.advance if advance is None else advance
    for index in range(first, stop):
        boundary()
        with np.load(source / ('windows/window_%08d.npz' % index), allow_pickle=False) as original:
            times = original['time'].copy()
        before = dict(runner.integration_metrics)
        tick = time.perf_counter()
        state, result = operation(state, channels, float(times[-1]), sample_times=times, recording_plan=plan)
        elapsed = time.perf_counter() - tick
        state.input_position['recorded_window_count'] = index + 1
        # All verification, checkpoint I/O and progress below are outside advance timing.
        verify_window(result, index)
        rows.append({'source_window': index, 'advance_seconds': elapsed,
                     'all_current_process_metadata_sha256': hashlib.sha256(json.dumps(result.metadata, sort_keys=True).encode()).hexdigest(),
                     'solver_seconds': runner.integration_metrics['integration_wall_seconds'] - before['integration_wall_seconds'],
                     'solver_calls': runner.integration_metrics['solver_calls'] - before['solver_calls'],
                     'nfev': runner.integration_metrics['nfev'] - before['nfev']})
        progress(label, completed_window=index, advance_seconds=elapsed)
        del result
    return state, rows


def main():
    began = time.perf_counter()
    tracked = ('retina_sim/hybrid.py', 'retina_sim/gpu_hybrid.py', 'retina_sim/gpu_dop853.py',
               'retina_sim/contracts.py', 'retina_sim/network.py', 'retina_sim/recording.py')
    before_sources = {name: file_sha(ROOT / name) for name in tracked}
    save('source_hashes_before', before_sources)
    boundary()
    if guard.last['available_physical_bytes'] < 15 * 1024**3:
        raise MemoryError('Original startup reserve')
    manifest = json.loads((source / 'manifest.json').read_bytes())
    cfg = RunConfig.from_dict(manifest['config'])
    if file_sha(cfg.media_path) != manifest['media_sha256']:
        raise ValueError('Original actual video changed')
    construction = replace(cfg, native_background_values=None, frontend_signature='')
    progress('ASSEMBLY')
    compiled, _, audit = atlas_population_network(ROOT, duration=.0001, light_context=construction.light_context(),
        owner_bounds_mm=cfg.owner_bounds(), halo_mm=.04, max_spatial_cells=cfg.max_spatial_cells,
        seed=cfg.seed, execution_profile=cfg.execution_profile, configuration_path=cfg.circuit_configuration_path)
    solver = SolverConfig(**audit['resolved_execution_profile']['solver'])
    del audit
    gc.collect()
    boundary()
    runner = make_runner(compiled, cfg.neural_integration_backend)
    if runner.model.model_hash != manifest['runtime']['model_hash'] or compiled.assembly_hash != manifest['assembly_hash']:
        raise ValueError('Actual model/assembly identity changed')
    save('backend_qualification', verify_cuda_execution(runner, solver, cfg.light_context()))
    progress('STRICT_INPUT_AND_START_RESTORE')
    packet, stores = load_streamed_input(source / 'input_stream_shards/manifest.json', compiled,
        cfg.light_context(), reference_id=compiled.global_disk_circuit.record['reference_id'])
    source_sha = packet.source_summary['source_sha256']
    if source_sha != '38cb38afd58deb93b433806d619f442602f84fc4237e3c8d1364ea27d2d33c6f':
        raise ValueError('Exact prepared input changed')
    prefix = ROOT / 'reports/advance_exclusive_profile_20261018T230442Z/prefix_20ms_checkpoint.json'
    prefix_sha = file_sha(prefix)
    state, start_receipt = restore(runner, prefix, source_sha)
    selected, _, _ = recording_cohort(compiled)
    plan = IndexedRecordingPlan(RecordingPlan(ports=(), states=None, cell_ids=selected, max_bytes=64 * 1024**2))
    if sum(len(plan.indices(spec.cell_ids)) for spec in runner.model.state_specs) != 37449:
        raise ValueError('Original recording scope changed')
    patched_advance, procedure = no_empty_link_advance(runner)
    save('execution_procedure', procedure)
    origin = state.input_position['stimulus_origin_seconds']

    def fresh_channels():
        return rebase_offline_inputs(packet, compiled, cfg.light_context(), stimulus_origin_seconds=origin,
            grid_origin_seconds=state.grid_origin_seconds)

    channels = fresh_channels()
    progress('CONTROL_20_TO_40MS')
    state, control = run_windows(runner, state, channels, plan, 10, 20, 'CONTROL_20_TO_40MS')
    checkpoint_file_spooled(runner, state, output / 'control_40ms_checkpoint.json')
    del state, channels
    gc.collect()
    if file_sha(prefix) != prefix_sha:
        raise ValueError('Shared strict starting checkpoint changed')
    state, patched_restore = restore(runner, prefix, source_sha)
    channels = fresh_channels()
    progress('PATCHED_20_TO_40MS')
    state, measured = run_windows(runner, state, channels, plan, 10, 20, 'PATCHED_20_TO_40MS', advance=patched_advance)
    checkpoint_file_spooled(runner, state, output / 'patched_40ms_checkpoint.json')
    hashes = {name: file_sha(output / (name + '_40ms_checkpoint.json')) for name in ('control', 'patched')}
    hashes['previous_independent_40ms'] = file_sha(ROOT / 'reports/advance_exclusive_profile_20261018T230442Z/control_40ms_checkpoint.json')
    if len(set(hashes.values())) != 1:
        raise AssertionError('Complete state/RNG/history checkpoint bytes differ')
    after_sources = {name: file_sha(ROOT / name) for name in tracked}
    if before_sources != after_sources:
        raise AssertionError('Original numerical source file changed')
    control_seconds = sum(row['advance_seconds'] for row in control)
    measured_seconds = sum(row['advance_seconds'] for row in measured)
    if any(left['all_current_process_metadata_sha256'] != right['all_current_process_metadata_sha256']
           or left['solver_calls'] != right['solver_calls'] or left['nfev'] != right['nfev']
           for left, right in zip(control, measured)):
        raise AssertionError('Paired metadata or integration operation counts differ')
    control_solver = sum(row['solver_seconds'] for row in control)
    patched_solver = sum(row['solver_seconds'] for row in measured)
    result = {'status': 'NO_DELAY_PUBLICATION_ACTUAL_20MS_EQUIVALENCE_AND_TIMING_PASS',
              'scope': 'Actual video20-40ms,10 original2ms windows each; same retained20ms checkpoint; not a complete0.5s benchmark',
              'timed_duration_seconds_each': .020, 'timed_video_start_seconds': .020,
              'timed_video_end_seconds': .040, 'timed_windows_each': 10,
              'original_video_exposure_transition_seconds': 1 / 30,
              'native_input_grid_seconds': .0001, 'record_sample_seconds': .001,
              'input_manifest_sha256': source_sha, 'model_hash': runner.model.model_hash,
              'assembly_hash': compiled.assembly_hash, 'start_restore': start_receipt,
              'patched_restore': patched_restore, 'execution_procedure': procedure,
              'shared20ms_checkpoint_sha256': prefix_sha,
              'original_numerical_source_files_unchanged': before_sources == after_sources,
              'all_observed_numeric_bytes_units_scientific_event_metadata_equal': True,
              'complete_endpoint_state_clock_RNG_history_checkpoint_SHA256': hashes,
              'control_advance_seconds': control_seconds, 'patched_advance_seconds': measured_seconds,
              'advance_seconds_saved': control_seconds - measured_seconds,
              'advance_reduction_percent': (1 - measured_seconds / control_seconds) * 100,
              'advance_speedup': control_seconds / measured_seconds,
              'control_solver_seconds_included': control_solver, 'patched_solver_seconds_included': patched_solver,
              'control_outside_solver_seconds': control_seconds - control_solver,
              'patched_outside_solver_seconds': measured_seconds - patched_solver,
              'outside_solver_seconds_saved': (control_seconds - control_solver) - (measured_seconds - patched_solver),
              'pairing_limit': 'One same-process sequential pair, no nested timing hooks; includes normal cache/order/system variation. No full0.5s speedup extrapolation',
              'all_paired_metadata_including_diagnostic_timers_equal': True,
              'solver_calls_each': sum(row['solver_calls'] for row in control),
              'nfev_each': sum(row['nfev'] for row in control),
              'control': control, 'patched': measured, 'memory': guard.last,
              'excluded_from_advance_timing': ['assembly/setup', 'strict restore', 'input prepare/rebase',
                  'verification', 'checkpoint encoding/I/O', 'NPZ disk write', 'progress and resource guard'],
              'overall_probe_seconds': time.perf_counter() - began}
    save('result', result)
    progress('COMPLETED')


try:
    main()
except BaseException as error:
    terminal(error)
    save('failure', {'type': type(error).__name__, 'error': str(error)})
    raise
finally:
    cleanup()
