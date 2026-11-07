"""Record one bounded 69#6 window from its frozen 5 s checkpoint."""
from pathlib import Path
import os
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import time
import traceback
import numpy as np
import scipy

from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_time_storage import load_bundle
from retina_sim.spatial_group_graph import digest
from retina_sim.spatial_group_model import BC, RGC
from tools.run_gin69_current_ac import CurrentRunner, load_current, advance, calendar, DEST, OUT, read, guard, atomic_json
from tools.run_hc_minimal_real_diagnostic_v1 import peak_memory_bytes


START, STOP, SAMPLE = 5., 5.5, .001
OUTPUT = DEST / 'internal_states'
CATEGORIES = ('native', 'inner', 'spike_states', 'clocks', 'input_position', 'observation_delay', 'readout_cursor')


def check_runtime():
    expected = Path(r'D:\anaconda\envs\snn_env\python.exe').resolve()
    if (Path(sys.executable).resolve() != expected or sys.version_info[:3] != (3, 11, 15)
            or np.__version__ != '1.26.4' or scipy.__version__ != '1.17.1'):
        raise ContractError('Use the frozen snn_env Python 3.11.15 / NumPy 1.26.4 / SciPy 1.17.1')
    if START != 5. or STOP != 5.5 or not 0 < STOP-START <= 1. or SAMPLE != .001:
        raise ContractError('Only the authorized 5.0–5.5 s window at 1 ms is allowed')
    return dict(executable=sys.executable, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        threads={name: os.environ[name] for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')})


def causal_hashes(arm):
    payload = arm.checkpoint_payload()
    return {name: canonical_hash(payload[name]) for name in CATEGORIES}


def collect(arm, t):
    """Observe the original accepted states without committing or advancing."""
    before = (arm.native_state.time_seconds, tuple(i.representation_id for i in arm.native_intervals))
    evaluation, _, _ = arm._native_evaluation(t, 'right')
    outputs = arm.inner.outputs(t, arm.inner_state)
    after = (arm.native_state.time_seconds, tuple(i.representation_id for i in arm.native_intervals))
    if before != after or arm.inner_time != t or arm.time_seconds != t:
        raise ContractError('Observation changed the accepted integration frontier')
    native = evaluation.outputs
    values = {}
    for key, source in (('V', 'Vis'), ('Vbar', 'Vbar'), ('release', 'release')):
        values['cones', key] = np.array(native[source], copy=True)
    values['HC', 'Vh'] = np.array(native['Vh'], copy=True)
    for key in ('w', 'z2'):
        values['HC', key] = np.array(native[key][arm.assembly.h1_rows], copy=True)
    values['cones', 'bridge_release'] = np.array(arm._release_cache, copy=True)
    for (pop, mechanism, field), value in arm.inner_state.items():
        if pop not in arm.assembly.populations:
            continue
        if field in ('V', 'terminal_release', 'release', 'A', 's', 'h'):
            key = 'release' if field == 'terminal_release' else field
            if (pop, key) in values:
                raise ContractError('Ambiguous population state field: '+str((pop, key)))
            values[pop, key] = np.array(value, copy=True)
    for pop in BC:
        values[pop, 'effective_release'] = np.array(outputs[pop, 'effective_release'], copy=True)
    for pop in RGC:
        for key in ('gE', 'gI', 'expected_rate_Hz'):
            values[pop, key] = np.array(outputs[pop, key], copy=True)
    return values, outputs


def field_metadata(arm, values):
    fields = {}
    for (pop, key), value in values.items():
        token = pop+'__'+key
        rows = np.arange(arm.assembly.populations[pop].group_count)
        if pop == 'HC' and key in ('w', 'z2'):
            rows = arm.assembly.h1_rows
        elif key == 'bridge_release':
            rows = arm.selection.cone_rows
        if value.shape != (len(rows),) or not np.isfinite(value).all():
            raise ContractError('Recorded field and original row mapping differ: '+token)
        unit = ('vH_native_voltage' if pop in ('cones', 'HC') and key in ('V', 'Vbar', 'Vh', 'w', 'z2')
            else 'V' if key == 'V' else 'S' if key in ('gE', 'gI') else 'Hz' if key == 'expected_rate_Hz'
            else 'dimensionless effective release' if key in ('release', 'effective_release', 'bridge_release') else 'dimensionless')
        label = dict(V='膜电位', Vbar='Cone Vbar 原生状态', Vh='H1/H2 原生反馈状态',
            w='H1 w 原生状态', z2='H1 z2 原生状态', release='模型释放状态',
            effective_release='BC 适应后有效释放', bridge_release='BC 桥接缓存 release',
            A='适应状态', s='输出门控', h='抑制状态', gE='兴奋电导', gI='抑制电导',
            expected_rate_Hz='原 RGC 期望放电率')[key]
        if pop == 'cones' and key == 'V':
            label = 'Cone 原生电压'
        source_kind = ('saved_bridge_cache' if key == 'bridge_release' else
            'frozen_native_equation_output' if pop in ('cones', 'HC') else
            'current_inner_output' if key in ('effective_release', 'gE', 'gI', 'expected_rate_Hz') else 'accepted_inner_state')
        fields[token] = dict(population=pop, key=key, array=token, rows=np.asarray(rows, dtype=int).tolist(),
            unit=unit, label=label, source_kind=source_kind, time='time_seconds', side='right')
    return fields


def run():
    started = time.perf_counter()
    environment = check_runtime()
    manifest_path = OUTPUT / 'manifest.json'
    if manifest_path.exists():
        existing = read(manifest_path)
        if (existing.get('status') == 'COMPLETE' and existing.get('start_seconds') == START
                and existing.get('stop_seconds') == STOP
                and digest(OUTPUT / existing['responses_file']) == existing['responses_sha256']):
            print(json.dumps(dict(status='REUSED_COMPLETE_WINDOW', manifest=str(manifest_path)), ensure_ascii=False), flush=True)
            return
        raise FileExistsError('Existing incomplete or different replay evidence retained: '+str(manifest_path))
    for name in ('OWNER_run.json', 'OWNER_recovery.json'):
        if (DEST/name).exists() and read(DEST/name).get('status') == 'RUNNING':
            raise ContractError('Original scientific execution owns this result; replay not started')
    OUTPUT.mkdir(exist_ok=True)
    checkpoint = DEST / 'CHECKPOINT_50000'
    original_paths = [DEST/name for name in ('PREFIX_45s.npz', 'BACKGROUND.json', 'readout/MODEL_FROZEN.json',
        'readout/FIXED_DEVELOPMENT_PREDICTIONS.npz', 'CHECKPOINT_50000/manifest.json', 'seed/manifest.json')]
    original_hashes = {str(path.relative_to(ROOT)): digest(path) for path in original_paths}
    metadata = dict(schema='retina-internal-state-window', status='PREPARING', start_seconds=START,
        stop_seconds=STOP, sample_seconds=SAMPLE, runtime=environment, fits=0,
        maximum_authorized_replay_seconds=1., scientific_parameters_changed=False,
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_manifest_sha256=digest(checkpoint/'manifest.json'))
    atomic_json(manifest_path, metadata)
    arm = None
    try:
        print(json.dumps(dict(status='LOADING_FROZEN_MODEL', target_seconds=STOP, replay_seconds=STOP-START)), flush=True)
        model, protocol, background = load_current()
        if protocol['stop_seconds'] != 45.:
            raise ContractError('Original protocol stop changed')
        saved = load_bundle(checkpoint, expected_kind='validation_evidence', expected_identity=background['identity'])['payload']
        if saved['controller']['time_seconds'] != START or saved['controller']['inner_time'] != START:
            raise ContractError('The required exact 5 s checkpoint is absent')
        arm = CurrentRunner.from_checkpoint(model.assembly, model.seed, protocol, saved, guard=guard)
        initial_hashes = causal_hashes(arm)
        for name in CATEGORIES:
            if initial_hashes[name] != canonical_hash(saved[name]):
                raise ContractError('Checkpoint recovery changed '+name)
        if arm.identity != saved['controller']['identity'] or 'rod_capture' in model.seed:
            raise ContractError('Frozen execution identity or Rod input changed')
        target_rows = np.flatnonzero(model.assembly.parent_population_rows['parasol_RGC_OFF'] == 7730)
        if target_rows.tolist() != [3] or model.assembly.populations['parasol_RGC_OFF'].counts[3] != 1:
            raise ContractError('Original 69#6 target row or membership changed')
        with np.load(DEST/'PREFIX_45s.npz', allow_pickle=False) as original:
            original_time = original['time_source_s'][50000:55001].copy()
            reference = {key: original[key][50000:55001].copy() for key in ('V', 'gE', 'gI')}
            expected_rate = original['expected_rate_base_Hz'][5000:5501].copy()
        np.testing.assert_array_equal(original_time, np.arange(50000, 55001)*.0001)
        initial_frontier = float(arm.native_state.time_seconds)
        first, initial_outputs = collect(arm, START)
        if causal_hashes(arm) != initial_hashes:
            raise ContractError('Initial observer changed scientific continuation state')
        np.testing.assert_array_equal(first['cones', 'release'][arm.selection.cone_rows], first['cones', 'bridge_release'])
        fields = field_metadata(arm, first)
        times = np.arange(5000, 5501)*.001
        traces = {pop+'__'+key: np.empty((len(value), len(times))) for (pop, key), value in first.items()}
        for (pop, key), value in first.items(): traces[pop+'__'+key][:, 0] = value
        populations = {}
        for pop in {item['population'] for item in fields.values()}:
            layout = model.assembly.populations[pop]
            populations[pop] = dict(ids=np.asarray(layout.group_ids).astype(str).tolist(),
                members=np.asarray(layout.counts, dtype=int).tolist(), coordinate_role=layout.coordinate_role,
                representation='independently_solved_BC_cell' if pop in BC else
                'effective_functional_pool' if pop.startswith(('same_', 'cross_')) else 'representative_group')
        metadata.update(status='RUNNING', model_identity=background['model_identity'], background_identity=background['identity'],
            execution_identity=arm.identity, initial_native_frontier_seconds=initial_frontier,
            parameter_identity=canonical_hash(model.seed['inner_parameters']), artifact_identity=model.assembly.artifact_sha256,
            exact_recovered_categories=list(CATEGORIES), original_protocol_stop_seconds=45.,
            source_seconds=START, records=1, fields=fields, populations=populations,
            responses_file='responses.npz', time_array='time_seconds', original_artifact_sha256=original_hashes,
            provenance=dict(tool_sha256=digest(Path(__file__)), runner_sha256=digest(ROOT/'tools/run_gin69_current_ac.py'),
                seed_manifest_sha256=background['seed_manifest_sha256'], input_policy_sha256=digest(OUT/'input_policy.json'),
                split_sha256=background['split_sha256'], unchanged_native_input_sha256=background['source_native_input_sha256'],
                numerical_protocol=protocol, prehistory='complete checkpoint continuation including original clocks, delay and RNG'))
        atomic_json(manifest_path, metadata)
        print(json.dumps(dict(status='EXACT_RECOVERY_PASS', wall_seconds=time.perf_counter()-started,
            native_frontier=initial_frontier, fields=len(fields))), flush=True)
        compared, recorded = 1, 1
        max_difference = dict.fromkeys(('V', 'gE', 'gI'), 0.)

        def compare(t, outputs):
            j = round((t-START)*10000)
            for key, port in (('V', 'voltage'), ('gE', 'gE'), ('gI', 'gI')):
                actual = float(outputs['parasol_RGC_OFF', port][3])
                delta = abs(actual-float(reference[key][j]))
                max_difference[key] = max(max_difference[key], delta)
                if actual != reference[key][j]:
                    raise ContractError(f'Replay differs from saved {key} at {t:.10f}: absolute difference {delta!r}')

        compare(START, initial_outputs)
        last_progress = time.perf_counter()
        loop_started = time.perf_counter()
        for end, flags in calendar(protocol):
            if end <= START: continue
            if end > STOP: break
            if end-START > 1. or end > 5.5:
                raise ContractError('Bounded replay stop exceeded')
            advance(arm, end, flags)
            if arm.native_state.time_seconds > STOP+.001:
                raise ContractError('Native lookahead exceeds original bounded allowance')
            if flags['rod']:
                outputs = arm.inner.outputs(end, arm.inner_state)
                compare(end, outputs); compared += 1
                tick = round(end*10000)
                if tick % 10 == 0:
                    if abs(end-times[recorded]) > 1e-12:
                        raise ContractError('Recorded original 1 ms index changed')
                    values, _ = collect(arm, end)
                    if set(values) != set(first):
                        raise ContractError('Recorded field set changed')
                    for (pop, key), value in values.items():
                        if not np.isfinite(value).all(): raise ContractError('Nonfinite recorded state')
                        traces[pop+'__'+key][:, recorded] = value
                    recorded += 1
            if time.perf_counter()-last_progress >= 20.:
                metadata.update(source_seconds=float(end), records=recorded, wall_seconds=time.perf_counter()-started,
                    peak_working_set_bytes=peak_memory_bytes())
                atomic_json(manifest_path, metadata)
                print(json.dumps({key: metadata[key] for key in ('status', 'source_seconds', 'records', 'wall_seconds', 'peak_working_set_bytes')}), flush=True)
                last_progress = time.perf_counter()
        if recorded != 501 or compared != 5001 or arm.time_seconds != STOP:
            raise ContractError('Bounded replay did not reach the exact recorded endpoint')
        np.testing.assert_array_equal(traces['parasol_RGC_OFF__expected_rate_Hz'][3], expected_rate)
        bridge_rows = arm.selection.cone_rows
        np.testing.assert_array_equal(traces['cones__release'][bridge_rows], traces['cones__bridge_release'])
        final_hashes = causal_hashes(arm)
        collect(arm, STOP)
        if causal_hashes(arm) != final_hashes:
            raise ContractError('Final observer changed scientific continuation state')
        for path in original_paths:
            if digest(path) != original_hashes[str(path.relative_to(ROOT))]:
                raise ContractError('Original scientific artifact changed: '+str(path))
        temporary = OUTPUT / 'responses.partial.npz'
        np.savez_compressed(temporary, time_seconds=times, **traces)
        destination = OUTPUT / 'responses.npz'
        if destination.exists(): raise FileExistsError('Existing recorded window retained')
        temporary.replace(destination)
        metadata.update(status='COMPLETE', source_seconds=STOP, records=recorded, responses_sha256=digest(destination),
            response_bytes=destination.stat().st_size, final_native_frontier_seconds=float(arm.native_state.time_seconds),
            wall_seconds=time.perf_counter()-started, replay_wall_seconds=time.perf_counter()-loop_started,
            peak_working_set_bytes=peak_memory_bytes(), source_data_modified=False,
            validation=dict(exact_checkpoint_recovery=True, observer_preserves_causal_state=True,
                target_V_gE_gI_exact_samples=compared, target_max_abs_difference=max_difference,
                original_expected_rate_exact=True, native_release_matches_bridge_exact=True, original_files_unchanged=True),
            limitations=['Only 5.0–5.5 s was replayed; other times retain their original recording coverage.',
                'Native values use the original accepted solver dense representation at actual public timestamps, not shifted frontier values.',
                'Cone/HC are representative groups; ordinary AC are functional pools; BC retains independently solved member identities.',
                'HC w/z2 only bind saved H1 rows; no H2 placeholder states are recorded.',
                'BC release is terminal_release before adaptation; effective_release is its separately recorded adapted model output.',
                'No refitting, scientific parameter changes or qualification change.'])
        atomic_json(manifest_path, metadata)
        print(json.dumps({key: metadata[key] for key in ('status', 'start_seconds', 'stop_seconds', 'records', 'response_bytes', 'wall_seconds', 'validation')}, ensure_ascii=False), flush=True)
    except BaseException:
        metadata.update(status='FAILED', wall_seconds=time.perf_counter()-started, error=traceback.format_exc())
        atomic_json(manifest_path, metadata)
        raise
    finally:
        if arm is not None: arm.close()


if __name__ == '__main__':
    run()
