"""Resume only the frozen current 69#6 run in its original scientific runtime.

The original runner source participates in its checkpoint identity and is left
unchanged. This entry adds recovery/recording orchestration, not model dynamics.
"""
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from datetime import datetime, timezone
import json
import time
import traceback

import numpy as np
import scipy

from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_time_storage import load_bundle, save_bundle
from retina_sim.spatial_group_graph import digest
from tools.run_gin69_current_ac import (CurrentRunner, load_current, advance, calendar,
    DEST, OUT, read, guard, atomic_json, fit)
from tools.run_hc_minimal_real_diagnostic_v1 import peak_memory_bytes
from tools.durable_research_runtime_v1 import process_identity


def runtime():
    expected = Path(r'D:\anaconda\envs\snn_env\python.exe').resolve()
    if (Path(sys.executable).resolve() != expected or sys.version_info[:3] != (3, 11, 15)
            or np.__version__ != '1.26.4' or scipy.__version__ != '1.17.1'):
        raise ContractError('Use the frozen snn_env Python 3.11.15 / NumPy 1.26.4 / SciPy 1.17.1; .venv is not this experiment')
    return dict(executable=sys.executable, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')})


def latest_checkpoint(bg):
    paths = [p for p in DEST.glob('CHECKPOINT_*') if p.is_dir() and p.name.removeprefix('CHECKPOINT_').isdigit()]
    for p in sorted(paths, key=lambda p:int(p.name.removeprefix('CHECKPOINT_')), reverse=True):
        # A partial or mismatched newest checkpoint is an error, not a silent fallback.
        saved = load_bundle(p, expected_kind='validation_evidence', expected_identity=bg['identity'])
        payload = saved['payload']; time_s = payload['controller']['time_seconds']
        if round(time_s*10000) != int(p.name.removeprefix('CHECKPOINT_')) or time_s < 5.:
            raise ContractError('Checkpoint directory and completed inner time differ')
        return p, saved
    raise ContractError('No complete frozen checkpoint; restart from zero is forbidden')


def run(*, check_only=False):
    started = time.perf_counter(); environment = runtime()
    if (DEST/'NEURAL_RESULT.json').exists() and read(DEST/'NEURAL_RESULT.json').get('status') == 'COMPLETE':
        raise FileExistsError('Completed current trajectory retained; select fit explicitly')
    model, protocol, bg = load_current()
    checkpoint, saved = latest_checkpoint(bg)
    payload = saved['payload']
    arm = CurrentRunner.from_checkpoint(model.assembly, model.seed, protocol, payload, guard=guard)
    try:
        categories = ('native', 'inner', 'spike_states', 'clocks', 'input_position', 'observation_delay', 'readout_cursor')
        restored = arm.checkpoint_payload(recording=payload['recording'])
        for key in categories:
            if canonical_hash(restored[key]) != canonical_hash(payload[key]):
                raise ContractError('Checkpoint restoration changed '+key)
        if arm.identity != payload['controller']['identity'] or 'rod_capture' in model.seed:
            raise ContractError('Original execution identity or engineering Rod input changed')
        pop = 'parasol_RGC_OFF'
        rows = np.flatnonzero(model.assembly.parent_population_rows[pop] == 7730)
        if len(rows) != 1 or model.assembly.populations[pop].counts[rows[0]] != 1:
            raise ContractError('Frozen 69#6 target member identity differs')
        row = int(rows[0]); initial_time = arm.time_seconds
        recording = payload['recording']; recorded = recording['recorded']
        if recorded != round(initial_time*10000)+1 or set(recording['arrays']) != {'V','gE','gI'}:
            raise ContractError('Checkpoint voltage recording frontier differs')
        arrays = {name:np.empty(450001) for name in recording['arrays']}
        for name, value in recording['arrays'].items():
            if value.shape != (recorded,) or not np.isfinite(value).all():
                raise ContractError('Incomplete saved voltage/conductance prefix')
            arrays[name][:recorded] = value
        expected = np.empty(45000)
        saved_expected = recording['expected']
        if saved_expected.shape != (round(initial_time*1000),):
            raise ContractError('Saved expected-rate prefix differs')
        expected[:len(saved_expected)] = saved_expected
        values = arm.inner.outputs(initial_time, arm.inner_state)
        for name, port in [('V','voltage'), ('gE','gE'), ('gI','gI')]:
            if values[pop,port][row] != arrays[name][recorded-1]:
                raise ContractError('Restored endpoint differs from persisted '+name)
        if initial_time < 45.:
            expected[round(initial_time*1000)] = values[pop,'expected_rate_Hz'][row]
        baseline = model.upstream.upstream.uniform_base
        receipt = read(ROOT/'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02/h2_priority_assets/RECEIPT.json')
        evidence = dict(status='EXACT_CHECKPOINT_RECOVERY_PASS', runtime=environment,
            root_cause='Wrong scientific runtime: .venv NumPy 2.4.6 rebuilt different floating-point parameter/state contents; original snn_env restores frozen identities exactly',
            comparison_stage='Builder H2PriorityModel, before source background or checkpoint restoration',
            wrong_runtime_uniform_identity='b202e31a5b36a493b8dd63de307d368bcf2115cf1ca480989f46b86159ed5520',
            expected_uniform_identity=receipt['uniform_model_identity'], actual_uniform_identity=baseline.identity,
            expected_model_identity=bg['model_identity'], actual_model_identity=model.identity,
            artifact=model.assembly.artifact_sha256, parameter_hash=canonical_hash(model.seed['inner_parameters']),
            source_background_identity=bg['identity'], source_seed_manifest_sha256=digest(DEST/'seed/manifest.json'),
            split_sha256=digest(OUT/'SPLIT_CONTRACT.json'), input_policy_sha256=digest(OUT/'input_policy.json'),
            rod_mode='original engineering; no rod_capture binding',
            checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_manifest_sha256=digest(checkpoint/'manifest.json'),
            execution_identity=arm.identity, restored_inner_seconds=initial_time,
            restored_native_frontier_seconds=arm.native_state.time_seconds, restored_recorded_samples=recorded,
            exact_restored_categories=list(categories), source_arrays_or_equations_modified=False,
            identity_checks_bypassed=False, background_recomputed_for_experiment=False,
            note='Builder reconstructs its diagnostic identity; actual formal source background is the unchanged saved seed',
            existing_parity=digest(DEST/'EXECUTION_PARITY.json'), recovery_wall_seconds=time.perf_counter()-started,
            peak_working_set_bytes=peak_memory_bytes(), authorization='User 2026-11-05 explicitly resumes only 69#6')
        if evidence['actual_uniform_identity'] != evidence['expected_uniform_identity']:
            raise ContractError('Uniform baseline still differs')
        atomic_json(DEST/'IDENTITY_RECOVERY.json', evidence)
        print(json.dumps({k:evidence[k] for k in ('status','restored_inner_seconds','execution_identity','recovery_wall_seconds')}), flush=True)
        if check_only:
            return
        # The saved prefix is copied byte-for-byte; no 0..5 s neural advancement.
        prefix_hash = canonical_hash(recording)
        simulation_started = time.perf_counter()
        for end, flags in calendar(protocol):
            if end <= arm.inner_time:
                continue
            advance(arm, end, flags)
            if not (flags['rod'] or end == 45.):
                continue
            j = round(end*10000)
            if j != recorded:
                raise ContractError('Native voltage sample omitted or repeated')
            values = arm.inner.outputs(end, arm.inner_state)
            for name, port in [('V','voltage'), ('gE','gE'), ('gI','gI')]:
                arrays[name][j] = values[pop,port][row]
            if j % 10 == 0 and j < 450000:
                expected[j//10] = values[pop,'expected_rate_Hz'][row]
            recorded += 1
            if j % 5000 == 0:
                progress = dict(status='RUNNING', pid=os.getpid(), source_seconds=end,
                    restored_from_seconds=initial_time, target_seconds=45., recorded=recorded,
                    wall_seconds=time.perf_counter()-started, neural_wall_seconds=time.perf_counter()-simulation_started,
                    peak_working_set_bytes=peak_memory_bytes(), source_identity=bg['identity'],
                    durable_source_seconds=float(checkpoint.name.removeprefix('CHECKPOINT_'))/10000,
                    automatic_fit=True)
                atomic_json(DEST/'PROGRESS.json', progress); print(json.dumps(progress), flush=True)
                arm.metrics['inner_counters'].clear()
            if j % 50000 == 0:
                checkpoint = DEST/('CHECKPOINT_'+str(j))
                current = arm.checkpoint_payload(recording=dict(recorded=recorded,
                    arrays={k:v[:recorded].copy() for k,v in arrays.items()}, expected=expected[:j//10].copy()))
                save_bundle(checkpoint, kind='validation_evidence', identity=bg['identity'], payload=current,
                    qualification=dict(default_adopted=False, fit_eligible=False))
        if recorded != 450001 or not all(np.isfinite(v).all() for v in (*arrays.values(), expected)):
            raise ContractError('Current continuous trajectory incomplete')
        if canonical_hash(dict(recorded=recording['recorded'],
                arrays={k:v[:recording['recorded']] for k,v in arrays.items()},
                expected=expected[:len(saved_expected)])) != prefix_hash:
            raise ContractError('Previously completed prefix changed')
        for seconds in (30, 45):
            j = seconds*10000
            path = DEST/f'PREFIX_{seconds}s.npz'
            if path.exists():
                receipt = read(DEST/f'PREFIX_{seconds}s.json')
                if receipt['sha256'] != digest(path) or receipt['background_identity'] != bg['identity']:
                    raise ContractError('Existing prefix identity differs; retained')
                with np.load(path, allow_pickle=False) as z:
                    for name in arrays:
                        np.testing.assert_array_equal(z[name], arrays[name][:j+1])
                    np.testing.assert_array_equal(z['time_source_s'], np.arange(j+1)*.0001)
                    np.testing.assert_array_equal(z['expected_rate_base_Hz'], expected[:j//10])
            else:
                np.savez_compressed(path, time_source_s=np.arange(j+1)*.0001,
                    **{k:v[:j+1] for k,v in arrays.items()}, expected_rate_base_Hz=expected[:j//10])
                atomic_json(DEST/f'PREFIX_{seconds}s.json', dict(status='COMPLETE_CONTINUOUS_PREFIX',
                    seconds=seconds, sha256=digest(path), background_identity=bg['identity'], target_group=7730,
                    source_cell='69#6', members=1, split_sha256=bg['split_sha256']))
        atomic_json(DEST/'NEURAL_RESULT.json', dict(status='COMPLETE', source_seconds=45.,
            restored_from_seconds=initial_time, computed_source_seconds=45.-initial_time,
            wall_seconds=time.perf_counter()-started, neural_wall_seconds=time.perf_counter()-simulation_started,
            peak_working_set_bytes=peak_memory_bytes(), identity=bg['identity'], model_identity=bg['model_identity'],
            target_row=row, recorded=recorded, original_prefix_preserved=True, optical_cache_reused=True,
            rod_mode='legacy_engineering', frozen_upstream=True, AC_calibrated=False, sealed_accessed=False,
            checks=evidence, source_sha256=digest(Path(__file__)), FIT_READY=False))
    finally:
        arm.close()


def summarize_fit():
    """Summarize the already frozen prediction arrays; no new fit or evaluation."""
    from tools.gin69_voltage_fit_support_v2 import metrics
    folder = DEST/'readout'
    frozen = read(folder/'MODEL_FROZEN.json')
    development = read(folder/'FIXED_DEVELOPMENT_RESULT.json')
    with np.load(folder/'FIXED_DEVELOPMENT_PREDICTIONS.npz', allow_pickle=False) as z:
        predictions = {k:z[k].copy() for k in z.files}
    with np.load(DEST/'PREFIX_45s.npz', allow_pickle=False) as z:
        original_probability = z['expected_rate_base_Hz']/1000
    counts = predictions['observed_counts']
    if counts.shape != (6, 45000) or original_probability.shape != (45000,):
        raise ContractError('Only approved 0..45 s prediction arrays can be summarized')
    if np.any(original_probability <= 0) or np.any(original_probability >= 1):
        raise ContractError('Original expected-rate probability cannot support finite marginal NLL')
    original_loss = np.where(counts, -np.log(original_probability), -np.log1p(-original_probability))
    rows = {}
    for label, lo, hi in (('train',0,30000), ('development',30000,45000)):
        row = dict(
            V_GLM=metrics(counts,predictions['probability'],predictions['conditional_loss'],lo,hi),
            constant=metrics(counts,predictions['constant_probability'],predictions['constant_conditional_loss'],lo,hi),
            original_RGC=metrics(counts,original_probability,original_loss,lo,hi))
        row['original_RGC']['marginal_NLL_nat_per1ms'] = row['original_RGC'].pop('conditional_NLL_nat_per1ms')
        observed = counts[:,lo:hi].mean(0).reshape(-1,20).mean(1)*1000
        row['observed_modulation_SD_Hz'] = float(observed.std())
        row['GLM_MSE_improvement_over_constant_Hz2'] = row['constant']['free_MSE_Hz2']-row['V_GLM']['free_MSE_Hz2']
        rows[label] = row
    if rows['development']['V_GLM'] != development['metrics']['V_GLM']:
        raise ContractError('Summary differs from fixed development evaluation')
    atomic_json(DEST/'METRICS.json',dict(status='TRAIN_AND_FIXED_DEVELOPMENT_COMPLETE',
        intervals=rows, selected_ridge=frozen['model']['ridge'], model=frozen['model'],
        neural=read(DEST/'NEURAL_RESULT.json'), MC=development['MC'],
        teacher_forced='conditional NLL only; real past spikes',
        free_history='same saved 4096 paths / 16 batches / seed 2026110106; self-generated history',
        original_RGC='Same neural input/model identity; exact history-distribution expected rate; marginal NLL is not teacher-forced',
        sealed_values_used=False, physiological_validation=False, FIT_READY=False,
        frozen_model_sha256=digest(folder/'MODEL_FROZEN.json'),
        predictions_sha256=digest(folder/'FIXED_DEVELOPMENT_PREDICTIONS.npz')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check','run','fit','complete'])
    args = parser.parse_args(); runtime()
    owner_path = DEST/'OWNER_recovery.json'
    for path in (owner_path, DEST/'OWNER_run.json', DEST/'OWNER_fit.json'):
        if path.exists():
            owner = read(path)
            if owner.get('status') == 'RUNNING' and process_identity(owner['pid'])['state'] in ('ALIVE','UNKNOWN'):
                raise RuntimeError('A current 69#6 owner is already running: '+str(path))
    began = time.perf_counter()
    atomic_json(owner_path, dict(status='RUNNING', pid=os.getpid(), mode=args.mode,
                                started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        if args.mode in ('check','run','complete'):
            run(check_only=args.mode == 'check')
        if args.mode in ('fit','complete'):
            fit()
            summarize_fit()
    except BaseException:
        atomic_json(owner_path, dict(status='FAILED', pid=os.getpid(), mode=args.mode,
            wall_seconds=time.perf_counter()-began, error=traceback.format_exc()))
        raise
    else:
        atomic_json(owner_path, dict(status='EXITED_0', pid=os.getpid(), mode=args.mode,
            wall_seconds=time.perf_counter()-began, peak_working_set_bytes=peak_memory_bytes()))


if __name__ == '__main__':
    main()
