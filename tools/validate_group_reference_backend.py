"""One bounded reference-only CUDA trajectory with fresh-process continuation.

This is integration evidence, not an old/new numerical or speed comparison.
The single 180s model wall clock includes checkpoint IO and process restoration.
"""
from pathlib import Path
import argparse
from collections import defaultdict
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.spatial_group_graph import digest
from retina_sim.spatial_group_model import GroupModelAssembly
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.group_time_storage import save_bundle, load_bundle
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_monitor import GroupTimeMonitor
from retina_sim.group_time_recording import GroupRecording
from tools.validate_group_time_candidate import (write, cpu, checkpoint_hash,
    preserve_stop_evidence, clock_events, sample_grid, event_snapshot_sides)

PREPARATION = ROOT/'reports/group_time_candidate_20261025_v1/preparation_r1'
ARTIFACT = ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2'
QUALIFICATION = dict(default_adopted=False, fit_eligible=False, coupled_background_qualified=False)


def protocol():
    p = json.loads((PREPARATION/'protocol.json').read_text(encoding='utf8'))
    if (p['start_seconds'], p['checkpoint_seconds'], p['stop_seconds']) != (4.993, 4.997, 5.001):
        raise ContractError('Fixed 8ms reference window changed')
    expected = dict(max_rss_bytes=16*1024**3, minimum_available_bytes=4*1024**3,
        max_whole_GPU_bytes=12*1024**3, total_preparation_wall_seconds=600,
        paired_test_wall_seconds=180)
    if p['runtime_limits'] != expected:
        raise ContractError('Original resource/wall limits changed')
    if digest(ROOT/p['numerical_budget']['path']) != p['numerical_budget']['sha256']:
        raise ContractError('Frozen numerical budget changed')
    p.update(schema='retipath-reference-backend-integration-1', execution_mode='reference',
        authorization='User authorized complete reference backend milestone and one bounded real CUDA integration',
        source_protocol_sha256=digest(PREPARATION/'protocol.json'),
        comparison='Single reference trajectory; no multirate arm and no claimed old/new trajectory equivalence')
    return p


def implementation_identity():
    # Small source files only. This closes the old driver's incomplete explicit
    # dependency list without hashing datasets or importing optional backends.
    paths = list((ROOT/'retina_sim').glob('*.py')) + [Path(__file__),
        ROOT/'tools/validate_group_time_candidate.py', ROOT/'tools/cone_hc_foveal_recording.py']
    return {str(p.relative_to(ROOT)).replace('\\','/'): digest(p) for p in sorted(paths)}


def enforce_wall(started_unix, now=None):
    if started_unix is not None and (time.time() if now is None else now)-started_unix > 180.:
        raise ContractError('Reference 180s wall limit reached; do not expand or retry budgets')


def restore_session(output, identity):
    if (output/'failure.json').exists() or (output/'REQUEST_STOP').exists() or (output/'result.json').exists():
        raise ContractError('Failed, stopped or completed run cannot be resumed')
    session = json.loads((output/'session_checkpoint.json').read_text(encoding='utf8'))
    expected = session.pop('session_hash')
    if canonical_hash(session) != expected or canonical_hash(session['identity']) != canonical_hash(identity):
        raise ContractError('Reference checkpoint session identity differs')
    enforce_wall(session['test_started_unix'])
    return session


def run(output, *, resume=False):
    process_started = time.perf_counter()
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or (output.exists() != resume):
        raise ContractError('Fresh output or this new run explicit resume required')
    if not resume:
        output.mkdir(parents=False)
    p = protocol()
    if resume and json.loads((output/'protocol.json').read_text(encoding='utf8')) != p:
        raise ContractError('Saved reference protocol changed')
    if not resume:
        write(output/'protocol.json', p)
    pins = json.loads((ROOT/'reports/spatial_group_graph_v02_20261025_v1/verification.json').read_text())['unchanged_pins']
    identity = dict(schema=p['schema'], group_manifest=digest(ARTIFACT/'manifest.json'),
        seed_manifest=digest(PREPARATION/'construction_seed/manifest.json'),
        protocol=canonical_hash(p), implementation=implementation_identity())
    if identity['group_manifest'] != p['group_manifest_sha256']:
        raise ContractError('Fixed group manifest changed')
    if resume:
        if json.loads((output/'prospective_identity.json').read_text()) != identity:
            raise ContractError('Resume implementation/protocol/seed changed')
    else:
        write(output/'prospective_identity.json', identity)
    stage = 'COLD_PREPARATION'
    arm = monitor = recorder = None
    phases = defaultdict(float)
    started = None
    public = p['start_seconds']
    last_validated = None
    next_event = 0
    recorded = set()
    last_evidence = {}
    restored = False
    session = None
    peak_gpu = 0
    peak_rss = 0
    guard_tick = -1.
    memory = MemoryGuard(16*1024**3, 4*1024**3)
    torch = None
    compiled_previous = {}
    def timed(name, fn):
        before = time.perf_counter()
        try:
            return fn()
        finally:
            phases[name] += time.perf_counter()-before
    def guard(force=False):
        nonlocal guard_tick, peak_gpu, peak_rss
        if (output/'REQUEST_STOP').exists():
            raise ContractError('Explicit reference acceptance stop')
        enforce_wall(started)
        if started is None and time.perf_counter()-process_started > 600.:
            raise ContractError('Original 600s cold preparation wall limit reached')
        now = time.perf_counter()
        if force or now-guard_tick >= 1.:
            if not memory.check():
                raise ContractError('Reference CPU memory guard: '+str(memory.stop_reason))
            peak_rss = max(peak_rss, memory.last.get('process_peak_rss_bytes', 0))
            used = None
            if torch is not None:
                free, total = torch.cuda.mem_get_info(0)
                used = total-free
                peak_gpu = max(peak_gpu, used)
                if used > 12*1024**3:
                    raise ContractError('Original whole-GPU 12GiB guard reached')
            for name, expected in pins.items():
                if digest(ROOT/name) != expected:
                    raise ContractError('Original STOP/source pin changed: '+name)
            write(output/'status.json', dict(stage=stage, public_time_seconds=public,
                last_validated_time_seconds=last_validated, model_forward_started=started is not None,
                test_elapsed_seconds=None if started is None else time.time()-started,
                resources=memory.last, whole_GPU_used_bytes=used, whole_GPU_sampled_peak_bytes=peak_gpu))
            guard_tick = now
    def metrics():
        return dict(measured_phase_seconds=dict(phases), neural_metrics={} if arm is None else arm.metrics,
            compiled_metrics={} if arm is None else arm.compiled.metrics,
            compiled_execution_profile={} if arm is None else arm.compiled.execution_profile,
            previous_process_compiled_metrics=compiled_previous,
            transport_metrics={} if arm is None else arm.transport.metrics,
            resources=memory.last, process_peak_rss_bytes=peak_rss, whole_GPU_sampled_peak_bytes=peak_gpu,
            test_wall_seconds=None if started is None else time.time()-started,
            process_wall_seconds=time.perf_counter()-process_started,
            timing_note='Outer phases are exclusive; neural/compiled/transport counters are nested or asynchronous and must not be added again')
    try:
        if resume:
            session = restore_session(output, identity)
            started = session['test_started_unix']
            public = session['public_time_seconds']
            last_validated = session['last_validated_time_seconds']
            next_event = session['next_event']
            recorded = set(session['recorded_times'])
            phases.update(session['phases'])
            peak_gpu = session['whole_GPU_sampled_peak_bytes']
            peak_rss = session['process_peak_rss_bytes']
            compiled_previous = session['compiled_execution_profile']
        guard(True)
        def cuda_setup():
            nonlocal torch
            import torch as module
            torch = module
            if not torch.cuda.is_available():
                raise ContractError('Real CUDA required; no CPU substitute')
            torch.cuda.set_device(0)
            torch.cuda.init()
        timed('cuda_initialization', cuda_setup)
        guard(True)
        seed_bundle = timed('seed_load', lambda: load_bundle(PREPARATION/'construction_seed', expected_kind='source_construction_seed'))
        seed = seed_bundle['payload']
        assembly = timed('group_assembly', lambda: GroupModelAssembly(ARTIFACT))
        guard(True)
        common = dict(mode='reference', torch=torch, device='cuda:0', guard=guard)
        times = sample_grid(p['start_seconds'], p['stop_seconds'], .0005)
        if resume:
            stage = 'COMPLETE_CHECKPOINT_RESTORE'
            bundle = timed('checkpoint_load', lambda: load_bundle(output/'reference_checkpoint',
                expected_kind='complete_model_checkpoint', expected_identity=identity))
            payload = bundle['payload']
            arm = timed('arm_restore', lambda: GroupTimeArm.from_checkpoint(assembly, seed, p, payload, **common))
            monitor = timed('monitor_restore', lambda: GroupTimeMonitor.from_payload(assembly, seed['states'],
                payload['monitor'], parameters=seed['inner_parameters'], torch=torch))
            recorder = timed('recording_restore', lambda: GroupRecording.restore(assembly, output/'reference',
                payload['recording'], artifactidentity=identity))
            rebuilt = timed('checkpoint_rebuild_witness', lambda: arm.checkpoint_payload(monitor=monitor.payload(),
                recording=recorder.checkpoint_payload()))
            hashes = timed('checkpoint_restore_hash', lambda: (checkpoint_hash(payload), checkpoint_hash(rebuilt)))
            if hashes[0] != hashes[1] or hashes[0] != session['checkpoint_payload_hash']:
                raise ContractError('Complete reference state/RNG/history/controller/recorder restoration differs')
            restored = True
            write(output/'restore_receipt.json', dict(status='COMPLETE_PAYLOAD_BITWISE_RESTORED', public_time_seconds=public,
                source_payload_hash=hashes[0], restored_payload_hash=hashes[1], fresh_process=True,
                checkpoint_writer_pid=session['writer_pid'], restore_pid=os.getpid(),
                elapsed_model_wall_seconds=time.time()-started))
            del bundle, payload, rebuilt
        else:
            arm = timed('arm_build', lambda: GroupTimeArm(assembly, seed, p, **common))
            monitor = timed('monitor_build', lambda: GroupTimeMonitor(assembly, seed['states'], parameters=seed['inner_parameters'], torch=torch))
            recorder = timed('recording_build', lambda: GroupRecording(assembly, output/'reference', np.asarray(times), artifactidentity=identity))
            write(output/'environment.json', dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0), pid=os.getpid(),
                initial_state_qualification=seed_bundle['qualification'], group_counts={k:v.group_count for k,v in assembly.populations.items()},
                compiled_metrics=arm.compiled.metrics, cold_prepare_seconds=time.perf_counter()-process_started))
            started = time.time()
        arm.compiled.profile_enabled = True
        guard(True)
        records = set(times)
        edges = {i/150. for i in range(748,752) if p['start_seconds'] < i/150. <= p['stop_seconds']}
        nodes = set(sample_grid(p['start_seconds'], p['stop_seconds'], .0001)) | records | edges | {p['start_seconds'], p['stop_seconds']}
        def observe(t, side, state=None, *, do_record=False):
            nonlocal public, last_validated, last_evidence
            guard()
            public = float(t)
            e = timed('snapshot_and_bridge', lambda: arm.snapshot(t, side, inner_state=state, queue_node=t in nodes))
            timed('complete_monitor', lambda: monitor.observe(t, side, e['states'], e['rhs'], transitions=e['transitions'],
                ports=e['ports'], event_generation=e['event_generation'], aux=e['aux'], rhs_kind=e['rhs_kind']))
            last_evidence = {'reference':e}
            if monitor.domain_failures or monitor.missing:
                raise ContractError('Original domain/history/gap or complete unit coverage monitor failed')
            last_validated = float(t)
            if do_record and t in records and t not in recorded:
                def record():
                    native = arm.record_native_values(t, side, snapshot=e)
                    recorder.record(t, native, e['inner_outputs'], delayedHC=cpu(e['delayed_HC']), eventlogs=arm.event_log)
                    arm.event_log = {pop:[] for pop in arm.event_log}
                timed('ordinary_recording', record)
                recorded.add(t)
        if not resume:
            stage = 'INITIAL_COMMON_SNAPSHOT_AND_SAMPLING'
            observe(public, 'right')
            timed('original_events', arm.initial_commit)
            observe(public, 'right', do_record=True)
            timed('consumer_ack', lambda: arm.ack(public))
        stage = 'REFERENCE_STIMULUS_ADVANCE'
        events = clock_events(p)
        for event_index in range(next_event, len(events)):
            end, rod, bc = events[event_index]
            begin = arm.inner_time
            if not begin < end:
                raise ContractError('Original event chronology lost')
            timed('native_compute', lambda: arm.ensure_native(end, 'left'))
            advance = timed('compiled_inner_compute', lambda: arm.advance_inner(begin, end))
            points = {t for t in nodes if begin < t < end}
            points.update(t for t in advance.accepted_times if begin < t < end)
            for interval in arm.native_intervals:
                for curve in interval.curves.values():
                    points.update(s.t1 for s in curve.segments if begin < s.t1 < end)
            for t in sorted(points):
                state = timed('inner_dense_readout', lambda: advance.evaluate(t, 'right'))
                if t in edges:
                    observe(t, 'left', state)
                observe(t, 'right', state, do_record=t in records)
                timed('consumer_ack', lambda: arm.ack(t))
            pre_side, post_side = event_snapshot_sides(end, edges)
            observe(end, pre_side, advance.state)
            if rod or bc:
                timed('original_events', lambda: arm.commit(end, rod=rod, bc=bc, rgc=bc,
                    rod_sample_time=max(p['start_seconds'], end-.0001+1e-13) if rod else None))
                observe(end, post_side, do_record=True)
            elif end in records:
                observe(end, 'right', do_record=True)
            timed('consumer_ack', lambda: arm.ack(end))
            public = end
            if not resume and end == p['checkpoint_seconds']:
                stage = 'COMPLETE_CHECKPOINT_EXPORT'
                guard(True)
                payload = timed('checkpoint_construct', lambda: arm.checkpoint_payload(monitor=monitor.payload(),
                    recording=recorder.checkpoint_payload()))
                witness = timed('checkpoint_save_hash', lambda: checkpoint_hash(payload))
                receipt = timed('checkpoint_save', lambda: save_bundle(output/'reference_checkpoint', kind='complete_model_checkpoint',
                    identity=identity, payload=payload, qualification=QUALIFICATION))
                guard(True)
                saved = dict(identity=identity, public_time_seconds=public, last_validated_time_seconds=last_validated,
                    next_event=event_index+1, test_started_unix=started, recorded_times=sorted(recorded),
                    phases=dict(phases), checkpoint_payload_hash=witness, writer_pid=os.getpid(),
                    whole_GPU_sampled_peak_bytes=peak_gpu, process_peak_rss_bytes=peak_rss,
                    compiled_execution_profile=dict(arm.compiled.execution_profile))
                write(output/'session_checkpoint.json', dict(saved, session_hash=canonical_hash(saved)))
                write(output/'checkpoint_receipt.json', dict(status='SAVED_FOR_FRESH_PROCESS_RESUME', time_seconds=public,
                    receipt=receipt, native_frontier=arm.native_state.time_seconds, metrics=metrics()))
                return dict(status='CHECKPOINT_SAVED_REQUIRES_NEW_PROCESS_RESUME', time_seconds=public)
        stage = 'FINAL_COMPLETE_MONITOR_AND_REPLAY'
        guard(True)
        if recorded != records or not restored or arm.time_seconds != p['stop_seconds']:
            raise ContractError('Ordinary record, restore or full trajectory coverage incomplete')
        report = timed('monitor_report', lambda: monitor.report(p['stop_seconds']-p['start_seconds']))
        timed('monitor_evidence_save', lambda: save_bundle(output/'complete_monitor_witness', kind='validation_evidence',
            identity=identity, payload=dict(report=report, monitor=monitor.payload()), qualification=QUALIFICATION))
        source = ROOT/assembly.manifest['configuration']['source_run']/'centered_inputs_v2'
        fm = np.load(source/'source_frame_to_unique.npy', allow_pickle=False)
        frames = np.load(source/'source_unique_gray8.npy', mmap_mode='r')
        indices = np.arange(748,752,dtype=np.int64)
        replay = timed('replay_finalize', lambda: recorder.finish(np.asarray(frames[fm[indices]]), indices.astype(float)/150., indices,
            metadata=dict(input_class='8ms参考后端集成检查', runtime={'stimulus_origin_seconds':0.},
                replay=dict(source_construction_initial=True, qualified_background=False, paired_validation=False))))
        guard(True)
        result = dict(status='REFERENCE_8MS_INTEGRATION_COMPLETE', scope_seconds=[p['start_seconds'],p['stop_seconds']],
            **metrics(), ordinary_recording_samples=len(recorded), observation_fields=45, complete_monitor_callbacks=monitor.count,
            domain_history_failures=monitor.domain_failures, checkpoint_restored=True, replay_receipt=replay,
            original_monitor_stationarity_passed=report.get('passed'),
            trajectory_equivalence='NOT_MEASURED_NO_PAIRED_BASELINE', speedup=None,
            source_initial_legal=True, **QUALIFICATION)
        write(output/'result.json', result)
        return result
    except Exception as exc:
        failure = dict(stage=stage, error=type(exc).__name__, message=str(exc),
            validated_public_time_seconds=last_validated, attempted_public_time_seconds=public,
            native_frontier=None if arm is None else arm.native_state.time_seconds,
            checkpoint_restored=restored, ordinary_recording_samples=len(recorded), **metrics(),
            source_initial_legal=True, trajectory_equivalence='NOT_MEASURED_NO_PAIRED_BASELINE', **QUALIFICATION)
        preserve_stop_evidence(output, identity, failure, {} if recorder is None else {'reference':recorder},
            {} if arm is None else {'reference':arm}, last_evidence)
        raise
    finally:
        if arm is not None:
            arm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--orchestrate', action='store_true')
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    if args.orchestrate:
        if args.resume or output.exists() or not output.is_relative_to(ROOT):
            raise ContractError('Orchestrator starts a fresh reference run only')
        command = [sys.executable, '-X', 'utf8', '-B', str(Path(__file__)), '--output', str(output)]
        first = subprocess.run(command)
        if first.returncode:
            raise SystemExit(first.returncode)
        if not (output/'checkpoint_receipt.json').is_file() or (output/'failure.json').exists():
            raise ContractError('First process did not complete its checkpoint')
        second = subprocess.run(command+['--resume'])
        raise SystemExit(second.returncode)
    result = run(output, resume=args.resume)
    print(json.dumps({k:result[k] for k in ('status','time_seconds','test_wall_seconds','checkpoint_restored') if k in result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
