"""One deterministic full7.1s neural trajectory per stimulus, compact target ports."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import math
import time
import numpy as np
from retina_sim.contracts import ContractError,canonical_hash
from tools.cell6_runtime_monitor_v2 import Cell6Monitor as GroupTimeMonitor
from retina_sim.group_time_storage import save_bundle,load_bundle
from tools.durable_research_runtime_v1 import atomic_json
from tools.cell6_halo_context_v1 import ROOT,BASE,MODEL,TARGET,load,protocol,read,write
from tools.cell6_compiled_inner_v1 import CompiledCell6Runner as Cell6Runner
from tools.validate_ac_vs_timescale import calendar


def witness(snapshot):return {'STATE__'+'__'.join(k):v.copy() for k,v in snapshot['states'].items()}


def capture(group,guard,journal,memory,*,output_root=None):
    experiment=BASE if output_root is None else Path(output_root)
    out=experiment/'cache'/('g'+str(group));started=time.perf_counter()
    segment_path=experiment/('SEGMENT_REQUEST_g'+str(group)+'.json')
    segment=read(segment_path) if segment_path.exists() else None
    if segment and Path(segment['capture_directory']).resolve()!=journal.directory.parent.resolve():raise ContractError('Segment worker ownership differs')
    recovery=None
    if out.exists():
        recovery=read(out/'RESUME_AUTHORIZATION.json')
        marker=out/('RESUME_STARTED_'+journal.directory.parent.name+'.json' if segment else 'RESUME_STARTED.json')
        if (out/'manifest.json').exists() or marker.exists():raise ContractError('Existing completed/attempted recovery retained')
        if Path(recovery['capture_directory']).resolve()!=journal.directory.parent.resolve():raise ContractError('Recovery controller does not own authorization')
        if segment:
            predecessor=Path(recovery['predecessor_capture'])
            if (read(predecessor/'TERMINAL.json')['status']!='EXITED_0'
                or read(predecessor/'DEBUG_TERMINAL.json')['native_failure_captured']
                or read(predecessor/'RESOURCE.json')['status']!='PASS'):
                raise ContractError('Process handoff requires clean predecessor; native-failed checkpoints excluded')
        prior=read(out/'INPUT_AND_MODEL_IDENTITY.json')
        if prior['identity']!=recovery['prefix_cache_identity']:raise ContractError('Recovery prefix identity changed')
        atomic_json(marker,dict(capture_directory=str(journal.directory.parent),checkpoint=recovery['checkpoint'],automatic_retry=False))
    else:out.mkdir(parents=True,exist_ok=False)
    source=read(BASE/'SOURCE_IDENTITY.json');case=source['cases'][group]
    assembly,seed=load();seed=deepcopy(seed);guard();p=protocol()
    target=np.flatnonzero(assembly.populations['parasol_RGC_ON'].group_ids==TARGET)
    if len(target)!=1 or assembly.populations['parasol_RGC_ON'].counts[target[0]]!=2:raise ContractError('Exact original target/members required')
    row=int(target[0]);background=seed['input_reference']['native_background_td']
    with np.load(BASE/'SOURCE_PREPARATION.npz',allow_pickle=False) as z:raw=z['g'+str(group)+'_stimulus'].copy()
    if hashlib.sha256(raw.tobytes()).hexdigest()!=case['stimulus_sha256']:raise ContractError('Actual source waveform differs')
    frame_indices={frame:max(0,math.floor((frame/150.)/.0001)) for frame in p['input_frame_indices']}
    if max(frame_indices.values())>=len(raw):raise ContractError('Clock mapping outside original waveform')
    seed['native_inputs']={frame:background*(1+.05*(float(raw[j])-float(raw[0]))) for frame,j in frame_indices.items()}
    if any(np.any(v<0) or not np.isfinite(v).all() for v in seed['native_inputs'].values()):raise ContractError('Original native input domain changed')
    identity=canonical_hash(dict(compiled_inner_adapter=file_sha(ROOT/'tools/cell6_compiled_inner_v1.py'),compiled_cache_code=file_sha(Path(__file__)),source=file_sha(BASE/'SOURCE_IDENTITY.json'),case=case,model_seed=file_sha(MODEL/'seed/manifest.json'),
        target=TARGET,protocol=p,input_payload=canonical_hash(seed['native_inputs']),monitor_code_sha256=file_sha(ROOT/'retina_sim/group_time_monitor.py'),code={n:file_sha(ROOT/'tools'/n) for n in ['cell6_raw_cache_v1.py','cell6_raw_runner_v1.py','cell6_halo_context_v1.py','cell6_raw_numerical_v1.py','cell6_runtime_monitor_v2.py']}))
    input_identity=dict(identity=identity,case=case,model=read(MODEL/'RESULT.json'),target=TARGET,
        local_row=row,members=2,target_is_named_halo_not_core=True,input_mapping=source['input_mapping'],frame_source_indices={str(k):v for k,v in frame_indices.items()},
        input_payload_sha256=canonical_hash(seed['native_inputs']),no_observed_spikes_used_in_forward=True,
        FIT_READY=False,background_qualified=False,production_adopted=False)
    if recovery:
        if any(canonical_hash(prior[k])!=canonical_hash(input_identity[k]) for k in input_identity if k!='identity'):
            raise ContractError('Recovery source/model/input differs from current attempt')
        input_identity['recovery']=recovery
    if recovery:atomic_json(out/'INPUT_AND_MODEL_IDENTITY.json',input_identity)
    else:write(out/'INPUT_AND_MODEL_IDENTITY.json',input_identity)
    arrays={k:np.empty(71001,dtype=np.float64) for k in ['V','gE','gI']};expected=np.empty(7100)
    recorded=bins=0;arm=None;checkpoints=[];numeric_records=[];numeric_times=[];restored_once=False
    try:
        if recovery:
            checkpoint=out/recovery['checkpoint']
            if file_sha(checkpoint/'manifest.json')!=recovery['checkpoint_manifest_sha256']:raise ContractError('Recovery checkpoint manifest changed')
            saved=load_bundle(checkpoint,expected_kind='validation_evidence',expected_identity=dict(cache_identity=recovery['prefix_cache_identity']))['payload']
            if saved['recording']['identity']!=recovery['prefix_cache_identity']:raise ContractError('Checkpoint recording source changed')
            arm=Cell6Runner.from_checkpoint(assembly,seed,p,saved,guard=guard)
            copied=arm.checkpoint_payload(monitor=saved['monitor'],recording=saved['recording'])
            categories=['native','inner','spike_states','clocks','input_position','observation_delay','readout_cursor']
            if any(canonical_hash(saved[k])!=canonical_hash(copied[k]) for k in categories):raise ContractError('Recovery changed complete scientific state')
            monitor=GroupTimeMonitor.from_payload(assembly,seed['states'],saved['monitor'],parameters=seed['inner_parameters'])
            recorded=saved['recording']['recorded'];bins=saved['recording']['bins']
            for name in arrays:arrays[name][:recorded]=saved['recording']['arrays'][name]
            expected[:bins]=saved['recording']['expected'];restored_once=True
            checkpoints=sorted(v.name for v in out.glob('CHECKPOINT_*') if v.is_dir());initial_max_rhs=None
            restore_name='RESTORE_CHECK_'+journal.directory.parent.name+'.json' if segment else 'RECOVERY_RESTORE_CHECK.json'
            write(out/restore_name,dict(roundtrip_exact=True,categories=categories,source_seconds=arm.inner_time,recorded=recorded,bins=bins,
                old_topology_or_failed_native_checkpoint_reused=False,prefix_cache_identity=recovery['prefix_cache_identity'],continued_cache_identity=identity))
        else:
            arm=Cell6Runner(assembly,seed,p,precision='standard',guard=guard)
            monitor=GroupTimeMonitor(assembly,seed['states'],parameters=seed['inner_parameters'])
        gate=read(experiment/'PAIR_GATE.json')
        if arm.identity!=gate['compiled_controller_identity']:raise ContractError('Formal compiled controller differs from accepted pair')
        def observe(t,side):
            snapshot=arm.snapshot(t,side,queue_node=True)
            monitor.observe(t,side,snapshot['states'],snapshot['rhs'],transitions=snapshot['transitions'],ports=snapshot['ports'],aux=snapshot['aux'],event_generation=snapshot['event_generation'])
            if monitor.domain_failures:raise ContractError('Original full-chain physical/history gate failed')
            return snapshot
        def record(snapshot,j):
            nonlocal recorded,bins
            if j!=recorded:raise ContractError('Source node repeated or skipped')
            for name,port,unit in [('V','voltage','V'),('gE','gE','S'),('gI','gI','S')]:
                if snapshot['port_units']['parasol_RGC_ON',port]!=unit:raise ContractError('Physical target port unit mismatch')
                arrays[name][j]=snapshot['ports']['parasol_RGC_ON',port][row]
            if j%10==0 and j<71000:
                expected[bins]=snapshot['ports']['parasol_RGC_ON','expected_rate_Hz'][row];bins+=1
            recorded+=1
        if not recovery:
            first=observe(0.,'right');record(first,0);arm.initial_commit();observe(0.,'right');arm.ack(0.)
            initial_max_rhs=max(float(abs(v).max(initial=0)) for k,v in first['rhs'].items() if k[0] in ('cones','HC'))
        for end,flags in calendar(p):
            if end<=arm.inner_time:continue
            guard();begin=arm.inner_time;arm.ensure_native(end,'left');arm.advance_inner(begin,end)
            before=observe(end,'left' if flags['frame'] or end==arm.stop else 'right')
            j=int(round(end/.0001));author=flags['rod'] or end==arm.stop
            if author and j==recorded:record(before,j)
            if flags['rod'] or flags['bc']:
                arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(0.,end-.0001+1e-13) if flags['rod'] else None)
            after=observe(end,'left' if end==arm.stop else 'right');arm.ack(end)
            if group==0 and author and 10000<=j<=10100:
                if len(numeric_times)==0 or j!=round(numeric_times[-1]/.0001):numeric_times.append(end);numeric_records.append(witness(after))
                if j==10100:
                    np.savez_compressed(out/'STANDARD_NUMERICAL_WINDOW.npz',time_seconds=np.asarray(numeric_times),
                        **{k:np.stack([v[k] for v in numeric_records],axis=1) for k in numeric_records[0]})
            checkpoint_due=author and (j in (500,10000,10100) or j>0 and j%5000==0 or end==arm.stop)
            if checkpoint_due:
                name='CHECKPOINT_'+str(j).zfill(7)
                if name not in checkpoints:
                    payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=dict(identity=identity,recorded=recorded,bins=bins,
                        arrays={k:v[:recorded].copy() for k,v in arrays.items()},expected=expected[:bins].copy()))
                    save_bundle(out/name,kind='validation_evidence',identity=dict(cache_identity=identity),payload=payload,qualification=dict(default_adopted=False,fit_eligible=False))
                    checkpoints.append(name);atomic_json(out/'LATEST_CHECKPOINT.json',dict(directory=name,cache_identity=identity,source_sample=j))
                    if not restored_once:
                        saved=load_bundle(out/name,expected_kind='validation_evidence')['payload']
                        restored=Cell6Runner.from_checkpoint(assembly,seed,p,saved,guard=guard)
                        copied=restored.checkpoint_payload(monitor=monitor.payload(),recording=payload['recording'])
                        for category in ['native','inner','spike_states','clocks','input_position','observation_delay','readout_cursor']:
                            if canonical_hash(saved[category])!=canonical_hash(copied[category]):restored.close();raise ContractError('Complete cache restore differs: '+category)
                        previous=arm;arm=restored;previous.close();restored_once=True
                        write(out/'RESTORE_CHECK.json',dict(time_seconds=end,roundtrip_exact=True,continued_remaining_original_trajectory=True,no_extra_replay=True))
            if author and j%1000==0:
                progress=dict(group=group,source_seconds=end,total_seconds=7.1,wall_seconds=time.perf_counter()-started,recorded=recorded,
                    native_seconds=arm.metrics['native_wall_seconds'],inner_seconds=arm.metrics['inner_wall_seconds'],memory=memory.last)
                atomic_json(out/'PROGRESS.json',progress);journal.heartbeat(force=True,phase='neural_g'+str(group),**progress)
                print(json.dumps(progress),flush=True)
            if segment and end==segment['stop_seconds'] and end<arm.stop:
                name='CHECKPOINT_'+str(j).zfill(7)
                if name not in checkpoints:raise ContractError('Process handoff requires original full checkpoint')
                atomic_json(out/('SEGMENT_COMPLETE_'+str(segment['index']).zfill(2)+'.json'),dict(
                    source_seconds=end,cache_identity=identity,checkpoint=name,checkpoint_manifest_sha256=file_sha(out/name/'manifest.json'),
                    full_cache_complete=False,normal_worker_exit_required=True,precision_or_state_reset=False))
                return
        if recorded!=71001 or bins!=7100 or not all(np.isfinite(v).all() for v in arrays.values()):raise ContractError('Full source coverage/finite values missing')
        np.savez_compressed(out/'endogenous.npz',time_source_s=np.arange(71001)*.0001,expected_rate_base_Hz=expected,**arrays)
        write(out/'END_HISTORY.json',arm.spike_states)
        events=[dict(population=pop,**e) for pop,es in arm.event_log.items() for e in es if e['time_seconds']<7.1]
        write(out/'GROUP_OBJECT_EVENTS.json',events)
        if group==0:
            from tools.cell6_raw_numerical_v1 import check_reference
            numerical=check_reference(assembly,seed,p,out,guard)
            if not numerical['passed']:raise ContractError('Original short numerical comparison failed')
        result=dict(status='COMPLETE_COMPILED_BACKEND_CELL6_CACHE',cache_identity=identity,group=group,stimulus_sha256=case['stimulus_sha256'],
            samples=recorded,bins=bins,duration_seconds=7.1,target=TARGET,members=2,units=dict(V='V',gE='S',gI='S'),
            arrays_sha256=file_sha(out/'endogenous.npz'),complete_research_restore=restored_once,physical_gate_passed=not monitor.domain_failures,
            initial_native_rhs_max=initial_max_rhs,neural_metrics=arm.metrics,wall_seconds=time.perf_counter()-started,
            one_deterministic_trajectory_for3_observation_repeats=True,old_neural_cache_reused=False,
            same_attempt_complete_checkpoint_recovered=bool(recovery),recovery=recovery,
            short_numerical_reference='Accepted independent50ms same-input compiled/reference pair; no new full tight claim',
            topology_changed=False,benchmark_role='same-existing-g1 formal verification; no new-cell coverage',compiled_inner_backend=type(arm.compiled).__name__,compiled_rhs_calls_current_instance=arm.compiled._calls,
            FIT_READY=False,background_qualified=False,production_adopted=False)
        write(out/'manifest.json',result)
    finally:
        if arm is not None:arm.close()
        if not (out/'manifest.json').exists():
            atomic_json(out/'PARTIAL_STATUS.json',dict(recorded=recorded,bins=bins,completed=False,last_checkpoint=checkpoints[-1] if checkpoints else None))


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
