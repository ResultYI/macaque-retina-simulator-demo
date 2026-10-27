"""One prospectively bound, complete-observer real-CUDA group A/B comparison."""
from pathlib import Path
import argparse
from collections import defaultdict
import json
import math
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np

from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.spatial_group_graph import digest
from retina_sim.spatial_group_model import GroupModelAssembly
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.group_time_storage import save_bundle, load_bundle
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_monitor import GroupTimeMonitor, GroupTimeComparator
from retina_sim.group_time_recording import GroupRecording


SOURCES = ('group_parameter_binding.py','group_native_physics.py','native_multirate.py',
    'group_inner_dynamics.py','group_inner_compiled.py','group_time_runner.py','group_time_transport.py',
    'group_time_monitor.py','group_time_recording.py','group_time_storage.py','continuous_state.py',
    'continuous_bridge.py','semianalytic_time.py','multirate_contract.py')


def write(path,value):
    path=Path(path);pending=path.with_name(path.name+'.writing')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    os.replace(pending,path)


def cpu(value):
    return value.detach().cpu().numpy().copy() if hasattr(value,'detach') else np.array(value,copy=True)


def checkpoint_hash(value, *, reuse_arrays=True, cache_payload_bytes=128*1024**2, diagnostics=None):
    """Full restore witness; this explicit checkpoint check may copy all GPU state."""
    from retina_sim.canonical_block_cache import CanonicalBlockCache
    from retina_sim.canonical_stream import stream_canonical_hash, _PRIMITIVES
    captures={};tensor_reads=0
    def version_of(v):
        try:return getattr(v,'_version',None)
        except RuntimeError:return None
    def host(v):
        nonlocal tensor_reads
        if hasattr(v,'detach'):
            identity=id(v);version=version_of(v)
            result=cpu(v)
            tensor_reads+=1
            if reuse_arrays:
                if version_of(v)!=version:raise ContractError('Frozen hash tensor changed during capture')
                current=(result.dtype.str,result.shape,result.tobytes(order='C'))
                if identity in captures and captures[identity][1:]!=(version,*current):
                    raise ContractError('Frozen hash tensor storage changed during capture')
                captures[identity]=(v,version,*current)
            return result
        if isinstance(v,dict):return {k:host(x) for k,x in v.items()}
        if isinstance(v,list):
            return v.copy() if reuse_arrays and all(type(x) in _PRIMITIVES for x in v) else [host(x) for x in v]
        if isinstance(v,tuple):
            return v if reuse_arrays and all(type(x) in _PRIMITIVES for x in v) else tuple(host(x) for x in v)
        return v
    cache=CanonicalBlockCache(cache_payload_bytes) if reuse_arrays else None
    result=stream_canonical_hash(host(value),block_cache=cache)
    if diagnostics is not None:
        diagnostics.update(tensor_current_reads=tensor_reads,tensor_conversion_reuse=False,
                           **({} if cache is None else cache.metrics()))
    return result


def preserve_stop_evidence(output,identity,failure,recorders,arms,last_evidence):
    """Keep ordinary windows and diagnostics; a guard stop is not a checkpoint."""
    output=Path(output)
    write(output/'failure.json',failure)
    try:
        cursors={m:r.checkpoint_payload() for m,r in recorders.items()}
        receipt=save_bundle(output/'stopped_diagnostic',kind='validation_evidence',identity=identity,
            payload=dict(last_completed_callback=last_evidence,recording_cursors=cursors,
                accepted_inner_frontiers={m:a.inner_time for m,a in arms.items()},
                accepted_inner_states={m:a.inner_state for m,a in arms.items()},complete_checkpoint=False),
            qualification=dict(default_adopted=False,fit_eligible=False))
        failure['stopped_diagnostic']=receipt
    except Exception as evidence_error:
        failure['stop_evidence_export_error']=str(evidence_error)
    write(output/'failure.json',failure)


def clock_events(protocol):
    """Original min-next-clock and original1e-12 due set; no input snapping."""
    start,stop,origin=protocol['start_seconds'],protocol['stop_seconds'],protocol['clock_origin_seconds']
    rod=int(round((start-origin)/.0001))+1;bc=int(round((start-origin)/.001))+1
    result=[]
    while True:
        rt,bt=origin+rod*.0001,origin+bc*.001
        t=min(rt,bt)
        if t>stop and abs(t-stop)>1e-12:break
        due_rod,due_bc=abs(rt-t)<1e-12,abs(bt-t)<1e-12
        result.append((min(t,stop),due_rod,due_bc))
        if due_rod:rod+=1
        if due_bc:bc+=1
        if result[-1][0]==stop:break
    if not result or result[-1][0]!=stop:result.append((stop,False,False))
    return result


def sample_grid(start,stop,period):
    first=int(np.ceil(start/period-1e-10));last=int(np.floor(stop/period+1e-10))
    return tuple(float(i*period) for i in range(first,last+1) if start<=i*period<=stop)


def event_snapshot_sides(time_seconds,input_edges):
    # One departing snapshot, one original commit, one incoming snapshot.
    return ('left' if time_seconds in input_edges else 'right'),'right'


def summary(result):
    worst=None
    for gate,row in result.get('per_cell',{}).items():
        if not all(k in row for k in ('reference','candidate','allowance')):continue
        delta=np.abs(np.asarray(row['candidate'])-np.asarray(row['reference']))
        bound=np.broadcast_to(np.asarray(row['allowance']),delta.shape)
        ratio=np.divide(delta,bound,out=np.where(delta==0,0.,np.inf),where=bound>0)
        if not ratio.size:continue
        i=int(np.argmax(ratio));v=float(ratio.ravel()[i])
        if worst is None or v>worst['_norm']:
            worst=dict(gate=gate,flat_index=i,reference=float(np.asarray(row['reference']).ravel()[i]),
                candidate=float(np.asarray(row['candidate']).ravel()[i]),difference=float(delta.ravel()[i]),
                allowance=float(bound.ravel()[i]),normalized=v if math.isfinite(v) else 'INFINITE',_norm=v)
    if worst is not None:worst.pop('_norm')
    return dict(status=result['status'],issues=result.get('issues',[]),stamp=result.get('stamp'),worst=worst)


def run(preparation,output,*,resume=False):
    preparation,output=Path(preparation).resolve(),Path(output).resolve()
    if not output.is_relative_to(ROOT) or (output.exists() and not resume) or (not output.exists() and resume):
        raise ContractError('Fresh validation output or explicit matching resume required')
    if not resume:output.mkdir(parents=False)
    protocol=json.loads((preparation/'protocol.json').read_text(encoding='utf8'))
    approved=ROOT/'reports/group_time_candidate_20261025_v1/approval.json'
    if digest(approved)!=protocol['approval_sha256']:raise ContractError('Direct approval identity differs')
    seed_bundle=load_bundle(preparation/'construction_seed',expected_kind='source_construction_seed')
    seed=seed_bundle['payload'];start,stop=protocol['start_seconds'],protocol['stop_seconds']
    assembly=GroupModelAssembly(ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2')
    if digest(assembly.root/'manifest.json')!=protocol['group_manifest_sha256']:raise ContractError('Approved group changed')
    sources={name:digest(ROOT/'retina_sim'/name) for name in SOURCES}
    identity=dict(schema='retipath-full-group-time-validation-1',group=assembly.artifact_sha256,
        seed_manifest=seed_bundle['manifest_hash'],protocol_sha256=digest(preparation/'protocol.json'),implementation=sources,
        validation_driver_sha256=digest(Path(__file__)))
    if not resume:
        write(output/'prospective_identity.json',identity)
    elif canonical_hash(json.loads((output/'prospective_identity.json').read_text()))!=canonical_hash(identity):
        raise ContractError('Resume implementation/protocol/seed changed')
    import torch
    if not torch.cuda.is_available():raise ContractError('Real CUDA required; no CPU replacement acceptance')
    device=torch.device('cuda:0')
    memory=MemoryGuard(16*1024**3,4*1024**3)
    test_started=None;guard_tick=0.;public=start;last_validated=None
    stage='ASSEMBLING_GROUP_ARMS';arms={};monitors={};recorders={}
    phases={mode:defaultdict(float) for mode in ('reference','candidate')}
    reports=[];recorded=set();next_event=0;checkpoint_restored=False;restore_hashes={};last_evidence={}
    pins=json.loads((ROOT/'reports/spatial_group_graph_v02_20261025_v1/verification.json').read_text())['unchanged_pins']
    def guard():
        nonlocal guard_tick
        if (output/'REQUEST_STOP').exists():raise ContractError('Explicit paired validation stop')
        if test_started is not None and time.time()-test_started>protocol['runtime_limits']['paired_test_wall_seconds']:
            raise ContractError('Paired wall limit reached; do not expand or retry budgets')
        now=time.perf_counter()
        if now-guard_tick>=1.:
            if not memory.check():raise ContractError('Paired CPU memory guard: '+str(memory.stop_reason))
            free,total=torch.cuda.mem_get_info(device)
            if total-free>12*1024**3:raise ContractError('Whole-GPU12GiB guard reached')
            for name,expected in pins.items():
                if digest(ROOT/name)!=expected:raise ContractError('Original STOP/source pin changed: '+name)
            write(output/'status.json',dict(stage=stage,public_time_seconds=public,model_forward_started=test_started is not None,
                elapsed_test_seconds=None if test_started is None else time.time()-test_started,resources=memory.last,
                whole_GPU_used_bytes=total-free))
            guard_tick=now
    def timed(mode,phase,fn):
        began=time.perf_counter();value=fn();phases[mode][phase]+=time.perf_counter()-began;return value
    recording_times=sample_grid(start,stop,.0005)
    grids=set(sample_grid(start,stop,.0001));records=set(recording_times)
    input_edges={i/150. for i in range(748,752) if start<i/150.<=stop}
    queue_nodes=grids|records|input_edges|{start,stop}
    events=clock_events(protocol)
    initial_states=seed['states']
    initial_ports=None;comparator=None
    try:
        guard()
        if resume:
            session=json.loads((output/'session_checkpoint.json').read_text(encoding='utf8'))
            sh=session.pop('session_hash')
            if canonical_hash(session)!=sh:raise ContractError('Pair checkpoint session identity differs')
            public,next_event=session['public_time_seconds'],session['next_event']
            last_validated=session['last_validated_time_seconds']
            test_started=session['test_started_unix'];reports=session['comparison_summaries'];recorded=set(session['recorded_times'])
            phases={m:defaultdict(float,session['phases'][m]) for m in phases}
            initial_ports=load_bundle(output/'initial_ports',expected_kind='validation_evidence')['payload']
        for mode in ('reference','candidate'):
            common=dict(mode=mode,torch=torch,device=device,guard=guard,
                shared_plans=None if mode=='reference' else arms['reference'].shared_plans)
            if resume:
                restored=load_bundle(output/(mode+'_checkpoint'),expected_kind='complete_model_checkpoint',
                    expected_identity=dict(identity,mode=mode))
                payload=restored['payload']
                arm=GroupTimeArm.from_checkpoint(assembly,seed,protocol,payload,**common)
                arms[mode]=arm
                monitors[mode]=GroupTimeMonitor.from_payload(assembly,initial_states,payload['monitor'],
                    parameters=seed['inner_parameters'],torch=torch)
                recorders[mode]=GroupRecording.restore(assembly,output/mode,payload['recording'],artifactidentity=identity)
                rebuilt=arm.checkpoint_payload(monitor=monitors[mode].payload(),recording=recorders[mode].checkpoint_payload())
                original_hash,rebuilt_hash=checkpoint_hash(payload),checkpoint_hash(rebuilt)
                if original_hash!=rebuilt_hash:
                    raise ContractError('Complete '+mode+' checkpoint restoration differs')
                restore_hashes[mode]=dict(original=original_hash,restored=rebuilt_hash)
            else:
                arm=GroupTimeArm(assembly,seed,protocol,**common)
                arms[mode]=arm
                monitors[mode]=GroupTimeMonitor(assembly,initial_states,parameters=seed['inner_parameters'],torch=torch)
                recorders[mode]=GroupRecording(assembly,output/mode,np.asarray(recording_times),artifactidentity=identity)
        if resume:
            checkpoint_restored=True
            # Restoration is observed without recommitting the boundary or
            # adding a duplicate ordinary record/monitor callback.
            write(output/'restore_receipt.json',dict(status='COMPLETE_ARM_CONTROLLERS_QUEUES_AND_RECORDERS_RESTORED',
                public_time_seconds=public,native_frontiers={m:a.native_state.time_seconds for m,a in arms.items()},
                consumer_watermarks={m:a.native_consumed for m,a in arms.items()},
                complete_payload_hashes=restore_hashes,model_forward_started=True))
        else:
            pre={m:a.snapshot(start,'right',queue_node=True) for m,a in arms.items()}
            initial_ports=dict(values={k:cpu(v) for k,v in pre['reference']['ports'].items()},units=pre['reference']['port_units'])
            save_bundle(output/'initial_ports',kind='validation_evidence',identity=identity,payload=initial_ports,
                qualification=dict(default_adopted=False,fit_eligible=False))
            test_started=time.time()
        comparator=GroupTimeComparator(monitors['reference'],initial_ports['values'],initial_ports['units'],parameters=seed['inner_parameters'])
        def observe(t,which,states=None,*,do_record=False):
            nonlocal public,last_validated,last_evidence
            guard();public=float(t)
            evidence={}
            for mode,arm in arms.items():
                kwargs=dict(queue_node=t in queue_nodes)
                if states is not None:kwargs['inner_state']=states[mode]
                evidence[mode]=timed(mode,'full_rhs_snapshot_and_bridge',lambda a=arm,k=kwargs:a.snapshot(t,which,**k))
                e=evidence[mode]
                timed(mode,'complete_monitor',lambda e=e,m=mode:monitors[m].observe(t,which,e['states'],e['rhs'],
                    transitions=e['transitions'],ports=e['ports'],event_generation=e['event_generation'],aux=e['aux'],rhs_kind=e['rhs_kind']))
            last_evidence=evidence
            compared=comparator.compare_pair(evidence['reference'],evidence['candidate'])
            reports.append(summary(compared))
            if compared['status']!='PASS':
                save_bundle(output/'first_failed_numeric_witness',kind='validation_evidence',identity=identity,
                    payload=dict(comparison=compared,reference=evidence['reference'],candidate=evidence['candidate']),
                    qualification=dict(default_adopted=False,fit_eligible=False))
                raise ContractError('Original numerical/causal gate '+compared['status']+': '+str(compared['issues'][:3]))
            if any(mon.domain_failures for mon in monitors.values()):raise ContractError('Original domain/history/gap monitor failed')
            last_validated=float(t)
            if do_record and t in records and t not in recorded:
                for mode,arm in arms.items():
                    e=evidence[mode]
                    values=arm.record_native_values(t,which,snapshot=e)
                    timed(mode,'ordinary_recording',lambda m=mode,e=e,v=values,a=arm:recorders[m].record(t,v,e['inner_outputs'],
                        delayedHC=cpu(e['delayed_HC']),eventlogs=a.event_log))
                    arm.event_log={pop:[] for pop in arm.event_log}
                recorded.add(t)
            return evidence
        if not resume:
            stage='INITIAL_COMMON_SNAPSHOT_AND_SAMPLING'
            observe(start,'right')
            for mode,arm in arms.items():timed(mode,'original_events',arm.initial_commit)
            observe(start,'right',do_record=True)
            for arm in arms.values():arm.ack(start)
        stage='PAIRED_STIMULUS_ADVANCE'
        for event_index in range(next_event,len(events)):
            end,rod,bc=events[event_index]
            begin=arms['reference'].inner_time
            if not begin<end:raise ContractError('Original event chronology lost')
            advances={}
            for mode,arm in arms.items():
                timed(mode,'native_compute',lambda a=arm:a.ensure_native(end,'left'))
                advances[mode]=timed(mode,'compiled_inner_compute',lambda a=arm:a.advance_inner(begin,end))
            points={t for t in queue_nodes if begin<t<end}
            for mode,arm in arms.items():
                points.update(t for t in advances[mode].accepted_times if begin<t<end)
                for interval in arm.native_intervals:
                    for curve in interval.curves.values():points.update(s.t1 for s in curve.segments if begin<s.t1<end)
            for t in sorted(points):
                states={m:r.evaluate(t,'right') for m,r in advances.items()}
                if t in input_edges:observe(t,'left',states)
                observe(t,'right',states,do_record=t in records)
                for arm in arms.values():arm.ack(t)
            endpoint={m:r.state for m,r in advances.items()}
            pre_side,post_side=event_snapshot_sides(end,input_edges)
            observe(end,pre_side,endpoint,do_record=False)
            if rod or bc:
                for mode,arm in arms.items():
                    timed(mode,'original_events',lambda a=arm:a.commit(end,rod=rod,bc=bc,rgc=bc,
                        rod_sample_time=max(start,end-.0001+1e-13) if rod else None))
                observe(end,post_side,do_record=True)
            elif end in records:observe(end,'right',do_record=True)
            for arm in arms.values():arm.ack(end)
            public=end
            if not resume and end==protocol['checkpoint_seconds']:
                stage='COMPLETE_CHECKPOINT_EXPORT'
                for mode,arm in arms.items():
                    payload=arm.checkpoint_payload(monitor=monitors[mode].payload(),recording=recorders[mode].checkpoint_payload())
                    save_bundle(output/(mode+'_checkpoint'),kind='complete_model_checkpoint',identity=dict(identity,mode=mode),
                        payload=payload,qualification=dict(default_adopted=False,fit_eligible=False,coupled_background_qualified=False))
                session=dict(public_time_seconds=public,next_event=event_index+1,test_started_unix=test_started,
                    last_validated_time_seconds=last_validated,comparison_summaries=reports,
                    recorded_times=sorted(recorded),phases={m:dict(v) for m,v in phases.items()})
                write(output/'session_checkpoint.json',dict(session,session_hash=canonical_hash(session)))
                write(output/'checkpoint_receipt.json',dict(status='SAVED_FOR_FRESH_PROCESS_RESUME',public_time_seconds=public,
                    native_frontiers={m:a.native_state.time_seconds for m,a in arms.items()},
                    unconsumed_native_intervals={m:len(a.native_intervals) for m,a in arms.items()}))
                return {'status':'CHECKPOINT_SAVED_REQUIRES_NEW_PROCESS_RESUME','time_seconds':public}
        stage='FINAL_COMPLETE_MONITOR_COMPARISON'
        final=comparator.compare_monitors(monitors['reference'],monitors['candidate'],stop-start)
        save_bundle(output/'complete_monitor_witness',kind='validation_evidence',identity=identity,payload=final,
            qualification=dict(default_adopted=False,fit_eligible=False))
        if final['status']!='PASS':raise ContractError('Complete original monitor statistics gate: '+str(final['issues'][:3]))
        if recorded!=records or not checkpoint_restored:raise ContractError('Ordinary recording or fresh-process restore coverage incomplete')
        source=ROOT/assembly.manifest['configuration']['source_run']/'centered_inputs_v2'
        fm=np.load(source/'source_frame_to_unique.npy',allow_pickle=False);frames=np.load(source/'source_unique_gray8.npy',mmap_mode='r')
        indices=np.arange(748,752,dtype=np.int64);times=indices.astype(np.float64)/150.
        replays={m:rec.finish(np.asarray(frames[fm[indices]]),times,indices,
            metadata=dict(input_class='8ms实际数值验收',runtime={'stimulus_origin_seconds':0.},
                replay=dict(source_construction_initial=True,qualified_background=False,paired_validation=True))) for m,rec in recorders.items()}
        hot={m:sum(v.values()) for m,v in phases.items()}
        result=dict(status='PAIRED_NUMERICAL_PASS' if hot['candidate']<=.8*hot['reference'] else 'NUMERICAL_PASS_PERFORMANCE_NOT_ESTABLISHED',
            numerical_status='PASS',performance_gate_fraction=.8,reference_hot_seconds=hot['reference'],candidate_hot_seconds=hot['candidate'],
            speedup=hot['reference']/hot['candidate'],stimulus_seconds=stop-start,checkpoint_restored=True,
            measured_phase_seconds={m:dict(v) for m,v in phases.items()},neural_metrics={m:a.metrics for m,a in arms.items()},
            transport_metrics={m:a.transport.metrics for m,a in arms.items()},comparisons=len(reports),comparison_summaries=reports,
            replay_receipts=replays,source_initial_legal=True,qualified_background=False,fit_eligible=False,default_adopted=False,
            extra_QA_readback_not_in_hot_phase_sum=True,whole_test_wall_seconds=time.time()-test_started)
        write(output/'result.json',result);return result
    except Exception as exc:
        failure=dict(stage=stage,error=type(exc).__name__,message=str(exc),
            validated_public_time_seconds=last_validated,attempted_public_time_seconds=public,
            native_frontiers={m:a.native_state.time_seconds for m,a in arms.items()},
            source_initial_legal=True,qualified_background=False,fit_eligible=False,default_adopted=False,
            comparison_summaries=reports,measured_phase_seconds={m:dict(v) for m,v in phases.items()},resources=memory.last,
            neural_metrics={m:a.metrics for m,a in arms.items()},
            transport_metrics={m:a.transport.metrics for m,a in arms.items() if a.transport is not None},
            ordinary_recording_samples=len(recorded),
            test_wall_seconds=None if test_started is None else time.time()-test_started)
        # A guard may stop inside an accepted interval. Preserve the last full
        # callback and ordinary files without falsely exporting a resumable
        # complete checkpoint at a mismatched public/inner frontier.
        preserve_stop_evidence(output,identity,failure,recorders,arms,last_evidence)
        raise
    finally:
        for mode in ('candidate','reference'):
            if mode in arms:arms[mode].close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',default='reports/group_time_candidate_20261025_v1/preparation_r1')
    parser.add_argument('--output',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();result=run(ROOT/args.preparation,ROOT/args.output,resume=args.resume)
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparison_summaries','neural_metrics')},ensure_ascii=False))


if __name__=='__main__':main()
