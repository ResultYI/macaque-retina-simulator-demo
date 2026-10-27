"""One fresh authorized CPU case prefix; shared science ROOT is read-only."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args();job=read(args.job)
    directory=Path(job['directory']).resolve();phase=Path(job['phase']).resolve()
    if not directory.is_relative_to(phase) or not phase.is_relative_to(ROOT):raise RuntimeError('Authorized private case directory required')
    from tools.validate_group_scc_radau import worker_handshake
    worker_handshake(directory,job,'case')
    from retina_sim.rgc_cpu_case_isolation import ScopedStageWorker,resources,enforce_private_writes
    from retina_sim.contracts import ContractError,canonical_hash
    from retina_sim.group_time_storage import load_bundle,save_bundle
    from retina_sim.group_time_runner import GroupTimeArm
    from retina_sim.group_time_monitor import GroupTimeMonitor
    from retina_sim.rgc_stage1_cache import write
    from tools.fit_rgc_stage1 import verify
    from tools.validate_ac_vs_native_support import load
    from tools.validate_ac_vs_equilibrium_start import initial_match
    from tools.validate_ac_vs_timescale import calendar
    from tools.validate_ac_vs import causal_hash
    from copy import deepcopy
    import numpy as np
    blocked=enforce_private_writes(directory);evidence=directory/'evidence';evidence.mkdir()
    began=time.perf_counter();cpu_began=time.process_time();before=resources();arm=None
    states={};ports={};stamps=[];checkpoint_wall=0.;checkpoint_CPU=0.;initial=None
    guard=ScopedStageWorker(ROOT,evidence,phase,9054,job)
    try:
      with guard:
        guard.check();spec=verify();case=next(c for c in spec['cases'] if c['case_id']==job['case_id'])
        computation=canonical_hash(dict(spec=spec,case=case))
        if computation!=job['computation_identity']:raise ContractError('Unmodified full spec+case identity differs')
        a,seed,parent=load(ROOT/spec['prepared_model']);seed=deepcopy(seed)
        if canonical_hash(seed)!=spec['source_seed_sha256']:raise ContractError('Frozen saved seed changed')
        drive=load_bundle(ROOT/case['drive_bundle'],expected_kind='validation_evidence')['payload']
        if canonical_hash(drive)!=case['drive_sha256']:raise ContractError('Frozen input payload changed')
        protocol=deepcopy(parent['protocol']);protocol.update(start_seconds=drive['model_origin_s'],stop_seconds=drive['model_stop_s'],
            checkpoint_seconds=drive['model_stop_s'],input_frame_indices=sorted(drive['native_inputs']),rgc_stage1_cache_identity=computation)
        seed['native_inputs']=deepcopy(drive['native_inputs']);seed['rod_inputs']=deepcopy(drive['rod_inputs'])
        if job.get('resume_checkpoint'):
            saved=load_bundle(job['resume_checkpoint'],expected_kind='complete_model_checkpoint')
            arm=GroupTimeArm.from_checkpoint(a,seed,protocol,saved['payload'],mode='reference',guard=guard.check)
            monitor=GroupTimeMonitor.from_payload(a,seed['states'],saved['payload']['monitor'],parameters=seed['inner_parameters'])
            if causal_hash(arm.checkpoint_payload(monitor=monitor.payload(),recording=saved['payload']['recording']))!=causal_hash(saved['payload']):raise ContractError('Restored complete causal payload differs before advance')
            rec=saved['payload']['recording'];stamps=rec['stamps'];initial=rec['initial_match']
            states={k:list(v) for k,v in rec['states'].items()};ports={k:list(v) for k,v in rec['ports'].items()};restore_match=True
        else:
            arm=GroupTimeArm(a,seed,protocol,mode='reference',guard=guard.check)
            monitor=GroupTimeMonitor(a,seed['states'],parameters=seed['inner_parameters']);initial,_=initial_match(arm,seed);restore_match=None
        if not initial['passed']:raise ContractError('Frozen complete initial state/history differs')
        build_elapsed=time.perf_counter()-began;hot=time.perf_counter();hot_CPU=time.process_time()
        def observe(t,side,label):
            s=arm.snapshot(t,side,queue_node=True)
            monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
            if monitor.domain_failures:raise ContractError('Original domain/history gate failed')
            stamps.append([float(t),label])
            for k,v in s['states'].items():states.setdefault(k,[]).append(np.array(v,copy=True))
            for pop in spec['recording']:
                for field in ('voltage','gE','gI'):
                    key=pop+'|'+field;ports.setdefault(key,[]).append(np.array(s['ports'][pop,field],copy=True))
        def recording():return dict(stamps=stamps,initial_match=initial,states={k:np.asarray(v) for k,v in states.items()},ports={k:np.asarray(v) for k,v in ports.items()})
        def checkpoint(name):
            nonlocal checkpoint_wall,checkpoint_CPU
            started=time.perf_counter();cpu_started=time.process_time();payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=recording())
            save_bundle(evidence/name,kind='complete_model_checkpoint',identity=dict(cache_identity=computation),payload=payload,qualification=dict(default_adopted=False,fit_eligible=False))
            checkpoint_wall+=time.perf_counter()-started;checkpoint_CPU+=time.process_time()-cpu_started
            return payload
        if not arm.initial_commit_done:
            observe(arm.start,'right','initial_before');arm.initial_commit();observe(arm.start,'right','initial_after');arm.ack(arm.start)
        target=arm.start+job['duration_seconds'];mid=arm.start+job['mid_checkpoint_seconds'] if job.get('mid_checkpoint_seconds') else None
        for end,flags in calendar(protocol):
            if end<=arm.inner_time:continue
            if end>target+1e-13:break
            begin=arm.inner_time;arm.ensure_native(end,'left');arm.advance_inner(begin,end)
            observe(end,'left' if flags['frame'] else 'right','pre')
            if flags['rod'] or flags['bc']:arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
            observe(end,'right','post');arm.ack(end)
            if mid is not None and abs(end-mid)<1e-13:checkpoint('mid_checkpoint')
            j=int(round((end-arm.start)/.0001))
            if j%100==0:write(evidence/'progress.json',dict(source_seconds=end-arm.start,elapsed_seconds=time.perf_counter()-began))
        if abs(arm.inner_time-target)>1e-13:raise ContractError('Frozen prefix endpoint absent from original calendar')
        hot_wall=time.perf_counter()-hot;hot_cpu=time.process_time()-hot_CPU
        payload=checkpoint('final_checkpoint')
        np.savez_compressed(evidence/'traces.npz',**{'state__'+'|'.join(k):np.asarray(v) for k,v in states.items()},**{'port__'+k:np.asarray(v) for k,v in ports.items()})
        units={'|'.join((b.population,b.mechanism,b.name)):b.unit for b in (*a.state_blocks,*a.probability_blocks,*a.auxiliary_z1_blocks)}
        write(evidence/'trace_metadata.json',dict(stamps=stamps,state_units=units,port_units={k:('V' if k.endswith('|voltage') else 'S') for k in ports},initial_match=initial))
        after=resources()
        try:
            from threadpoolctl import threadpool_info
            pools=threadpool_info()
        except ImportError:pools=None
        result=dict(status='COMPLETE_SHORT_PREFIX_NOT_FULL_CACHE',case_id=case['case_id'],computation_identity=computation,arm_identity=arm.identity,
            orchestration_identity=job['orchestration_identity'],source_ROOT=str(ROOT),pid=os.getpid(),duration_seconds=job['duration_seconds'],
            build_load_seconds=build_elapsed,hot_including_midcheckpoint_seconds=hot_wall,hot_CPU_seconds=hot_cpu,checkpoint_write_wall_seconds=checkpoint_wall,checkpoint_write_CPU_seconds=checkpoint_CPU,
            worker_total_wall_seconds=time.perf_counter()-began,worker_total_CPU_seconds=time.process_time()-cpu_began,resources_before=before,resources_after=after,peak_RSS_bytes=guard.memory.peak_rss_bytes,
            metrics=arm.metrics,initial_match_passed=initial['passed'],restore_initial_match=restore_match,complete_causal_sha256=causal_hash(payload),
            raw_absolute_monitor=monitor.report(job['duration_seconds']),thread_environment={k:os.environ.get(k) for k in job['thread_policy']},threadpools=pools,
            shared_write_attempts=blocked,old_CPU_STOP_preserved=(ROOT/'reports/rgc_stage1_complete_v1/REQUEST_STOP').exists(),old_GPU_STOP_preserved=(ROOT/'reports/rgc_stage1_gpu_v1/REQUEST_STOP').exists(),
            GPU_used=False,FIT_READY=False)
        write(evidence/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status','case_id','pid','duration_seconds','hot_including_midcheckpoint_seconds','checkpoint_write_wall_seconds','worker_total_wall_seconds','worker_total_CPU_seconds','peak_RSS_bytes']}),flush=True)
    except Exception as e:
        write(evidence/'FAILURE.json',dict(type=type(e).__name__,reason=str(e),source_seconds=None if arm is None else arm.inner_time-arm.start,blocked_writes=blocked));raise
    finally:
        if arm is not None:arm.close()

if __name__=='__main__':main()
