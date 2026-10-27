"""Finite same-source CPU/CUDA prefix and separate-process CUDA restore checks."""
from pathlib import Path
from copy import deepcopy
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_monitor import GroupTimeMonitor,frozen_budget
from retina_sim.group_time_storage import load_bundle,save_bundle
from retina_sim.rgc_stage1_cache import StageWorker,write
from retina_sim.ac_vs_gpu import bind_gpu,host_payload,initial_match_on_device
from tools.validate_ac_vs_native_support import load
from tools.validate_ac_vs_equilibrium_start import initial_match
from tools.validate_ac_vs_timescale import calendar
from tools.validate_ac_vs import causal_hash
from tools.fit_rgc_stage1 import verify

OUT=ROOT/'reports/rgc_stage1_gpu_adaptation_v1'
VAL=OUT/'validation_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def cpu(v):return np.asarray(v.detach().cpu().numpy() if hasattr(v,'detach') else v)

def setup():
    spec=verify();case=spec['cases'][0];a,seed,identity=load(ROOT/spec['prepared_model']);seed=deepcopy(seed)
    drive=load_bundle(ROOT/case['drive_bundle'],expected_kind='validation_evidence')['payload']
    if canonical_hash(drive)!=case['drive_sha256']:raise ContractError('Frozen actual g0 drive changed')
    seed['native_inputs']=deepcopy(drive['native_inputs']);seed['rod_inputs']=deepcopy(drive['rod_inputs'])
    protocol=deepcopy(identity['protocol']);protocol.update(start_seconds=drive['model_origin_s'],stop_seconds=drive['model_stop_s'],
        checkpoint_seconds=drive['model_stop_s'],input_frame_indices=sorted(drive['native_inputs']),rgc_stage1_cache_identity=canonical_hash(dict(spec=spec,case=case)))
    return spec,a,seed,protocol

def run(mode):
    directory=VAL/mode;directory.mkdir(parents=True,exist_ok=False)
    prereg=read(OUT/'PRE_REGISTERED_CHECKS.json');t0=time.perf_counter();spec,a,seed,protocol=setup();assembly_seconds=time.perf_counter()-t0
    torch=None;context_seconds=0.
    if mode!='cpu':
        t=time.perf_counter();import torch
        if not torch.cuda.is_available():raise ContractError('CUDA unavailable; no fallback')
        torch.cuda.init();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();context_seconds=time.perf_counter()-t;a=bind_gpu(a)
    execution='reference' if mode=='cpu' else 'gpu_dop853';kwargs=dict(mode=execution)
    if torch is not None:kwargs.update(torch=torch,device=torch.device('cuda:0'))
    states={};ports={};stamps=[];units={};initial=None;arm=None
    with StageWorker(ROOT,directory,OUT,spec['single_worker_safety_seconds']) as worker:
        try:
            if mode=='gpu-resume':
                saved=load_bundle(VAL/'gpu-split'/'checkpoint',expected_kind='complete_model_checkpoint')
                arm=GroupTimeArm.from_checkpoint(a,seed,protocol,saved['payload'],guard=worker.check,**kwargs)
                monitor=GroupTimeMonitor.from_payload(a,seed['states'],saved['payload']['monitor'],parameters=seed['inner_parameters'],torch=torch)
                restore_initial_match=causal_hash(host_payload(arm.checkpoint_payload(monitor=monitor.payload())))==causal_hash(saved['payload'])
                if not restore_initial_match:raise ContractError('Restored complete CUDA causal state differs before advance')
                with np.load(VAL/'gpu-split'/'traces.npz',allow_pickle=False) as z:
                    meta=read(VAL/'gpu-split'/'trace_metadata.json');stamps=meta['stamps'];units=meta['state_units'];initial=meta['initial_match']
                    states={tuple(k.split('|')):list(z['state__'+k].copy()) for k in units}
                    ports={k:list(z['port__'+k].copy()) for k in meta['port_units']}
            else:
                arm=GroupTimeArm(a,seed,protocol,guard=worker.check,**kwargs)
                monitor=GroupTimeMonitor(a,seed['states'],parameters=seed['inner_parameters'],torch=torch);initial,_=initial_match_on_device(arm,seed)
                if not initial['passed']:raise ContractError('GPU/CPU initial state/history contract differs')
                restore_initial_match=None
            hot=time.perf_counter()
            def observe(t,side,label):
                s=arm.snapshot(t,side,queue_node=True)
                monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
                if monitor.domain_failures:raise ContractError('Original physical/domain/history gate failed')
                stamps.append([float(t),label])
                for k,v in s['states'].items():states.setdefault(k,[]).append(cpu(v).copy())
                for pop in spec['recording']:
                    for field in ('voltage','gE','gI'):
                        k=pop+'|'+field;ports.setdefault(k,[]).append(cpu(s['ports'][pop,field]).copy())
                return s
            if not arm.initial_commit_done:
                observe(arm.start,'right','initial_before');arm.initial_commit();observe(arm.start,'right','initial_after');arm.ack(arm.start)
            offset=prereg['checkpoint_offset_seconds'] if mode=='gpu-split' else prereg['duration_seconds']
            target=arm.start+offset
            for end,flags in calendar(protocol):
                if end<=arm.inner_time:continue
                if end>target+1e-13:break
                begin=arm.inner_time;arm.ensure_native(end,'left');arm.advance_inner(begin,end)
                observe(end,'left' if flags['frame'] else 'right','pre')
                if flags['rod'] or flags['bc']:
                    arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
                observe(end,'right','post');arm.ack(end)
            if abs(arm.inner_time-target)>1e-13:raise ContractError('Frozen prefix endpoint missing from original calendar')
            if torch is not None:torch.cuda.synchronize()
            hot_seconds=time.perf_counter()-hot
            payload=arm.checkpoint_payload(monitor=monitor.payload())
            save_bundle(directory/'checkpoint',kind='complete_model_checkpoint',identity=dict(arm=arm.identity),payload=payload,qualification=dict(default_adopted=False,fit_eligible=False,FIT_READY=False))
            units={'|'.join((b.population,b.mechanism,b.name)):b.unit for b in (*a.state_blocks,*a.probability_blocks,*a.auxiliary_z1_blocks)}
            np.savez_compressed(directory/'traces.npz',**{'state__'+'|'.join(k):np.asarray(v) for k,v in states.items()},**{'port__'+k:np.asarray(v) for k,v in ports.items()})
            write(directory/'trace_metadata.json',dict(stamps=stamps,state_units=units,port_units={k:('V' if k.endswith('|voltage') else 'S') for k in ports},initial_match=initial))
            result=dict(mode=mode,execution=execution,hot_seconds=hot_seconds,assembly_load_seconds=assembly_seconds,CUDA_import_context_seconds=context_seconds,
                arm_build_seconds=arm.metrics['build_seconds'],runtime_metrics=arm.metrics,initial_match_passed=initial['passed'],restore_initial_match=restore_initial_match,
                causal_final_sha256=causal_hash(host_payload(payload)),raw_absolute_monitor=monitor.report(offset),actual_simulated_prefix_seconds=arm.inner_time-arm.start,
                new_process=True,physical_qualification=False,FIT_READY=False)
            if torch is not None:result['GPU']=dict(device=torch.cuda.get_device_name(),precision='float64',runtime=arm.gpu_rhs.module.identity,
                RHS_metrics=arm.gpu_rhs.metrics,compiled=arm.compiled.metrics,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_total_bytes=list(torch.cuda.mem_get_info()))
            write(directory/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('runtime_metrics','raw_absolute_monitor','GPU')}),flush=True)
        except Exception as e:
            write(directory/'FAILURE.json',dict(type=type(e).__name__,reason=str(e),source_seconds=None if arm is None else arm.inner_time-arm.start));raise
        finally:
            if arm is not None:arm.close()

def compare():
    spec,a,seed,protocol=setup();cm=read(CPU_DIR/'trace_metadata.json');gm=read(VAL/'gpu/trace_metadata.json');rm=read(VAL/'gpu-resume/trace_metadata.json')
    if cm['stamps']!=gm['stamps'] or gm['stamps']!=rm['stamps']:raise ContractError('CPU/GPU source event/observation ordering differs')
    rows=[];ports=[];exact=True
    with np.load(CPU_DIR/'traces.npz') as c,np.load(VAL/'gpu/traces.npz') as g,np.load(VAL/'gpu-resume/traces.npz') as r:
        for key,unit in cm['state_units'].items():
            k=tuple(key.split('|'));budget=frozen_budget(seed['states'][k],unit)[2];delta=np.abs(c['state__'+key]-g['state__'+key]);norm=float(np.max(delta/budget)) if delta.size else 0.
            same=np.array_equal(g['state__'+key],r['state__'+key]);exact &= same
            rows.append(dict(state=key,unit=unit,max_absolute_difference=float(delta.max()) if delta.size else 0.,original_budget_norm=norm,passed=norm<=1,GPU_restore_exact=same))
        for key,unit in cm['port_units'].items():
            budget=frozen_budget(c['port__'+key][0],unit)[2];delta=np.abs(c['port__'+key]-g['port__'+key]);norm=float(np.max(delta/budget)) if delta.size else 0.
            same=np.array_equal(g['port__'+key],r['port__'+key]);exact &= same
            ports.append(dict(port=key,unit=unit,original_budget_norm=norm,passed=norm<=1,GPU_restore_exact=same))
    cr=read(CPU_DIR/'RESULT.json');gr=read(VAL/'gpu/RESULT.json');rr=read(VAL/'gpu-resume/RESULT.json')
    cp=load_bundle(CPU_DIR/'checkpoint',expected_kind='complete_model_checkpoint')['payload'];gp=load_bundle(VAL/'gpu/checkpoint',expected_kind='complete_model_checkpoint')['payload']
    sampled_exact=canonical_hash(cp['spike_states'])==canonical_hash(gp['spike_states'])
    clocks_exact=canonical_hash(cp['clocks'])==canonical_hash(gp['clocks'])
    causal_exact=gr['causal_final_sha256']==rr['causal_final_sha256']
    passed=all(x['passed'] for x in rows+ports) and exact and sampled_exact and clocks_exact and causal_exact and rr['restore_initial_match']
    result=dict(passed=bool(passed),states=rows,ports=ports,CPU_GPU_sampled_history_RNG_exact=sampled_exact,CPU_GPU_clocks_exact=clocks_exact,
        GPU_new_process_final_causal_exact=causal_exact,GPU_core_and_all_state_trajectories_restore_exact=exact,
        CPU_hot_seconds=cr['hot_seconds'],GPU_hot_seconds=gr['hot_seconds'],measured_hot_speedup=cr['hot_seconds']/gr['hot_seconds'],GPU_build_and_context_seconds=gr['arm_build_seconds']+gr['CUDA_import_context_seconds'],
        scope='10.1ms fixed source actualg0 prefix, original budgets, no longer-window or global background upgrade',FIT_READY=False)
    write(VAL/'COMPARISON.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('states','ports')}))
    if not passed:raise ContractError('Frozen GPU numerical/restore acceptance failed; evidence retained')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu','gpu-split','gpu-resume','compare']);p.add_argument('--attempt',default='validation_v1');p.add_argument('--cpu-reference');args=p.parse_args()
    if not args.attempt.replace('_','').isalnum():raise ContractError('Simple new validation attempt name required')
    VAL=OUT/args.attempt
    CPU_DIR=(OUT/args.cpu_reference) if args.cpu_reference else VAL/'cpu'
    compare() if args.mode=='compare' else run(args.mode)
