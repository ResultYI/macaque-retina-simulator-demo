"""One actual50ms old/new pair, retaining original numerical/event contracts."""
from pathlib import Path
from copy import deepcopy
import faulthandler,json,os,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.run_cell6_matched_baselines_v1 import attachment_gate,read,sha,write,BASE,REFERENCE


def run(out):
    import numpy as np
    from tools.cell6_halo_context_v1 import load,protocol
    from tools.cell6_raw_runner_v1 import Cell6Runner
    from tools.cell6_compiled_inner_v1 import CompiledCell6Runner,PHYSICAL
    from tools.cell6_runtime_monitor_v2 import Cell6Monitor
    from tools.validate_ac_vs_timescale import calendar
    from retina_sim.background_checkpoint import MemoryGuard
    from retina_sim.contracts import canonical_hash
    from retina_sim.group_time_storage import load_bundle,save_bundle
    from retina_sim.group_time_monitor import frozen_budget
    from tools.durable_research_runtime_v1 import DurableRun
    journal=DurableRun(out/'runtime','one_real_compiled_inner_pair');memory=MemoryGuard(24*1024**3,8*1024**3)
    started=time.perf_counter()
    def guard():
        if (out/'STOP').exists():raise RuntimeError('Explicit STOP')
        if not memory.check():raise MemoryError(str(memory.last))
        journal.heartbeat(memory=memory.last)
    def phase(name,**values):
        journal.heartbeat(force=True,phase=name,**values);print(json.dumps(dict(phase=name,**values)),flush=True)
    phase('LOADING_FROZEN_G1')
    assembly,seed=load();seed=deepcopy(seed);p=protocol();folder=REFERENCE/'cache/g1'
    identity=read(folder/'INPUT_AND_MODEL_IDENTITY.json')
    with np.load(BASE/'SOURCE_PREPARATION.npz',allow_pickle=False) as z:raw=z['g1_stimulus'].copy()
    seed['native_inputs']={int(f):seed['input_reference']['native_background_td']*(1+.05*(float(raw[j])-float(raw[0]))) for f,j in identity['frame_source_indices'].items()}
    if canonical_hash(seed['native_inputs'])!=identity['input_payload_sha256']:raise RuntimeError('Actual input program differs')
    source=folder/'CHECKPOINT_0010000';saved=load_bundle(source,expected_kind='validation_evidence')['payload'];stop=1.05
    write(out/'PROTOCOL_FROZEN.json',dict(schema='compiled-inner-one-pair-v1',group=1,interval_seconds=[1.,stop],
        source_checkpoint=str(source),source_checkpoint_sha256=sha(source/'manifest.json'),input_program_sha256=identity['input_payload_sha256'],
        state_budgets='Original frozen_budget(seed,state_unit)',rhs_parity=dict(rtol=5e-14,atol=2e-13,source='tests/test_ac_vs_gpu.py'),
        native_and_inner_solver='same IncrementDenseDOP853',FP64=True,fastmath=False,event_phase_and_PCG_unchanged=True,no_production_default_change=True))
    results={};trajectories={}
    try:
        for name in ('reference','compiled'):
            phase('BUILDING',arm=name);tick=time.perf_counter()
            arm=(Cell6Runner.from_checkpoint(assembly,seed,p,saved,guard=guard) if name=='reference'
                else CompiledCell6Runner.from_reference_checkpoint(assembly,seed,p,saved,guard=guard))
            setup=time.perf_counter()-tick;baseline=deepcopy(arm.metrics);before_calls=getattr(arm.compiled,'_calls',0)
            monitor=Cell6Monitor(assembly,seed['states'],parameters=seed['inner_parameters']);records=[];times=[]
            def observe(t,side):
                s=arm.snapshot(t,side,queue_node=True)
                monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
                if monitor.domain_failures:raise RuntimeError('Original physical/history gate failed')
                return s
            try:
                phase('ADVANCING',arm=name,from_seconds=1.,to_seconds=stop);tick=time.perf_counter()
                initial=observe(1.,'right');times.append(1.);records.append({k:v.copy() for k,v in initial['states'].items()})
                for end,flags in calendar(p):
                    if end<=1. or end>stop:continue
                    guard();begin=arm.inner_time;arm.ensure_native(end,'left');arm.advance_inner(begin,end)
                    observe(end,'left' if flags['frame'] else 'right')
                    if flags['rod'] or flags['bc']:
                        arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(0.,end-.0001+1e-13) if flags['rod'] else None)
                    after=observe(end,'right');arm.ack(end)
                    if flags['rod']:times.append(end);records.append({k:v.copy() for k,v in after['states'].items()})
                warm=time.perf_counter()-tick;arrays={k:np.stack([v[k] for v in records],axis=1) for k in records[0]};trajectories[name]=arrays
                np.savez_compressed(out/(name+'_STATES.npz'),time_seconds=np.asarray(times),**{'STATE__'+'__'.join(k):v for k,v in arrays.items()})
                payload=arm.checkpoint_payload(monitor=monitor.payload(),recording={})
                results[name]=dict(setup_wall_seconds=setup,warm_wall_seconds=warm,
                    metrics_delta={k:arm.metrics[k]-baseline[k] for k in arm.metrics if isinstance(arm.metrics[k],(int,float))},
                    event_generation_delta=arm.event_generation-saved['controller']['event_generation'],
                    PCG_and_history=deepcopy(arm.spike_states),event_log=deepcopy(arm.event_log),
                    compiled_rhs_calls=getattr(arm.compiled,'_calls',0)-before_calls,compiled_build=getattr(arm,'compiled_build',None),
                    transfer=getattr(arm,'transfer_evidence',None),final_controller_identity=arm.identity,
                    final_physical_hashes={k:canonical_hash(payload[k]) for k in PHYSICAL})
                if name=='compiled':
                    save_bundle(out/'COMPILED_CHECKPOINT',kind='validation_evidence',identity=dict(execution=arm.identity),payload=payload,qualification=dict(default_adopted=False,fit_eligible=False))
                    restored=CompiledCell6Runner.from_checkpoint(assembly,seed,p,load_bundle(out/'COMPILED_CHECKPOINT',expected_kind='validation_evidence')['payload'],guard=guard)
                    try:
                        copied=restored.checkpoint_payload(monitor=payload['monitor'],recording={})
                        results[name]['compiled_checkpoint_roundtrip_exact']=all(canonical_hash(copied[k])==canonical_hash(payload[k]) for k in PHYSICAL)
                    finally:restored.close()
            finally:arm.close()
            phase('ARM_COMPLETE',arm=name,warm_wall_seconds=warm)
        units={(b.population,b.mechanism,b.name):b.unit for b in (*assembly.state_blocks,*assembly.probability_blocks,*assembly.auxiliary_z1_blocks)}
        errors=[]
        for key,reference in trajectories['reference'].items():
            budget=frozen_budget(seed['states'][key],units[key])[2];error=np.max(abs(trajectories['compiled'][key]-reference),axis=1)
            errors.append(dict(field=list(key),rows=len(error),nodes=reference.shape[1],max_error=float(error.max(initial=0)),
                max_budget_ratio=float(np.max(error/budget,initial=0)),passed=bool(np.all(error<=budget))))
        discrete=lambda states:{pop:{k:v for k,v in fields.items() if k!='conditional_rate_Hz'} for pop,fields in states.items()}
        histories=canonical_hash(discrete(results['reference']['PCG_and_history']))==canonical_hash(discrete(results['compiled']['PCG_and_history']))
        rate_checks=[]
        for pop,fields in results['reference']['PCG_and_history'].items():
            a=np.asarray(fields['conditional_rate_Hz']);b=np.asarray(results['compiled']['PCG_and_history'][pop]['conditional_rate_Hz'])
            pa=seed['inner_parameters']['populations'][pop]
            Bv=frozen_budget(seed['states'][pop,'RGC_spike_history','V'],'V')[2]
            budget=pa['alpha']/pa['beta']*Bv+8192*np.finfo(float).eps*np.maximum(abs(a),1.)
            rate_checks.append(dict(population=pop,max_error=float(np.max(abs(a-b),initial=0)),
                max_budget_ratio=float(np.max(abs(a-b)/budget,initial=0)),passed=bool(np.all(abs(a-b)<=budget))))
        events=canonical_hash(results['reference']['event_log'])==canonical_hash(results['compiled']['event_log'])
        passed=all(v['passed'] for v in errors+rate_checks) and histories and events and results['compiled']['compiled_checkpoint_roundtrip_exact'] and results['compiled']['compiled_rhs_calls']>0
        write(out/'PAIR_RESULT.json',dict(passed=passed,interval_seconds=[1,stop],nodes=len(times),states=errors,
            maximum_budget_ratio=max(v['max_budget_ratio'] for v in errors),exact_PCG_history=histories,conditional_rate_checks=rate_checks,exact_events=events,
            arms=results,total_wall_seconds=time.perf_counter()-started,peak_RSS_bytes=memory.peak_rss_bytes,
            warm_total_speedup=results['reference']['warm_wall_seconds']/results['compiled']['warm_wall_seconds'],
            inner_integral_speedup=results['reference']['metrics_delta']['inner_wall_seconds']/results['compiled']['metrics_delta']['inner_wall_seconds'],
            numerical_thresholds_unchanged=True,FIT_READY=False,background_qualified=False,production_adopted=False))
        if not passed:raise RuntimeError('Pair failed original numerical/event/restore contract; no formal cache')
        journal.finish('PASS');phase('PAIR_PASS')
    except BaseException as error:
        write(out/'FAILURE.json',dict(error=repr(error)));journal.finish('FAILED');raise


if __name__=='__main__':
    nonce,folder=sys.argv[1:];out=Path(folder)
    with (out/'NATIVE_FAULTHANDLER.log').open('x',encoding='utf8') as stream:
        faulthandler.enable(file=stream,all_threads=True);attachment_gate(nonce,out);run(out)
