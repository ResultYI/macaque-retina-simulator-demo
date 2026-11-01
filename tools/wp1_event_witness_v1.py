"""Recover only missing sampler witnesses using the original checkpointed paths."""
from pathlib import Path
from copy import deepcopy
import faulthandler,json,os,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.run_cell6_matched_baselines_v1 import attachment_gate,read,sha,write,BASE,REFERENCE


def run(out,index):
    import numpy as np
    from tools.cell6_halo_context_v1 import load,protocol
    from tools.cell6_raw_runner_v1 import Cell6Runner
    from tools.cell6_compiled_kernel_guard_v2 import GuardedCompiledCell6Runner,install_guard
    from tools.cell6_runtime_monitor_v2 import Cell6Monitor
    from tools.validate_ac_vs_timescale import calendar
    from retina_sim.background_checkpoint import MemoryGuard
    from retina_sim.contracts import canonical_hash
    from retina_sim.group_time_storage import load_bundle,save_bundle
    from retina_sim.group_time_monitor import GroupTimeComparator
    from tools.durable_research_runtime_v1 import DurableRun
    accel=ROOT/'reports/CELL6_COMPILED_INNER_V1';parent=out.parent
    differences=read(accel/'FORMAL_EVENT_SCOPE.json')['changes']
    old=REFERENCE/'cache/g1';new=accel/'formal_g1/cache/g1'
    common={p.name for p in old.glob('CHECKPOINT_*')}&{p.name for p in new.glob('CHECKPOINT_*')}
    grouped={}
    for difference in differences:
        tick=int(round(difference['reference']['bin_start_seconds']*1000))
        eligible=[n for n in common if int(n.split('_')[1])<=tick*10]
        checkpoint=max(eligible,key=lambda n:int(n.split('_')[1]))
        grouped.setdefault(checkpoint,[]).append(difference)
    plans=sorted(grouped.items(),key=lambda item:int(item[0].split('_')[1]));checkpoint,changes=plans[index]
    wanted={int(round(v['reference']['bin_start_seconds']*1000)) for v in changes};stop=max(wanted)/1000.
    start=int(checkpoint.split('_')[1])*.0001
    journal=DurableRun(out/'runtime','recover_original_event_witness');memory=MemoryGuard(24*1024**3,8*1024**3)
    def guard():
        if (out/'STOP').exists():raise RuntimeError('Explicit STOP')
        if not memory.check():raise MemoryError(str(memory.last))
        journal.heartbeat(memory=memory.last)
    def phase(name,**values):
        journal.heartbeat(force=True,phase=name,window=index,**values)
        print(json.dumps(dict(phase=name,window=index,**values)),flush=True)
    assembly,seed=load();seed=deepcopy(seed);p=protocol();identity=read(old/'INPUT_AND_MODEL_IDENTITY.json')
    with np.load(BASE/'SOURCE_PREPARATION.npz',allow_pickle=False) as z:raw=z['g1_stimulus'].copy()
    seed['native_inputs']={int(f):seed['input_reference']['native_background_td']*(1+.05*(float(raw[j])-float(raw[0]))) for f,j in identity['frame_source_indices'].items()}
    if canonical_hash(seed['native_inputs'])!=identity['input_payload_sha256']:raise RuntimeError('Frozen input program differs')
    write(out/'PROTOCOL_FROZEN.json',dict(window=index,checkpoint=checkpoint,start=start,stop=stop,bins=sorted(wanted),
        reference_checkpoint_sha256=sha(old/checkpoint/'manifest.json'),compiled_checkpoint_sha256=sha(new/checkpoint/'manifest.json'),
        same_source_input=True,original_calendar_and_snapshots=True,no_sampler_or_tolerance_changes=True,
        purpose='Recover existing last_draws for changed event timestamps only; not a full-duration new numerical claim'))
    all_witnesses={};began=time.perf_counter()
    try:
        for name,folder,runner in [('reference',old,Cell6Runner),('compiled',new,GuardedCompiledCell6Runner)]:
            phase('BUILDING',arm=name,source_seconds=start)
            saved=load_bundle(folder/checkpoint,expected_kind='validation_evidence')['payload']
            arm=runner.from_checkpoint(assembly,seed,p,saved,guard=guard)
            if name=='reference':install_guard(arm.inner)
            if canonical_hash(arm.checkpoint_payload()['spike_states'])!=canonical_hash(saved['spike_states']):raise RuntimeError('Prefix history changed')
            monitor=Cell6Monitor.from_payload(assembly,seed['states'],saved['monitor'],parameters=seed['inner_parameters'])
            retained={};full_events=read(folder/'GROUP_OBJECT_EVENTS.json')
            def observe(t,side):
                s=arm.snapshot(t,side,queue_node=True)
                monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
                if monitor.domain_failures:raise RuntimeError('Original physical/history gate failed')
            try:
                for end,flags in calendar(p):
                    if end<=arm.inner_time:continue
                    if end>stop+1e-13:break
                    guard();arm.ensure_native(end,'left');arm.advance_inner(arm.inner_time,end)
                    observe(end,'left' if flags['frame'] or end==arm.stop else 'right')
                    if flags['rod'] or flags['bc']:
                        arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(0.,end-.0001+1e-13) if flags['rod'] else None)
                    tick=int(round(end*1000))
                    if flags['bc'] and tick in wanted:
                        events=[dict(population=pop,**event) for pop,events in arm.last_events.items() for event in events]
                        original_events=[e for e in full_events if int(round(e['bin_start_seconds']*1000))==tick]
                        if canonical_hash(events)!=canonical_hash(original_events):
                            write(out/(name+'_REPLAY_MISMATCH.json'),dict(bin=tick,original=original_events,replayed=events))
                            raise RuntimeError('Supplement must reproduce the original cached events exactly')
                        retained[str(tick)]=deepcopy(arm.last_draws)
                        phase('BIN_WITNESS_SAVED',arm=name,source_seconds=end,bin=tick)
                    observe(end,'left' if end==arm.stop else 'right');arm.ack(end)
                if set(retained)!=set(map(str,wanted)):raise RuntimeError('Requested witness coverage incomplete')
                save_bundle(out/(name+'_WITNESSES'),kind='validation_evidence',identity=dict(window=index,arm=name),payload=retained,qualification=dict(default_adopted=False,fit_eligible=False))
                all_witnesses[name]=retained
            finally:arm.close()
        comparator=GroupTimeComparator(monitor,{},{},parameters=seed['inner_parameters']);issues=[];numeric_checks=[]
        def issue(kind,gate,detail):issues.append(dict(status=kind,gate=gate,detail=str(detail)))
        def numeric(name,a,b,allowed):
            a=np.asarray(a);b=np.asarray(b);allowed=np.asarray(allowed);error=abs(a-b)
            numeric_checks.append(dict(name=name,max_budget_ratio=float(np.max(error/allowed,initial=0)),passed=bool(np.all(error<=allowed))))
            if not np.all(error<=allowed):issue('FAIL',name,'Original propagated budget exceeded')
        for tick in sorted(all_witnesses['reference']):
            for pop in all_witnesses['reference'][tick]:
                comparator._draw(pop,all_witnesses['reference'][tick][pop],all_witnesses['compiled'][tick][pop],numeric,issue)
        passed=not issues and all(x['passed'] for x in numeric_checks)
        write(out/'WITNESS_RESULT.json',dict(passed=passed,window=index,start=start,stop=stop,bins=sorted(wanted),issues=issues,numeric_checks=numeric_checks,
            original_cached_events_reproduced_bitwise=True,original_GroupTimeComparator_draw_used=True,
            root_cause='Continuous FP64 backend yields tiny rate differences; original sampler converts same uniform to time by division',
            event_formula_or_acceptance_threshold_changed=False,wall_seconds=time.perf_counter()-began,peak_RSS_bytes=memory.peak_rss_bytes))
        if not passed:raise RuntimeError('Original sampler witness contract not passed')
        journal.finish('PASS');phase('WINDOW_PASSED')
    except BaseException as error:
        write(out/'FAILURE.json',dict(error=repr(error)));journal.finish('FAILED');raise


if __name__=='__main__':
    nonce,folder,number=sys.argv[1:];out=Path(folder)
    with (out/'NATIVE_FAULTHANDLER.log').open('x',encoding='utf8') as stream:
        faulthandler.enable(file=stream,all_threads=True);attachment_gate(nonce,out);run(out,int(number))
