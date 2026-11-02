"""Actual50ms source movie, original standard/tight profiles, complete states."""
from copy import deepcopy
import time
import numpy as np
from retina_sim.contracts import canonical_hash
from retina_sim.group_time_storage import save_bundle,load_bundle
from retina_sim.group_time_monitor import frozen_budget
from retina_sim.ac_vs_monitor import DynamicDefectMonitor
from tools.cell6_runtime_monitor_v3 import Cell6Monitor
from tools.cell6_compiled_kernel_guard_v2 import GuardedCompiledCell6Runner
from tools.cell6_compiled_inner_v1 import PHYSICAL,CLOSED_INSTANCES
from tools.integrated_far_readouts_v1 import Readouts
from tools.validate_ac_vs_timescale import calendar
from tools.run_external_first_frame_v1 import TypedDerivativeMonitor,segment_midpoint
from tools.validate_ac_vs import dynamic_observe
from tools.gin_radial_context_v1 import ROOT,BASE,read,write,load,protocol
from tools.prepare_gin_radial_candidate_v1 import sha

def trajectory(precision,guard,journal):
    start=time.perf_counter();out=BASE/'forward'/precision;out.mkdir(parents=True);phase=time.perf_counter();assembly,seed=load();seed=deepcopy(seed);p=protocol()
    with np.load(BASE/'background/MOVIE_NATIVE_INPUTS.npz',allow_pickle=False) as z:drive=z['live_native_drive'].copy()
    seed['native_inputs']={i:drive[i].copy() for i in p['input_frame_indices']}
    rod=float(seed['input_reference']['rod_background_rstar_s']);nr=assembly.populations['rods'].group_count
    seed['rod_inputs']={i:np.full(nr,rod) for i in p['input_frame_indices']}
    load_seconds=time.perf_counter()-phase;identity=canonical_hash(dict(source=sha(BASE/'INPUT_APPROXIMATION_MANIFEST.json'),optics=sha(BASE/'optical/CONFIG.json'),
        actual_capture=sha(BASE/'optical/CAPTURE.mat'),initial_seed=sha(BASE/'background/seed/manifest.json'),protocol=p,input_payload=canonical_hash(seed['native_inputs']),
        code={n:sha(ROOT/'tools'/n) for n in ('gin_radial_context_v1.py','gin_radial_initialize_v1.py','gin_radial_forward_v1.py','cell6_compiled_kernel_guard_v2.py','cell6_compiled_inner_v1.py','cell6_raw_runner_v1.py','cell6_runtime_monitor_v3.py')},precision=precision))
    write(out/'IDENTITY.json',dict(identity=identity,precision=precision,protocol=p,source_frames_zero_based=list(range(751,759)),
        no_observed_spikes_in_forward=True,initial_state=canonical_hash(seed['states']),initial_PCG_history=canonical_hash(seed['spike_states']),
        role='50ms actual-source conditional radial cost/numeric check; not60s trial reproduction',FIT_READY=False,background_qualified=False,production_adopted=False))
    arm=None;times=[];states=[];records=[];record_times=[];checkpoints=[];restore=None;write_seconds=0.;metrics=[];builds=[];closed_start=len(CLOSED_INSTANCES)
    try:
        phase=time.perf_counter();arm=GuardedCompiledCell6Runner(assembly,seed,p,precision=precision,guard=guard);construction=time.perf_counter()-phase;builds.append(arm.compiled_build)
        monitor=Cell6Monitor(assembly,seed['states'],parameters=seed['inner_parameters']);reader=Readouts(arm)
        defects=DynamicDefectMonitor(assembly,seed['states']);mapped=TypedDerivativeMonitor(arm,defects)
        def observe(t,side):
            snapshot=arm.snapshot(t,side,queue_node=True)
            monitor.observe(t,side,snapshot['states'],snapshot['rhs'],transitions=snapshot['transitions'],ports=snapshot['ports'],aux=snapshot['aux'],event_generation=snapshot['event_generation'])
            if monitor.domain_failures:raise ValueError('Original finite/domain/history gate failed: '+str(monitor.domain_failures))
            return snapshot
        def record(t,snapshot):
            if times and round(t/.0001)==round(times[-1]/.0001):raise ValueError('Duplicate authoring node')
            times.append(t);states.append({'STATE__'+'__'.join(k):v.copy() for k,v in snapshot['states'].items()})
            if round(t/.0001)%10==0:record_times.append(t);records.append(reader.observe(t,snapshot))
        first=observe(0.,'right');arm.initial_commit();first=observe(0.,'right');record(0.,first);arm.ack(0.)
        phase=time.perf_counter()
        for end,flags in calendar(p):
            guard();begin=arm.inner_time;arm.ensure_native(end,'left');advance=arm.advance_inner(begin,end)
            if precision=='tight':
                for curve in advance.dense_segments:
                    t=segment_midpoint(curve);dynamic_observe(arm,advance,t,mapped,'left' if t==advance.t1 else 'right')
                if not defects.report()['passed']:raise ValueError('Original dynamic-defect budget failed')
            observe(end,'left' if flags['frame'] or end==arm.stop else 'right')
            if flags['rod'] or flags['bc']:
                arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(0.,end-.0001+1e-13) if flags['rod'] else None)
            after=observe(end,'left' if end==arm.stop else 'right');arm.ack(end)
            if flags['rod'] or end==arm.stop:record(end,after)
            j=round(end/.0001)
            if (flags['rod'] or end==arm.stop) and j in (250,500):
                t0=time.perf_counter();payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=dict(identity=identity,nodes=len(times)))
                path=out/('CHECKPOINT_'+str(j).zfill(7));save_bundle(path,kind='validation_evidence',identity=dict(candidate=identity),payload=payload,qualification=dict(default_adopted=False,fit_eligible=False))
                checkpoints.append(path.name);write_seconds+=time.perf_counter()-t0
                if precision=='standard' and j==250:
                    saved=load_bundle(path,expected_kind='validation_evidence')['payload'];restored=GuardedCompiledCell6Runner.from_checkpoint(assembly,seed,p,saved,guard=guard)
                    copied=restored.checkpoint_payload(monitor=monitor.payload(),recording=payload['recording'])
                    if any(canonical_hash(saved[k])!=canonical_hash(copied[k]) for k in PHYSICAL):restored.close();raise ValueError('Complete scientific checkpoint roundtrip differs')
                    metrics.append(dict(arm.metrics));builds.append(restored.compiled_build);arm.close();arm=restored;reader=Readouts(arm);mapped=TypedDerivativeMonitor(arm,defects)
                    restore=dict(time_seconds=end,exact_categories=list(PHYSICAL),resumed_remaining_original_trajectory=True,no_state_history_or_PCG_reset=True)
                journal.heartbeat(force=True,phase=precision,source_seconds=end,total_seconds=.05)
                print(precision,end,'of0.05',flush=True)
        loop_seconds=time.perf_counter()-phase;metrics.append(dict(arm.metrics));events=[dict(population=pop,**e) for pop,es in arm.event_log.items() for e in es if e['time_seconds']<.05]
        if len(times)!=501 or len(records)!=51:raise ValueError('Short authoring/recording grid coverage differs')
        t0=time.perf_counter();np.savez_compressed(out/'NUMERICAL_STATES.npz',time_seconds=np.asarray(times),**{k:np.stack([x[k] for x in states],axis=1) for k in states[0]})
        np.savez_compressed(out/'RESPONSE.npz',time_seconds=np.asarray(record_times),**{k:np.stack([x[k] for x in records],axis=1) for k in records[0]})
        write(out/'FIELDS.json',reader.fields);write(out/'GROUP_OBJECT_EVENTS.json',events);write(out/'END_HISTORY.json',arm.spike_states);write(out/'MONITOR.json',monitor.report(.0001))
        write_seconds+=time.perf_counter()-t0
        summary=dict(status='COMPLETE_SHORT_TRAJECTORY',identity=identity,precision=precision,simulated_seconds=.05,authoring_nodes=len(times),layer_recording_nodes=len(records),
            state_fields=len(states[0]),state_values=sum(len(v) for v in states[0].values()),fields=len(records[0]),event_objects=len(events),load_context_seconds=load_seconds,
            first_construction_seconds=construction,compiled_builds=builds,loop_wall_seconds_including_monitor_checkpoint_restore=loop_seconds,metrics_instances=metrics,
            write_seconds=write_seconds,wall_seconds=time.perf_counter()-start,restore=restore,checkpoints=checkpoints,
            domain_history_finite_passed=not monitor.domain_failures,dynamic_defect=defects.report() if precision=='tight' else None,
            compiled_RHS_calls=arm.compiled._calls+sum(x['compiled_rhs_calls'] for x in CLOSED_INSTANCES[closed_start:]),metrics_are_cumulative_across_checkpoint=True,
            python_continuous_RHS_fallback_calls=0,FIT_READY=False,background_qualified=False,production_adopted=False)
        write(out/'RESULT.json',summary);return summary
    finally:
        if arm is not None:arm.close()

def run(guard,journal):
    standard=trajectory('standard',guard,journal);tight=trajectory('tight',guard,journal)
    a0,seed=load();units={(b.population,b.mechanism,b.name):b.unit for b in (*a0.state_blocks,*a0.probability_blocks,*a0.auxiliary_z1_blocks)};fields=[]
    with np.load(BASE/'forward/standard/NUMERICAL_STATES.npz',allow_pickle=False) as a,np.load(BASE/'forward/tight/NUMERICAL_STATES.npz',allow_pickle=False) as b:
        if not np.array_equal(np.round(a['time_seconds']/.0001),np.round(b['time_seconds']/.0001)):raise ValueError('Numerical authoring grids differ')
        for n in a.files:
            if n=='time_seconds':continue
            k=tuple(n.split('__')[1:]);budget=frozen_budget(seed['states'][k],units[k])[2];err=np.max(abs(a[n]-b[n]),axis=1)
            fields.append(dict(field=n,unit=units[k],rows=len(err),nodes=501,max_budget_ratio=float(np.max(err/budget,initial=0)),passed=bool(np.all(err<=budget))))
    ea=read(BASE/'forward/standard/GROUP_OBJECT_EVENTS.json');eb=read(BASE/'forward/tight/GROUP_OBJECT_EVENTS.json')
    report=dict(passed=all(x['passed'] for x in fields) and tight['dynamic_defect']['passed'],scope='All saved full-chain state scalars at501100us authoring nodes across[0,50]ms; original standard/tight tolerances and frozen budgets',
        state_fields=len(fields),state_scalars=sum(f['rows'] for f in fields),maximum_budget_ratio=max(f['max_budget_ratio'] for f in fields),fields=fields,
        event_lists_identical=canonical_hash(ea)==canonical_hash(eb),event_counts=dict(standard=len(ea),tight=len(eb)),
        events_scope='Generated group-object Bernoulli events, not independently measured member spikes; event equality reported separately from continuous-state numerical gate',
        no_budget_or_threshold_change=True,full60s_or600s_numerics_claimed=False,FIT_READY=False,background_qualified=False,production_adopted=False)
    write(BASE/'SHORT_NUMERICAL_CHECK.json',report)
    if not report['passed']:raise ValueError('Original full-chain short numerical state/defect gate failed')
