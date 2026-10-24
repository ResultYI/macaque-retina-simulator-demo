"""Single20ms paired continuation: fixed costs separated, no qualification run."""
from pathlib import Path
from functools import partial
from copy import copy
from contextlib import contextmanager
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.benchmark_resident_from_receipt import main,BASE,digest


def stream_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def sample_times(start,stop,np):
    first,last=round(start/.001),round(stop/.001)
    if first%2 or last-first!=20:raise ValueError('Frozen20ms window aligned to original millisecond clock required')
    values=[first*.001]
    for tick in range(first,last,2):
        values.extend((tick*.001+.000031,(tick+1)*.001,tick*.001+.00123,(tick+2)*.001))
    return np.array(values)


def restore_actual(runner,directory):
    path=directory/('runtime_export_'+runner.model.model_hash[:20]+'_receipt.json')
    receipt=json.loads(path.read_bytes());raw=directory/receipt['file']
    if (Path(receipt['file']).name!=receipt['file'] or receipt['actual_complete_codec_output'] is not True or
        receipt['fit_eligible'] is not False or raw.stat().st_size!=receipt['size'] or stream_sha(raw)!=receipt['sha256']):
        raise ValueError('Authentic actual checkpoint receipt/file mismatch')
    # The original/current loader checks full identities, layout and payload.
    return runner.restore(raw.read_text(encoding='utf8')),dict(receipt=receipt,path=str(path),receipt_sha256=digest(path))


@contextmanager
def actual_draw_capture(compiled,expected,rows,cost):
    """Trace ONLY inside original tiny CPU RGC sampling calls, never solver/RHS.

    The entire wrapped RGC-call duration is reported as a conservative upper
    bound on added observation overhead. No random draw is replayed/recreated.
    """
    from retina_sim.contracts import canonical_hash
    saved=[]
    def wrap(old,cid):
        code=old.__func__.__code__
        def invoke(t,V,aux):
            tick=time.perf_counter()
            if sys.getprofile() is not None:raise RuntimeError('Existing profiler cannot be overwritten')
            def capture(frame,event,arg):
                if event!='return' or frame.f_code is not code or arg is None:return
                q=frame.f_locals;n=q['self'];updated,events=arg
                event_times={e['cell_index']:e['time_seconds'] for e in events}
                common=dict(component=cid,bin_start_seconds=float(q['t']),dt=float(n.dt),
                    voltage_state=cid+'.'+n.state_specs[0].state_name,history_hash=canonical_hash(q['history']),
                    rng_before_hash=canonical_hash(q['aux']['rng_state']),rng_after_hash=canonical_hash(updated['rng_state']))
                for i,cell in enumerate(n.state_specs[0].cell_ids):
                    k=len(rows)
                    if k>=len(expected):raise ValueError('Unexpected original RGC draw')
                    rows.append(dict(common,id=expected[k]['id'],cell_id=cell,cell_index=i,alpha=float(n.alpha[i]),
                        beta=float(n.beta[i]),u=float(q['u'][i]),rate_Hz=float(q['rate'][i]),probability=float(q['p'][i]),
                        event_decision=bool(q['counts'][i]),event_time_seconds=event_times.get(i),draw_index=k,
                        actual_voltage_V=float(q['V'][i])))
            sys.setprofile(capture)
            try:return old(t,V,aux)
            finally:
                sys.setprofile(None);cost['entire_wrapped_RGC_seconds_upper_bound']+=time.perf_counter()-tick;cost['RGC_calls']+=1
        return invoke
    try:
        for cid,c in compiled.components.items():
            n=c.native_model
            if hasattr(n,'sample_spikes'):
                saved.append((n,'sample_spikes' in n.__dict__,n.__dict__.get('sample_spikes')))
                n.sample_spikes=wrap(n.sample_spikes,cid)
        yield
    finally:
        for n,present,old in saved:
            if present:n.sample_spikes=old
            else:del n.sample_spikes


def stage(context,channels,policy,check,save,begun,torch,np,*,output):
    from retina_sim.network import NetworkStimulus
    from retina_sim.resident_qa_harness import _schedule,_monitor_for,_recording_witness,_compare_recording
    from retina_sim.recording import RecordingPlan
    from retina_sim.resident_recording_selection import indexed_recording_plan
    from retina_sim.contracts import canonical_hash
    reference,candidate=context['reference'],context['candidate'];snapshot=context['snapshot']
    restore_times={};provenance={};states={}
    for arm,runner,directory in [('reference',reference,BASE/'full_r6/qa'),('candidate',candidate,BASE/'full_r9/qa')]:
        check();tick=time.perf_counter();states[arm],provenance[arm]=restore_actual(runner,directory)
        torch.cuda.synchronize();restore_times[arm]=time.perf_counter()-tick
    start=states['reference'].time_seconds;stop=28*.001
    if start!=.008 or states['candidate'].time_seconds!=start:raise ValueError('Authentic8ms source states required')
    times=sample_times(start,stop,np)
    cells=[];ports=[]
    for cid in states['reference'].spike_states:
        cells.extend(reference.compiled.components[cid].state_specs[0].cell_ids)
        ports.extend(cid+'.'+p for p in reference.compiled.components[cid].outputs);ports.append(cid+'.conditional_rate')
    recording=indexed_recording_plan(RecordingPlan(states=(),ports=tuple(ports),cell_ids=tuple(cells)))
    stimulus=NetworkStimulus(reference.compiled,channels,states['reference'],stop,
        states['reference'].solver_config.input_interpolation,reference.model.evidence_tag)
    _,intervals,ticks=_schedule(reference,states['reference'],stimulus,times,stop)
    expected=[]
    for t,due in ticks:
        for cid in states['reference'].spike_states:
            if cid not in due:continue
            n=reference.compiled.components[cid].native_model
            for i,cell in enumerate(n.state_specs[0].cell_ids):
                expected.append(dict(id='long_draw_%04d'%len(expected),component=cid,cell_id=cell,
                    bin_start_seconds=float(t),dt=float(n.dt),voltage_state=cid+'.'+n.state_specs[0].state_name,
                    cell_index=i,alpha=float(n.alpha[i]),beta=float(n.beta[i])))
    save('prospective_continuation_plan.json',dict(start=start,stop=stop,duration=.020,sample_times=times.tolist(),
        intervals=intervals,draws=expected,checkpoint_provenance=provenance,restore_seconds=restore_times,
        source_contract_hash=context['plan']['contract_hash'],binding=candidate.binding,
        arm_model_hashes=context['plan']['arm_identity_hashes'],retained_candidate='hot_r1',
        monitor_reference='Original frozen0ms reference, no new scales',
        events='Actual original RGC return locals captured only inside RGC function; no RNG replay; overhead upper bound retained in raw timer',
        budget='Original frozen state/recording/draw formulas unchanged; no full trajectory QA or qualification',
        timing='One paired20ms continuous advance per arm, same41-stamp density/ordinary RGC output/full monitor. Build, restore, transport prime, monitor init, advance, final report separated.'))
    primes={}
    for arm,runner in [('reference',reference),('candidate',candidate)]:
        check();tick=time.perf_counter();monitor=_monitor_for(runner,snapshot,policy)
        runner.advance(states[arm],channels,start+.0001,sample_times=np.array([start,start+.000031,start+.0001]),
            recording_plan=RecordingPlan(states=(),ports=(),cell_ids=()),state_observer=monitor.observe)
        torch.cuda.synchronize();primes[arm]=time.perf_counter()-tick
        if states[arm].time_seconds!=start:raise ValueError('Functional prime mutated original checkpoint time')
    values={};rows={};measurements={}
    for arm,runner in [('reference',reference),('candidate',candidate)]:
        check();torch.cuda.synchronize();tick=time.perf_counter();monitor=_monitor_for(runner,snapshot,policy)
        torch.cuda.synchronize();initialization=time.perf_counter()-tick
        observed=[];draws=[];cost=dict(RGC_calls=0,entire_wrapped_RGC_seconds_upper_bound=0.)
        def observe(t,y,bundle):
            if len(observed)%20==0:check()
            observed.append(float(t));monitor.observe(t,y,bundle)
        before={k:runner.integration_metrics.get(k,0) for k in ('solver_calls','nfev')}
        tick=time.perf_counter()
        try:
            with actual_draw_capture(runner.compiled,expected,draws,cost):
                ending,result=runner.advance(states[arm],channels,stop,sample_times=times,
                    recording_plan=recording,state_observer=observe)
            torch.cuda.synchronize();advance=time.perf_counter()-tick
        finally:save(arm+'_actual_draws.json',draws)
        tick=time.perf_counter();report=monitor.report(stop-start);torch.cuda.synchronize();report_time=time.perf_counter()-tick
        rows[arm]=draws;values[arm]=(ending,result,report,observed)
        measurements[arm]=dict(restore_seconds=restore_times[arm],prime_excluded_seconds=primes[arm],
            monitor_initialization_seconds=initialization,continuous_advance_seconds=advance,
            monitor_report_seconds=report_time,total_runtime_with_fixed_init_seconds=initialization+advance+report_time,
            solver_intervals=runner.integration_metrics.get('solver_calls',0)-before['solver_calls'],
            integration_RHS_calls=runner.integration_metrics.get('nfev',0)-before['nfev'],
            full_monitor_observations=len(observed),actual_draws=len(draws),event_count=result.metadata['event_count'],
            event_observation_upper_bound=cost)
        save('continuation_progress.json',dict(completed_arm=arm,measurement=measurements[arm],wall_seconds=time.perf_counter()-begun))
        save(arm+'_raw_monitor_report.json',report)
    c=copy(context['comparator_template']);c.issues=[];c.worst_status='PASS';c.closed=False;c.issue_overflow=False
    c.seen={k:[] for k in c.seen};c.spike_cells=set();c.plan=dict(c.plan,draws=expected);evidence=[];c.sink=evidence.append
    for arm in rows:
        if len(rows[arm])!=len(expected):raise ValueError('Long-window actual draw count differs')
    for a,b in zip(rows['reference'],rows['candidate']):
        spec=c.states[a['voltage_state']];_,bound=c._budget(spec,slice(a['cell_index'],a['cell_index']+1))
        if not np.isfinite(b['actual_voltage_V']) or abs(a['actual_voltage_V']-b['actual_voltage_V'])>float(bound[0]):
            c._issue('FAIL','long.actual_RGC_voltage','Actual sampled V exceeds frozen initial budget')
        c.compare_draw({k:v for k,v in a.items() if k!='actual_voltage_V'},{k:v for k,v in b.items() if k!='actual_voltage_V'})
    a,z=values['reference'],values['candidate']
    c._exact(a[3],z[3],'long.complete_monitor_callback_times')
    for key in ('grid_origin_seconds','sample_index','component_tick_indices'):
        c._exact(getattr(a[0],key),getattr(z[0],key),'long.clock.'+key)
    for cid,aux in a[0].spike_states.items():
        for key in ('history','rng_state','last_bin_counts','observation_scheme'):
            c._exact(aux[key],z[0].spike_states[cid][key],'long.RGC.'+cid+'.'+key)
        identity=lambda seq:[{k:v for k,v in e.items() if k!='time_seconds'} for e in seq]
        c._exact(identity(aux['pending_events']),identity(z[0].spike_states[cid]['pending_events']),'long.pending_event_identity.'+cid)
    flat_a=reference._flat(a[0]);flat_z=z[0].y.detach().cpu().numpy();offset=0;max_ratio=0.
    for spec in reference.model.state_specs:
        end=offset+spec.shape[0];reg=c.states[spec.state_name]
        c._numeric(flat_a[offset:end],flat_z[offset:end],reg,'long.final.'+spec.state_name)
        _,budget=c._budget(reg,slice(None))
        if end>offset:max_ratio=max(max_ratio,float(np.max(np.abs(flat_z[offset:end]-flat_a[offset:end])/budget)))
        offset=end
    np.save(output/'reference_final_state.npy',flat_a,allow_pickle=False);np.save(output/'candidate_final_state.npy',flat_z,allow_pickle=False)
    # Include validated original draws for authentic inherited pending events.
    c.plan=dict(c.plan,draws=context['plan']['draws']+expected)
    _compare_recording(c,_recording_witness(a[1]),_recording_witness(z[1]))
    for key in ('passed','domain_failures','observations'):c._exact(a[2][key],z[2][key],'long.monitor.'+key)
    for key in ('solver_intervals','integration_RHS_calls'):c._exact(measurements['reference'][key],measurements['candidate'][key],'long.work.'+key)
    save('actual_draw_comparisons.json',evidence)
    save('continuation_checks.json',dict(status=c.worst_status,issues=[x.__dict__ for x in c.issues],
        full_final_state_max_error_over_frozen_budget=max_ratio,actual_draws=len(expected),
        full_final_state_files={n:stream_sha(output/n) for n in ('reference_final_state.npy','candidate_final_state.npy')},
        scope='Final full state, selected traces, actual draws/events/RNG/history/clocks/work counts and complete monitor predicates. Not all full-state stage points/qualification.',
        original8ms_verdict_unchanged='INCONCLUSIVE',production_acceptance=False))
    left,right=measurements['reference'],measurements['candidate']
    return dict(status='FOCUSED_VARIANT_COMPLETED',kind='ONE_PAIRED_20MS_CONTINUATION',start=start,stop=stop,
        measurements=measurements,continuous_advance_speedup=left['continuous_advance_seconds']/right['continuous_advance_seconds'],
        runtime_with_fixed_init_speedup=left['total_runtime_with_fixed_init_seconds']/right['total_runtime_with_fixed_init_seconds'],
        focused_check_status=c.worst_status,issues=[x.__dict__ for x in c.issues],
        full_final_state_max_error_over_frozen_budget=max_ratio,performance_measured=True,production_acceptance=False,
        original_numerical_status='INCONCLUSIVE',qualification_repeated=False,spike_coverage_retried=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-continuation',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    raise SystemExit(main(args.output,after_prepare=partial(stage,output=args.output)))
