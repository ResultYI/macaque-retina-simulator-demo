"""Bounded natural event-path coverage from authentic 8ms checkpoints.

The original 8ms verdict remains immutable. This checks actual RGC voltage,
rate/probability/event-time propagation and exact RNG/history on a continuation;
it is not a second full-state trajectory QA or background/fitting qualification.
"""
from copy import copy
from pathlib import Path
import hashlib,json,sys,time


def supplement(context,channels,policy,check,save,base,*,stop=.064):
    import numpy as np
    from .contracts import canonical_hash
    from .recording import RecordingPlan
    from .resident_qa_harness import _monitor_for
    reference,candidate=context['reference'],context['candidate']
    def restore(runner,directory):
        key='runtime_export_'+runner.model.model_hash[:20]
        receipt=json.loads((directory/(key+'_receipt.json')).read_bytes());path=directory/receipt['file']
        if Path(receipt['file']).name!=receipt['file'] or path.stat().st_size!=receipt['size']:
            raise ValueError('Actual event continuation checkpoint size/path differs')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for b in iter(lambda:stream.read(1024**2),b''):h.update(b)
        if h.hexdigest()!=receipt['sha256']:raise ValueError('Actual event continuation checkpoint changed')
        return runner.restore(path.read_text(encoding='utf8'))
    states=[restore(reference,base/'full_r6/qa'),restore(candidate,base/'full_r9/qa')]
    start=states[0].time_seconds
    if start!=.008 or states[1].time_seconds!=start:raise ValueError('Authentic 8ms continuation required')
    cells=[tuple(c) for c in context['plan']['rgc_cells']]
    original=json.loads((base/'full_r9/qa/outcome.json').read_bytes())
    covered={tuple(c) for c in original['observed_spike_cells']}
    ports=tuple(cid+'.'+suffix for cid in states[0].spike_states for suffix in ('spike_count','window_rate'))
    recording=RecordingPlan(states=(),ports=ports,cell_ids=tuple(c[1] for c in cells))
    monitors=[_monitor_for(r,context['snapshot'],policy) for r in (reference,candidate)]
    expected=[]
    for tick in range(9,int(round(stop/.001))+1):
        for cid in states[0].spike_states:
            native=reference.compiled.components[cid].native_model
            for i,cell in enumerate(native.state_specs[0].cell_ids):
                expected.append(dict(id='continued_%05d'%len(expected),component=cid,cell_id=cell,
                    bin_start_seconds=float(states[0].grid_origin_seconds+tick*native.dt),dt=float(native.dt),
                    voltage_state=cid+'.'+native.state_specs[0].state_name,cell_index=i,
                    alpha=float(native.alpha[i]),beta=float(native.beta[i])))
    plan=dict(schema='resident_natural_event_supplement_v1',start=start,maximum_stop=stop,chunk_seconds=.004,
        seed_changed=False,stimulus_changed=False,physiology_changed=False,
        actual_original_checkpoint=True,actual_candidate_checkpoint=True,draws=expected,
        source_contract_sha256=context['plan']['contract_hash'],initial_scales='Unchanged frozen 0ms QA initial state',
        source_plan_sha256=context['plan']['plan_sha256'],
        scope='Actual sampled voltage prerequisite + original rate/probability/event-time bounds, exact RNG/history/order, flushed outputs. Complete original monitor executes at all callbacks. Original 8ms verdict is not rewritten.')
    save('natural_event_plan.json',plan)
    validator=copy(context['comparator_template']);validator.plan=dict(validator.plan,draws=expected)
    validator.issues=[];validator.worst_status='PASS';validator.issue_overflow=False;validator.closed=False
    validator.seen={key:[] for key in validator.seen};validator.spike_cells=set()
    evidence=[];validator.sink=evidence.append
    codes={c.native_model.sample_spikes.__func__.__code__ for c in reference.compiled.components.values() if hasattr(c.native_model,'sample_spikes')}
    owners={id(c.native_model):cid for cid,c in reference.compiled.components.items() if hasattr(c.native_model,'sample_spikes')}
    positions=[0,0];timestamp_lists=[[],[]];all_rows=[[],[]]
    def tracer(arm):
        def profile(frame,event,arg):
            if event!='return' or frame.f_code not in codes or arg is None:return
            loc=frame.f_locals;native=loc['self'];cid=owners[id(native)];updated,events=arg
            by_cell={e['cell_index']:e['time_seconds'] for e in events}
            common=dict(component=cid,bin_start_seconds=float(loc['t']),dt=float(native.dt),
                voltage_state=cid+'.'+native.state_specs[0].state_name,
                history_hash=canonical_hash(loc['history']),rng_before_hash=canonical_hash(loc['aux']['rng_state']),
                rng_after_hash=canonical_hash(updated['rng_state']))
            for i,cell in enumerate(native.state_specs[0].cell_ids):
                index=positions[arm]
                if index>=len(expected):raise ValueError('Unplanned additional continuation draw')
                record=dict(common,id=expected[index]['id'],cell_id=cell,cell_index=i,
                    alpha=float(native.alpha[i]),beta=float(native.beta[i]),u=float(loc['u'][i]),
                    rate_Hz=float(loc['rate'][i]),probability=float(loc['p'][i]),event_decision=bool(loc['counts'][i]),
                    event_time_seconds=by_cell.get(i),draw_index=index)
                row=dict(record,actual_voltage_V=float(loc['V'][i]));all_rows[arm].append(row)
                positions[arm]+=1
                if arm:
                    a=all_rows[0][index];spec=validator.states[record['voltage_state']]
                    _,bound=validator._budget(spec,slice(i,i+1))
                    if not np.isfinite(row['actual_voltage_V']) or abs(row['actual_voltage_V']-a['actual_voltage_V'])>float(bound[0]):
                        validator._issue('FAIL','event_voltage_prerequisite','Actual sampled voltage exceeds unchanged frozen initial budget')
                    validator.compare_draw({k:v for k,v in a.items() if k!='actual_voltage_V'},record)
                    if validator.worst_status!='PASS':raise ValueError('Natural event gate failed; no redraw or threshold change')
        return profile
    results=[];current=start;begun=time.perf_counter();wall_reserve_stop=False
    try:
        while current<stop-1e-12 and len(covered)<len(cells):
            if time.perf_counter()-context['run_begun']>680:
                wall_reserve_stop=True
                break
            target=min(stop,float(int(round(current/.001))+4)*.001);pair=[]
            for arm,runner in enumerate((reference,candidate)):
                check();before=len(all_rows[arm]);stamps=[]
                if sys.getprofile() is not None:raise RuntimeError('Exclusive event-return profile required')
                def observe(t,y,bundle,arm=arm):
                    check();stamps.append(float(t));monitors[arm].observe(t,y,bundle)
                sys.setprofile(tracer(arm))
                try:
                    states[arm],result=runner.advance(states[arm],channels,target,
                        sample_times=np.array([current,target]),recording_plan=recording,state_observer=observe)
                finally:
                    sys.setprofile(None)
                    save('natural_event_'+('reference' if arm==0 else 'candidate')+'_draws.json',all_rows[arm])
                timestamp_lists[arm].extend(stamps);pair.append((result,stamps,len(all_rows[arm])-before))
            if pair[0][1:]!=pair[1][1:]:raise ValueError('Natural continuation callback/draw schedule differs')
            for key in ('sample_index','component_tick_indices','grid_origin_seconds'):
                if canonical_hash(getattr(states[0],key))!=canonical_hash(getattr(states[1],key)):
                    raise ValueError('Natural continuation exact clock differs')
            for cid,aux in states[0].spike_states.items():
                for key in ('history','rng_state','last_bin_counts'):
                    if canonical_hash(aux[key])!=canonical_hash(states[1].spike_states[cid][key]):
                        raise ValueError('Natural continuation exact RNG/history differs')
            for cid,events in pair[0][0].metadata['spike_events'].items():
                a=[(e['cell_id'],e['bin_start_seconds'],e['bin_stop_seconds']) for e in events]
                b=[(e['cell_id'],e['bin_start_seconds'],e['bin_stop_seconds']) for e in pair[1][0].metadata['spike_events'][cid]]
                if a!=b:raise ValueError('Natural flushed-event cell/order/bin identity differs')
                covered.update((cid,e['cell_id']) for e in events)
            for key in pair[0][0].outputs_with_units:
                if not np.array_equal(pair[0][0].outputs_with_units[key].values,pair[1][0].outputs_with_units[key].values):
                    raise ValueError('Natural recorded count/rate/time differs')
            current=target
            results.append(dict(stop=current,covered_cells=sorted(covered),draws=positions.copy()))
            save('natural_event_progress.json',dict(chunks=results,wall_seconds=time.perf_counter()-begun))
        reports=[m.report(current-start) for m in monitors] if current>start else []
        if reports and (reports[0]['passed']!=reports[1]['passed'] or reports[0]['domain_failures']!=reports[1]['domain_failures']):
            raise ValueError('Natural full-monitor predicates differ')
        outcome=dict(status='EVENT_GATES_COMPLETED' if len(covered)==len(cells) else 'EVENT_COVERAGE_REMAINS_INCOMPLETE',
            original_8ms_verdict_unchanged=True,start=start,stop=current,maximum_stop=stop,draws=positions,
            covered_cells=sorted(covered),missing_cells=[c for c in cells if c not in covered],
            issues=[x.__dict__ for x in validator.issues],full_monitor_reports=reports,
            full_monitor_callbacks=[len(x) for x in timestamp_lists],wall_seconds=time.perf_counter()-begun,
            stopped_for_hot_wall_reserve=wall_reserve_stop,
            physiology_fit=False,production_acceptance=False)
    except Exception as exc:
        outcome=dict(status='EVENT_SUPPLEMENT_STOPPED',error_type=type(exc).__name__,detail=str(exc),
            start=start,last_complete_pair_stop=current,draws=positions,covered_cells=sorted(covered),
            issues=[x.__dict__ for x in validator.issues],original_8ms_verdict_unchanged=True,
            wall_seconds=time.perf_counter()-begun)
    save('natural_event_evidence.json',evidence);save('natural_event_outcome.json',outcome)
    return outcome
