"""One g0 short reference on actual input, no six-case full tight repetition."""
from pathlib import Path
import json
import numpy as np
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_time_monitor import GroupTimeMonitor,frozen_budget
from retina_sim.ac_vs_monitor import DynamicDefectMonitor
from retina_sim.group_time_storage import load_bundle
from tools.cell6_raw_runner_v1 import Cell6Runner
from tools.cell6_raw_cache_v1 import witness
from tools.validate_ac_vs_timescale import calendar
from tools.run_external_first_frame_v1 import TypedDerivativeMonitor,segment_midpoint
from tools.validate_ac_vs import dynamic_observe


def recover_standard_window(assembly,seed,protocol,out,guard,saved):
    arm=Cell6Runner.from_checkpoint(assembly,seed,protocol,saved,guard=guard)
    monitor=GroupTimeMonitor(assembly,seed['states'],parameters=seed['inner_parameters'])
    records=[];times=[]
    def observe(t,side):
        s=arm.snapshot(t,side,queue_node=True)
        monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
        if monitor.domain_failures:raise ContractError('Recovered standard window physical gate failed')
        return s
    try:
        times.append(1.);records.append(witness(observe(1.,'right')))
        for end,flags in calendar(protocol):
            if end<=1. or end>1.01:continue
            guard();begin=arm.inner_time;arm.ensure_native(end,'left');arm.advance_inner(begin,end)
            observe(end,'left' if flags['frame'] else 'right')
            if flags['rod'] or flags['bc']:
                arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(0.,end-.0001+1e-13) if flags['rod'] else None)
            s=observe(end,'right');arm.ack(end)
            if flags['rod'] and round(end/.0001)!=round(times[-1]/.0001):times.append(end);records.append(witness(s))
        expected=load_bundle(out/'CHECKPOINT_0010100',expected_kind='validation_evidence')['payload']
        actual=arm.checkpoint_payload(monitor=monitor.payload(),recording={})
        categories=['native','inner','spike_states','clocks','observation_delay','input_position','readout_cursor']
        if any(canonical_hash(actual[k])!=canonical_hash(expected[k]) for k in categories):
            raise ContractError('Reconstructed standard window differs from actual saved continuation')
        np.savez_compressed(out/'STANDARD_NUMERICAL_WINDOW.npz',time_seconds=np.asarray(times),
            **{k:np.stack([v[k] for v in records],axis=1) for k in records[0]})
        (out/'STANDARD_WINDOW_RECONSTRUCTION.json').write_text(json.dumps(dict(scope='only original required 10ms standard numerical window',
            full_saved_endpoint_exact=True,categories=categories,nodes=len(times),full_trajectory_repeated=False),indent=2),encoding='utf8')
    finally:arm.close()


def check_reference(assembly,seed,protocol,out,guard):
    saved=load_bundle(out/'CHECKPOINT_0010000',expected_kind='validation_evidence')['payload']
    if not (out/'STANDARD_NUMERICAL_WINDOW.npz').exists():recover_standard_window(assembly,seed,protocol,out,guard,saved)
    arm=Cell6Runner.from_checkpoint(assembly,seed,protocol,saved,guard=guard)
    original=arm.checkpoint_payload(monitor=saved['monitor'],recording=saved['recording'])
    for k in ['native','inner','spike_states','clocks','observation_delay','input_position']:
        if canonical_hash(original[k])!=canonical_hash(saved[k]):raise ContractError('Short reference fork loses complete standard prefix')
    fork=arm.tighter_continuation();monitor=GroupTimeMonitor(assembly,seed['states'],parameters=seed['inner_parameters'])
    defects=DynamicDefectMonitor(assembly,seed['states']);mapped=TypedDerivativeMonitor(arm,defects)
    records=[];times=[]
    def observe(t,side):
        s=arm.snapshot(t,side,queue_node=True)
        monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
        if monitor.domain_failures:raise ContractError('Original short numerical physical gate failed')
        return s
    try:
        times.append(1.);records.append(witness(observe(1.,'right')))
        for end,flags in calendar(protocol):
            if end<=1. or end>1.01:continue
            guard();begin=arm.inner_time;arm.ensure_native(end,'left');advance=arm.advance_inner(begin,end)
            for curve in advance.dense_segments:
                t=segment_midpoint(curve);dynamic_observe(arm,advance,t,mapped,'left' if t==advance.t1 else 'right')
                if not defects.report()['passed']:raise ContractError('Original short dynamic-defect budget failed')
            observe(end,'left' if flags['frame'] else 'right')
            if flags['rod'] or flags['bc']:
                arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(0.,end-.0001+1e-13) if flags['rod'] else None)
            s=observe(end,'right');arm.ack(end)
            if flags['rod']:
                if round(end/.0001)!=round(times[-1]/.0001):times.append(end);records.append(witness(s))
        arrays={k:np.stack([r[k] for r in records],axis=1) for k in records[0]}
        np.savez_compressed(out/'TIGHT_NUMERICAL_WINDOW.npz',time_seconds=np.asarray(times),**arrays)
        with np.load(out/'STANDARD_NUMERICAL_WINDOW.npz',allow_pickle=False) as a:
            if not np.array_equal(np.round(a['time_seconds']/.0001),np.round(np.asarray(times)/.0001)):raise ContractError('Short reference nodes differ')
            fields=[];units={(b.population,b.mechanism,b.name):b.unit for b in (*assembly.state_blocks,*assembly.probability_blocks,*assembly.auxiliary_z1_blocks)}
            for name,v in arrays.items():
                key=tuple(name.split('__')[1:]);budget=frozen_budget(seed['states'][key],units[key])[2]
                error=np.max(abs(v-a[name]),axis=1);ratio=float(np.max(error/budget,initial=0))
                fields.append(dict(field=name,rows=len(v),nodes=len(times),max_budget_ratio=ratio,passed=bool(np.all(error<=budget))))
        report=dict(passed=all(v['passed'] for v in fields) and defects.report()['passed'],scope='g0 actual first-active10ms [1,1.01]s from common exact full1s continuation',
            fork=fork,standard=dict(rtol=1e-6,atol=1e-9,max_step=.0001),tight=dict(rtol=1e-7,atol=1e-10,max_step=.00005),
            state_fields=len(fields),state_values=sum(f['rows'] for f in fields),nodes=len(times),
            maximum_budget_ratio=max(f['max_budget_ratio'] for f in fields),fields=fields,dynamic_defect=defects.report(),
            no_gate_or_budget_change=True,full_7point1s_or_other_cases_tight_claimed=False)
        (out/'SHORT_NUMERICAL_CHECK.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        return report
    finally:arm.close()
