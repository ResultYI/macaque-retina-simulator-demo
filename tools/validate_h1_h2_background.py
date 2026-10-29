"""Bounded joint-background qualification using the already delivered local graph."""
from pathlib import Path
from copy import deepcopy
import argparse
import importlib.util
import json
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.parameter_system import data
from retina_sim.parameter_migration import export_bundle
from retina_sim.group_time_storage import save_bundle,load_bundle
from retina_sim.group_time_monitor import GroupTimeMonitor,rhs_allowance,frozen_budget
from retina_sim.h1_h2_effective import H1H2GroupRunner,bind_parameters,MECHANISM,UNIT
from retina_sim.h1_h2_background import initialize_joint_background,INITIALIZATION
from retina_sim.spatial_group_model import BC
from retina_sim.spatial_group_graph import digest
from tools.validate_ac_vs_native_support import load
from tools.validate_h1_h2_effective import tick

DELIVERY=ROOT/'reports/h2_integration_20261028_v1/delivery'
DIAGNOSIS=ROOT/'reports/h2_s_on_bc_defect_20261028_v1'


def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):Path(p).write_text(json.dumps(data(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def assembly_from_saved(seed):
    # Reuse the previous independently verified local-only loader; no global HC work.
    helper=DIAGNOSIS/'diagnose.py'
    expected=read(DIAGNOSIS/'CHANGE_MANIFEST.json')['new_diagnostic_files']['diagnose.py']['sha256']
    if digest(helper)!=expected:raise ContractError('Prior frozen local loader differs')
    spec=importlib.util.spec_from_file_location('h2_saved_local_loader',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parent,_,source=load(ROOT/'reports/ac_vs_native_support_v1/prepared_v1')
    return module.local_saved_assembly(parent,seed),source


def protocol_for(seed,source):
    p=deepcopy(source['protocol']);origin=p['clock_origin_seconds']
    start=int(round((seed['time_seconds']-origin)/.0001))
    p['stop_seconds']=origin+(start+60)*.0001
    return p,start


def snapshot_payload(arm,s):
    return dict(states=s['states'],rhs=s['rhs'],ports=s['ports'],spike_states=deepcopy(arm.spike_states))


def compare(a,b):
    errors={}
    for category in ('states','rhs','ports'):
        if set(a[category])!=set(b[category]):raise ContractError('Checkpoint snapshot registry differs')
        errors[category]=max(float(np.max(abs(v-b[category][k]),initial=0.)) for k,v in a[category].items())
    errors['spike_states_exact']=canonical_hash(a['spike_states'])==canonical_hash(b['spike_states'])
    if any(errors[k] for k in ('states','rhs','ports')) or not errors['spike_states_exact']:
        raise ContractError('Checkpoint continuation differs: '+str(errors))
    return errors


def run_background(a,seed,p,start,out):
    arm=H1H2GroupRunner(a,seed,p);restored=None
    try:
        initial=arm.snapshot(arm.start,'right',queue_node=True)
        monitor=GroupTimeMonitor(a,initial['states'],parameters=seed['inner_parameters'])
        def observe(s):
            monitor.observe(s['time_seconds'],s['side'],s['states'],s['rhs'],transitions=s['transitions'],
                ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
            if monitor.domain_failures:raise ContractError('Background monitor domain failure: '+str(monitor.domain_failures))
        observe(initial);report=monitor.report(.006)
        target=('S_ON_BC','HistoryBipolarTerminal','history_0')
        budgets=[]
        for k,row in monitor.rows.items():
            if row['budget'] is None:raise ContractError('Missing frozen numerical scale: '+str(k))
            f=initial['rhs'][k];allow=rhs_allowance(row['scale'],row['budget'],np.zeros_like(f),f)
            budgets.append(float(np.max(abs(f)/allow,initial=0.)))
        map_ratio=max(float(np.max(abs(initial['transitions'][k]-initial['states'][k])/monitor.rows[k]['budget'],initial=0.))
            for k in monitor.transition_keys)
        write(out/'INITIAL_EQUILIBRIUM.json',dict(monitor=report,max_RHS_allowance_ratio=max(budgets),
            maximum_pure_transition_error_over_B=map_ratio,target_initial=initial['states'][target],
            old_target_norm=77.27515150881214,new_target_norm=next(r['discrete_defect_norm'] for r in report['states']
                if r['state']==':'.join(target))))
        if not report['passed'] or max(budgets)>1 or map_ratio>1:
            raise ContractError('Independent initial background criterion failed; no further audit or tuning')
        arm.initial_commit();post=arm.snapshot(arm.start,'right',queue_node=True);observe(post)
        first={}
        for pop in BC:
            keys=[(pop,'HistoryBipolarTerminal','history_'+str(i)) for i in range(3)]
            first[pop]=max(float(np.max(abs(post['states'][k]-initial['states'][k]),initial=0.)) for k in keys)
        if any(first.values()):raise ContractError('First background BC event changed initialized history')
        write(out/'FIRST_BC_EVENT.json',dict(time=arm.start,side='right',BC_tick=int(round(arm.start/.001)),
            all_BC_history_max_change=first,target_before=initial['states'][target],target_after=post['states'][target]))
        arm.ack(arm.start);errors=None
        for j in range(1,61):
            index=start+j;end=p['clock_origin_seconds']+index*.0001
            for s in tick(arm,end,index):observe(s)
            if j==30:
                save_bundle(out/'checkpoint',kind='complete_model_checkpoint',
                    identity=dict(initialized_model=seed['joint_background_initialization']['initialized_model_id'],execution=arm.identity),
                    payload=arm.checkpoint_payload(monitor=monitor.payload()),
                    qualification=dict(default_adopted=False,fit_eligible=False,local_background_qualified=True))
                loaded=load_bundle(out/'checkpoint',expected_kind='complete_model_checkpoint')['payload']
                restored=H1H2GroupRunner.from_checkpoint(a,seed,p,loaded)
            elif j>30:
                tick(restored,end,index)
                errors=compare(snapshot_payload(arm,arm.snapshot(end,'right')),snapshot_payload(restored,restored.snapshot(end,'right')))
        final=arm.snapshot(p['stop_seconds'],'right');report=monitor.report(.006)
        write(out/'ABSOLUTE_MONITOR.json',report)
        save_bundle(out/'endpoint',kind='validation_evidence',identity=dict(initialized_model=seed['joint_background_initialization']['initialized_model_id']),
            payload=snapshot_payload(arm,final),qualification=dict(default_adopted=False,fit_eligible=False))
        write(out/'BACKGROUND_RUN.json',dict(duration_seconds=.006,observations=monitor.count,
            absolute_monitor_passed=report['passed'],maximum_norm=report['maximum_norm'],dominant=report['dominant'],
            checkpoint_continuation=errors,local_background_qualified=report['passed'],
            dynamic_numerical_qualification=False,space_closure_qualification=False,physiological_calibration=False,FIT_READY=False))
        if not report['passed']:raise ContractError('Another absolute defect remains; record and stop without a second audit')
        return report
    finally:
        arm.close()
        if restored is not None:restored.close()


def run_stimulus(seed,source,out):
    a,_=assembly_from_saved(seed);s=deepcopy(seed);p,start=protocol_for(s,source)
    first=int(np.floor((s['time_seconds']-p['input_origin_seconds'])*150.))
    lm=np.flatnonzero(a.types['cones']!='S');srows=np.flatnonzero(a.types['cones']=='S')
    for frame,v in s['native_inputs'].items():
        if frame>first:v[lm]*=1.15
    s['controlled_stimulus']=dict(kind='LM_selective_native_Td_15pct',first_changed_frame=first+1,
        S_native_input_held=True,background_initialization=seed['joint_background_initialization']['initialized_model_id'])
    bind_parameters(a,s,ROOT,p)
    arm=H1H2GroupRunner(a,s,p)
    try:
        initial=arm.snapshot(arm.start,'right',queue_node=True);arm.initial_commit();arm.ack(arm.start)
        data=[]
        for j in range(1,61):
            index=start+j;t=p['clock_origin_seconds']+index*.0001
            _,snap=tick(arm,t,index)
            data.append((t,snap['states']['HC',MECHANISM,'h2'].copy(),snap['ports']['cones','release'].copy()))
        h=np.array([v[1] for v in data]);r=np.array([v[2] for v in data]);dh=h-initial['states']['HC',MECHANISM,'h2']
        dr=r[:,srows]-initial['ports']['cones','release'][srows]
        hb=frozen_budget(initial['states']['HC',MECHANISM,'h2'],UNIT)[2]
        rb=frozen_budget(initial['ports']['cones','release'][srows],'dimensionless')[2]
        passed=np.min(np.mean(dh,axis=1))<0 and np.max(abs(dh)/hb)>1 and np.max(dr/rb)>1
        np.savez_compressed(out/'SHORT_H2_STIMULUS.npz',time=[v[0] for v in data],h2=h,cone_release=r,
            cone_types=a.types['cones'],H2_group_ids=a.populations['HC'].group_ids[a.h2_rows])
        result=dict(passed=bool(passed),duration_seconds=.006,input_step_fraction=.15,
            stimulus_unit='native Td diagnostic only',first_changed_frame=first+1,
            max_h2_change=float(np.max(abs(dh))),most_negative_mean_h2_change=float(np.min(np.mean(dh,axis=1))),
            max_S_release_increase=float(np.max(dr)),H2_budget_ratio=float(np.max(abs(dh)/hb)),
            S_release_budget_ratio=float(np.max(dr/rb)),
            dynamic_mechanism_observed=bool(passed),dynamic_numerical_qualification=False)
        write(out/'SHORT_H2_STIMULUS.json',result)
        if not passed:raise ContractError('Short H2 causal mechanism not resolved above the original numerical budget')
        return result
    finally:arm.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-check',action='store_true');args=parser.parse_args();out=args.output.resolve()
    if args.resume_check:
        seed=load_bundle(out/'seed',expected_kind='source_construction_seed')['payload']
        a,source=assembly_from_saved(seed);p,start=protocol_for(seed,source)
        payload=load_bundle(out/'checkpoint',expected_kind='complete_model_checkpoint')['payload']
        expected=load_bundle(out/'endpoint',expected_kind='validation_evidence')['payload']
        arm=H1H2GroupRunner.from_checkpoint(a,seed,p,payload)
        try:
            for index in range(start+31,start+61):tick(arm,p['clock_origin_seconds']+index*.0001,index)
            result=compare(snapshot_payload(arm,arm.snapshot(p['stop_seconds'],'right')),expected)
            write(out/'FRESH_PROCESS_RESTORE.json',dict(passed=True,comparison=result,initialization_recomputed=False))
            print('FRESH_PROCESS_RESTORE_PASS',flush=True)
        finally:arm.close()
        return
    out.mkdir(parents=True,exist_ok=False)
    try:
        outer=load_bundle(DELIVERY/'joint_seed',expected_kind='source_construction_seed')['payload']
        old_hash=canonical_hash(outer);a,source=assembly_from_saved(outer);p,start=protocol_for(outer,source)
        seed,bundle=initialize_joint_background(a,outer,ROOT,p)
        if canonical_hash(outer)!=old_hash:raise ContractError('Saved outer seed was mutated')
        if canonical_hash(seed['inner_parameters'])!=canonical_hash(outer['inner_parameters']):raise ContractError('Parameters changed')
        export_bundle(bundle,out/'parameters')
        old_manifest=read(DELIVERY/'parameters/manifest.json')
        with np.load(DELIVERY/'parameters/resolved_fp64.npz') as z:
            errors={name:float(np.max(abs(bundle.values[name]-z[old_manifest['array_keys'][name]]),initial=0.)) for name in bundle.values}
        if any(errors.values()):raise ContractError('Resolved physiological values changed')
        deltas={':'.join(k):dict(max_abs=float(np.max(abs(v-outer['states'][k]),initial=0.)),changed=bool(np.any(v!=outer['states'][k])))
            for k,v in seed['states'].items()}
        write(out/'INITIAL_STATE_DIFFERENCES.json',deltas)
        write(out/'INITIALIZATION_IDENTITY.json',seed['joint_background_initialization'])
        write(out/'PARAMETER_COMPARISON.json',dict(all_resolved_values_exact=True,array_count=len(errors),
            maximum_difference=0.,parameters_sha256=canonical_hash(seed['inner_parameters']),
            original_binding=old_manifest['binding_id'],new_context_binding=bundle.identity,
            reason='same physical values; constant-background input and newly sampled stationary prehistory have a new RunContext'))
        save_bundle(out/'seed',kind='source_construction_seed',identity=seed['joint_background_initialization'],payload=seed,
            qualification=dict(default_adopted=False,fit_eligible=False,local_background_qualified=False))
        print('NEW_INNER_BACKGROUND_CONSTRUCTED',flush=True)
        report=run_background(a,seed,p,start,out);print('CONSTANT_BACKGROUND_PASS',report['maximum_norm'],flush=True)
        stimulus=run_stimulus(seed,source,out);print('SHORT_H2_STIMULUS_PASS',flush=True)
        write(out/'RESULT.json',dict(status='LOCAL_JOINT_BACKGROUND_INITIALIZATION_QUALIFIED',
            initialized_model=INITIALIZATION,initialized_model_id=seed['joint_background_initialization']['initialized_model_id'],
            local_background_qualified=True,absolute_monitor_passed=True,absolute_maximum_norm=report['maximum_norm'],
            target_old_norm=77.27515150881214,target_new_norm=next(r['discrete_defect_norm'] for r in report['states'] if r['state']=='S_ON_BC:HistoryBipolarTerminal:history_0'),
            dynamic_mechanism_passed=stimulus['passed'],dynamic_numerical_qualified=False,
            infinite_space_closure_qualified=False,physiological_calibration=False,FIT_READY=False,
            parameters_fitted=False,outer_fixed_point_recomputed=False))
    except BaseException as exc:
        write(out/'FAILED.json',dict(error=str(exc),traceback=traceback.format_exc(),
            parameters_changed=False,monitor_changed=False,FIT_READY=False,stop=True))
        raise


if __name__=='__main__':main()
