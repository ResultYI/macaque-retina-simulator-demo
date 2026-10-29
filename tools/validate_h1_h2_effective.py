"""Typed actual-local H1/H2 validation; no fitting or long/full-retina advance."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.integrate import solve_ivp

from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.parameter_system import data
from retina_sim.parameter_migration import export_bundle
from retina_sim.group_time_storage import save_bundle,load_bundle
from retina_sim.group_time_monitor import GroupTimeMonitor,frozen_budget,rhs_allowance
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.spatial_group_model import BC
from retina_sim.group_inner_dynamics import BC_MECH
from retina_sim.h1_h2_effective import (MODEL,LEGACY,MECHANISM,QUALIFICATION,UNIT,
    H1H2Assembly,H1H2GroupRunner,H1H2Native,verify_typed_source,bind_parameters,save_artifact,matrix_id)
from tools.validate_ac_vs_native_support import load


def write(path,value):
    Path(path).write_text(json.dumps(data(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def say(stage,**values): print(json.dumps(dict(stage=stage,**values)),flush=True)


def diagnose(assembly,seed,label,rows,output):
    """Only diagnostic cone-native excitation changes; formal gray calibration stays frozen."""
    native=H1H2Native(assembly);native.boundary=deepcopy(seed['h1_h2_model']['boundary'])
    t0=seed['time_seconds'];pulse=t0+.010;stop=t0+.150
    bg=assembly.global_light[assembly.native_rows['cones']].copy()
    stepped=bg.copy();stepped[rows]*=1.15
    y,anchor=native.construction_point(type('Seed',(),dict(states=seed['states'],time_seconds=t0))(),stepped,pulse,0)
    first=np.linspace(t0,pulse,11)
    solution=solve_ivp(lambda t,x:native.rhs(x,t,anchor),(t0,pulse),y,
        method='DOP853',rtol=1e-6,atol=1e-9,max_step=.0001,t_eval=first)
    if not solution.success:raise ContractError('Controlled native step failed: '+solution.message)
    R,E=anchor.at(pulse)
    recovery=AnalyticREAnchor(pulse,R,E,bg,stop,1)
    after=np.linspace(pulse,stop,141)
    second=solve_ivp(lambda t,x:native.rhs(x,t,recovery),(pulse,stop),solution.y[:,-1],
        method='DOP853',rtol=1e-6,atol=1e-9,max_step=.0001,t_eval=after)
    if not second.success:raise ContractError('Controlled native recovery failed: '+second.message)
    times=np.r_[solution.t,second.t[1:]]
    packed=np.column_stack((solution.y,second.y[:,1:]))
    h=packed[native.slices['Vh']]
    releases=[]
    for i,t in enumerate(times):
        active=anchor if t<=pulse else recovery
        ev=native.evaluate(packed[:,i],t,active)
        if not np.isfinite(ev.rhs).all():raise ContractError('Nonfinite controlled RHS')
        releases.append(ev.outputs['release'])
    release=np.asarray(releases).T
    cone=assembly.types['cones']; srows=np.flatnonzero(cone=='S')
    h1,h2=h[assembly.h1_rows],h[assembly.h2_rows]
    d1=h1-h1[:,[0]];d2=h2-h2[:,[0]];sr=release[srows]-release[srows,[0]][:,None]
    # Signal existence is judged against the original frozen state budget.
    b1=frozen_budget(h1[:,0],UNIT)[2];b2=frozen_budget(h2[:,0],UNIT)[2]
    br=frozen_budget(release[srows,0],'dimensionless')[2]
    peak=int(np.argmax(np.max(np.abs(d2),axis=0)))
    result=dict(label=label,diagnostic_only=True,input_unit='td',step_fraction=.15,
        selected_cone_groups=len(rows),duration_seconds=.150,pulse_seconds=.010,
        max_h1_change=float(np.max(np.abs(d1))),max_h2_change=float(np.max(np.abs(d2))),
        h1_peak_budget_ratio=float(np.max(np.abs(d1)/b1[:,None])),
        h2_peak_budget_ratio=float(np.max(np.abs(d2)/b2[:,None])),
        h2_mean_change_at_peak=float(np.mean(d2[:,peak])),
        S_release_peak_increase=float(np.max(sr)),S_release_peak_decrease=float(np.min(sr)),
        S_release_budget_ratio=float(np.max(np.abs(sr)/br[:,None])),
        recovery_peak_index=peak,recovery_final_norm=float(np.linalg.norm(d2[:,-1])),
        peak_norm=float(np.max(np.linalg.norm(d2,axis=0))),
        recovery_moves_toward_background=bool(np.linalg.norm(d2[:,-1]) < np.max(np.linalg.norm(d2,axis=0))),
        finite_domain=True,nfev=solution.nfev+second.nfev)
    np.savez_compressed(output/(label+'.npz'),time_seconds=times,h1=h1,h2=h2,cone_release=release,
        cone_Vis=packed[native.slices['Vis']],cone_Vbar=packed[native.slices['Vbar']],
        H1_group_ids=assembly.populations['HC'].group_ids[assembly.h1_rows],
        H2_group_ids=assembly.populations['HC'].group_ids[assembly.h2_rows],
        cone_group_ids=assembly.populations['cones'].group_ids,cone_types=cone,
        native_state_unit=np.array(UNIT),meaning=np.array('effective feedback state; not physical membrane mV'))
    say('CONTROLLED_'+label,max_h1=result['max_h1_change'],max_h2=result['max_h2_change'])
    return result


def tick(arm,end,index):
    arm.ensure_native(end,'left');arm.advance_inner(arm.inner_time,end)
    left=arm.snapshot(end,'left',queue_node=True)
    event=index%10==0
    arm.commit(end,rod=True,bc=event,rgc=event,
        rod_sample_time=arm.protocol['clock_origin_seconds']+(index-1)*.0001+1e-13)
    right=arm.snapshot(end,'right',queue_node=True);arm.ack(end)
    if event:
        for pop in BC:
            route=next(r for r in arm.inner.routes if r.pre=='cones' and r.post==pop)
            expected=arm.inner.mean[route.name]@left['ports']['cones','release']
            if not np.array_equal(expected,left['ports'][pop,'drive']):
                raise ContractError('Native H1/H2 release was not received by the actual BC drive')
            history=right['states'][pop,BC_MECH,'history_0']
            if not np.array_equal(history,expected-arm.inner.par(pop,'baseline')):
                raise ContractError('BC event did not consume the actual new cone release')
    return left,right


def actual_runner(assembly,seed,protocol,output):
    protocol=deepcopy(protocol)
    origin=protocol['clock_origin_seconds']
    start_tick=int(round((seed['time_seconds']-origin)/.0001))
    if seed['time_seconds'] != origin+start_tick*.0001:
        raise ContractError('Current source seed does not lie on its original rod clock')
    protocol['stop_seconds']=origin+(start_tick+60)*.0001
    arm=H1H2GroupRunner(assembly,seed,protocol)
    restored=None
    try:
        initial=arm.snapshot(arm.start,'right',queue_node=True)
        monitor=GroupTimeMonitor(assembly,initial['states'],parameters=seed['inner_parameters'])
        def observe(s):
            monitor.observe(s['time_seconds'],s['side'],s['states'],s['rhs'],transitions=s['transitions'],
                ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
            if monitor.domain_failures:raise ContractError('Local monitor domain failed: '+str(monitor.domain_failures))
        observe(initial);arm.initial_commit();observe(arm.snapshot(arm.start,'right',queue_node=True));arm.ack(arm.start)
        traces=[];restored_errors=[]
        for relative_index in range(1,61):
            index=start_tick+relative_index
            end=origin+index*.0001
            for s in tick(arm,end,index):observe(s)
            current=arm.snapshot(end,'right')
            traces.append((end,current['states']['HC',MECHANISM,'h1'],current['states']['HC',MECHANISM,'h2'],
                           current['ports']['cones','release']))
            if relative_index==30:
                identity=dict(model=MODEL,artifact=assembly.artifact_sha256,execution=arm.identity)
                save_bundle(output/'checkpoint',kind='complete_model_checkpoint',identity=identity,
                    payload=arm.checkpoint_payload(monitor=monitor.payload()),
                    qualification=dict(default_adopted=False,fit_eligible=False,physiological_calibration=False))
                loaded=load_bundle(output/'checkpoint',expected_kind='complete_model_checkpoint',expected_identity=identity)
                restored=H1H2GroupRunner.from_checkpoint(assembly,seed,protocol,loaded['payload'])
                bad=deepcopy(loaded['payload']);bad['controller']['hc_model']['model']=LEGACY
                try:H1H2GroupRunner.from_checkpoint(assembly,seed,protocol,bad)
                except ContractError:pass
                else:raise ContractError('Legacy/new checkpoint identity was accepted')
            elif relative_index>30:
                tick(restored,end,index)
                other=restored.snapshot(end,'right')
                for category in ('states','rhs','ports'):
                    if set(current[category])!=set(other[category]):raise ContractError('Restore changed registry')
                    for key,v in current[category].items():
                        err=float(np.max(np.abs(v-other[category][key]),initial=0.))
                        restored_errors.append(err)
                        if err:raise ContractError('Restored continuation differs at '+str(key))
                if canonical_hash(arm.spike_states)!=canonical_hash(restored.spike_states):
                    raise ContractError('Checkpoint changed sampled RNG/event/history state')
        report=monitor.report(.006);write(output/'monitor.json',report)
        np.savez_compressed(output/'actual_runner_6ms.npz',time_seconds=np.array([r[0] for r in traces]),
            h1=np.array([r[1] for r in traces]),h2=np.array([r[2] for r in traces]),
            cone_release=np.array([r[3] for r in traces]))
        return dict(duration_seconds=.006,native_max_step_seconds=.0001,BC_RGC_period_seconds=.001,
            actual_group_time_runner=True,start_seconds=arm.start,clock_origin_seconds=origin,
            checkpoint_at_seconds=origin+(start_tick+30)*.0001,checkpoint_relative_seconds=.003,
            restored_continuation_max_error=max(restored_errors),checkpoint_wrong_model_rejected=True,
            H2_in_actual_RHS=True,independent_h1_h2_states=True,
            finite_domain=not report['domain_failures'],monitor_missing_scales=report['missing_scales'],
            absolute_monitor_passed=report['passed'],absolute_monitor_not_promoted=True,
            cone_release_received_by_BC=True,clock_indices=arm.clock_indices,
            rod_RBC_AII_code_parameters_routes_unchanged=True)
    finally:
        arm.close()
        if restored is not None:restored.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    try:
        parent,old_seed,source=load(ROOT/'reports/ac_vs_native_support_v1/prepared_v1')
        typed,evidence=verify_typed_source(parent.base,ROOT);write(out/'TYPE_PROVENANCE.json',evidence)
        model=H1H2Assembly(parent,parent.base,typed,evidence,engineering=True)
        save_artifact(model,typed,out/'typed_artifact');say('TYPED_ARTIFACT_COMPLETE')
        seed=model.seed(old_seed);write(out/'JOINT_EQUILIBRIUM.json',seed['h1_h2_model'])
        bundle=bind_parameters(model,seed,ROOT,source['protocol']);export_bundle(bundle,out/'parameters')
        save_bundle(out/'joint_seed',kind='source_construction_seed',identity=dict(model=MODEL,artifact=model.artifact_sha256,
            parameter_binding_id=bundle.identity,source_sha256=canonical_hash(model.connection_identity)),payload=seed,
            qualification=dict(default_adopted=False,fit_eligible=False,qualified_background=False,
                qualified_outer_background=True,physiological_calibration=False,FIT_READY=False))
        say('JOINT_EQUILIBRIUM',residual=seed['h1_h2_model']['full_parent_equilibrium']['residual'])
        # True initial state/RHS residual check against existing unit-specific budgets.
        native=H1H2Native(model);native.boundary=model.native_boundary
        bg=model.global_light[model.native_rows['cones']]
        y,anchor=native.construction_point(type('Seed',(),dict(states=seed['states'],time_seconds=seed['time_seconds']))(),
            bg,seed['time_seconds']+.2,0)
        f=native.rhs(y,seed['time_seconds'],anchor);allowances=[]
        units=['vH_native_precursor','vH_native_current',UNIT,'vH_native_conductance',UNIT,UNIT,UNIT,UNIT,UNIT]
        for name,unit in zip(native.slices,units):
            initial=y[native.slices[name]];scale,_,budget=frozen_budget(initial,unit)
            allowances.extend(rhs_allowance(scale,budget,np.zeros_like(initial),f[native.slices[name]]))
        ratio=float(np.max(np.abs(f)/np.asarray(allowances)))
        if ratio>1:raise ContractError('Joint initial RHS exceeds original numerical allowance: '+str(ratio))
        diagnostics={}
        for label in ('S','L','M','LM'):
            rows=np.flatnonzero(np.isin(model.types['cones'],list(label)))
            if not len(rows):raise ContractError('No actual selective '+label+' cone groups')
            diagnostics[label]=diagnose(model,seed,label,rows,out)
        ablation=H1H2Assembly(parent,parent.base,typed,evidence,engineering=True,feedback_scale=0.)
        ablated_seed=ablation.seed(old_seed);bind_parameters(ablation,ablated_seed,ROOT,source['protocol'])
        diagnostics['LM_H2_feedback_off']=diagnose(ablation,ablated_seed,'LM_H2_feedback_off',
            np.flatnonzero(model.types['cones']!='S'),out)
        checks=dict(S_drives_H2=diagnostics['S']['h2_peak_budget_ratio']>1,
            S_does_not_drive_H1=diagnostics['S']['h1_peak_budget_ratio']<=1,
            LM_drive_H1_and_H2=all(diagnostics[k]['h1_peak_budget_ratio']>1 and diagnostics[k]['h2_peak_budget_ratio']>1 for k in ('L','M')),
            H2_inputs_same_sign=all(diagnostics[k]['h2_mean_change_at_peak']<0 for k in ('S','L','M')),
            LM_surround_increases_fixed_S_release=diagnostics['LM']['S_release_peak_increase']>0 and diagnostics['LM']['S_release_budget_ratio']>1,
            H2_ablation_removes_LM_to_S=diagnostics['LM_H2_feedback_off']['S_release_budget_ratio']<=1,
            recovery=all(v['recovery_moves_toward_background'] for v in diagnostics.values()))
        write(out/'CONTROLLED_FUNCTION.json',dict(checks=checks,diagnostics=diagnostics,
            tolerances='existing frozen_budget for signal existence; directional predictions have no fitted threshold'))
        if not all(checks.values()):raise ContractError('Controlled H2 mechanism acceptance failed: '+str(checks))
        say('CONTROLLED_MECHANISM_PASS')
        actual=actual_runner(model,seed,source['protocol'],out);write(out/'ACTUAL_RUNNER.json',actual)
        result=dict(status='ENGINEERING_H1_H2_IMPLEMENTED_SHORT_LOCAL_FUNCTION_PASS',
            H2_structure_integrated=True,H2_short_function_pass=True,H2_physiologically_calibrated=False,
            eligible_for_next_full_layer_parameter_fit=False,FIT_READY=False,absolute_monitor=False,
            Cell6='INCONCLUSIVE; historical failures retained',models=dict(legacy=LEGACY,new=MODEL),
            new_artifact=model.artifact_sha256,parameter_binding_id=bundle.identity,
            parameters=dict(records=len(bundle.registry.records),arrays=len(bundle.values),qualification=QUALIFICATION),
            local_counts=dict(cones=len(model.types['cones']),H1=len(model.h1_rows),H2=len(model.h2_rows)),
            joint_equilibrium=seed['h1_h2_model']['full_parent_equilibrium'],initial_RHS_allowance_ratio=ratio,
            controlled_checks=checks,actual_runner=actual,wall_seconds=time.perf_counter()-began,
            full_retina_dynamic_advances=0,fit_runs=0,default_changed=False,
            limitations=['H2 spectral input weights retain source associations; no calibrated S/LM gain',
                'H2 base tau and effective scales borrowed across types, not measured H2 physiology',
                'H2->LM feedback disabled; H2->S enabled',
                'Finite local support with held joint-background exterior; recurrent full-retina closure not certified',
                'CPU joint reference only; structured Radau, forced waveform and CUDA H2 are explicitly unsupported'])
        write(out/'RESULT.json',result);say('COMPLETE',status=result['status'],wall_seconds=result['wall_seconds'])
    except BaseException as exc:
        write(out/'FAILED.json',dict(status='FAILED_NO_QUALIFICATION_UPGRADE',error=str(exc),
            traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-began,parameters_tuned=False,
            DC_offset_added=False,implicit_clipping_applied=False,FIT_READY=False))
        raise


if __name__=='__main__':main()
