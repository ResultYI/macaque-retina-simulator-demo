"""One existing50ms spatial native-light protocol through all integrated layers."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from tools.integrated_far_context_v1 import BASE,HC,ROOT,load,protocol,read,write
from tools.integrated_far_runner_v1 import IntegratedRunner
from tools.integrated_far_readouts_v1 import Readouts
from tools.run_external_first_frame_v1 import TypedDerivativeMonitor,segment_midpoint
from tools.validate_ac_vs_timescale import calendar
from tools.validate_ac_vs import dynamic_observe
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_time_monitor import GroupTimeMonitor
from retina_sim.ac_vs_monitor import DynamicDefectMonitor
from retina_sim.group_time_storage import save_bundle,load_bundle
from retina_sim.spatial_group_graph import digest


class WitnessMonitor(DynamicDefectMonitor):
    def observe(self,states,rhs,derivatives,**kwargs):
        super().observe(states,rhs,derivatives,**kwargs)
        if not self.report()['passed']:
            self.failure_derivatives={key:value.copy() for key,value in derivatives.items()}


def run(precision,guard,journal,memory,*,output_name=None,diagnostic_until=None):
    out=BASE/(output_name or precision);out.mkdir(exist_ok=False)
    assembly,seed=load();guard();p=protocol()
    previous=read(HC/'hc_response_validation.json');stimulus=previous['stimulus']
    if stimulus['duration_s']!=.05 or stimulus['contrast']!=.1:raise ContractError('Original short spatial protocol changed')
    center=np.asarray(previous['center_deg'])
    mask=np.linalg.norm(assembly.populations['cones'].centroid_mm/.223-center,axis=1)<=stimulus['radius_deg']
    if int(mask.sum())!=stimulus['driven_cone_groups']:raise ContractError('Original spatial mask differs')
    background=seed['input_reference']['native_background_td'];target=background.copy();target[mask]*=1.1
    seed['native_inputs']={i:target.copy() for i in p['input_frame_indices']}
    rod=seed['input_reference']['rod_background_rstar_s']
    seed['rod_inputs']={i:np.full(assembly.populations['rods'].group_count,rod,dtype=np.float64) for i in p['input_frame_indices']}
    input_descriptor=dict(source=str((HC/'hc_response_validation.json').relative_to(ROOT)),source_sha256=digest(HC/'hc_response_validation.json'),
        exact_previous_duration_and_mask=True,mask_group_ids=assembly.populations['cones'].group_ids[mask],
        target_input=target,constant_native_holds='Same physical target over8 consumer holds; no R/E or neural reset',
        scope='Optical-equivalent engineering light step from frozen S0; not calibrated external visual contrast')
    write(out/'INPUT.json',input_descriptor)
    arm=None;records=[];times=[];sides=[];roundtrip=False
    try:
        arm=IntegratedRunner(assembly,seed,p,precision=precision,guard=guard)
        physical=GroupTimeMonitor(assembly,seed['states'],parameters=seed['inner_parameters'])
        defects=WitnessMonitor(assembly,seed['states']);mapped=TypedDerivativeMonitor(arm,defects)
        reader=Readouts(arm)
        populations={name:dict(group_ids=pop.group_ids,represented_members=pop.counts,
            coordinate_mm=getattr(pop,'centroid_mm',None),core_rows=assembly.core_population_rows.get(name,np.array([],dtype=int)))
            for name,pop in assembly.populations.items()}
        write(out/'IDENTITY.json',dict(schema='single-far-integrated-layer-response-v1',runner=arm.identity,
            HC_configuration=assembly.lateral_configuration,background_seed_sha256=canonical_hash(seed),
            numerical_profile=arm.numeric_profile,input_protocol=p,stimulus=input_descriptor,populations=populations,
            units='Native HC feedback states have no mV calibration; BC/AC/AII/RGC SI units retained',
            group_objects_not_independent_fine_cells=True,FIT_READY=False,background_qualified=False,production_adopted=False))
        def observe(t,side,record=False):
            snapshot=arm.snapshot(t,side,queue_node=True)
            physical.observe(t,side,snapshot['states'],snapshot['rhs'],transitions=snapshot['transitions'],
                ports=snapshot['ports'],aux=snapshot['aux'],event_generation=snapshot['event_generation'])
            if physical.domain_failures:raise ContractError('Original physical/history gate failed')
            if record:
                records.append(reader.observe(t,snapshot));times.append(float(t));sides.append(side)
            return snapshot
        def packet():
            return dict(time_seconds=np.asarray(times),side=np.asarray(sides),
                **{key:np.stack([v[key] for v in records],axis=1) for key in records[0]})
        observe(arm.start,'right');arm.initial_commit();observe(arm.start,'right',True);arm.ack(arm.start)
        for index,(end,flags) in enumerate(calendar(p)):
            if end<=arm.inner_time:continue
            guard();begin=arm.inner_time;arm.ensure_native(end,'left');advance=arm.advance_inner(begin,end)
            for curve in advance.dense_segments:
                mid=segment_midpoint(curve)
                witness=dynamic_observe(arm,advance,mid,mapped,'left' if mid==advance.t1 else 'right')
                if not defects.report()['passed']:
                    report=defects.report()
                    write(out/'FAILED_DYNAMIC_DEFECT.json',dict(interval=[begin,end],report=report))
                    save_bundle(out/'failed_numerical_witness',kind='validation_evidence',identity=dict(runner=arm.identity),
                        payload=dict(states=witness['states'],rhs=witness['rhs'],derivatives=defects.failure_derivatives,time_seconds=mid,defects=report),
                        qualification=dict(default_adopted=False,fit_eligible=False))
                    raise ContractError('Original dynamic numerical defect budget failed')
            endpoint=end==arm.stop;side='left' if endpoint or flags['frame'] else 'right'
            observe(end,side,flags['frame'])
            if flags['rod'] or flags['bc']:
                arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],
                    rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
            observe(end,'left' if endpoint else 'right',flags['record'] or flags['frame'] or endpoint);arm.ack(end)
            if diagnostic_until is not None and end>=diagnostic_until:
                write(out/'DIAGNOSTIC_WINDOW.json',dict(reached_seconds=end,full_protocol_completed=False,defects=defects.report()))
                return
            if abs(end/.005-round(end/.005))<1e-9:
                label=f'{round(end*1000):03d}ms';prefix=out/('RESPONSE_THROUGH_'+label+'.npz')
                np.savez_compressed(prefix,**packet())
                payload=arm.checkpoint_payload(recording=dict(curve_prefix=prefix.name,sha256=digest(prefix),records=len(records)))
                save_bundle(out/('CHECKPOINT_'+label),kind='validation_evidence',identity=dict(runner=arm.identity),payload=payload,
                    qualification=dict(default_adopted=False,fit_eligible=False,FIT_READY=False,background_qualified=False,
                        scope='Complete integrated research continuation with native frontier/queues/history/PCG; not production checkpoint'))
                if not roundtrip:
                    saved=load_bundle(out/('CHECKPOINT_'+label),expected_kind='validation_evidence')['payload']
                    restored=IntegratedRunner.from_checkpoint(assembly,seed,p,saved,guard=guard)
                    try:
                        copied=restored.checkpoint_payload(recording=saved['recording'])
                        for category in ('native','inner','spike_states','clocks','input_position','observation_delay','readout_cursor','integrated_research_contract'):
                            if canonical_hash(payload[category])!=canonical_hash(copied[category]):raise ContractError('Complete continuation roundtrip differs: '+category)
                    except BaseException:
                        restored.close();raise
                    if precision=='tight':
                        old=arm;arm=restored;old.close()
                        mapped=TypedDerivativeMonitor(arm,defects);reader=Readouts(arm)
                    else:restored.close()
                    roundtrip=True;write(out/'CHECKPOINT_ROUNDTRIP.json',dict(passed=True,time_seconds=end,
                        no_extra_forward_replay=True,continued_planned_trajectory_from_restored_checkpoint=precision=='tight',
                        all_native_frontier_and_PC_G_history_preserved=True))
                journal.heartbeat(force=True,phase=precision,accepted_seconds=end,memory=memory.last)
                print(f'{precision}: {end*1000:.1f}/50ms',flush=True)
        np.savez_compressed(out/'RESPONSE.npz',**packet())
        write(out/'FIELD_CONTRACT.json',reader.fields)
        events=[dict(population=pop,**event) for pop,values in arm.event_log.items() for event in values if event['time_seconds']<arm.stop]
        write(out/'GROUP_OBJECT_EVENTS.json',events)
        write(out/'END_HISTORY.json',deepcopy(arm.spike_states))
        write(out/'RESULT.json',dict(status='PASS',reached_seconds=arm.inner_time,records=len(records),
            full_research_checkpoint_roundtrip=roundtrip,physical_gate_passed=not physical.domain_failures,
            dynamic_defect=defects.report(),timings=arm.metrics,events_published_strictly_before_stop=len(events),
            endpoint_original_commit_preserved=True,post_stop_pending_events_in_checkpoint_only=True,
            no_new_physiology_or_stimulus=True,FIT_READY=False,background_qualified=False,production_adopted=False))
    finally:
        if records and not (out/'RESPONSE.npz').exists():np.savez_compressed(out/'PARTIAL_RESPONSE.npz',**packet())
        if arm is not None:arm.close()
