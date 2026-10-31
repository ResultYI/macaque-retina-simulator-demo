"""One existing external first-frame exposure through true Cone/HC/inner dynamics."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys,time,json,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cross_position_context_v1 import load_background,read,write
from tools.validate_ac_vs_timescale import calendar
from tools.validate_ac_vs import dynamic_observe
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_time_monitor import GroupTimeMonitor
from retina_sim.ac_vs_monitor import DynamicDefectMonitor
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.hc_lateral_interface import LateralHCResearchRunner
from retina_sim.h1_h2_effective import MECHANISM
from retina_sim.group_time_storage import save_bundle
from retina_sim.rgc_functional_v1 import RGCFunctionalV1
from retina_sim.spatial_group_model import RGC
from retina_sim.spatial_group_graph import digest
from retina_sim.continuous_state import Dop853Dense
from tools.durable_research_runtime_v1 import DurableRun


def segment_midpoint(curve: Dop853Dense) -> float:
    """ACVSInner returns owned typed curves, rather than SciPy solver objects."""
    return curve.t0+(curve.t1-curve.t0)/2


class TypedDerivativeMonitor:
    """The same native carrier-to-H1/H2 mapping used by the existing snapshot."""
    def __init__(self,arm,monitor):self.arm,self.monitor=arm,monitor
    def observe(self,states,rhs,derivatives,**kwargs):
        for name in ('w','z2'):derivatives['HC',MECHANISM,name]=derivatives.pop(('HC','reduced_native',name))
        carrier=derivatives.pop(('HC','reduced_native','Vh'))
        derivatives['HC',MECHANISM,'h1']=carrier[self.arm.assembly.h1_rows]
        derivatives['HC',MECHANISM,'h2']=carrier[self.arm.assembly.h2_rows]
        self.monitor.observe(states,rhs,derivatives,**kwargs)


def run(prepared,position,out,*,max_wall_seconds=820):
    out.mkdir(exist_ok=False);started=time.perf_counter();memory=MemoryGuard(24*1024**3,8*1024**3)
    journal=DurableRun(out/'runtime_model','neural-'+position)
    status='FAILED';arm=None;records=[];times=[];sides=[];events=[];phase='load'
    continuous=False;chunk_start=0;chunks=[]
    def guard():
        if time.perf_counter()-started>max_wall_seconds:raise TimeoutError('Bounded process wall-time guard')
        if not memory.check():raise MemoryError(str(memory.stop_reason))
        metrics={} if arm is None else arm.metrics
        journal.heartbeat(status='RUNNING',phase=phase,accepted_seconds=None if arm is None else arm.inner_time,
            memory=memory.last,native_integration_seconds=metrics.get('native_wall_seconds',0.),
            inner_integration_seconds=metrics.get('inner_wall_seconds',0.))
    def save_records():
        if records:
            if continuous:
                storage=list(chunks)
                if chunk_start<len(records):
                    part=records[chunk_start:];path=out/'RESPONSE_PARTIAL.npz'
                    np.savez_compressed(path,time_seconds=np.asarray(times[chunk_start:]),side=np.asarray(sides[chunk_start:]),
                        **{key:np.stack([r[key] for r in part],axis=1) for key in part[0]})
                    storage.append(dict(file=path.name,records=len(part),sha256=digest(path)))
                (out/'RESPONSE_INDEX.json').write_text(json.dumps(dict(chunks=storage,records=len(records),
                    concatenation_axis='time:0 for time_seconds/side;1 for port arrays',no_duplicate_full_copy=True),indent=2),encoding='utf8')
            else:
                np.savez_compressed(out/'RESPONSE.npz',time_seconds=np.asarray(times),side=np.asarray(sides),
                    **{key:np.stack([r[key] for r in records],axis=1) for key in records[0]})
    try:
        guard();prep=read(prepared/'PREPARATION.json');binding=read(prepared/position/'BINDING.json')
        for relative,expected in prep['source_pins'].items():
            if digest(ROOT/relative)!=expected:raise ContractError('Bound external fixture source changed: '+relative)
        if digest(prepared/position/'DRIVE.npz')!=binding['drive_sha256']:raise ContractError('External input drive changed')
        a,seed,source=load_background(Path(binding['prepared']),Path(binding['background']));guard()
        seed=deepcopy(seed);protocol=prep['protocol'];frames=protocol['input_frame_indices']
        with np.load(prepared/position/'DRIVE.npz',allow_pickle=False) as z:drive=z['group_native_drive'].copy()
        if drive.ndim==1:
            seed['native_inputs']={index:drive.copy() for index in frames}
        elif drive.ndim==2 and drive.shape==(120,a.populations['cones'].group_count) and frames==list(range(600)):
            continuous=True
            seed['native_inputs']={index:drive[index//5].copy() for index in frames}
        else:raise ContractError('Only existing one-exposure or frozen120-frame input is supported')
        rod=float(seed['input_reference']['rod_background_rstar_s'])
        seed['rod_inputs']={index:np.full(a.populations['rods'].group_count,rod) for index in frames}
        phase='construct_true_native_runner';arm=LateralHCResearchRunner(a,seed,protocol,guard=guard)
        physical=GroupTimeMonitor(a,seed['states'],parameters=seed['inner_parameters'])
        dynamic=DynamicDefectMonitor(a,seed['states']);mapped=TypedDerivativeMonitor(arm,dynamic)
        readout=object.__new__(RGCFunctionalV1);readout.model=arm.inner
        readout.populations=tuple(p for p in RGC if arm.inner.counts[p]);readout.fields=readout._fields()
        types={p:dict(group_ids=a.populations[p].group_ids.tolist(),groups=a.populations[p].group_count,
            represented_members=a.populations[p].counts.tolist()) for p in readout.populations}
        identity=dict(experiment=prep['experiment_id'],position=position,runner=arm.identity,
            initialized_model=source['initialized_model_id'],protocol=protocol,fields=readout.fields,types=types,
            true_Cone_HC_dynamics=True,internal_release_clamped=False,input_fixture=prep['fixture'],
            continuous_IVP_no_per_frame_reset=True,connector_source_sha256=digest(Path(__file__)),
            terminal_sample='Optical input left side; original endpoint discrete clock commits retained; published events strictly before stop',
            full_checkpoint_supported=False,FIT_READY=False,background_qualified=False,production_adopted=False)
        write(out/'IDENTITY.json',identity)
        def observe(t,side,record=False):
            snapshot=arm.snapshot(t,side,queue_node=True)
            physical.observe(t,side,snapshot['states'],snapshot['rhs'],transitions=snapshot['transitions'],
                ports=snapshot['ports'],aux=snapshot['aux'],event_generation=snapshot['event_generation'])
            if physical.domain_failures:raise ContractError('Original physical/history/conservation gate failed')
            if record:
                values=readout.observe_ports(snapshot['inner_outputs'])
                for key,value in snapshot['ports'].items():
                    if key[0] not in RGC:values['ports__'+':'.join(key)]=np.asarray(value).copy()
                records.append(values);times.append(float(t));sides.append(side)
            return snapshot
        initial=observe(arm.start,'right');arm.initial_commit();observe(arm.start,'right',True);arm.ack(arm.start)
        phase='advance_existing_exposure';edge_checks=[];next_chunk=1.
        for index,(end,flags) in enumerate(calendar(protocol)):
            if end<=arm.inner_time:continue
            guard();begin=arm.inner_time;arm.ensure_native(end,'left');advance=arm.advance_inner(begin,end)
            for curve in advance.dense_segments:
                mid=segment_midpoint(curve)
                witness=dynamic_observe(arm,advance,mid,mapped,'left' if mid==advance.t1 else 'right')
                if not dynamic.report()['passed']:
                    save_bundle(out/'failed_numerical_witness',kind='validation_evidence',identity=identity,
                        payload=dict(states=witness['states'],rhs=witness['rhs'],time_seconds=mid,defects=dynamic.report()),
                        qualification=dict(default_adopted=False,fit_eligible=False))
                    raise ContractError('Original dynamic numerical defect budget failed')
            endpoint=end==arm.stop;side='left' if endpoint or flags['frame'] else 'right'
            before=observe(end,side,flags['frame'])
            if flags['rod'] or flags['bc']:
                arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],
                    rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
            last=observe(end,'left' if endpoint else 'right',flags['record'] or flags['frame'] or endpoint)
            if flags['frame']:
                jump=max(float(np.max(abs(last['native_outputs'][k]-before['native_outputs'][k]),initial=0)) for k in ['X','C','Vis','g_i','Vbar','w','z2','Vh','z1_aux'])
                edge_checks.append(dict(time_seconds=end,maximum_continuous_native_jump=jump))
            arm.ack(end)
            if index%500==0:
                progress=dict(position=position,accepted_seconds=end,wall_seconds=time.perf_counter()-started,memory=memory.last)
                (out/'PROGRESS.json').write_text(json.dumps(progress,indent=2),encoding='utf8');print(json.dumps(progress),flush=True)
            if end>=next_chunk:
                part=records[chunk_start:]
                path=out/('RESPONSE_THROUGH_%04dms.npz'%round(end*1000))
                np.savez_compressed(path,
                    time_seconds=np.asarray(times[chunk_start:]),side=np.asarray(sides[chunk_start:]),
                    **{key:np.stack([r[key] for r in part],axis=1) for key in part[0]})
                chunks.append(dict(file=path.name,records=len(part),sha256=digest(path),
                    first_sample_seconds=float(times[chunk_start]),last_sample_seconds=float(times[-1])))
                chunk_start=len(records);next_chunk+=1.
        phase='write_outputs';write_started=time.perf_counter()
        journal.heartbeat(force=True,phase=phase,accepted_seconds=arm.inner_time)
        save_records();write(out/'PORT_UNITS.json',{':'.join(k):v for k,v in last['port_units'].items()})
        for pop in readout.populations:
            events.extend(dict(type=pop,**event) for event in arm.event_log[pop] if event['time_seconds']<arm.stop)
        write(out/'GROUP_OBJECT_EVENTS.json',events)
        save_bundle(out/'STATE_END',kind='validation_evidence',identity=identity,
            payload=dict(states=last['states'],spike_states=deepcopy(arm.spike_states),clocks=dict(arm.clock_indices),
                time_seconds=arm.inner_time,scope='End of existing optical exposure; not complete-resume checkpoint'),
            qualification=dict(default_adopted=False,fit_eligible=False,FIT_READY=False))
        response={p:dict(maximum_voltage_change_mV=float(np.max(abs(records[-1][p+'__voltage']-records[0][p+'__voltage']))*1000),
            maximum_source_rate_change_Hz=float(np.max(abs(records[-1][p+'__expected_binary_count_rate']-records[0][p+'__expected_binary_count_rate'])))) for p in readout.populations}
        ports={key:float(np.max(abs(records[-1][key]-records[0][key]),initial=0)) for key in records[0] if key.startswith('ports__')}
        status='PASS'
        stage='EXISTING_EXTERNAL_FIRST_EXPOSURE_END_TO_END_COMPLETE' if arm.stop==1/30 else 'EXISTING_EXTERNAL_FROZEN_4S_VIDEO_END_TO_END_COMPLETE'
        write(out/'RESULT.json',dict(status=stage,position=position,
            reached_seconds=arm.inner_time,duration_seconds=arm.stop-arm.start,recorded_nodes=len(records),
            physical_history_conservation_passed=not physical.domain_failures,dynamic_numerical_defect=dynamic.report(),
            input_hold_boundary_checks=edge_checks,published_group_object_events=len(events),
            actual_RGC_response=response,maximum_end_change_by_port=ports,clocks=arm.clock_indices,
            no_extra_exposure_or_partial_bin_events_added=True,single_reference_only=True,
            independent_fine_cell_spikes_claimed=False,controlled_optical_contrast_calibration=False,
            FIT_READY=False,background_qualified=False,production_adopted=False))
        print(json.dumps(dict(position=position,status=status,reached_seconds=arm.inner_time)),flush=True)
        output_write_seconds=time.perf_counter()-write_started;phase='complete'
    except BaseException as exc:
        save_records();write(out/'FAILURE.json',dict(phase=phase,error=str(exc),traceback=traceback.format_exc(),
            time_seconds=None if arm is None else arm.inner_time,parameters_or_tolerances_changed=False))
        raise
    finally:
        propagating=sys.exc_info()[0] is not None
        if propagating:status='FAILED'
        metrics={} if arm is None else arm.metrics
        close_error=None
        if arm is not None:
            try:arm.close()
            except BaseException as exc:close_error=exc;status='FAILED'
        wall=time.perf_counter()-started
        timing=dict(native_neural_integration_seconds=metrics.get('native_wall_seconds',0.),
            inner_neural_integration_seconds=metrics.get('inner_wall_seconds',0.),
            runner_build_seconds=metrics.get('build_seconds',0.),
            final_output_seconds=locals().get('output_write_seconds'),
            other_load_monitor_recording_seconds=wall-metrics.get('native_wall_seconds',0.)-metrics.get('inner_wall_seconds',0.))
        write(out/'RESOURCE.json',dict(status=status,wall_seconds=wall,phase=phase,timings=timing,
            memory=memory.last,memory_stop_reason=memory.stop_reason,automatic_retry=False,
            per_process_limit_seconds=max_wall_seconds,FIT_READY=False,background_qualified=False,production_adopted=False))
        journal.finish(status,phase=phase,timings=timing,memory=memory.last,
            close_error=None if close_error is None else str(close_error))
        if close_error is not None and not propagating:raise close_error


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--position',choices=['near','far'],required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'reports'):raise ContractError('Fresh reports output required')
    run(args.preparation.resolve(),args.position,output)
