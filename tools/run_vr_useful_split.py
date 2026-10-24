"""Explicit actual-input candidate worker; missing real capture blocks import.

No default/backend change. One existing round, original qualification, one raw
100ms background-prefix chunk, no fitting or automatic continuation to65s.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def rod_input_breakpoints(grid_origin, start, stop):
    import numpy as np
    ticks=grid_origin+np.arange(int(np.ceil((stop-grid_origin)/.0001))+1)*.0001
    return tuple(float(t) for t in ticks if start<t<stop)


def require_frontend(directory):
    """Read-only preflight before torch/Numba, graph assembly or model work."""
    from retina_sim.contracts import ContractError
    directory=Path(directory).resolve()
    if not directory.is_relative_to(ROOT):raise ContractError('Current project run required')
    budget=json.loads((directory/'round_budget.json').read_bytes())
    from retina_sim.useful_runtime_budget import effective_deadline
    deadline=effective_deadline(directory,budget)
    if deadline is not None and time.time()>=deadline:raise ContractError('Original useful-round deadline has expired')
    p=directory/'linear_inputs/frontend_receipt.json'
    if not p.is_file():raise ContractError('ACTUAL_LINEAR_FRONTEND_RECEIPT_ABSENT: no graph/JIT/model started')
    frontend=json.loads(p.read_bytes())
    if frontend['policy_sha256']!=budget['policy_sha256'] or frontend['physical_calibration_claim'] is not False:
        raise ContractError('Actual frontend does not match approved engineering input')
    for name,digest in frontend['files'].items():
        q=directory/'linear_inputs'/name
        if q.name!=name or hashlib.sha256(q.read_bytes()).hexdigest()!=digest:
            raise ContractError('Actual capture content changed: '+name)
    return budget,frontend


def main(directory, *, resume_qualified=False, source_handler=None):
    budget,frontend=require_frontend(directory)
    from tools.live_status_io import write_json
    directory=Path(directory).resolve();output=directory/'source_execution'
    from retina_sim.useful_runtime_budget import effective_deadline,runtime_limits
    deadline=effective_deadline(directory,budget)
    if deadline is not None and deadline-time.time()<360:
        status=json.loads((directory/'status.json').read_bytes())
        status.update(stage='REAL_OPTICAL_INPUT_SAVED_INSUFFICIENT_COLD_SOURCE_RESERVE',
            model_forward_started=False,fit_eligible=False,remaining_wall_seconds=budget['deadline_unix']-time.time())
        write_json(directory/'status.json',status,replace=True)
        return
    if output.exists() and not resume_qualified:
        raise FileExistsError('Keep the prior source execution; explicit restore is required')
    if resume_qualified and not (output/'background/head.json').is_file():
        raise FileNotFoundError('Explicit qualified resume requires the existing background head')
    output.mkdir(exist_ok=resume_qualified)
    status=dict(stage='ACTUAL_GRAPH_PREPARATION',pid=os.getpid(),round_budget=budget,
        model_forward_started=False,fit_eligible=False,active_matlab_pid=None,
        active_execution_goal=runtime_limits(directory,budget).get('authorized_goal','bounded_first_useful_capture'),
        qualified_background_resume_requested=resume_qualified,background_integral_rerun_requested=not resume_qualified,
        effective_deadline_unix=deadline,time_deadline_enforced=deadline is not None)
    write_json(directory/'status.json',status,replace=True)
    with (output/f'watchdog-{os.getpid()}.log').open('xb') as log:
        subprocess.Popen([sys.executable,str(ROOT/'tools/watch_useful_capture_process.py'),
            '--run-dir',str(directory),'--root-pid',str(os.getpid())],stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        import numpy as np
        from dataclasses import replace
        from retina_platform.config import RunConfig
        from retina_sim.contracts import ContractError,canonical_hash,SolverConfig,Quantity
        from retina_sim.spatial import InputChannel
        from retina_sim.atlas_population_network import atlas_population_network
        from retina_sim.gpu_compact import CompactCudaHybridRunner
        from retina_sim.resident_cusparse import make_candidate
        from retina_sim.source_graph_contract import SourceGraphContract,source_rf_mm
        from retina_sim.resident_background import bind_resident_background
        from retina_sim.cone_hc_split_contract import ConeHCSplitContract,USEFUL_SCOPE
        from retina_sim.cone_hc_split_runner import ConeHCSplitRunner
        from retina_sim.split_execution_guard import SplitExecutionGuard
        from retina_sim.split_source_background import SplitBackgroundCheckpointStore,prepare_split_source_background
        from retina_sim.split_boundary_source import SplitSourceSession
        from retina_sim.split_boundary_replay_gate import CacheReadoutComparator
        from tools.vr_teacher_handoff_v1.fit_integration_v1.capture_preparation_v2.handoff import approved_policy
        from tools.vr_teacher_handoff_v1.fit_integration_v1.capture_preparation_v2.linear_gray import LinearGrayScene
        from tools.vr_teacher_handoff_v1.fit_integration_v1.pilot import Gray8Chunks
        approval=json.loads((directory/'input_policy_approval.json').read_bytes())
        handoff,objects=approved_policy(ROOT,approval)
        ref=json.loads((directory/'actual_global_reference.json').read_bytes())
        if canonical_hash(ref)!=frontend['global_reference_numeric_sha256']:
            raise ContractError('Actual numerical ISET global reference changed after capture')
        input_dir=directory/'linear_inputs'
        expected=np.load(input_dir/'cone_expected_counts.npy',allow_pickle=False,mmap_mode='r')
        background=np.load(input_dir/'cone_background_counts.npy',allow_pickle=False,mmap_mode='r')
        table=json.loads((ROOT/'data/atlas/human_disk3mm_typed_tables_v1/cones/manifest.json').read_bytes())
        cone_types=np.load(ROOT/table['extras']['subtype']['path'],allow_pickle=False,mmap_mode='r')
        type_index=np.array([{'L':0,'M':1,'S':2}[x.decode('ascii')] for x in cone_types])
        alpha=np.asarray(ref['q_ref_LMS_rstar_s'])[type_index]/100.
        native_background=np.asarray(background)/(1/150.)/alpha
        if not np.array_equal(native_background,np.load(input_dir/'cone_background_native_td.npy',allow_pickle=False,mmap_mode='r')):
            raise ContractError('Actual native background differs from the registered capture bridge')
        if expected.shape!=(16,len(alpha)) or not np.isfinite(native_background).all() or np.any(native_background<=0):
            raise ContractError('Actual full-cone input arrays are incomplete')
        a=dict(scope=USEFUL_SCOPE,input_policy_approved=True,target_graph_approved=True,
            default_backend_change=False,fit_or_upload_authorized=False,
            user_authorization_reference=approval['trusted_conversation_evidence_id'],
            input_policy_receipt_sha256=hashlib.sha256((input_dir/'frontend_receipt.json').read_bytes()).hexdigest(),
            target_graph_receipt_sha256=handoff['references']['capture_configuration']['sha256'],
            execution_protocol_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        a.update(runtime_limits(directory,budget))
        guard=SplitExecutionGuard(a,stop_file=directory/'REQUEST_STOP',output_directory=directory)
        guard.require(force=True)
        cfg=RunConfig.project_default(ROOT)
        policy=replace(cfg.background_policy(),max_wall_seconds=3600.) # budgets only, same scientific criteria
        cfg.native_background_values=tuple(native_background)
        cfg.native_background_td=objects['input_policy']['outside']['uniform_scene_photopic_reference_td_before_cell_capture']
        cfg.rod_background_rstar_s=objects['input_policy']['rod']['outside_rate_Rstar_per_s']
        # Compatibility scalar, never used for a cone drive in this native-only
        # graph. All cone inputs use the actual per-cell captured vector above.
        cfg.cone_background_rstar_s=ref['q_ref_LMS_rstar_s'][0]*(19/255/.5)
        cfg.frontend_signature=canonical_hash(frontend);light=cfg.light_context()
        config_path=handoff['references']['capture_configuration']['path']
        configuration=objects['capture_configuration']
        compiled,_,audit=atlas_population_network(ROOT,duration=12.,light_context=light,
            owner_bounds_mm=configuration['owner_bounds_mm'],halo_mm=configuration['halo_mm'],
            max_spatial_cells=600000,seed=cfg.seed,execution_profile='atlas_isetbio_geometry_v1',
            configuration_path=config_path)
        guard.require(force=True)
        rows=np.asarray(compiled.global_disk_view['selected_rows']['cones'])
        if not np.array_equal(rows,np.arange(len(alpha))):
            raise ContractError('Native cone ordering differs from the exact captured optical order; no implicit remap')
        for cid,required in configuration['owner_global_rows'].items():
            actual=np.asarray(compiled.global_disk_view['selected_rows'][cid])
            if not np.array_equal(np.sort(actual),np.sort(required)):
                raise ContractError('Actual graph does not retain the exact original8 plus comparison4 cohort')
        for cell in objects['cells']['cells']:
            pop=compiled.populations[cell['student_component']]
            ids=list(pop.cells.cell_id)
            if cell['student_cell_id'] not in ids:raise ContractError('Approved comparison target absent in actual graph')
            i=ids.index(cell['student_cell_id'])
            if not np.array_equal(source_rf_mm(pop)[i],cell['rf_mm']):
                raise ContractError('Actual comparison RF differs; no nearest-node reselection')
        compact=CompactCudaHybridRunner(compiled)
        solver=SolverConfig(**audit['resolved_execution_profile']['solver'])
        snapshot=compact.initialize(solver)
        source_contract=SourceGraphContract(compact,configuration_path=ROOT/config_path,
            configuration_sha256=a['target_graph_receipt_sha256'])
        a['source_reference_graph_binding']=source_contract.source_graph_binding
        graph_path=output/'source_graph_binding.json'
        if graph_path.exists():
            if json.loads(graph_path.read_bytes())!=source_contract.source_graph_binding:
                raise ContractError('Restored source graph differs from the saved approved graph')
        else:write_json(graph_path,source_contract.source_graph_binding)
        retained=make_candidate(compact,numerical_contract=source_contract,purpose='approved_source_binding')
        base=bind_resident_background(retained,snapshot,light,policy)
        a.update(source_resident_model_hash=retained.model.model_hash,baseline_binding_hash=retained.binding['binding_hash'],
            approved_input_policy_sha256_by_phase={'uniform_background':canonical_hash({k:v.input_contract for k,v in base.channels.items()})})
        contract=ConeHCSplitContract.for_useful_chunks(a)
        split=ConeHCSplitRunner.from_resident(retained,purpose='bounded_source',numerical_contract=contract)
        state=split.import_resident_state(base.state,retained=retained,reference_state=base.reference_state,
            channels=base.channels,channel_data_hashes={k:canonical_hash(v.at(0.)) for k,v in base.channels.items()},monitor_policy=policy)
        store=SplitBackgroundCheckpointStore(ROOT,split,base.reference_state,light,policy,
            directory=output/'background',create=not resume_qualified)
        status.update(stage='ORIGINAL_FULL_BACKGROUND_QUALIFICATION',model_forward_started=True)
        write_json(directory/'status.json',status,replace=True)
        def background_progress(current,window):
            status.update(background_absolute_seconds=current.time_seconds,
                last_completed_background_window=window,resources=dict(guard.last),updated_unix=time.time())
            write_json(directory/'status.json',status,replace=True)
        if resume_qualified:
            head=json.loads((store.directory/'head.json').read_bytes())
            qualified_path=store.directory/head['snapshot']
            state,envelope=store.load(qualified_path,require_converged=True)
            if envelope['receipt_sha256']!=head['receipt_sha256']:
                raise ContractError('Qualified background head receipt changed')
            report=envelope['report']
            background_progress(state,report['windows'][-1])
        else:
            state,report,qualified_path=prepare_split_source_background(store,state,base.channels,
                reference_state=base.reference_state,guard=guard,
                max_prepare_advance_seconds=policy.max_sim_seconds if deadline is None else 2.,
                on_progress=background_progress)
        if not report['converged']:
            status.update(stage='UNQUALIFIED_PROGRESS_SAVED_NO_CAPTURE',checkpoint=str(qualified_path))
            write_json(directory/'status.json',status,replace=True);return
        if source_handler is not None:
            return source_handler(split=split,state=state,store=store,base=base,authorization=a,
                policy=policy,audit=audit,objects=objects,frontend=frontend,directory=directory,
                output=output,guard=guard,status=status)
        if deadline is not None and time.time()+180>=deadline:
            status.update(stage='QUALIFIED_BACKGROUND_SAVED_INSUFFICIENT_FIRST_CHUNK_RESERVE',checkpoint=str(qualified_path))
            write_json(directory/'status.json',status,replace=True);return
        raw_origin=state.time_seconds
        intake=Path(objects['received_assets']['local_intake_root'])
        index=json.loads((intake/objects['received_assets']['stimulus_index']['path']).read_bytes())
        reader=Gray8Chunks(intake,index);adapter=LinearGrayScene(ROOT,approval)
        frames=[reader.frame(i) for i in range(16)]
        channels={};hashes={};edges=np.arange(17,dtype=np.float64)/150.
        absolute_edges=raw_origin+edges
        native_cache={};rod_cache={}
        for external in compiled.spec.external_inputs:
            component=compiled.components[external.target_component];port=component.inputs[external.target_port]
            old=base.channels[external.channel]
            metadata=dict(old.input_contract,declared_constant_background=False,
                vr_policy_sha256=budget['policy_sha256'],frontend_receipt_sha256=a['input_policy_receipt_sha256'],
                raw_zero_absolute_seconds=raw_origin,recorded_background_frames=750)
            if port.unit=='td':
                def evaluate(t,port=port,old=old):
                    if t<raw_origin:return Quantity(native_background,'td',port.meaning,old.evidence_tag)
                    i=int(np.searchsorted(absolute_edges,t,side='right')-1)
                    if not 0<=i<16:raise ContractError('Actual prepared cone prefix exhausted; no implicit extension')
                    if i not in native_cache:
                        native_cache.clear();native_cache[i]=Quantity(np.asarray(expected[i])/(1/150.)/alpha,'td',port.meaning,old.evidence_tag)
                    return native_cache[i]
                breaks=tuple(absolute_edges[:-1])
                channels[external.channel]=InputChannel(0.,raw_origin+edges[-1],evaluate,port.unit,port.meaning,
                    'zero_order_hold',old.evidence_tag,metadata,breaks)
            elif component.time_contract.kind=='author_discrete' and external.target_port=='light':
                pop=compiled.populations['rod'];lookup={x:i for i,x in enumerate(pop.cells.cell_id)}
                xy=source_rf_mm(pop)[[lookup[x] for x in port.cell_ids]]
                def rod_evaluate(t,xy=xy,port=port,old=old):
                    tick=np.floor((t-state.grid_origin_seconds)/.0001)
                    absolute=state.grid_origin_seconds+tick*.0001
                    if absolute<raw_origin:return old.at(t)
                    i=int(np.searchsorted(absolute_edges,absolute,side='right')-1)
                    if not 0<=i<16:raise ContractError('Actual prepared rod prefix exhausted')
                    if i not in rod_cache:
                        rod_cache.clear();rod_cache[i]=Quantity(adapter.independent_rod_rates(frames[i],xy),port.unit,port.meaning,old.evidence_tag)
                    return rod_cache[i]
                metadata.update(grid_step_seconds=.0001,includes_background=True)
                channels[external.channel]=InputChannel(0.,raw_origin+edges[-1],rod_evaluate,port.unit,port.meaning,
                    'zero_order_hold',old.evidence_tag,metadata,
                    rod_input_breakpoints(state.grid_origin_seconds,old.start_seconds,raw_origin+edges[-1]))
            else:channels[external.channel]=old
            hashes[external.channel]=canonical_hash(dict(frontend=frontend,channel=external.channel,
                policy=channels[external.channel].input_contract))
        b=dict(a,approved_input_policy_sha256_by_phase=dict(a['approved_input_policy_sha256_by_phase'],
            movie=canonical_hash({k:v.input_contract for k,v in channels.items()})))
        split.replace_useful_runtime_permit(ConeHCSplitContract.for_useful_chunks(b))
        identity=dict(stimulus_hash=objects['pilot_contract']['input']['index_sha256'],
            calibration_hash=a['input_policy_receipt_sha256'],spatial_mapping_hash=frontend['capture_geometry_sha256'],
            background_hash=store.binding_hash)
        session=SplitSourceSession.from_qualified_background(store,qualified_path,directory=output/'chunks',
            reference_state=base.reference_state,background_channels=base.channels,movie_channels=channels,
            channel_data_hashes=hashes,identity=identity,registry=audit['registry'],guard=guard)
        status.update(stage='FIRST_RAW_BACKGROUND_PREFIX_CHUNK',raw_origin_seconds=raw_origin)
        write_json(directory/'status.json',status,replace=True)
        def comparator(current,times,plan):
            return CacheReadoutComparator.for_runner(split,base.reference_state,policy,current,times,plan)
        stop=state.grid_origin_seconds+(round((raw_origin-state.grid_origin_seconds)/.0001)+1000)*.0001
        commit=session.advance_one(stop,comparator_factory=comparator)
        status.update(stage='FIRST_100MS_RAW_BACKGROUND_PREFIX_COMMITTED_STOP',commit=commit,
            finished_unix=time.time(),active_matlab_pid=None,qualified_background=True,
            active_movie_data_seconds=0.,fit_eligible=False)
        write_json(directory/'status.json',status,replace=True)
    except Exception as exc:
        status.update(stage='SOURCE_STOPPED_WITH_DIAGNOSTIC',exception_type=type(exc).__name__,
            message=str(exc),finished_unix=time.time())
        write_json(directory/'status.json',status,replace=True);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--resume-qualified',action='store_true')
    args=p.parse_args();main(args.run_dir,resume_qualified=args.resume_qualified)
