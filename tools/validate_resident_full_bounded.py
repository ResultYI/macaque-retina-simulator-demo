"""One supervised representative resident QA; no fitting or default adoption.

Requires the completed small diagnostic, immutable regression receipt and a
900-second external process supervisor. Always writes a new result directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(output):
    if os.environ.get('RETINA_RESIDENT_VALIDATION_AUTHORIZED')!='user_finished_game_20261022':
        raise PermissionError('Explicit released gaming restriction required')
    output.mkdir(parents=True,exist_ok=False)
    begun=time.perf_counter()
    report=dict(status='STARTING',pid=os.getpid(),fit_eligible=False,performance_validated=False)
    def save(name,value):
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    def phase(name,**values):
        report.update(phase=name,wall_seconds=time.perf_counter()-begun,**values)
        save('progress.json',report);print(name,flush=True)
    try:
        phase('resource_and_small_run_preconditions')
        small=json.loads((ROOT/'reports/resident_runtime_validation_20261022_v1/small_r2/outcome.json').read_bytes())
        if small['status']!='SMALL_CHAIN_CHECKS_COMPLETED_NOT_FULL_QA':
            raise RuntimeError('Existing small chain has not completed')
        import numpy as np
        import numba
        import torch
        from dataclasses import replace
        from retina_platform.config import RunConfig
        from retina_sim.atlas_population_network import atlas_population_network
        from retina_sim.background_checkpoint import MemoryGuard
        from retina_sim.background import constant_background_channels
        from retina_sim.contracts import SolverConfig,Quantity
        from retina_sim.gpu_compact import CompactCudaHybridRunner
        from retina_sim.recording import RecordingPlan
        from retina_sim.resident_recording_selection import indexed_recording_plan,selection_identity
        from retina_sim.resident_contract import ResidentNativeContract
        from retina_sim.resident_derived_ports import ribbon_descriptors,ADDENDUM_SHA256
        from retina_sim.resident_regression_evidence import load_reference_regression_receipt,RECEIPT
        from retina_sim.resident_qa_harness import run_resident_qa
        numba.set_num_threads(1);torch.set_num_threads(1)
        guard=MemoryGuard(12*1024**3,8*1024**3)
        def check():
            if not guard.check():raise RuntimeError('Memory budget: '+str(guard.stop_reason))
            if time.perf_counter()-begun>840:raise TimeoutError('840 s internal full-run limit')
            if shutil.disk_usage(output).free<8*1024**3:raise RuntimeError('8 GiB disk reserve')
            if torch.cuda.is_initialized() and torch.cuda.memory_allocated()>6*1024**3:
                raise RuntimeError('6 GiB CUDA allocation budget')
            if (ROOT/'STOP').exists() or (ROOT/'STOP.txt').exists():raise RuntimeError('STOP present; preserved')
            return True
        check()
        if shutil.disk_usage(output).free<72*1024**3:
            raise RuntimeError('64 GiB spool plus 8 GiB disk reserve required')
        receipt=load_reference_regression_receipt(receipt_path=ROOT/RECEIPT)
        contract=ResidentNativeContract()
        save('resource_budget.json',dict(max_rss_bytes=12*1024**3,minimum_available_bytes=8*1024**3,
            CUDA_allocation_limit_bytes=6*1024**3,QA_spool_limit_bytes=64*1024**3,
            minimum_disk_reserve_bytes=8*1024**3,QA_wall_seconds=600,outer_wall_seconds=900,
            initial_memory=guard.last,regression_receipt_sha256=receipt['receipt_sha256']))
        cfg=RunConfig.project_default(ROOT);light=cfg.light_context()
        configuration_path='configurations/natural_movie_matched_cohort_v1.json'
        configuration=json.loads((ROOT/configuration_path).read_bytes())
        phase('assemble_unchanged_full_dependency_graph')
        compiled,channels,audit=atlas_population_network(ROOT,duration=12.,light_context=light,
            owner_bounds_mm=configuration['owner_bounds_mm'],halo_mm=.04,max_spatial_cells=600000,
            seed=cfg.seed,execution_profile='atlas_isetbio_geometry_v1',configuration_path=configuration_path)
        check()
        counts={key:len(pop.cells.cell_id) for key,pop in compiled.populations.items()}
        if len(compiled.components)!=24 or counts['cones']!=545658 or counts['HC']!=344901:
            raise RuntimeError('Full graph differs; no cropping or fallback')
        phase('original_compact_runner_and_unchanged_initial_state',counts=counts,
              components=len(compiled.components),assembly_hash=compiled.assembly_hash)
        runner=CompactCudaHybridRunner(compiled)
        solver=SolverConfig(**audit['resolved_execution_profile']['solver'])
        contract.validate_solver(solver);contract.validate_reference(runner)
        snapshot=runner.initialize(solver)
        policy=cfg.background_policy()
        # Assembly channels are not the author-total-rate diagnostic source.
        # Use the same established helper as the original bounded full profile.
        channels=constant_background_channels(compiled,light,12.)
        for external in compiled.spec.external_inputs:
            component=compiled.components[external.target_component]
            if component.time_contract.kind=='author_discrete':
                channel=channels[external.channel]
                if (channel.interpolation!='zero_order_hold' or
                        channel.input_contract.get('grid_step_seconds')!=component.time_contract.fixed_step_seconds or
                        channel.input_contract.get('includes_background') is not True):
                    raise RuntimeError('Original author total-rate channel preflight failed')
        # Use the unchanged current initial state. Old checkpoints carry old
        # repaired-reference identities and are never relabelled or migrated.
        source_arrays={};frame=1/150
        source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        # Execute the exact archived constructor rather than relabel new input
        # code with an old identity. All scientific plan fields must match.
        from retina_sim.resident_reference_reuse import archived_input_loop
        archived_source=ROOT/'reports/resident_runtime_validation_20261022_v1/full_r3_supervisor/sources/validate_resident_full_bounded.py'
        archived_input_loop(archived_source,'778be9e8fa84bccc56633b436fa9fcbd042735f7ccd9333f92afb21f60f0a72d',
            dict(np=np,Quantity=Quantity,replace=replace,channels=channels,source_arrays=source_arrays,frame=frame))
        descriptors=ribbon_descriptors(compiled)
        selected=set()
        for name in ('cones','HC'):
            selected.add(compiled.populations[name].cells.cell_id[0])
        for cid in snapshot.spike_states:
            selected.update(compiled.components[cid].state_specs[0].cell_ids)
        for descriptor in descriptors.values():selected.update(descriptor['cell_ids'])
        plan=indexed_recording_plan(RecordingPlan(cell_ids=tuple(sorted(selected))))
        samples=np.array([0.,.000031,.001,.00317,frame,.00723,.008])
        phase('one_prospectively_bound_full_numerical_comparison',
              state_scalars=sum(x.size for x in snapshot.ordered_state_arrays),
              solver=solver.__dict__,selected_recording_cell_count=len(selected),
              derived_port_addendum_sha256=ADDENDUM_SHA256,initial_state_modified=False,
              checkpoint_migrated=False)
        save('recording_selection.json',selection_identity(plan))
        check()
        # Pure preparation calls compile the existing CPU kernels outside the
        # Python QA trace. They do not advance clocks/state or consume RNG.
        phase('prepare_existing_kernels_outside_QA_trace')
        from retina_sim.network import NetworkStimulus
        from retina_sim.resident_qa_harness import _monitor_for
        from retina_sim.contracts import canonical_hash
        initial_digest=canonical_hash(snapshot)
        preparation_start=time.perf_counter()
        stimulus=NetworkStimulus(compiled,channels,snapshot,.008,solver.input_interpolation,runner.model.evidence_tag)
        bundle=stimulus.at(snapshot.time_seconds)
        y=runner._flat(snapshot)
        runner.partition(snapshot.time_seconds,torch.as_tensor(y,dtype=torch.float64,device=runner.partition.gather.device),bundle)
        cold_monitor=_monitor_for(runner,snapshot,policy)
        cold_monitor.observe(snapshot.time_seconds,y,bundle)
        torch.cuda.synchronize()
        if canonical_hash(snapshot)!=initial_digest:raise RuntimeError('Preparation mutated initial state/aux')
        save('cold_kernel_preparation.json',dict(seconds=time.perf_counter()-preparation_start,
            complete_initial_state_and_aux_unchanged=True,extra_pure_RHS_and_ephemeral_monitor_only=True,
            excluded_from_main_callback_schedule=True,excluded_from_hot_timing=True))
        del cold_monitor,y,bundle,stimulus
        check()
        contexts=[]
        result=run_resident_qa(runner,snapshot,snapshot,channels,policy,
            prepared_sources=source_arrays,sample_times=samples,stop=.008,background_stop=.001,
            frame_boundaries=[frame],recording_plan=plan,output_directory=output/'qa',
            max_spool_bytes=64*1024**3,continuation_stop=.0081,on_boundary=check,
            execution_authorized=True,restriction_release_evidence='User: ok, game finished; parent authorized bounded required validation',
            max_qa_seconds=600,reference_regression_receipt=ROOT/RECEIPT,context_sink=contexts.append,
            reuse_reference_main=ROOT/'reports/resident_runtime_validation_20261022_v1/full_r6/qa',
            reuse_reference_complete=True,candidate_native_sparse_plan=True)
        report.update(status='FULL_QA_RETURNED',result=result,memory=guard.last,
                      CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=source_sha:
            raise RuntimeError('Input evaluator source changed during the comparison')
        # Lack of an event for a cell is coverage INCONCLUSIVE, not an output
        # mismatch. It never becomes a numerical PASS or backend adoption. A
        # diagnostic timing may follow only when every executed numeric/exact
        # gate completed, with that coverage limitation retained explicitly.
        only_spike_coverage=(result['status']=='INCONCLUSIVE' and result.get('issues') and
            all(issue['gate']=='spike_coverage' for issue in result['issues']))
        if contexts and (result['status']=='PASS' or only_spike_coverage):
            phase('bounded_same_process_hot_E2E',correctness_status=result['status'])
            # Ordinary target output scope: the existing eight RGC cells. Full
            # native/inner monitoring remains identical in both timed arms.
            rgc_cells=[];rgc_ports=[]
            for cid in snapshot.spike_states:
                rgc_cells.extend(compiled.components[cid].state_specs[0].cell_ids)
                rgc_ports.extend(cid+'.'+port for port in compiled.components[cid].outputs)
                rgc_ports.append(cid+'.conditional_rate')
            ordinary=indexed_recording_plan(RecordingPlan(states=(),ports=tuple(rgc_ports),cell_ids=tuple(rgc_cells)))
            report['performance']=hot_comparison(contexts[0],snapshot,channels,policy,ordinary,
                check,save,begun,torch,np,result['status'])
            report['performance_measured']=report['performance']['status']=='THREE_PAIRED_HOT_FULL_MODEL_CLIPS'
            report['performance_validated']=(report['performance_measured'] and result['status']=='PASS'
                and report['performance']['engineering_value_gate']['passed'])
    except Exception as exc:
        report.update(status='STOPPED',error_type=type(exc).__name__,detail=str(exc),traceback=traceback.format_exc())
    report['wall_seconds']=time.perf_counter()-begun
    save('outcome.json',report);print(json.dumps(report),flush=True)
    return 0 if report['status']=='FULL_QA_RETURNED' and report['result']['status']=='PASS' else 1


def hot_comparison(context,snapshot,channels,policy,recording,check,save,begun,torch,np,correctness_status):
    """Three paired full-model 2ms clips. No per-operator trace or QA I/O timed."""
    from retina_sim.background import _WindowMonitor,NUMERICAL_UNIT_SCALES
    from retina_sim.resident_qa_harness import _monitor_for,_recording_witness,_compare_recording
    from retina_sim.contracts import canonical_hash
    from retina_sim.resident_recording_selection import selection_identity
    reference,candidate=context['reference'],context['candidate']
    initial=context['initial'];plan=context['plan']
    import gc,copy
    gc.collect()
    if time.perf_counter()-begun>750:
        return dict(status='NOT_RUN_INSUFFICIENT_WALL_RESERVE',correctness_status=correctness_status)
    # Existing QA has already compiled and exercised dense paths and kernels.
    # One 100us transport/controller priming pair, never a background warmup.
    from retina_sim.recording import RecordingPlan
    for arm,runner,state in [('reference',reference,snapshot),('candidate',candidate,initial)]:
        check();monitor=_monitor_for(runner,snapshot,policy)
        runner.advance(state,channels,.0001,recording_plan=RecordingPlan(ports=(),states=(),cell_ids=()),
                       state_observer=monitor.observe)
    times=np.array([0.,.000031,.001,.00123,.002]);rows=[]
    initial_by_name={s.state_name:a for s,a in zip(reference.model.state_specs,snapshot.ordered_state_arrays)}
    def execute(arm):
        check();runner,state=(reference,snapshot) if arm=='reference' else (candidate,initial)
        before_metrics={key:runner.integration_metrics.get(key,0) for key in ('solver_calls','nfev')}
        totals=dict(integration=0.,history_events=0.,monitor_observe=0.,monitor_report=0.)
        originals={name:getattr(runner,name) for name in ('_integrate','_sample')}
        present={name:name in runner.__dict__ for name in originals}
        stored={name:runner.__dict__.get(name) for name in originals}
        def wrap(fun,key):
            def call(*a,**kw):
                tick=time.perf_counter()
                try:return fun(*a,**kw)
                finally:totals[key]+=time.perf_counter()-tick
            return call
        runner._integrate=wrap(originals['_integrate'],'integration')
        runner._sample=wrap(originals['_sample'],'history_events')
        try:
            torch.cuda.synchronize();tick=time.perf_counter()
            monitor=_monitor_for(runner,snapshot,policy)
            final,result=runner.advance(state,channels,.002,sample_times=times,
                recording_plan=recording,state_observer=wrap(monitor.observe,'monitor_observe'))
            report=wrap(monitor.report,'monitor_report')(.002)
            torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        finally:
            for name in originals:
                if present[name]:setattr(runner,name,stored[name])
                else:delattr(runner,name)
        totals['other_including_monitor_init_recording']=elapsed-sum(totals.values())
        return final,result,report,dict(arm=arm,E2E_seconds=elapsed,scopes=totals,
            solver_calls=runner.integration_metrics.get('solver_calls',0)-before_metrics['solver_calls'],
            integration_RHS_calls=runner.integration_metrics.get('nfev',0)-before_metrics['nfev'])
    for pair in range(3):
        order=('reference','candidate') if pair%2==0 else ('candidate','reference')
        values={}
        for arm in order:
            values[arm]=execute(arm)
            save('hot_progress.json',dict(pair=pair,completed_arm=arm,measurement=values[arm][3],
                                         wall_seconds=time.perf_counter()-begun))
        a,b=values['reference'],values['candidate']
        for key in ('solver_calls','integration_RHS_calls'):
            if a[3][key]!=b[3][key]:raise RuntimeError('Hot integration work differs: '+key)
        y=reference._flat(a[0]);z=b[0].y.detach().cpu().numpy()
        offset=0;maximum=0.
        for spec in reference.model.state_specs:
            stop=offset+spec.shape[0];reg=NUMERICAL_UNIT_SCALES[spec.unit]
            scale=np.maximum(np.abs(initial_by_name[spec.state_name]),reg['scale_floor'])
            allowance=.01*np.minimum(snapshot.solver_config.atol+snapshot.solver_config.rtol*scale,
                                     reg['atol']+policy.relative_tolerance*scale)
            ratio=float(np.max(np.abs(y[offset:stop]-z[offset:stop])/allowance)) if stop>offset else 0.
            maximum=max(maximum,ratio);offset=stop
        if maximum>1 or not np.isfinite(z).all():raise RuntimeError('Hot repeated final-state numerical gate')
        for key in ('grid_origin_seconds','sample_index','component_tick_indices'):
            if canonical_hash(getattr(a[0],key))!=canonical_hash(getattr(b[0],key)):
                raise RuntimeError('Hot exact clock gate: '+key)
        for cid,aux in a[0].spike_states.items():
            other=b[0].spike_states[cid]
            for key in ('history','rng_state','last_bin_counts'):
                if canonical_hash(aux[key])!=canonical_hash(other[key]):raise RuntimeError('Hot RNG/history gate')
        if a[2]['passed']!=b[2]['passed'] or a[2]['domain_failures']!=b[2]['domain_failures']:
            raise RuntimeError('Hot monitor predicate gate')
        if a[1].metadata['event_count']!=b[1].metadata['event_count']:
            raise RuntimeError('Hot recorded event-count gate')
        validator=copy.copy(context['comparator_template'])
        validator.issues=[];validator.worst_status='PASS';validator.issue_overflow=False;validator.closed=False
        validator.sink=lambda record:None
        _compare_recording(validator,_recording_witness(a[1]),_recording_witness(b[1]))
        if validator.worst_status!='PASS':
            save('hot_recording_failure.json',dict(pair=pair,issues=[issue.__dict__ for issue in validator.issues]))
            raise RuntimeError('Repeated hot typed recording gate')
        pair_row=dict(pair=pair,order=order,reference=a[3],candidate=b[3],
            speedup=a[3]['E2E_seconds']/b[3]['E2E_seconds'],
            final_state_max_error_over_frozen_budget=maximum,
            repeated_typed_recording_check='PASS',
            event_count=a[1].metadata['event_count'])
        rows.append(pair_row);save('hot_pairs.json',rows)
    import statistics
    left=[x['reference']['E2E_seconds'] for x in rows]
    right=[x['candidate']['E2E_seconds'] for x in rows]
    return dict(status='THREE_PAIRED_HOT_FULL_MODEL_CLIPS',neural_seconds_per_run=.002,
        pairs=rows,reference_median_seconds=statistics.median(left),candidate_median_seconds=statistics.median(right),
        speedup_of_medians=statistics.median(left)/statistics.median(right),
        engineering_value_gate=dict(required_hot_wall_reduction=.2,
            observed_hot_wall_reduction=1-statistics.median(right)/statistics.median(left),
            passed=statistics.median(right)<=.8*statistics.median(left)),
        correctness_status=correctness_status,full_gate_pass=correctness_status=='PASS',
        monitor='Complete original checks and report; original fixed thresholds',
        recording_selection=selection_identity(recording),
        scope='545658 cones/344901 HC/5096368 FP64 states, existing eight RGC output cells, same 2ms two-level input clip, NumPy/Numba threads=1; integration/events/RNG/complete monitor/report/recording included',
        excluded='Cold graph assembly/plans; QA hashing/traces/spool; explicit whole-state export/checkpoint and optical movie preparation',
        per_scope_wall='Inclusive host scopes including waits; not GPU kernel busy times',
        production_default_switched=False,fit_eligible=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute-full-qa',action='store_true',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(main(args.output))
