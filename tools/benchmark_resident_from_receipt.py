"""Bounded hot timing from an immutable completed numerical receipt.

No numerical trajectory QA is repeated. The only permitted source difference
is a proven byte-identical canonical encoder; all physics/solver/observer/native
plan sources and complete actual initial/input/monitor/model bindings must match.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'reports/resident_runtime_validation_20261022_v1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_completed_numeric_receipt(outcome,plan):
    if outcome.get('contract_hash')!=plan['contract_hash'] or outcome.get('plan_hash')!=plan['plan_sha256']:
        raise ValueError('Numerical receipt contract/plan differs')
    coverage_only=(outcome.get('status')=='INCONCLUSIVE' and outcome.get('issues') and
                   all(x['gate']=='spike_coverage' and x['status']=='INCONCLUSIVE' for x in outcome['issues']))
    if outcome.get('status')!='PASS' and not coverage_only:raise ValueError('Numerical/exact gates incomplete or failed')
    expected={key:len(plan[key]) for key in ('points','intervals','draws','monitors','checkpoint_ids','failure_cases','ownership_ids')}
    if outcome.get('compared')!=expected:raise ValueError('Numerical receipt witness coverage incomplete')


def main(output,*,after_prepare=None):
    if os.environ.get('RETINA_RESIDENT_VALIDATION_AUTHORIZED')!='user_finished_game_20261022':
        raise PermissionError('Explicit released gaming restriction required')
    output.mkdir(parents=True,exist_ok=False);begun=time.perf_counter()
    report=dict(status='STARTING',pid=os.getpid(),performance_validated=False,fit_eligible=False)
    def save(name,value):(output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    def phase(name,**fields):
        report.update(phase=name,wall_seconds=time.perf_counter()-begun,**fields);save('progress.json',report);print(name,flush=True)
    try:
        previous=json.loads((BASE/'full_r9_supervisor/supervisor.json').read_bytes())
        compatibility=json.loads((BASE/'initial_array_hash_r1/after.json').read_bytes())
        for name,old in previous['source_sha256'].items():
            if name=='tools/supervise_resident_validation.py':continue
            current=digest(ROOT/name)
            if current!=old:
                if (name!='retina_sim/canonical_stream.py' or compatibility['old_encoder_sha256']!=old or
                    compatibility['new_encoder_sha256']!=current or compatibility['exact_same_canonical_hash'] is not True):
                    raise ValueError('Unreviewed source change since completed QA: '+name)
        import numpy as np,numba,torch
        from dataclasses import replace
        from retina_platform.config import RunConfig
        from retina_sim.atlas_population_network import atlas_population_network
        from retina_sim.background_checkpoint import MemoryGuard
        from retina_sim.background import constant_background_channels
        from retina_sim.contracts import SolverConfig,Quantity
        from retina_sim.gpu_compact import CompactCudaHybridRunner
        from retina_sim.recording import RecordingPlan
        from retina_sim.resident_cusparse import make_candidate
        from retina_sim.resident_recording_selection import indexed_recording_plan
        from retina_sim.resident_reference_reuse import archived_input_loop
        from retina_sim.resident_qa_harness import WitnessSpool,hash_prepared_sources,_monitor_for
        from retina_sim.resident_qa import ResidentQAComparator
        from tools.validate_resident_full_bounded import hot_comparison
        numba.set_num_threads(1);torch.set_num_threads(1)
        guard=MemoryGuard(12*1024**3,8*1024**3)
        def check():
            if not guard.check():raise RuntimeError('Memory budget: '+str(guard.stop_reason))
            if time.perf_counter()-begun>840:raise TimeoutError('840 s internal hot-receipt budget')
            if (ROOT/'STOP').exists() or (ROOT/'STOP.txt').exists():raise RuntimeError('STOP present; preserved')
            if shutil.disk_usage(output).free<8*1024**3:raise RuntimeError('8 GiB disk reserve')
            if torch.cuda.is_initialized() and torch.cuda.memory_allocated()>6*1024**3:raise RuntimeError('6 GiB CUDA cap')
            return True
        check();phase('assemble_identical_full_graph_for_hot_measurement')
        cfg=RunConfig.project_default(ROOT);light=cfg.light_context()
        configuration_path='configurations/natural_movie_matched_cohort_v1.json'
        configuration=json.loads((ROOT/configuration_path).read_bytes())
        compiled,_,audit=atlas_population_network(ROOT,duration=12.,light_context=light,
            owner_bounds_mm=configuration['owner_bounds_mm'],halo_mm=.04,max_spatial_cells=600000,
            seed=cfg.seed,execution_profile='atlas_isetbio_geometry_v1',configuration_path=configuration_path)
        check();reference=CompactCudaHybridRunner(compiled)
        solver=SolverConfig(**audit['resolved_execution_profile']['solver']);snapshot=reference.initialize(solver)
        policy=cfg.background_policy();channels=constant_background_channels(compiled,light,12.)
        source_arrays={};frame=1/150
        archived_input_loop(BASE/'full_r3_supervisor/sources/validate_resident_full_bounded.py',
            '778be9e8fa84bccc56633b436fa9fcbd042735f7ccd9333f92afb21f60f0a72d',
            dict(np=np,replace=replace,Quantity=Quantity,channels=channels,source_arrays=source_arrays,frame=frame))
        candidate=make_candidate(reference)
        initial=candidate.import_reference_state(snapshot,reference_state=snapshot,channels=channels,
            channel_data_hashes=hash_prepared_sources(source_arrays),monitor_policy=policy)
        reader=WitnessSpool.open_existing(BASE/'full_r9/qa',guard=check)
        with reader.read('prospective_plan') as plan:
            numerical=json.loads((BASE/'full_r9/qa/outcome.json').read_bytes())
            require_completed_numeric_receipt(numerical,plan)
            if candidate.binding!=plan['binding'] or dict(reference=reference.model.model_hash,candidate=candidate.model.model_hash)!=plan['arm_identity_hashes']:
                raise ValueError('Actual model/graph/weights/initial/reference/input/solver binding differs from completed QA')
            template=ResidentQAComparator(plan,evidence_sink=lambda _:None)
            context=dict(reference=reference,candidate=candidate,initial=initial,snapshot=snapshot,plan=plan,comparator_template=template,run_begun=begun,build_audit=audit)
            save('accepted_numerical_receipt.json',dict(path=str(BASE/'full_r9/qa/outcome.json'),
                sha256=digest(BASE/'full_r9/qa/outcome.json'),status=numerical['status'],issues=numerical['issues'],
                plan_sha256=plan['plan_sha256'],actual_binding_exact=True,compatibility=compatibility,
                cold_graph_seconds=time.perf_counter()-begun))
            if after_prepare is not None:
                phase('focused_implementation_stage_from_preserved_baseline')
                report.update(after_prepare(context,channels,policy,check,save,begun,torch,np))
                report['wall_seconds']=time.perf_counter()-begun
                report['memory']=guard.last
                report['CUDA_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                save('outcome.json',report)
                print(json.dumps(report),flush=True)
                return 0 if report['status']=='FOCUSED_VARIANT_COMPLETED' else 1
            phase('bounded_natural_event_supplement_from_authentic_checkpoints')
            from retina_sim.resident_event_supplement import supplement
            report['natural_event_supplement']=supplement(context,channels,policy,check,save,BASE)
            if report['natural_event_supplement']['status']=='EVENT_SUPPLEMENT_STOPPED':
                raise RuntimeError('Natural event supplement stopped; inspect retained evidence before timing')
            check()
            phase('pure_dense_transport_priming_outside_timing')
            for runner,state in ((reference,snapshot),(candidate,initial)):
                monitor=_monitor_for(runner,snapshot,policy)
                runner.advance(state,channels,.0001,sample_times=np.array([0.,.000031,.0001]),
                    recording_plan=RecordingPlan(ports=(),states=(),cell_ids=()),state_observer=monitor.observe)
            cells=[];ports=[]
            for cid in snapshot.spike_states:
                cells.extend(compiled.components[cid].state_specs[0].cell_ids)
                ports.extend(cid+'.'+port for port in compiled.components[cid].outputs);ports.append(cid+'.conditional_rate')
            ordinary=indexed_recording_plan(RecordingPlan(states=(),ports=tuple(ports),cell_ids=tuple(cells)))
            phase('three_same_condition_hot_pairs_from_verified_receipt')
            performance=hot_comparison(context,snapshot,channels,policy,ordinary,check,save,begun,torch,np,numerical['status'])
            report.update(status='HOT_RECEIPT_RUN_COMPLETED',performance=performance,
                numerical_status=numerical['status'],numerical_issues=numerical['issues'],
                performance_measured=performance['status']=='THREE_PAIRED_HOT_FULL_MODEL_CLIPS',
                numerical_trajectory_QA_repeated=False,memory=guard.last,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        report.update(status='STOPPED',error_type=type(exc).__name__,detail=str(exc),traceback=traceback.format_exc())
    report['wall_seconds']=time.perf_counter()-begun;save('outcome.json',report);print(json.dumps(report),flush=True)
    return 0 if report['status']=='HOT_RECEIPT_RUN_COMPLETED' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-hot-from-receipt',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();raise SystemExit(main(a.output))
