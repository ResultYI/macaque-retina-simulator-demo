"""Execute every full_target target through actual connected equations and states."""
from pathlib import Path
import sys,json,argparse,hashlib,os,time
# Process-local default avoids BLAS worker overhead for small matrix/vector operations.
# Honor caller-specified thread counts and never modify the persistent environment.
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ.setdefault(key,'1')
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from retina_sim.contracts import SolverConfig
from retina_sim.recording import NPZWindowRecorder,RecordingPlan
from examples.full_target_network import full_target_network
from tools.run_full_network_fixture import serializable
from retina_sim.hybrid import HybridRunner
from retina_sim.runtime_profiles import LightContext,execution_profile

def main():
    p=argparse.ArgumentParser(description='Mechanism-complete macaque center/periphery model using transparent effective approximation profile; synthetic default input')
    p.add_argument('--duration',type=float,default=.006,help='Default6ms engineering execution recording; not physiological validation');p.add_argument('--window',type=float,default=.002)
    p.add_argument('--sample-step',type=float,default=.001,help='Explicit recording grid,s; use finer sampling for sub-ms event kinetics')
    p.add_argument('--effective-cones',type=int,default=128,help='Engineering preview count; source profile integrates density instead');p.add_argument('--effective-rods',type=int,default=384,help='Engineering preview count; source profile integrates density instead')
    p.add_argument('--bounds',nargs=4,type=float,default=[.004,7.2,-.04,.04],metavar=('XMIN','XMAX','YMIN','YMAX'),help='Engineering preview corridor,mm; source profile uses --owner-bounds and --halo-mm')
    p.add_argument('--cone-density-scale',type=float,default=None,help='Engineering preview density scaling only; rejected in source profile')
    p.add_argument('--rod-density-scale',type=float,default=None,help='Engineering preview density scaling only; rejected in source profile')
    p.add_argument('--secondary-gap-scale',type=float,default=0.,help='Optional physical rod-cone terminal gap multiplier;0means explicitly primary-only')
    p.add_argument('--ablate',action='append',default=[],choices=['ampa','gap','glycine'],help='Explicit primary-route ablation; all baseline mechanism code remains available')
    p.add_argument('--seed',type=int,default=20261010);p.add_argument('--output',type=Path)
    p.add_argument('--warmup',type=float,default=.002,help='Default2ms numerical continuity warmup, not a guaranteed physiological steady state; old long run: --warmup .2 --duration .12 --window .01')
    p.add_argument('--profile',choices=['retina_contact_geometry_effective_v1','macaque_constrained_native_v1','engineering_preview_v1'],default='retina_contact_geometry_effective_v1')
    p.add_argument('--owner-bounds',nargs=4,type=float,default=None)
    p.add_argument('--halo-mm',type=float,default=None)
    p.add_argument('--max-spatial-cells',type=int,default=None)
    p.add_argument('--hc-geometry-profile',choices=['source_patch_v1','hc_joint_synthetic_v1'],default=None)
    p.add_argument('--hc-arbor-profile',choices=['explicit_footprints','central_OPL_equal_area_W2000','W2000_equal_area_bounded_edges_047_104'],default=None)
    p.add_argument('--ipl-geometry-profile',choices=['retina_contact_geometry_effective_v1','p_chart_typed_effective_ipl_v1','legacy_point_ipl_v1'],default=None)
    p.add_argument('--requested-owner-output',action='append',choices=['midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON'])
    p.add_argument('--hc-geometry-json',default='',help='Explicit canonical/finite-footprint/morphology JSON; no inferred central arbor')
    p.add_argument('--hc-h1-radius-mm',type=float,default=None,help='Explicit project finite disk span; default unconfigured')
    p.add_argument('--hc-h2-radius-mm',type=float,default=None,help='Explicit project finite disk span; default unconfigured')
    p.add_argument('--native-background-td',type=float,default=100.)
    p.add_argument('--cone-background-rstar-s',type=float,default=1000.)
    p.add_argument('--rod-background-rstar-s',type=float,default=5.)
    p.add_argument('--background-initialization',choices=['fixed_prerun','bounded_convergence'],default='fixed_prerun')
    p.add_argument('--background-max-sim-seconds',type=float,default=10.)
    p.add_argument('--background-max-wall-seconds',type=float,default=60.)
    p.add_argument('--background-policy-json',default='')
    p.add_argument('--background-require-converged',action='store_true')
    a=p.parse_args()
    if a.duration<=0 or a.window<=0 or a.sample_step<=0 or a.warmup<0:p.error('Positive duration/window/sampling and nonnegative warmup required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=a.output or ROOT/'reports/full_target_runs'/stamp
    out.mkdir(parents=True,exist_ok=False)
    started=datetime.now(timezone.utc).isoformat();wall_start=time.perf_counter()
    progress_file=out/'progress.jsonl'
    phase_wall={}
    def progress(phase,t=None,**extra):
        row={'phase':phase,'utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-wall_start,'simulation_time_seconds':t,**extra}
        with progress_file.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        (out/'run_status.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    progress('building',duration_seconds=a.duration,warmup_seconds=a.warmup,effective_cones=a.effective_cones,effective_rods=a.effective_rods,input_class='synthetic',purpose='Finite-resolution engineering execution')
    total=a.duration+a.warmup
    if a.background_initialization=='bounded_convergence':
        from retina_platform.config import RunConfig
        policy=RunConfig(background_policy_path=a.background_policy_json,
            background_max_sim_seconds=a.background_max_sim_seconds,background_max_wall_seconds=a.background_max_wall_seconds,
            background_require_converged=a.background_require_converged).background_policy()
        total=a.duration+max(a.warmup,policy.max_sim_seconds)
    selected=execution_profile(ROOT,a.profile)
    hc_defaults=selected.get('HC_defaults',{})
    light=LightContext(a.native_background_td,a.cone_background_rstar_s,a.rod_background_rstar_s,
        background_source='CLI_declared_background')
    try:
        compiled,channels,audit=full_target_network(ROOT,effective_cones=a.effective_cones,effective_rods=a.effective_rods,seed=a.seed,duration=total,adaptation_seconds=a.warmup,ablations=a.ablate,secondary_gap_scale=a.secondary_gap_scale,bounds_mm=a.bounds,cone_density_scale=a.cone_density_scale,rod_density_scale=a.rod_density_scale,light_context=light,execution_profile=selected['profile_id'],owner_bounds_mm=a.owner_bounds or selected['region']['owner_bounds_mm'],halo_mm=a.halo_mm if a.halo_mm is not None else selected['region']['halo_mm'],runtime_profile=selected['parameter_profile'],max_spatial_cells=a.max_spatial_cells if a.max_spatial_cells is not None else selected['budgets']['max_population_cells'],hc_geometry_profile=a.hc_geometry_profile or hc_defaults.get('geometry_profile','source_patch_v1'),hc_geometry_path=a.hc_geometry_json or hc_defaults.get('geometry_json',''),hc_h1_radius_mm=a.hc_h1_radius_mm,hc_h2_radius_mm=a.hc_h2_radius_mm,hc_arbor_profile=a.hc_arbor_profile or hc_defaults.get('arbor_profile','explicit_footprints'),ipl_geometry_profile=a.ipl_geometry_profile or ('retina_contact_geometry_effective_v1' if a.profile=='retina_contact_geometry_effective_v1' else 'legacy_point_ipl_v1'),requested_owner_outputs=tuple(a.requested_owner_output or ['midget_RGC_ON']))
    except Exception as error:
        partial=getattr(error,'partial_build_audit',None)
        if partial is not None:
            (out/'partial_build_audit.json').write_text(json.dumps(partial,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (out/'failure.json').write_text(json.dumps({'status':'BUILD_BLOCKED','error':str(error),
            'invocation':vars(a)|{'output':str(out)},'time_integration':'NOT_RUN','physiological_validation':'NOT_RUN'},
            ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        progress('build_blocked',error=str(error),time_integration='NOT_RUN')
        raise

    (out/'hc_geometry.json').write_text(json.dumps(audit['spatial_profile'].get('HC_geometry',{'profile_id':'engineering_preview'}),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    config=SolverConfig(**selected['solver'])
    resolved={'schema_version':'1.0','execution':audit['resolved_execution_profile'],'light_context':audit['light_context'],
        'parameter_defaults':audit['registry'].runtime_profile.data,
        'resolved_parameters':{cid:serializable(c.parameters.records) for cid,c in compiled.components.items()},
        'source_records':serializable(tuple(audit['registry'].sources.values())),
        'geometry':audit['spatial_profile'],'rod_inner_geometry':audit['rod_inner_geometry'],'IPL_geometry':audit['IPL_geometry'],
        'resource_estimate':audit['resource_estimate'],'density_integrals':audit['density_integrals'],
        'regional_support':audit['regional_support'],'physical_reference_applicability':audit['physical_reference_applicability'],
        'glycine_source_condition':audit['glycine_source_condition']}
    (out/'resolved_profile.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'ipl_geometry.json').write_text(json.dumps(audit['IPL_geometry'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'owner_scopes.json').write_text(json.dumps(audit['population_owner_mask'],indent=2)+'\n',encoding='utf-8')
    phase_wall['build_seconds']=time.perf_counter()-wall_start
    runner=HybridRunner(compiled);state=runner.initialize(config)
    plan=RecordingPlan(max_bytes=selected['budgets'].get('max_recording_window_bytes',128*1024*1024))
    state.input_position={'background_context_hash':light.background_context_hash,'stimulus_origin_seconds':a.warmup}
    (out/'initial_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf-8')
    progress('initialized',0.,state_scalars=sum(s.shape[0] for s in runner.model.state_specs),components=len(compiled.components))
    preparation_report=None;stimulus_origin=a.warmup
    if a.background_initialization=='bounded_convergence':
        from retina_sim.background import constant_background_channels,prepare_background,rebase_stimulus_channels
        background=constant_background_channels(compiled,light,policy.max_sim_seconds+a.duration)
        state,preparation_report=prepare_background(runner,state,background,policy,
            on_window=lambda current,row:progress('background_preparation',current.time_seconds,maximum_norm=row['maximum_norm']))
        stimulus_origin=state.time_seconds;total=stimulus_origin+a.duration
        resolved['light_context']=dict(resolved['light_context'],requested_fixed_prerun_seconds=a.warmup,
            prerun_seconds=preparation_report['elapsed_seconds'],adaptation_status=preparation_report['status'],
            inner_initialization_mode='bounded_uniform_background_preparation',whole_network_steady_state_claim=False)
        resolved['background_initialization']=preparation_report
        (out/'resolved_profile.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (out/'background_initialization.json').write_text(json.dumps(preparation_report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (out/'prepared_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf-8')
        channels=rebase_stimulus_channels(channels,background,original_origin_seconds=a.warmup,
            stimulus_origin_seconds=stimulus_origin,stop_seconds=total)
        if policy.require_converged and not preparation_report['converged']:
            (out/'final_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf-8')
            (out/'summary.json').write_text(json.dumps({'engineering_execution':'INITIALIZATION_NOT_REACHED',
                'background_initialization':preparation_report,'windows':0,'last_valid_state_retained':True},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            progress('initialization_not_reached',state.time_seconds,status=preparation_report['status']);raise SystemExit(2)
    elif a.warmup:
        # Warmup is explicit continuous running, not a reset/new dark start per window.
        while state.time_seconds<a.warmup-1e-12:
            state,_=runner.advance(state,channels,min(a.warmup,state.time_seconds+a.window),recording_plan=RecordingPlan(ports=(),states=(),cell_ids=()))
            progress('warming_up',state.time_seconds,solver_metrics=dict(runner.integration_metrics))
    recorder=NPZWindowRecorder(out/'windows')
    while state.time_seconds<total-1e-12:
        stop=min(total,state.time_seconds+a.window)
        times=np.r_[np.arange(state.time_seconds,stop,a.sample_step),stop]
        times=np.array(sorted(set(float(t) for t in times)))
        state,result=runner.advance(state,channels,stop,sample_times=times,recording_plan=plan)
        recorder.consume(result)
        progress('recording',state.time_seconds,windows=recorder.windows,solver_metrics=dict(runner.integration_metrics))
    ck=runner.checkpoint(state);(out/'final_checkpoint.json').write_text(ck,encoding='utf-8')
    missing=[x for x in compiled.spec.required_mechanisms if not any(x in c.layers for c in compiled.components.values())]
    manifest={'model_id':compiled.spec.model_id,'assembly_hash':compiled.assembly_hash,'input_class':'synthetic','scientific_validation':'NOT_RUN','spec':serializable(compiled.spec),'parameter_evidence':{cid:serializable(c.parameters.records) for cid,c in compiled.components.items()},'source_records':serializable(tuple(audit['registry'].sources.values())),'geometry_and_route_audits':serializable({k:v for k,v in audit.items() if k not in ('registry','spec')}),'input_contracts':{k:v.input_contract for k,v in channels.items()},'background_initialization':preparation_report,'stimulus_origin_seconds':stimulus_origin,'grid_origin_seconds':state.grid_origin_seconds}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-wall_start,'phase_wall_seconds':phase_wall,'solver_metrics':dict(runner.integration_metrics),'invocation':vars(a)|{'output':str(out)},'purpose':'Short finite-resolution engineering execution; long durations available explicitly','engineering_execution':'PASS','unimplemented_target_layers':missing,'target_count':len(compiled.spec.required_mechanisms),'executed_components':len(compiled.components),'executed_layers':sorted(set(x for c in compiled.components.values() for x in c.layers)),'mechanism_bindings':{cid:{'mechanism':c.mechanism,'layers':list(c.layers),'profile_id':c.parameters.profile.profile_id} for cid,c in compiled.components.items()},'gap_contacts':sum(len(g.left_cell_ids) for g in compiled.spec.gap_junctions),'individual_first_synapse_gates':audit['first_synapse_contacts'],'chemical_synapse_edges':sum(x['per_edge_states'] for x in audit['chemical_synapses'].values()),'clock':compiled.clock,'windows':recorder.windows,'adaptation_interval_seconds':a.warmup,'recording_interval_seconds':[a.warmup,total],'checkpoint_sha256':hashlib.sha256(ck.encode()).hexdigest(),'input_class':'synthetic','parameter_classes_separate':True,'physiological_calibration':'NOT_COMPLETE','independent_animal_validation':'NOT_RUN','animal_fits':0,'runtime_code_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'retina_sim').glob('*.py')}}
    summary['invocation']['output']=str(out)
    summary.update(adaptation_interval_seconds=stimulus_origin,recording_interval_seconds=[stimulus_origin,total],
        stimulus_origin_seconds=stimulus_origin,grid_origin_seconds=state.grid_origin_seconds,
        background_initialization_status=preparation_report['status'] if preparation_report else 'FIXED_PRERUN_NO_CONVERGENCE_CLAIM')
    if missing:raise RuntimeError('Full target missing actual code layers: '+str(missing))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    progress('completed',state.time_seconds,windows=recorder.windows)
    print(json.dumps({'output':str(out),'target_layers':len(compiled.spec.required_mechanisms),'unimplemented':missing,'components':len(compiled.components),'windows':recorder.windows,'physiological_calibration':'NOT_COMPLETE'},ensure_ascii=False))

if __name__=='__main__':main()
