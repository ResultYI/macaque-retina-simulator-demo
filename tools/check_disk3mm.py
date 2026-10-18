"""Minimum representative assembly and real input/response checks.

Matched responses retain the existing numerical background criteria. This does
not compare physiology or adopt defaults. Actual blockers are never zero-filled.
"""
from pathlib import Path
import argparse,gc,json,os,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.disk_graph_storage import save_json

REGIONS={'central':(0.,0.,.02),'transition':(1.,0.,.08),'near_peripheral':(2.85,0.,.1)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region',choices=['all',*REGIONS],default='all')
    parser.add_argument('--circuit-configuration',type=Path,default=Path('configurations/atlas_disk3mm_v1.json'))
    parser.add_argument('--qualification-only',action='store_true',help='Inspect exact graph dependencies without assembly, optical capture or neural execution')
    parser.add_argument('--capture-transient',action='store_true',help='One100us real ISET/neural window per supported region; explicitly unadapted')
    parser.add_argument('--media',type=Path)
    parser.add_argument('--output',type=Path,default=Path('reports/disk3mm_validation_20261016'))
    parser.add_argument('--figure',action='store_true',help='One final overview from actual outputs')
    parser.add_argument('--check-pathway-response',action='store_true',help='Matched20ms static-image/background response after existing background convergence on each selected region')
    parser.add_argument('--resource-profile',type=Path,help='Explicit execution-resource budgets; numerical convergence criteria remain unchanged')
    parser.add_argument('--resume-background',type=Path,help='Resume the identity-matching full background checkpoint for a single region')
    parser.add_argument('--stop-on-blocker',action='store_true',help='Retain the first blocked region and stop before starting another expensive region')
    args=parser.parse_args();output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Validation output must stay inside project')
    config_path=(ROOT/args.circuit_configuration).resolve()
    if not config_path.is_relative_to(ROOT):raise ValueError('Circuit configuration must stay inside project')
    if not config_path.exists():raise RuntimeError('Production source geometry is still pending; existing generator/checkpoints are retained')
    if (args.capture_transient or args.check_pathway_response) and (args.media is None or not args.media.is_file()):raise ValueError('Provide actual media for the ISET check')
    if output.exists():raise FileExistsError('Validation delivery retained; use a new output directory for an authorized repair rerun')
    output.mkdir(parents=True);results={};traces={};responses={}
    from retina_platform.config import RunConfig
    from retina_sim.disk_circuit import GlobalDiskCircuit
    from retina_sim.atlas_population_network import atlas_population_network
    configuration=json.loads(config_path.read_bytes());circuit=GlobalDiskCircuit(ROOT,configuration['global_graph'])
    resource_values={}
    if args.resource_profile:
        resource_path=(ROOT/args.resource_profile).resolve()
        if not resource_path.is_relative_to(ROOT):raise ValueError('Resource profile outside project')
        resource=json.loads(resource_path.read_bytes())
        resource_values=resource['run_config']
        allowed={'background_memory_limit_mb','background_min_available_mb','background_max_wall_seconds','background_checkpoint_interval_seconds','neural_integration_backend'}
        if resource.get('schema')!='disk3mm_validation_resources_v1' or not set(resource_values)<=allowed:
            raise ValueError('Resource profile may only select explicit resources and a named numerical backend')
        for key,value in resource.get('process_environment',{}).items():
            if key!='OPENBLAS_NUM_THREADS' or value!='1' or os.environ.get(key)!=value:
                raise ValueError('Launch process with the explicitly declared resource environment; no silent runtime override')
    if args.resume_background and args.region=='all':raise ValueError('Resume must name exactly one original region')
    for name in REGIONS if args.region=='all' else [args.region]:
        compiled=initial_channels=audit=frontend=runner=state=result=view=archive=media=None
        x,y,width=REGIONS[name];bounds=(x-width/2,x+width/2,y-width/2,y+width/2);directory=output/name;directory.mkdir()
        started=time.monotonic();report={'region':name,'bounds_RF_mm':bounds,'global_graph':circuit.binding,
            'background_preparation':'NOT_PERFORMED',
            'steady_state_claim':False,'physiological_validation':False,'neural_windows':0,'default_adopted':False}
        try:
            print('REGION_GRAPH_QUALIFICATION',name,flush=True)
            view=circuit.owner_view(bounds,empty_chemical_input_policy=configuration.get('empty_chemical_input_policy','reject_empty_chemical_input_v1'))
            print('REGION_GRAPH_QUALIFIED',name,'nodes='+str(view['global_node_count']),'ready='+str(view['runtime_ready']),flush=True)
            report.update(selected_global_nodes=view['global_node_count'],empty_output_samples=view['empty_output_samples'],
                unsupported_rows=view['unsupported_rows'],empty_chemical_input_policy=view['empty_chemical_input_policy'],
                declared_empty_chemical_sums=view['declared_empty_chemical_sums'])
            if args.qualification_only:
                report['status']='GRAPH_QUALIFICATION_PASS_PENDING_RUNTIME' if view['runtime_ready'] else 'SOURCE_GEOMETRY_UNSUPPORTED'
                report['claim']='Exact frozen graph qualification only; no optical capture, background preparation or neural execution'
                results[name]=report;save_json(directory/'check.json',report);view=None;gc.collect()
                if args.stop_on_blocker and not report['status'].startswith('GRAPH_QUALIFICATION_PASS'):break
                continue
            if not view['runtime_ready']:
                report['status']='SOURCE_GEOMETRY_UNSUPPORTED';results[name]=report;save_json(directory/'check.json',report)
                if args.stop_on_blocker:break
                continue
            config=RunConfig.from_dict({'execution_profile':'atlas_isetbio_geometry_v1','circuit_configuration_path':str(config_path.relative_to(ROOT)),
                'owner_x_mm':x,'owner_y_mm':y,'owner_width_mm':width,'owner_height_mm':width,'max_spatial_cells':2000000,
                'duration_seconds':.0001,'window_seconds':.0001,'sample_seconds':.0001,'warmup_seconds':0.,
                'media_path':'' if args.media is None else str(args.media.resolve()),
                'background_checkpoint_path':'' if args.resume_background is None else str(args.resume_background.resolve()),**resource_values})
            print('REGION_ASSEMBLY_BEGIN',name,flush=True)
            compiled,initial_channels,audit=atlas_population_network(ROOT,duration=.0001,light_context=config.light_context(),owner_bounds_mm=bounds,
                halo_mm=.04,max_spatial_cells=config.max_spatial_cells,seed=config.seed,execution_profile=config.execution_profile,
                configuration_path=config.circuit_configuration_path)
            report.update(status='ASSEMBLY_PASS',components=len(compiled.components),populations={k:len(p.cells.cell_id) for k,p in compiled.populations.items()},
                actual_geometry=audit['atlas_runtime']['compiled_geometry_identity'])
            print('REGION_ASSEMBLY_PASS',name,'components='+str(len(compiled.components)),flush=True)
            if args.capture_transient or args.check_pathway_response:
                from retina_platform.media import MediaSource
                from retina_platform.backend import _gaze
                from retina_platform.disk_frontend import DiskMixedReferenceFrontend
                from retina_platform.disk_run_audit import prepare_global_audit
                from retina_sim.execution_backends import make_runner
                from retina_sim.contracts import SolverConfig
                from retina_sim.recording import RecordingPlan,NPZWindowRecorder
                media=MediaSource(config.media_path);gaze,_=_gaze(config)
                frontend=DiskMixedReferenceFrontend(config,compiled,media,gaze,ROOT,input_path=directory/'input.npz',on_status=lambda value:print(name,value,flush=True))
                archive=prepare_global_audit(compiled,audit,directory)
                save_json(directory/'resolved_profile.json',{'execution':audit['resolved_execution_profile'],
                    'parameter_profile':config.parameter_profile,'actual_values':archive['parameter_evidence'],
                    'global_graph':circuit.binding,'actual_geometry':report['actual_geometry'],
                    'source_spec':archive['source_spec'],'physiological_validation':False})
                save_json(directory/'manifest.json',{'schema':'disk3mm_representative_run_v1','configuration':config.to_dict(),
                    'assembly_hash':compiled.assembly_hash,'source_spec':archive['source_spec'],
                    'actual_links':[{'producer':q.specification.source_component,'port':q.specification.source_port,
                        'consumer':q.specification.target_component,'input':q.specification.target_port,
                        'shape':list(q.matrix.shape),'edges':q.matrix.nnz} for q in compiled.links],
                    'ISET':frontend.summary,'physiological_validation':False})
                try:
                    frontend.set_origin(0.,0.)
                    if args.check_pathway_response:
                        from disk3mm_response_check import paired_response
                        response,traces[name]=paired_response(config,compiled,audit,frontend,media,directory/'matched_response')
                        responses[name]=dict(response,region=name)
                        report.update(status='ISET_NEURAL_MATCHED_RESPONSE_PASS_NUMERICALLY_PREPARED',neural_windows=2,
                            elapsed_sim_seconds=response['duration_seconds'],stimulus_origin_seconds=response['stimulus_origin_seconds'],
                            background_preparation='CONVERGED_UNDER_EXISTING_NUMERICAL_CRITERIA',
                            background_checkpoint=response['background_checkpoint'],branch_restores=response['branch_restores'])
                    else:
                        runner=make_runner(compiled,config.neural_integration_backend)
                        state=runner.initialize(SolverConfig(**audit['resolved_execution_profile']['solver']))
                        state.input_position={'background_context_hash':config.light_context().background_context_hash,'stimulus_origin_seconds':0.}
                        state,result=runner.advance(state,frontend.window(0.,.0001),.0001,sample_times=np.array([0.,.0001]),
                            recording_plan=RecordingPlan(states=(),cell_ids=compiled.platform_recording_cell_ids,max_bytes=64*1024**2))
                        if not all(np.all(np.isfinite(q.values)) for q in result.outputs_with_units.values()):raise RuntimeError('Nonfinite actual transient output')
                        NPZWindowRecorder(directory/'windows').consume(result)
                        report.update(status='ISET_NEURAL_TRANSIENT_PASS_UNADAPTED',neural_windows=1,elapsed_sim_seconds=state.time_seconds)
                        traces[name]={key:(result.outputs_with_units['time'].values,q.values,q.unit) for key,q in result.outputs_with_units.items()
                            if 'RGC' in key and key.endswith('.voltage')}
                    report.update(ISET=frontend.summary,stochastic_statistics='INSUFFICIENT_SHORT_REPRESENTATIVE_WINDOWS')
                finally:
                    media.close()
        except Exception as error:
            report.update(status='BLOCKED',error=type(error).__name__+': '+str(error))
            background_report=directory/'matched_response/background_initialization.json'
            if background_report.exists():
                report['background_preparation']=json.loads(background_report.read_bytes())['status']
            if args.check_pathway_response:
                responses[name]={'status':'BLOCKED','region':name,'error':report['error']}
        report['wall_seconds']=time.monotonic()-started;results[name]=report;save_json(directory/'check.json',report)
        compiled=initial_channels=audit=frontend=runner=state=result=view=archive=media=None
        gc.collect()  # Sequential ROIs must not retain the previous full dependency view.
        save_json(output/'delivery.json',{'regions':results,'full3mm_runtime_complete':False,'default_changed':False,
            'claim':'Representative mechanism/input verification only; no steady-state or physiological calibration claim'})
        if args.stop_on_blocker and report['status'] in ('BLOCKED','SOURCE_GEOMETRY_UNSUPPORTED'):break
    save_json(output/'delivery.json',{'regions':results,'matched_pathway_responses':responses,'full3mm_runtime_complete':False,'default_changed':False,
        'claim':'Representative mechanism/input verification only; no steady-state or physiological calibration claim'})
    if args.figure:
        # Use the existing GUI Pillow renderer; the approved environment has
        # no matplotlib, and verification must not install or upgrade it.
        from PIL import Image,ImageDraw
        from retina_platform.plotting import curve_plot,font
        height=100+250*max(1,max((len(traces.get(name,{})) for name in results),default=0))
        figure=Image.new('RGB',(480*len(results),height),'white');draw=ImageDraw.Draw(figure)
        draw.text((14,10),'Actual representative outputs; numerical convergence is not physiological validation',fill='#202a35',font=font(14))
        for column,(name,report) in enumerate(results.items()):
            x=column*480;draw.text((x+14,40),name+' / '+report['status'],fill='#202a35',font=font(10))
            for row,(key,(times,values,unit)) in enumerate(traces.get(name,{}).items()):
                if not len(values):continue
                scale=1000 if unit=='V' else 1
                panel=curve_plot(times,np.mean(values,axis=0)*scale,(480,250),title=key,
                    unit=('stimulus-background ' if args.check_pathway_response else '')+('mV' if unit=='V' else unit))
                figure.paste(panel,(x,85+250*row))
        figure.save(output/'response.png')
    print(json.dumps({name:value['status'] for name,value in results.items()}),flush=True)

if __name__=='__main__':main()
