"""One bounded preparation on the production graph, then same-state bright/dark steps."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
import argparse,json,sys,time,hashlib
import numpy as np


def run_steps(root,out,runner,store,bundle,config,audit,guard):
    from retina_sim.background import constant_background_channels
    from retina_sim.background_checkpoint import atomic_json
    from retina_sim.recording import RecordingPlan,NPZWindowRecorder
    from retina_platform.backend import _components,_runtime_ports
    from retina_platform.results import ResultStore
    from retina_platform.plotting import curve_plot,font
    from PIL import Image,ImageDraw
    duration=.020;sample=.0001;branches={};recorded={};source_bytes=runner.checkpoint(store.load(bundle,require_converged=True)[0])
    source_sha=hashlib.sha256(source_bytes.encode()).hexdigest()
    for label,factor in [('bright',1.1),('dark',.9)]:
        if not guard.check():raise RuntimeError('Step memory safety stop: '+str(guard.last))
        state,_=store.load(bundle,require_converged=True);origin=state.time_seconds
        assert runner.checkpoint(state)==source_bytes
        state.input_position.update(phase='controlled_step',stimulus_origin_seconds=origin,step_factor=factor)
        path=out/label;path.mkdir();(path/'initial_checkpoint.json').write_text(source_bytes,encoding='utf8')
        light=replace(config.light_context(),native_background_td=config.native_background_td*factor,
            cone_background_rstar_s=config.cone_background_rstar_s*factor,rod_background_rstar_s=config.rod_background_rstar_s*factor)
        channels=constant_background_channels(runner.compiled,light,origin+duration+.001)
        manifest={'schema':'retina-platform-1','config':config.to_dict(),'required_layers':list(runner.compiled.spec.required_mechanisms),
            'components':_components(runner.compiled,audit['population_owner_mask']),'atlas_runtime':audit['atlas_runtime'],
            'assembly_hash':runner.compiled.assembly_hash,'scientific_validation':'NOT_RUN','media':{'kind':'controlled_uniform_step'},
            'recording':{'sample_seconds':sample,'window_seconds':.001},'runtime':{'stimulus_origin_seconds':origin,'grid_origin_seconds':state.grid_origin_seconds},
            'same_background_checkpoint_sha256':source_sha,'input':{'class':'explicit_independent_consumer_uniform_step','factor':factor,
                'native_td':light.native_background_td,'cone_Rstar_s':light.cone_background_rstar_s,'rod_Rstar_s':light.rod_background_rstar_s,
                'applied_external_channels':{k:{'unit':v.unit,'meaning':v.meaning,'value':v.input_contract['value']} for k,v in channels.items()},
                'physical_cone_Rstar_reference_used':'physical_cone_light' in channels,
                'duration_seconds':duration,'physical_scene_calibration':False,'parameter_changes':False}}
        _runtime_ports(manifest,runner.compiled,state);atomic_json(path/'manifest.json',manifest)
        recorder=NPZWindowRecorder(path/'windows');wall=time.perf_counter();start_metrics=dict(runner.integration_metrics)
        for j in range(20):
            if time.perf_counter()-wall>=120. or not guard.check():raise RuntimeError('Controlled step safety budget reached')
            stop=origin+(j+1)*.001;times=origin+np.arange(j*10,(j+1)*10+1)*sample
            for i,t in enumerate(times):
                coincident=[state.grid_origin_seconds+round((t-state.grid_origin_seconds)/p)*p for p in runner.periods.values() if abs(state.grid_origin_seconds+round((t-state.grid_origin_seconds)/p)*p-t)<1e-12]
                if coincident:times[i]=min(coincident)
            times[0]=state.time_seconds;times[-1]=stop
            state,result=runner.advance(state,channels,stop,sample_times=times,recording_plan=RecordingPlan(states=(),max_bytes=64*1024**2))
            assert all(np.all(np.isfinite(q.values)) for q in result.outputs_with_units.values())
            recorder.consume(result)
            (path/'latest_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf8')
        final=runner.checkpoint(state);(path/'final_checkpoint.json').write_text(final,encoding='utf8')
        summary={'status':'completed','simulation_time_seconds':state.time_seconds,'stimulus_origin_seconds':origin,'step_duration_seconds':duration,
            'components':len(runner.compiled.components),'required_layers':manifest['required_layers'],'unimplemented_layers':[],
            'windows':recorder.windows,'state_reset_between_windows':False,'wall_seconds':time.perf_counter()-wall,
            'checkpoint_sha256':hashlib.sha256(final.encode()).hexdigest(),'same_background_checkpoint_sha256':source_sha,
            'scientific_validation':'NOT_RUN','input':manifest['input']}
        atomic_json(path/'summary.json',summary);branches[label]=summary
        rs=ResultStore(path);recorded[label]={k:rs.history(k+'.voltage',0) for k in
            ('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON')}
        print(json.dumps({'phase':'step_completed','branch':label,'path':str(path),'wall_seconds':summary['wall_seconds']}),flush=True)
    image=Image.new('RGB',(1200,1360),'#f8fafc');draw=ImageDraw.Draw(image)
    draw.text((18,10),'Same saved background · +10% / -10% input · recorded ΔV in µV · 20 ms',font=font(16),fill='#243447')
    for row,key in enumerate(recorded['bright']):
        for col,label in enumerate(('bright','dark')):
            t,y=recorded[label][key];image.paste(curve_plot(t-t[0],(y-y[0])*1e6,(600,260),title=f'{label}: {key}, cell 0, ΔV / µV',unit='µV'),(600*col,42+260*row))
    draw.text((18,1342),'µV is a unit conversion of saved V; relative onset time. Numerical response, no calibration claim.',font=font(11),fill='#64748b')
    image.save(out/'bright_dark_steps.png');return branches


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--wall-seconds',type=float,default=1800.);p.add_argument('--memory-mb',type=float,default=4096.)
    args=p.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from retina_platform.config import RunConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.hybrid import HybridRunner
    from retina_sim.contracts import SolverConfig
    from retina_sim.background_checkpoint import MemoryGuard,prepare_with_checkpoint,atomic_json,BackgroundCheckpointStore
    out=(args.output or root/'reports/population_background_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=RunConfig.project_default(root);cfg.media_path=str(root/'examples/platform_media/synthetic_rgb.png')
    cfg.background_initialization_mode='bounded_convergence';cfg.background_require_converged=True;cfg.background_max_wall_seconds=args.wall_seconds
    cfg.background_memory_limit_mb=args.memory_mb;cfg.validate();policy=cfg.background_policy();light=cfg.light_context()
    resume=args.resume or (Path(cfg.background_checkpoint_path) if cfg.background_checkpoint_path else None)
    if resume is not None and not resume.is_absolute():resume=root/resume
    guard=MemoryGuard(int(args.memory_mb*1024**2),int(cfg.background_min_available_mb*1024**2))
    if not guard.check():raise RuntimeError('Insufficient initial memory reserve')
    compiled,_,audit=atlas_population_network(root,duration=policy.max_sim_seconds+1.,light_context=light,
        owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed)
    runner=HybridRunner(compiled);reference=runner.initialize(SolverConfig(**audit['resolved_execution_profile']['solver']))
    store=BackgroundCheckpointStore(root,runner,reference,light,policy);coverage=store.coverage();guard.check()
    request={'authorization':'Sentinel_6a4313c8bed48191ba4aaec294527b8a','configuration':cfg.to_dict(),'artifact':audit['atlas_runtime']['graph_binding'],
        'coverage':coverage,'binding':store.binding,'initial_memory':guard.last,'wall_limit_seconds':args.wall_seconds,
        'memory_limit_bytes':guard.max_rss_bytes,'checks_are_cooperative_at_valid_state_boundaries':True,'resume_source':str(resume) if resume else None,
        'steps_only_after_convergence':{'bright_factor':1.1,'dark_factor':.9,'duration_seconds_each':.02,'branch_wall_cap_seconds':120.},
        'physiological_validation':'NOT_RUN','AII_capacity_deficit_retained':True,'HC_gap_unchanged':True}
    atomic_json(out/'request.json',request);print(json.dumps({'phase':'background_started','path':str(out),'state_blocks':coverage['state_blocks'],
        'covered_blocks':coverage['covered_blocks'],'minimum_seconds':coverage['timing_requirements']['minimum_required_seconds'],'memory':guard.last,'wall_limit_seconds':args.wall_seconds}),flush=True)
    def progress(state,row):
        atomic_json(out/'progress.json',dict(row,time_seconds=state.time_seconds))
        last=row.get('last_window') or {};print(json.dumps({'phase':row['phase'],'time_seconds':state.time_seconds,'wall_seconds':row['wall_seconds'],
            'maximum_norm':last.get('maximum_norm'),'dominant':last.get('dominant'),'memory':row['memory']}),flush=True)
    state,report,store=prepare_with_checkpoint(root,runner,reference,light,policy,out/'background_checkpoint.json',resume_path=resume,guard=guard,on_progress=progress)
    atomic_json(out/'background_initialization.json',report);outcome={'background':report,'steps':'NOT_RUN_NOT_CONVERGED','path':str(out)}
    atomic_json(out/'outcome.json',outcome)
    if report['converged']:
        outcome['steps']=run_steps(root,out,runner,store,out/'background_checkpoint.json',cfg,audit,guard)
        atomic_json(out/'outcome.json',outcome)
    print(json.dumps({'phase':'finished','status':report['status'],'time_seconds':state.time_seconds,'budget_exit':report['budget_exit'],
        'dominant':report['dominant'],'path':str(out),'steps_executed':report['converged']}),flush=True)

if __name__=='__main__':main()
