"""One 200us old/new input comparison from the existing prepared background."""
from pathlib import Path
from dataclasses import replace
import sys,json,hashlib,csv,time,argparse
import numpy as np

def main():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from retina_platform.config import RunConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.hybrid import HybridRunner
    from retina_sim.contracts import SolverConfig
    from retina_sim.background_checkpoint import BackgroundCheckpointStore
    from retina_sim.background import constant_background_channels
    from retina_sim.recording import RecordingPlan
    from retina_sim.offline_input import load_offline_inputs,rebase_offline_inputs,SCHEMA
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=root/'reports/offline_input_smoke_20261013')
    out=parser.parse_args().output
    if out.exists() and any(out.iterdir()):raise FileExistsError('Existing smoke files are preserved; select another output directory')
    out.mkdir(exist_ok=True)
    protect=[p for p in (root/'configurations').glob('*.json')]+[root/'retina_sim'/n for n in ['vh2005.py','rod_pathway.py','inner.py','hybrid.py','background.py']]
    before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protect}
    cfg=RunConfig.project_default(root);light=cfg.light_context();wall=time.perf_counter()
    compiled,_,audit=atlas_population_network(root,duration=.002,light_context=light,owner_bounds_mm=cfg.owner_bounds(),
        halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed)
    runner=HybridRunner(compiled);reference=runner.initialize(SolverConfig(**audit['resolved_execution_profile']['solver']))
    store=BackgroundCheckpointStore(root,runner,reference,light,cfg.background_policy())
    cache=root/cfg.background_checkpoint_path;state,cached=store.load(cache,require_converged=True)
    origin=state.time_seconds;checkpoint=runner.checkpoint(state);checkpoint_sha=hashlib.sha256(checkpoint.encode()).hexdigest()
    previous=root/'reports/population_background_runs/20261012_authorized/bright'
    saved=json.loads((previous/'summary.json').read_bytes())
    inputs={'native_light':{'unit':'td','value':saved['input']['native_td']},'rod_light':{'unit':'R*/s','value':saved['input']['rod_Rstar_s']}}
    assert saved['same_background_checkpoint_sha256']==checkpoint_sha
    duration=.0002;edges=np.arange(3)*.0001;arrays={};declarations={}
    ref_id=audit['atlas_runtime']['source_atlas']['reference_id']
    for external in compiled.spec.external_inputs:
        name=external.channel;component=compiled.components[external.target_component];port=component.inputs[external.target_port]
        pid=next(k for k,p in compiled.populations.items() if tuple(p.cells.cell_id)==tuple(port.cell_ids));pop=compiled.populations[pid]
        value=inputs[name]['value'];counts=port.unit=='R*/s';unit='R*/window' if counts else port.unit
        arrays[name+'_positions']=pop.cells.rf_position.values.copy();arrays[name+'_time']=edges.copy()
        arrays[name+'_values']=(np.full((2,len(port.cell_ids)),value)*np.diff(edges)[:,None] if counts else np.full((3,len(port.cell_ids)),value))
        declarations[name]={'population_id':pid,'cell_ids':list(port.cell_ids),'data_axes':['time','cell'],'coordinate_role':'rf',
            'coordinate_unit':'mm','coordinate_frame':'functional_visual_field','reference_id':ref_id,'coordinate_registration':'existing_project_RF_verbatim',
            'positions_key':name+'_positions','values_key':name+'_values','unit':unit,'input_stage':'photoisomerization' if counts else 'retinal_illuminance',
            'evidence_tag':port.evidence_tag,'value_semantics':'expected','photon_noise':{'owner':'none','model':'none'},
            'mechanism_ownership':{'photopigment_capture':'upstream' if counts else 'consumer','phototransduction':'consumer','neural_noise':'consumer'},
            'background':{'includes_background':True,'unit':port.unit,'value':cfg.rod_background_rstar_s if counts else cfg.native_background_td,'source_id':light.background_source},
            'prehistory':{'kind':'declared_uniform_background','stop_seconds':0.},
            'time':{'representation':'integration_windows' if counts else 'samples','reference':'stimulus_relative_seconds','source_origin_seconds':origin,
                'start_seconds':0.,'stop_seconds':duration,'axis_key':name+'_time','interpolation':'zero_order_hold' if counts else 'linear','source_grid_step_seconds':.0001}}
    manifest={'schema':SCHEMA,'version':'1.0','producer':{'name':'Previously saved controlled bright input','version':'20261012',
        'source_id':'existing_bright_summary','source_sha256':hashlib.sha256((previous/'summary.json').read_bytes()).hexdigest(),
        'scope':'Engineering compatibility fixture, no new physiological experiment'},'background_context_hash':light.background_context_hash,'channels':declarations}
    filename=out/'input.npz';np.savez_compressed(filename,manifest_json=np.array(json.dumps(manifest)),**arrays)
    offline=load_offline_inputs(filename,compiled,light,reference_id=ref_id)
    (out/'source_summary.json').write_text(json.dumps(offline.source_summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    stop=origin+duration
    background=constant_background_channels(compiled,light,stop+.001)
    channels=rebase_offline_inputs(offline,compiled,light,stimulus_origin_seconds=origin,grid_origin_seconds=state.grid_origin_seconds)
    old=constant_background_channels(compiled,replace(light,native_background_td=inputs['native_light']['value'],rod_background_rstar_s=inputs['rod_light']['value']),stop+.001)
    times=origin+np.arange(3)*.0001
    for i,t in enumerate(times):
        coincident=[state.grid_origin_seconds+round((t-state.grid_origin_seconds)/p)*p for p in runner.periods.values() if abs(state.grid_origin_seconds+round((t-state.grid_origin_seconds)/p)*p-t)<1e-12]
        if coincident:times[i]=min(coincident)
    times[0]=origin;times[-1]=stop
    for name in channels:
        for t in times:np.testing.assert_allclose(channels[name].at(float(t)).values,old[name].at(float(t)).values,rtol=2e-15,atol=0)
        np.testing.assert_array_equal(channels[name].at(origin-.0001).values,background[name].at(origin-.0001).values)
    outputs=[];states=[]
    for selected in (old,channels):
        initial,_=store.load(cache,require_converged=True);assert runner.checkpoint(initial)==checkpoint
        initial.input_position.update(phase='offline_compatibility_smoke',stimulus_origin_seconds=origin)
        final,result=runner.advance(initial,selected,stop,sample_times=times,recording_plan=RecordingPlan(states=(),max_bytes=64*1024**2))
        outputs.append(result.outputs_with_units);states.append(final)
    maximum=0.
    for a,b in zip(states[0].ordered_state_arrays,states[1].ordered_state_arrays):
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14);maximum=max(maximum,float(np.max(np.abs(a-b))))
    assert states[0].spike_states==states[1].spike_states and states[0].component_tick_indices==states[1].component_tick_indices
    for name in outputs[0]:np.testing.assert_allclose(outputs[0][name].values,outputs[1][name].values,rtol=1e-12,atol=1e-14)
    saved_errors={}
    for name in ['midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON']:
        with (previous/(name+'_cell0_voltage.csv')).open(encoding='utf8') as stream:rows=list(csv.DictReader(stream))[:3]
        expected=np.array([float(r['voltage_V']) for r in rows]);actual=outputs[1][name+'.voltage'].values[0,:]
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)
        saved_errors[name]=float(np.max(np.abs(actual-expected)))
    assert before=={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protect}
    report={'status':'PASS','components':len(compiled.components),'cached_background_seconds':origin,'cached_checkpoint_sha256':checkpoint_sha,
        'each_smoke_duration_seconds':duration,'old_new_runs':2,'maximum_state_difference':maximum,'saved_first_window_voltage_errors_V':saved_errors,
        'state_rng_history_clock_preserved':True,'protected_files_unchanged':before,'wall_seconds':time.perf_counter()-wall,
        'new_background_preparation':False,'physiological_parameter_changes':False,'comparison_scope':'Offline engineering interface only; previously saved short bright response reused'}
    (out/'check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='protected_files_unchanged'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
