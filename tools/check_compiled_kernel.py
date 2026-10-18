"""Finite 20 ms Python/Numba comparison; cached input/state, no MATLAB/prewarm."""
from pathlib import Path
from dataclasses import replace
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.profile_bk_window import BK,PRIOR,sha,save,zero_metrics
from tools.check_runtime_optimizations import structure,difference
from retina_platform.config import RunConfig
from retina_sim.atlas_population_network import atlas_population_network
from retina_sim.contracts import SolverConfig,canonical_hash,ContractError
from retina_sim.hybrid import HybridRunner,RetinaState
from retina_sim.network import NetworkStimulus
from retina_sim.offline_input import load_offline_inputs,rebase_offline_inputs
from retina_sim.recording import RecordingPlan

OUT=ROOT/'reports/compiled_kernel_20ms_20261015'
BASE=ROOT/'reports/bk_optimization_20ms_20261015'

def transfer(runner,raw,solver):
    reference=runner.initialize(solver);p=raw['payload'];d=p['dynamic']['state']
    assert d['state_spec_hash']==reference.state_spec_hash and d['parameter_hash']==reference.parameter_hash
    # The archived initial checkpoint predates the preceding VH context-only
    # optimization too. Assembly-derived identities differ; the caller verifies
    # every numerical graph/parameter/layout field via the structural signature.
    dynamic=replace(reference.dynamic,time_seconds=d['time_seconds'],ordered_state_arrays=tuple(np.asarray(x) for x in d['ordered_state_arrays']))
    return RetinaState(dynamic,runner.compiled.assembly_hash,p['grid_origin_seconds'],p['sample_index'],
        json.loads(json.dumps(p['spike_states'])),json.loads(json.dumps(p['delay_states'])),
        json.loads(json.dumps(p['input_position'])),json.loads(json.dumps(p['link_delay_states'])),dict(p['component_tick_indices']))

def event_comparison(a,b):
    structure_equal=True;maximum=0.;count=0
    for cid,events in a.items():
        if len(events)!=len(b[cid]):structure_equal=False;continue
        for x,y in zip(events,b[cid]):
            count+=1
            structure_equal &= all(x[k]==y[k] for k in x if k!='time_seconds')
            maximum=max(maximum,abs(x['time_seconds']-y['time_seconds']))
    return {'count':count,'cell_order_counts_and_bin_endpoints_exact':bool(structure_equal),'event_time_max_absolute_seconds':maximum,
        'whole_events_exact':a==b}

def final_state_comparison(comparisons):
    a=json.loads((OUT/'python_checkpoint.json').read_bytes())['payload']
    b=json.loads((OUT/'numba_checkpoint.json').read_bytes())['payload']
    names=[k for k in comparisons if k.startswith('state:')]
    aa=a['dynamic']['state']['ordered_state_arrays'];bb=b['dynamic']['state']['ordered_state_arrays']
    assert len(names)==len(aa)==len(bb)
    states={}
    for key,x,y in zip(names,aa,bb):
        metric=difference(x,y);metric['unit']=comparisons[key]['unit']
        metric['max_scaled_error']=metric['max_absolute']/comparisons[key]['trajectory_scale'];states[key]=metric
    clocks={k:canonical_hash(a[k])==canonical_hash(b[k]) for k in ['grid_origin_seconds','sample_index','component_tick_indices','input_position','link_delay_states']}
    delays={}
    for key,x in a['delay_states'].items():
        y=b['delay_states'][key]
        delays[key]={'history_times_exact':x['history_times']==y['history_times'],
            'history_values':difference(x['history_values'],y['history_values'])}
    assert max(x['max_scaled_error'] for x in states.values())<1e-10 and all(clocks.values())
    assert all(x['history_times_exact'] for x in delays.values())
    return {'state_blocks':states,'clock_input_and_link_history_exact':clocks,'observation_delay_history':delays}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():raise RuntimeError('Completed evidence already exists')
    old=json.loads(PRIOR.read_bytes());cfg=RunConfig.from_dict(old['config']);light=cfg.light_context()
    source=json.loads((BK/'manifest.json').read_bytes())['input_segments'][0]['source'];packet_path=Path(source['source_file'])
    assert sha(packet_path)==source['source_sha256']
    with np.load(packet_path,allow_pickle=False) as data:manifest=json.loads(data['manifest_json'].item())
    new_light=replace(light,calibration='approved_mixed_reference:'+manifest['producer']['frontend_signature'])
    compiled,_,audit=atlas_population_network(ROOT,duration=.075,light_context=light,owner_bounds_mm=cfg.owner_bounds(),
        halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed)
    solver=SolverConfig(**audit['resolved_execution_profile']['solver'])
    python=HybridRunner(compiled,compute_backend='python');native=HybridRunner(compiled,compute_backend='numba')
    print('COMPILED',json.dumps(native.model.compute_metadata),flush=True)
    if not (OUT/'first_compile.json').exists():save(OUT/'first_compile.json',native.model.compute_metadata)
    assert structure(python)==structure(native)==json.loads((BASE/'before.json').read_bytes())['structural_signature']
    raw=json.loads((BASE/'initial_checkpoint.json').read_bytes());assert canonical_hash(raw['payload'])==raw['payload_hash']
    bundle=load_offline_inputs(packet_path,compiled,new_light,reference_id=manifest['channels']['native_light']['reference_id'])
    states={name:transfer(runner,raw,solver) for name,runner in [('python',python),('numba',native)]}
    origin=states['python'].time_seconds;stop=round((origin+.02)/.0001)*.0001
    channels=rebase_offline_inputs(bundle,compiled,new_light,stimulus_origin_seconds=origin,grid_origin_seconds=states['python'].grid_origin_seconds)
    stimulus=NetworkStimulus(compiled,channels,states['python'],stop,solver.input_interpolation,python.model.evidence_tag)
    y=python._flat(states['python']);z=y.copy();z[0]*=1.005;z[2*1465]*=1.001
    point=[]
    for a in (y,z,y):
        b=stimulus.at(origin);ref=python.model.rhs_flat(origin,a,b,python.model.parameters,python.model.graph)
        got=native.model.rhs_flat(origin,a,b,native.model.parameters,native.model.graph)
        scale=np.maximum(np.abs(ref),1.)
        error=float(np.max(np.abs(ref-got)/scale));point.append({'max_scaled_error':error,'max_abs_error':float(np.max(np.abs(ref-got)))})
        assert error<1e-10,point
    print('RHS_POINTS_PASS',point,flush=True)
    times=np.arange(round(origin/.001),round(stop/.001)+1)*.001;times[0]=origin;times[-1]=stop
    # Same recording work for both paths; all state/output trajectories at 1 ms.
    plan=RecordingPlan(max_bytes=64*1024**2)
    records={};finals={};results={};schedules={}
    for name,runner in [('python',python),('numba',native)]:
        stamp=OUT/(name+'.json');data_path=OUT/(name+'_outputs.npz');checkpoint_path=OUT/(name+'_checkpoint.json')
        if stamp.exists():
            records[name]=json.loads(stamp.read_bytes());finals[name]=runner.restore(checkpoint_path.read_text());continue
        schedule=[];zero_metrics(runner);tick=time.perf_counter()
        final,result=runner.advance(states[name],channels,stop,sample_times=times,recording_plan=plan,
            state_observer=lambda t,y,b:schedule.append(t))
        wall=time.perf_counter()-tick;serialized=runner.checkpoint(final)
        assert runner.checkpoint(runner.restore(serialized))==serialized
        checkpoint_path.write_text(serialized,encoding='utf8')
        np.savez_compressed(data_path,**{k:q.values for k,q in result.outputs_with_units.items()})
        row={'wall_seconds':wall,'integration_metrics':dict(runner.integration_metrics),'compute_backend':runner.model.compute_metadata,
            'schedule_times':schedule,'spike_events':result.metadata['spike_events'],
            'units':{k:q.unit for k,q in result.outputs_with_units.items()},'checkpoint_roundtrip_exact':True}
        save(stamp,row);records[name]=row;finals[name]=final
        print(name.upper()+'_COMPLETE',wall,runner.integration_metrics,flush=True)
    comparisons={};maximum_scaled=0.
    with np.load(OUT/'python_outputs.npz') as a,np.load(OUT/'numba_outputs.npz') as b:
        assert set(a.files)==set(b.files)
        for key in a.files:
            metric=difference(a[key],b[key]);unit=records['python']['units'][key]
            floor={'V':.001,'A':1e-12,'S':1e-12,'pA':1.,'s':1e-3,'F':1e-12}.get(unit,1.)
            # A trajectory envelope selected for engineering regression, not an
            # inference of global error from the solver's local tolerances.
            scale=max(float(np.max(np.abs(a[key]))),floor);metric['trajectory_scale']=scale
            metric['max_scaled_trajectory_error']=metric['max_absolute']/scale;metric['unit']=unit
            maximum_scaled=max(maximum_scaled,metric['max_scaled_trajectory_error']);comparisons[key]=metric
    p=finals['python'];n=finals['numba']
    rng=all(p.spike_states[k]['rng_state']==n.spike_states[k]['rng_state'] and p.spike_states[k]['history']==n.spike_states[k]['history'] and p.spike_states[k]['last_bin_counts']==n.spike_states[k]['last_bin_counts'] for k in p.spike_states)
    events=event_comparison(records['python']['spike_events'],records['numba']['spike_events'])
    schedule=difference(records['python']['schedule_times'],records['numba']['schedule_times'])
    assert maximum_scaled<1e-10 and rng and events['cell_order_counts_and_bin_endpoints_exact'] and events['event_time_max_absolute_seconds']<1e-12 and schedule['bitwise_equal']
    summary={'status':'PASS','window_seconds':.02,'first_compile':json.loads((OUT/'first_compile.json').read_bytes()),
        'python_seconds':records['python']['wall_seconds'],'numba_seconds':records['numba']['wall_seconds'],
        'speedup':records['python']['wall_seconds']/records['numba']['wall_seconds'],
        'integration_metrics':{k:v['integration_metrics'] for k,v in records.items()},
        'comparison':comparisons,'final_state_comparison':final_state_comparison(comparisons),'maximum_scaled_trajectory_error':maximum_scaled,'trajectory_limit':1e-10,
        'trajectory_criterion':'max absolute trajectory difference / max(reference trajectory magnitude, stated unit floor); measured on this window only; not a solver-local-tolerance global guarantee',
        'RHS_point_checks':point,'RNG_and_spike_count_history_exact':rng,'events':events,'schedule':schedule,
        'state_blocks':len(p.ordered_state_arrays),'checkpoint_roundtrips_exact':True,
        'structural_signature':structure(python),'input_sha256':sha(packet_path),'initial_checkpoint_sha256':sha(BASE/'initial_checkpoint.json'),
        'MATLAB_calls':0,'background_prewarm_runs':0,'reference_replays':1,'compiled_replays':1,
        'checkpoint_semantics':'Production implementation identities remain strict. Test-only transfer changes model identity after exact graph/parameter/state registry checks. No source checkpoint overwritten.',
        'limitations':['One 20 ms window, all state/output samples at1 ms','Events/history/recording/input interpolation and SciPy integration remain Python',
            'Compiled RHS supports only verified factory implementations without algebraic closure/edge delays; auto mode reports fallback for other graphs']}
    save(OUT/'summary.json',summary)
    print('COMPARISON_PASS',json.dumps({k:summary[k] for k in ['python_seconds','numba_seconds','speedup','maximum_scaled_trajectory_error','events','RNG_and_spike_count_history_exact']}),flush=True)

if __name__=='__main__':main()
