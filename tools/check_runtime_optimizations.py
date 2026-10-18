"""One before/after 20 ms replay using the archived B-K input and background.

The after phase explicitly transfers the verified before state to the new code
identity for this regression only. Production checkpoint validation is unchanged.
"""
from pathlib import Path
from dataclasses import asdict, replace
import argparse, hashlib, json, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.profile_bk_window import BK, PRIOR, CLASSES, sha, save, zero_metrics
from retina_platform.config import RunConfig
from retina_sim.atlas_population_network import atlas_population_network
from retina_sim.background_checkpoint import BackgroundCheckpointStore
from retina_sim.contracts import ContractError, SolverConfig, canonical_hash
from retina_sim.hybrid import HybridRunner, RetinaState
from retina_sim.offline_input import load_offline_inputs,rebase_offline_inputs
from retina_sim.recording import RecordingPlan

OUT=ROOT/'reports/bk_optimization_20ms_20261015'
FILES=('hybrid.py','contracts.py','network.py','vh2005.py')

def structure(runner):
    c=runner.compiled
    return canonical_hash({'parameters':runner.model.parameters.parameter_hash,
        'states':runner.model.state_specs,'clock':c.clock,
        'components':{cid:{'mechanism':x.mechanism,'inputs':x.inputs,'outputs':x.outputs,
            'parameters':x.parameters.parameter_hash,'time':x.time_contract,
            'internal_dependencies':x.internal_dependencies} for cid,x in c.components.items()},
        'populations':{k:v.cells for k,v in c.populations.items()},
        'links':[{'spec':x.specification,'matrix_data':x.matrix.data,'matrix_indices':x.matrix.indices,
            'matrix_indptr':x.matrix.indptr,'shape':x.matrix.shape} for x in c.links]})

def difference(a,b):
    a=np.asarray(a);b=np.asarray(b);d=np.abs(a-b)
    return {'bitwise_equal':bool(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()),
        'max_absolute':float(d.max()) if d.size else 0.,
        'max_relative_to_before':float(np.max(d/np.maximum(np.abs(a),1e-30))) if d.size else 0.}

def compare_checkpoints(before,after):
    a=json.loads(before)['payload'];b=json.loads(after)['payload']
    arrays_a=a['dynamic']['state']['ordered_state_arrays'];arrays_b=b['dynamic']['state']['ordered_state_arrays']
    names=[x['name'] for x in json.loads((OUT/'before.json').read_text())['state_layout']]
    assert len(arrays_a)==len(arrays_b)==len(names)
    blocks={name:difference(x,y) for name,x,y in zip(names,arrays_a,arrays_b)}
    aux={k:{'equal':a[k]==b[k]} for k in ['grid_origin_seconds','sample_index','component_tick_indices','input_position']}
    def nested(x,y,path=''):
        if isinstance(x,dict):return {k:nested(x[k],y[k],path+'.'+k) for k in x}
        if isinstance(x,list):
            try:
                xa=np.asarray(x,dtype=float);ya=np.asarray(y,dtype=float)
                return difference(xa,ya)
            except (ValueError,TypeError):return [nested(u,v,path) for u,v in zip(x,y)]
        return {'equal':x==y} if not isinstance(x,float) else difference([x],[y])
    for k in ['spike_states','delay_states','link_delay_states']:aux[k]=nested(a[k],b[k])
    # Normalize only implementation identities and their derived checksum.
    # canonical_hash preserves signed zero and all nested history/RNG fields.
    b['assembly_hash']=a['assembly_hash']
    for key in ['model_hash','graph_hash']:b['dynamic']['state'][key]=a['dynamic']['state'][key]
    b['dynamic']['payload_hash']=canonical_hash(b['dynamic']['state'])
    return {'state_blocks':blocks,'auxiliary':aux,
        'payload_exact_after_code_identity_normalization':canonical_hash(a)==canonical_hash(b),
        'code_identity_fields_expected_to_change':['assembly_hash','dynamic.state.model_hash','dynamic.state.graph_hash']}

def main(phase):
    if phase=='before':OUT.mkdir(parents=True,exist_ok=False)
    if (OUT/(phase+'.json')).exists():raise RuntimeError('Phase already recorded; preserve original evidence')
    old=json.loads(PRIOR.read_bytes());cfg=RunConfig.from_dict(old['config']);old_light=cfg.light_context()
    source=json.loads((BK/'manifest.json').read_bytes())['input_segments'][0]['source'];packet_path=Path(source['source_file'])
    assert sha(packet_path)==source['source_sha256']
    with np.load(packet_path,allow_pickle=False) as packet:packet_manifest=json.loads(packet['manifest_json'].item())
    new_light=replace(old_light,calibration='approved_mixed_reference:'+packet_manifest['producer']['frontend_signature'])
    compiled,_,audit=atlas_population_network(ROOT,duration=.075,light_context=old_light,
        owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed)
    solver=SolverConfig(**audit['resolved_execution_profile']['solver']);runner=HybridRunner(compiled);reference=runner.initialize(solver)
    signature=structure(runner);store=BackgroundCheckpointStore(ROOT,runner,reference,old_light,cfg.background_policy())
    background=Path(old['runtime']['prepared_background_cache']);cache_rejected=False
    if phase=='before':
        state,_=store.load(background,require_converged=True)
        for name in FILES:(OUT/('before_'+name)).write_bytes((ROOT/'retina_sim'/name).read_bytes())
    else:
        before=json.loads((OUT/'before.json').read_bytes())
        assert signature==before['structural_signature']
        assert sha(background)==before['background_sha256'] and sha(packet_path)==before['input_sha256']
        try:store.load(background,require_converged=True)
        except ContractError:cache_rejected=True
        else:raise AssertionError('Code identity changed but old cache was accepted')
        raw=json.loads((OUT/'initial_checkpoint.json').read_bytes());assert canonical_hash(raw['payload'])==raw['payload_hash']
        p=raw['payload'];d=p['dynamic']['state']
        assert d['state_spec_hash']==reference.state_spec_hash and d['parameter_hash']==reference.parameter_hash
        assert d['solver_config']==asdict(solver)
        dynamic=replace(reference.dynamic,time_seconds=d['time_seconds'],ordered_state_arrays=tuple(np.array(x) for x in d['ordered_state_arrays']))
        state=RetinaState(dynamic,compiled.assembly_hash,p['grid_origin_seconds'],p['sample_index'],p['spike_states'],
            p['delay_states'],p['input_position'],p['link_delay_states'],p['component_tick_indices'])
    state.input_position.update(phase='media_stimulus',stimulus_origin_seconds=state.time_seconds,input_segment=0,
        foreground_background_context_hash=new_light.background_context_hash)
    initial=runner.checkpoint(state)
    if phase=='before':(OUT/'initial_checkpoint.json').write_text(initial,encoding='utf8')
    bundle=load_offline_inputs(packet_path,compiled,new_light,reference_id=packet_manifest['channels']['native_light']['reference_id'])
    channels=rebase_offline_inputs(bundle,compiled,new_light,stimulus_origin_seconds=state.time_seconds,grid_origin_seconds=state.grid_origin_seconds)
    origin=state.time_seconds;stop=round((origin+.02)/.0001)*.0001
    times=np.arange(round(origin/.001),round(stop/.001)+1)*.001;times[0]=origin;times[-1]=stop
    plan=RecordingPlan(ports=tuple(cid+'.'+p for cid in CLASSES for p in ['conditional_rate','voltage']),states=(),max_bytes=64*1024**2)
    zero_metrics(runner);tick=time.perf_counter();final,result=runner.advance(state,channels,stop,sample_times=times,recording_plan=plan)
    wall=time.perf_counter()-tick;checkpoint=runner.checkpoint(final)
    (OUT/(phase+'_checkpoint.json')).write_text(checkpoint,encoding='utf8')
    np.savez_compressed(OUT/(phase+'_outputs.npz'),**{k:q.values for k,q in result.outputs_with_units.items()})
    row={'phase':phase,'wall_seconds':wall,'integration_metrics':runner.integration_metrics,
        'structural_signature':signature,'input_sha256':sha(packet_path),'background_sha256':sha(background),
        'solver':asdict(solver),'state_layout':result.metadata['global_state_layout'],
        'spike_events':result.metadata['spike_events'],'clock':compiled.clock,
        'core_sha256':{name:sha(ROOT/'retina_sim'/name) for name in FILES},
        'old_cache_rejected_by_production_identity_check':cache_rejected,
        'MATLAB_calls':0,'prewarm_runs':0,'test_only_state_identity_transfer':phase=='after'}
    if phase=='before':
        with np.load(ROOT/'reports/bk_profile_20ms_20261015/comparison_outputs.npz') as baseline:
            row['original_baseline_outputs']={k:difference(baseline['normal:'+k],q.values) for k,q in result.outputs_with_units.items()}
    else:
        with np.load(OUT/'before_outputs.npz') as baseline:
            row['output_comparison']={k:difference(baseline[k],q.values) for k,q in result.outputs_with_units.items()}
        row['state_comparison']=compare_checkpoints((OUT/'before_checkpoint.json').read_text(),checkpoint)
        row['spike_events_equal']=row['spike_events']==before['spike_events']
        row['speedup_vs_same_session_before']=before['wall_seconds']/wall
        row['speedup_vs_original_10_8277606']=10.8277606/wall
        # Round-trip continuity at the same terminal checkpoint, with no extra advance.
        restored=runner.restore(checkpoint);row['checkpoint_roundtrip_bitwise_equal']=runner.checkpoint(restored)==checkpoint
    save(OUT/(phase+'.json'),row)
    print(json.dumps({k:v for k,v in row.items() if k in ['phase','wall_seconds','integration_metrics','speedup_vs_same_session_before',
        'old_cache_rejected_by_production_identity_check','checkpoint_roundtrip_bitwise_equal','spike_events_equal']},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['before','after'],required=True)
    main(parser.parse_args().phase)
