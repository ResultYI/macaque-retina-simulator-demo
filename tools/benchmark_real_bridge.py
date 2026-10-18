"""Sequential real-graph CUDA bridge comparison, never checkpoint migration.

The original arm is exported after the prepared A benchmark. The compact arm
rebuilds exactly that frozen graph in a fresh child, independently initializes
the original factory values and compares every state/input before execution.
One complete warmup and one100us recorded window per arm; no new background.
"""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,cProfile,gc,hashlib,json,pstats,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    from tools.benchmark_compiler_snapshot import use_v7_compiler
    use_v7_compiler()
from retina_sim.contracts import canonical_hash,SolverConfig
from retina_sim.network import InputBundle
from retina_sim.recording import RecordingPlan
from retina_sim.recording_indices import IndexedRecordingPlan
from retina_sim.background_checkpoint import MemoryGuard
from tools.disk3mm_response_check import observation_path

AUX=('time_seconds','assembly_hash','grid_origin_seconds','sample_index','component_tick_indices',
     'spike_states','delay_states','link_delay_states','input_position')


def save_json(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,allow_nan=False,indent=2)+'\n',encoding='utf-8')


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def aux(state):return {name:getattr(state,name) for name in AUX}


def persist_state(path,state):
    np.savez(str(path)+'_arrays.npz',**{'state_%03d'%i:array for i,array in enumerate(state.ordered_state_arrays)})
    # Ordinary architecture auxiliary payloads contain JSON-native lists/
    # tuples/scalars. json.dump streams without assembling another giant string.
    with Path(str(path)+'_aux.json').open('w',encoding='utf-8') as stream:
        json.dump(aux(state),stream,sort_keys=True,allow_nan=False,separators=(',',':'))
    return {'array_file':Path(str(path)+'_arrays.npz').name,'array_file_sha256':file_sha(str(path)+'_arrays.npz'),
        'aux_file':Path(str(path)+'_aux.json').name,'aux_file_sha256':file_sha(str(path)+'_aux.json'),
        'array_tuple_hash':canonical_hash(state.ordered_state_arrays),'aux_hash':canonical_hash(aux(state)),
        'scalars':sum(a.size for a in state.ordered_state_arrays)}


def require_state(path,state):
    with np.load(str(path)+'_arrays.npz',allow_pickle=False) as values:
        if len(values.files)!=len(state.ordered_state_arrays):raise AssertionError('State block count changed')
        for i,array in enumerate(state.ordered_state_arrays):np.testing.assert_array_equal(values['state_%03d'%i],array)
    expected=json.loads(Path(str(path)+'_aux.json').read_bytes())
    if canonical_hash(expected)!=canonical_hash(aux(state)):raise AssertionError('Full event/history/RNG/delay/input auxiliary state changed')


def static_input(captured):
    channels={}
    for name,channel in captured.items():
        value=channel.at(.00005)
        channels[name]=replace(channel,start_seconds=0.,stop_seconds=.002,
            evaluator=lambda t,value=value:value,breakpoints_seconds=(),
            input_contract=dict(channel.input_contract,short_execution_only_static_rate_repeated=True))
    return channels


def binding(runner,state,channels):
    return {'assembly_hash':runner.compiled.assembly_hash,'graph_hash':runner.model.graph.graph_hash,
        'parameter_hash':runner.model.parameters.parameter_hash,'state_schema_hash':state.dynamic.state_spec_hash,
        'clock_hash':canonical_hash(runner.compiled.clock),'solver':asdict(state.solver_config),
        'input_hash':canonical_hash({name:{'value':channel.at(.00005),'contract':channel.input_contract,
            'interpolation':channel.interpolation} for name,channel in channels.items()})}


def rhs_case(runner,state,channels,output,expected=None):
    import torch
    from retina_sim.gpu_compact import compact_release_layout
    p=runner.partition;initial=runner._flat(state);bundle=InputBundle({name:channel.at(.00005) for name,channel in channels.items()})
    summary=[]
    for label,values in (('initial',initial),('perturbed',initial.copy())):
        if label=='perturbed':values[:p.offset]*=1.000001
        device=torch.as_tensor(values.copy(),dtype=torch.float64,device='cuda')
        derivative=p(.00005,device,bundle).cpu().numpy()
        q=bundle.channels[p.native_external.channel];light=torch.as_tensor(q.values.copy(),dtype=torch.float64,device='cuda')
        _,full_release=p.native(device[:p.offset],light)
        used,private,consumers=compact_release_layout(runner.model,p.cid,p.native.nc)
        full=full_release.cpu().numpy();parts={'RHS':derivative}
        links=[link for link in runner.compiled.links if link.specification.source_component==p.cid]
        for i,link in enumerate(links):
            spec=link.specification;key=(spec.target_component,spec.target_port)
            mapped=next(item for item in private[key] if item.specification is spec)
            old=link.matrix@full;new=mapped.matrix@full[used]
            np.testing.assert_array_equal(old,new);parts['consumer_%03d'%i]=new
        np.savez(output/('rhs_'+label+'.npz'),**parts)
        if expected is not None:
            with np.load(expected/('rhs_'+label+'.npz'),allow_pickle=False) as previous:
                assert set(previous.files)==set(parts)
                for key,value in parts.items():np.testing.assert_array_equal(previous[key],value)
        summary.append({'state':label,'RHS_scalars':len(derivative),'all_consumer_drives':consumers,
            'RHS_sha256':hashlib.sha256(derivative.tobytes()).hexdigest(),
            'all_RHS_and_consumer_arrays_exact_to_legacy':expected is not None})
    return summary


def execute_case(runner,initial,captured,audit,config,output,guard,expected=None):
    output.mkdir();channels=static_input(captured)
    # Use the same independently initialized phase metadata in both arms.
    initial.input_position={'phase':'fresh_initial_state_short_execution_benchmark'}
    science=binding(runner,initial,channels)
    target,chosen,bc,_=observation_path(runner.compiled,audit)
    plan=IndexedRecordingPlan(RecordingPlan(states=tuple(name+'.terminal_release' for name in bc),
        cell_ids=frozenset(chosen),max_bytes=64*1024**2))
    if expected is not None:
        previous=json.loads((expected/'case.json').read_bytes())
        if science!=previous['frozen_scientific_binding']:raise AssertionError('Frozen graph/parameters/schema/clock/input/solver changed')
        require_state(expected/'initial',initial)
    initial_receipt=persist_state(output/'initial',initial)
    rhs=rhs_case(runner,initial,channels,output,expected)
    if not guard.check():raise MemoryError('Original reserve reached before real short window')
    times=np.array([0.,.00005,.0001]);points=[]
    def observe(t,y,bundle):
        if not guard.check():raise MemoryError('Original memory reserve reached during real short window')
        points.append(float(t))
    print('REAL_BRIDGE_COMPLETE_WINDOW_WARMUP_BEGIN',flush=True)
    warmed,_=runner.advance(initial,channels,.0001,sample_times=times,recording_plan=plan,state_observer=observe)
    del warmed;gc.collect();points.clear();before=dict(runner.integration_metrics)
    import torch
    torch.cuda.synchronize();tick=time.perf_counter()
    end,result=runner.advance(initial,channels,.0001,sample_times=times,recording_plan=plan,state_observer=observe)
    torch.cuda.synchronize();wall=time.perf_counter()-tick
    measured={'wall_seconds':wall,'observer_times':list(points),
        **{key:runner.integration_metrics[key]-before[key] for key in ('solver_calls','nfev','integration_wall_seconds')}}
    end_receipt=persist_state(output/'end',end)
    signals={key:value.values for key,value in result.outputs_with_units.items()}
    np.savez(output/'outputs.npz',**signals)
    declarations={key:{'unit':value.unit,'meaning':value.meaning,'evidence_tag':value.evidence_tag,
        'coordinate_frame':value.coordinate_frame,'recorded_cell_ids':result.metadata.get('recorded_cell_ids',{}).get(key)}
        for key,value in result.outputs_with_units.items()}
    save_json(output/'output_declarations.json',declarations)
    if expected is not None:
        require_state(expected/'end',end)
        if declarations!=json.loads((expected/'output_declarations.json').read_bytes()):raise AssertionError('Output units/IDs/meaning changed')
        with np.load(expected/'outputs.npz',allow_pickle=False) as previous:
            assert set(previous.files)==set(signals)
            for key,value in signals.items():np.testing.assert_array_equal(previous[key],value)
        previous_case=json.loads((expected/'case.json').read_bytes())
        for key in ('observer_times','solver_calls','nfev'):
            if measured[key]!=previous_case['measurement'][key]:raise AssertionError('Numerical stage/observer schedule changed')
    # One same-state diagnostic repeat for residual bottlenecks. Its overhead
    # is excluded from the unprofiled comparison above and is reported apart.
    profiler=cProfile.Profile();profile_before=dict(runner.integration_metrics);points.clear();profile_tick=time.perf_counter()
    profiler.enable()
    repeated,repeated_result=runner.advance(initial,channels,.0001,sample_times=times,recording_plan=plan,state_observer=observe)
    profiler.disable();profile_wall=time.perf_counter()-profile_tick
    for first,second in zip(end.ordered_state_arrays,repeated.ordered_state_arrays):np.testing.assert_array_equal(first,second)
    if canonical_hash(aux(end))!=canonical_hash(aux(repeated)):raise AssertionError('Profile repeat changed causal full state')
    for key,value in result.outputs_with_units.items():np.testing.assert_array_equal(value.values,repeated_result.outputs_with_units[key].values)
    if (points!=measured['observer_times'] or runner.integration_metrics['nfev']-profile_before['nfev']!=measured['nfev']):
        raise AssertionError('Profile repeat changed observer/solver schedule')
    profiler.dump_stats(str(output/'same_100us_profile.pstats'))
    stats=pstats.Stats(profiler);top=[]
    for key,(primitive,total,self_seconds,cumulative,callers) in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:35]:
        filename,line,name=key;path=Path(filename)
        try:filename=path.resolve().relative_to(ROOT).as_posix()
        except (ValueError,OSError):filename=path.name
        top.append({'function':filename+':'+str(line)+':'+name,'calls':total,'self_seconds':self_seconds,'cumulative_seconds':cumulative})
    del repeated,repeated_result;gc.collect()
    report={'status':'REAL_FULL_GRAPH_COMPACT_PAIR_EXACT_PASS' if expected else 'REAL_FULL_GRAPH_LEGACY_CASE_EXPORTED',
        'frozen_scientific_binding':science,'execution_model_hash':runner.model.model_hash,
        'components':len(runner.compiled.components),'native_cones':runner.partition.native.nc,
        'native_HC':runner.partition.native.nh,
        'population_counts':{key:len(pop.cells.cell_id) for key,pop in runner.compiled.populations.items()},
        'selected_dependency_nodes':getattr(runner.compiled,'global_disk_view',{}).get('global_node_count'),
        'actual_compiler_sha256':file_sha(sys.modules['retina_sim.assembly'].__file__),
        'scope':'One100us real complete recorded window from original factory initial state; not adapted, no background preparation',
        'initial':initial_receipt,'end':end_receipt,'RHS_and_consumer_checks':rhs,'measurement':measured,
        'protocol':'Each arm in a separate sequential process; one complete100us warmup discarded, then one matched100us measured window; common indexed recording, all original clocks/history/RNG/solver settings.',
        'checkpoint_migration':False,'input':'Existing v7 real ISET static-image expected-rate capture; no new export',
        'source_manifest':'reports/human_disk3mm_floor_transition_v7/transition/manifest.json',
        'source_manifest_sha256':file_sha(ROOT/'reports/human_disk3mm_floor_transition_v7/transition/manifest.json'),
        'input_capture_sha256':file_sha(ROOT/'reports/human_disk3mm_floor_transition_v7/transition/input.npz'),
        'memory':dict(guard.last),'integration_metadata':runner.model.compute_metadata,
        'residual_profile':{'wall_seconds_with_profiler':profile_wall,'same_full_state_causal_aux_outputs_and_schedule_exact':True,
            'top_cumulative_functions':top,'inclusive_times_overlap_do_not_add':True,'profiler_time_is_not_throughput':True},
        'procedure_source_sha256':file_sha(__file__),'default_changed':False,'physiological_validation':False}
    if expected is not None:
        report['legacy_measurement']=previous_case['measurement']
        report['short_window_speedup']=previous_case['measurement']['wall_seconds']/wall
        report['full_registered_arrays_outputs_units_IDs_causal_aux_RNG_history_delays_exact']=True
        report['compact_layout']=runner.partition.layout;report['compact_statistics']=runner.partition.statistics
    save_json(output/'case.json',report)
    print('REAL_BRIDGE_CASE_COMPLETE',json.dumps({'status':report['status'],'measurement':measured,
        'speedup':report.get('short_window_speedup'),'state_scalars':end_receipt['scalars']}),flush=True)
    return {'path':str(output.relative_to(ROOT)),'status':report['status'],'measurement':measured}


def run_legacy_bridge(runner,reference,captured,audit,config,output,guard):
    return execute_case(runner,reference,captured,audit,config,output,guard)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--legacy',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();legacy=(ROOT/args.legacy).resolve();output=(ROOT/args.output).resolve()
    if not legacy.is_relative_to(ROOT) or not output.is_relative_to(ROOT) or output.exists():raise ValueError('Use retained legacy case and fresh project output')
    previous=json.loads((legacy/'case.json').read_bytes())
    from tools.benchmark_execution_procedures import available_windows_commit_bytes
    guard=MemoryGuard(14336*1024**2,8192*1024**2);guard.check()
    if guard.last['available_physical_bytes']<15*1024**3 or available_windows_commit_bytes()<7*1024**3:
        raise MemoryError('Original physical/commit admission failed before compact model allocation')
    from retina_platform.config import RunConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.execution_backends import make_runner,CUDA_COMPACT
    from retina_sim.disk_input_packet import load_global_input
    original=ROOT/'reports/human_disk3mm_floor_transition_v7/transition'
    if file_sha(original/'manifest.json')!=previous['source_manifest_sha256'] or file_sha(original/'input.npz')!=previous['input_capture_sha256']:
        raise AssertionError('Source manifest/capture changed between sequential arms')
    manifest=json.loads((original/'manifest.json').read_bytes());config=RunConfig.from_dict(manifest['configuration'])
    construction=replace(config,native_background_values=None,frontend_signature='');setup={};tick=time.perf_counter()
    print('REAL_COMPACT_ASSEMBLY_BEGIN',flush=True)
    compiled,_,audit=atlas_population_network(ROOT,duration=.0001,light_context=construction.light_context(),
        owner_bounds_mm=config.owner_bounds(),halo_mm=.04,max_spatial_cells=config.max_spatial_cells,
        seed=config.seed,execution_profile=config.execution_profile,configuration_path=config.circuit_configuration_path)
    setup['assembly_seconds']=time.perf_counter()-tick;print('REAL_COMPACT_ASSEMBLY_COMPLETE',setup,flush=True)
    if not guard.check():raise MemoryError('Original reserve reached after compact assembly')
    tick=time.perf_counter();runner=make_runner(compiled,CUDA_COMPACT)
    initial=runner.initialize(SolverConfig(**audit['resolved_execution_profile']['solver']))
    setup['runner_initialize_seconds']=time.perf_counter()-tick
    packet=load_global_input(original/'input.npz',compiled,config.light_context(),reference_id=compiled.global_disk_circuit.record['reference_id'])
    execute_case(runner,initial,packet.channels,audit,config,output,guard,expected=legacy)
    save_json(output/'setup.json',setup)


if __name__=='__main__':main()
