"""One real process: reverse-order strict restore and2ms recording comparison."""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,functools,gc,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.benchmark_compiler_snapshot import use_v7_compiler
COMPILER=use_v7_compiler()
from retina_sim.background_checkpoint import MemoryGuard,BackgroundCheckpointStore
from retina_sim.contracts import SolverConfig,canonical_hash
from retina_sim.assembly import RetinaBuilder
from retina_sim.recording import RecordingPlan
from retina_sim.recording_indices import IndexedRecordingPlan
from tools.benchmark_execution_procedures import available_windows_commit_bytes
from tools.disk3mm_response_check import observation_path,aligned_response_times


def save(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def fingerprint(state):
    identity={key:getattr(state.dynamic,key) for key in ('state_spec_hash','graph_hash','parameter_hash','model_hash',
        'model_id','model_version','input_interpolation','evidence_tag','optional_rng_state')}
    identity['solver']=asdict(state.solver_config)
    auxiliary={key:getattr(state,key) for key in ('time_seconds','assembly_hash','grid_origin_seconds','sample_index',
        'component_tick_indices','spike_states','delay_states','link_delay_states','input_position')}
    return {'identity':identity,'arrays_hash':canonical_hash(state.ordered_state_arrays),'aux_hash':canonical_hash(auxiliary),
        'scalars':sum(a.size for a in state.ordered_state_arrays)}


def instrumented_load(store,path,mode):
    original_dumps,original_loads=json.dumps,json.loads;large=[]
    def dumps(value,*args,**kwargs):
        tick=time.perf_counter();result=original_dumps(value,*args,**kwargs);duration=time.perf_counter()-tick
        if len(result)>1024**2:large.append({'operation':'dumps','bytes_or_characters':len(result),'seconds':duration,
            'caller':Path(sys._getframe(1).f_code.co_filename).name+':'+sys._getframe(1).f_code.co_name})
        return result
    def loads(value,*args,**kwargs):
        tick=time.perf_counter();result=original_loads(value,*args,**kwargs);duration=time.perf_counter()-tick
        if len(value)>1024**2:large.append({'operation':'loads','bytes_or_characters':len(value),'seconds':duration,
            'caller':Path(sys._getframe(1).f_code.co_filename).name+':'+sys._getframe(1).f_code.co_name})
        return result
    json.dumps,json.loads=dumps,loads;tick=time.perf_counter()
    try:state,value=store.load(path,require_converged=True,decode_mode=mode)
    finally:json.dumps,json.loads=original_dumps,original_loads
    wall=time.perf_counter()-tick
    return state,value,{'wall_seconds':wall,'phases':dict(store.last_load_timings),
        'procedure':dict(store.last_load_procedure),'large_JSON_calls':large,
        'large_JSON_total_seconds':sum(row['seconds'] for row in large)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():raise ValueError('Use a fresh project output')
    output.mkdir(parents=True);start=time.perf_counter();guard=MemoryGuard(14336*1024**2,8192*1024**2)
    guard.check()
    if guard.last['available_physical_bytes']<15*1024**3 or available_windows_commit_bytes()<7*1024**3:
        save(output/'result.json',{'status':'MEMORY_ADMISSION_BLOCKED','memory':guard.last});return
    def check():
        if not guard.check():raise MemoryError('Original memory reserve reached')
    from retina_platform.config import RunConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.execution_backends import make_runner
    from retina_sim.disk_input_packet import load_global_input
    original=ROOT/'reports/human_disk3mm_floor_transition_v7/transition';setup={}
    manifest=json.loads((original/'manifest.json').read_bytes());config=RunConfig.from_dict(manifest['configuration'])
    construction=replace(config,native_background_values=None,frontend_signature='');builds=[];build_depth=[0]
    original_build=RetinaBuilder.build
    @functools.wraps(original_build)
    def measured_build(self,*values,**kwargs):
        tick=time.perf_counter();outer=build_depth[0]==0;build_depth[0]+=1
        try:result=original_build(self,*values,**kwargs)
        finally:build_depth[0]-=1
        if outer:builds.append({'seconds':time.perf_counter()-tick,'components':len(result.components),
            'links':len(result.links),'native_components':[cid for cid,c in result.components.items() if hasattr(c.native_model,'A')]})
        return result
    print('RESTORE_PAIR_ASSEMBLY_BEGIN',flush=True);tick=time.perf_counter();RetinaBuilder.build=measured_build
    try:
        compiled,_,audit=atlas_population_network(ROOT,duration=.0001,light_context=construction.light_context(),
            owner_bounds_mm=config.owner_bounds(),halo_mm=.04,max_spatial_cells=config.max_spatial_cells,
            seed=config.seed,execution_profile=config.execution_profile,configuration_path=config.circuit_configuration_path)
    finally:RetinaBuilder.build=original_build
    setup['assembly_seconds']=time.perf_counter()-tick;setup['top_level_builder_calls']=builds;check()
    print('RESTORE_PAIR_ASSEMBLY_COMPLETE',json.dumps(setup),flush=True);tick=time.perf_counter()
    runner=make_runner(compiled,config.neural_integration_backend)
    setup['runner_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
    solver=SolverConfig(**audit['resolved_execution_profile']['solver']);reference=runner.initialize(solver)
    setup['reference_initialize_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
    store=BackgroundCheckpointStore(ROOT,runner,reference,config.light_context(),config.background_policy())
    setup['strict_binding_construction_seconds']=time.perf_counter()-tick;check();tick=time.perf_counter()
    packet=load_global_input(original/'input.npz',compiled,config.light_context(),reference_id=compiled.global_disk_circuit.record['reference_id'])
    setup['input_capture_strict_load_seconds']=time.perf_counter()-tick
    target,chosen,bc,_=observation_path(compiled,audit)
    ordinary=RecordingPlan(states=tuple(name+'.terminal_release' for name in bc),cell_ids=frozenset(chosen),max_bytes=64*1024**2)
    indexed=IndexedRecordingPlan(ordinary)
    setup['unique_setup_total_before_restore_seconds']=time.perf_counter()-start
    save(output/'setup.json',setup);print('RESTORE_PAIR_SETUP_COMPLETE',json.dumps(setup),flush=True)
    checkpoint=original/'matched_response/background_checkpoint.json'
    print('RESTORE_PAIR_LEGACY_WARMUP_BEGIN',flush=True);tick=time.perf_counter()
    warmed,value=store.load(checkpoint,require_converged=True,decode_mode='legacy')
    warm_seconds=time.perf_counter()-tick;del warmed,value;gc.collect();check()
    print('RESTORE_PAIR_LEGACY_WARMUP_COMPLETE',warm_seconds,flush=True)
    loads={};traces={};states={};ends={};output_hashes={}
    for mode,plan in (('parsed',indexed),('legacy',ordinary)):
        print('RESTORE_PAIR_MEASURE_BEGIN',mode,flush=True);check()
        state,value,metrics=instrumented_load(store,checkpoint,mode);loads[mode]=metrics
        states[mode]=fingerprint(state)
        for array,raw in zip(state.ordered_state_arrays,value['checkpoint']['payload']['dynamic']['state']['ordered_state_arrays']):
            np.testing.assert_array_equal(array.view(np.uint64),np.asarray(raw,dtype=np.float64).view(np.uint64))
        # Every restored causal field comes from the source whose full nested
        # hashes were checked above; keep the returned envelope independent.
        auxiliary={key:getattr(state,key) for key in ('grid_origin_seconds','sample_index','component_tick_indices',
            'spike_states','delay_states','link_delay_states','input_position')}
        if canonical_hash(auxiliary)!=value['auxiliary_time_rng_history_hash']:raise AssertionError('Restored causal payload changed')
        del value;gc.collect();check()
        origin=state.time_seconds;duration=.002;stop=origin+duration;channels={}
        for name,channel in packet.channels.items():
            q=channel.at(.00005);channels[name]=replace(channel,start_seconds=origin,stop_seconds=stop,
                evaluator=lambda t,q=q:q,breakpoints_seconds=(),
                input_contract=dict(channel.input_contract,short_execution_only_static_rate_repeated=True))
        times,_=aligned_response_times(runner,state.grid_origin_seconds,origin,duration)
        def observe(t,y,bundle):check()
        # One whole same-input2ms warmup for each plan; discard its state.
        warm,_=runner.advance(state,channels,stop,sample_times=times,recording_plan=plan,state_observer=observe)
        del warm;gc.collect();before=dict(runner.integration_metrics);tick=time.perf_counter()
        end,result=runner.advance(state,channels,stop,sample_times=times,recording_plan=plan,state_observer=observe)
        traces[mode]={'wall_seconds':time.perf_counter()-tick,**{key:runner.integration_metrics[key]-before[key]
            for key in ('solver_calls','nfev','integration_wall_seconds')}}
        ends[mode]=fingerprint(end)
        output_hashes[mode]={'outputs_hash':canonical_hash(result.outputs_with_units),'metadata_hash':canonical_hash(result.metadata)}
        np.savez(output/('end_'+mode+'.npz'),**{'state_%03d'%i:array for i,array in enumerate(end.ordered_state_arrays)})
        print('RESTORE_PAIR_ARM_COMPLETE',mode,json.dumps({'load':metrics,'trace':traces[mode]}),flush=True)
        del state,end,result;gc.collect();check()
        save(output/'partial.json',{'setup':setup,'loads':loads,'traces':traces,'states':states,'ends':ends})
    assert states['parsed']==states['legacy'] and ends['parsed']==ends['legacy'] and output_hashes['parsed']==output_hashes['legacy']
    assert traces['parsed']['nfev']==traces['legacy']['nfev'] and traces['parsed']['solver_calls']==traces['legacy']['solver_calls']
    with np.load(output/'end_parsed.npz',allow_pickle=False) as first,np.load(output/'end_legacy.npz',allow_pickle=False) as second:
        for key in first.files:np.testing.assert_array_equal(first[key].view(np.uint64),second[key].view(np.uint64))
    report={'status':'REAL_STRICT_RESTORE_AND_REVERSE_RECORDED_PAIR_EXACT_PASS','setup':setup,'loads':loads,'traces':traces,
        'warmup_legacy_load_seconds_discarded':warm_seconds,'state_fingerprints':states,'endpoint_fingerprints':ends,
        'output_fingerprints':output_hashes,'restore_speedup_single_pair':loads['legacy']['wall_seconds']/loads['parsed']['wall_seconds'],
        'recorded_index_speedup_reverse_pair':traces['legacy']['wall_seconds']/traces['parsed']['wall_seconds'],
        'protocol':'One large graph process; full legacy restore warmup discarded; parsed-first then legacy-second measured once each. Each2ms recorded arm has one same-input full-window warmup. No new background; complete original checks retained.',
        'scope':'Original-HC v7 frozen transition,4,967,023 states, original GPU-v1;311MB checkpoint strict restore and2ms static captured image',
        'frozen_compiler':COMPILER,'model_hash':runner.model.model_hash,'checkpoint_file_sha256':hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),
        'no_default_switch':True,'B_backend':'Not used; retained optional','new_long_background':False,
        'actual_total_process_wall_seconds_including_warmups_parity_and_artifact_IO':time.perf_counter()-start,'memory':guard.last}
    save(output/'result.json',report)
    print('RESTORE_PAIR_COMPLETE',json.dumps({'status':report['status'],'restore_speedup':report['restore_speedup_single_pair'],
        'recorded_reverse_speedup':report['recorded_index_speedup_reverse_pair'],'setup':setup,'loads':loads,'traces':traces}),flush=True)


if __name__=='__main__':main()
