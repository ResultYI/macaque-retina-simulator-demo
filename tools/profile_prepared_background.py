"""Short exact-model wall/profile diagnosis using an existing converged state.

No new optical capture, background preparation, parameter, cell or solver change.
One unprofiled1ms chunk and the same1ms chunk under cProfile are compared.
"""
from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
import argparse,cProfile,json,pstats,sys,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_platform.config import RunConfig
from retina_sim.atlas_population_network import atlas_population_network
from retina_sim.execution_backends import make_runner
from retina_sim.contracts import SolverConfig
from retina_sim.background import constant_background_channels,_WindowMonitor
from retina_sim.background_checkpoint import BackgroundCheckpointStore
from retina_sim.recording import RecordingPlan
from tools.disk3mm_response_check import aligned_response_times


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reports/performance_profile_20261018')
    parser.add_argument('--compare-schema-cache',action='store_true',
        help='Compare the original uncached validation with the immutable-registry cache on the same model/state')
    args=parser.parse_args();output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():raise ValueError('Use a new retained profile output directory')
    output.mkdir();setup={};tick=time.perf_counter()
    original=ROOT/'reports/human_disk3mm_floor_transition_v7/transition'
    manifest=json.loads((original/'manifest.json').read_bytes())
    config=RunConfig.from_dict(manifest['configuration'])
    construction=replace(config,native_background_values=None,frontend_signature='')
    print('SHORT_PROFILE_ASSEMBLY_BEGIN; no ISET or background run',flush=True)
    compiled,_,audit=atlas_population_network(ROOT,duration=.0001,light_context=construction.light_context(),
        owner_bounds_mm=config.owner_bounds(),halo_mm=.04,max_spatial_cells=config.max_spatial_cells,
        seed=config.seed,execution_profile=config.execution_profile,configuration_path=config.circuit_configuration_path)
    setup['assembly_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
    print('SHORT_PROFILE_ASSEMBLY_COMPLETE',setup['assembly_seconds'],flush=True)
    runner=make_runner(compiled,config.neural_integration_backend)
    solver=SolverConfig(**audit['resolved_execution_profile']['solver'])
    reference=runner.initialize(solver);policy=config.background_policy()
    reference.input_position={'background_context_hash':config.light_context().background_context_hash}
    store=BackgroundCheckpointStore(ROOT,runner,reference,config.light_context(),policy)
    setup['runner_reference_binding_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
    print('SHORT_PROFILE_RUNNER_BINDING_COMPLETE',setup['runner_reference_binding_seconds'],flush=True)
    checkpoint=original/'matched_response/background_checkpoint.json'
    baseline,saved=store.load(checkpoint,require_converged=True)
    setup['strict_full_checkpoint_load_seconds']=time.perf_counter()-tick
    del saved
    print('SHORT_PROFILE_STRICT_CHECKPOINT_REUSE_PASS',baseline.time_seconds,runner.model.model_hash,flush=True)
    start=baseline.time_seconds;stop=start+.001
    channels=constant_background_channels(compiled,config.light_context(),stop+.001)
    times,_=aligned_response_times(runner,baseline.grid_origin_seconds,start,.001)
    plan=RecordingPlan(ports=(),states=(),cell_ids=())
    def run_chunk():
        monitor=_WindowMonitor(runner,reference,policy)
        before=dict(runner.integration_metrics);begin=time.perf_counter()
        state,result=runner.advance(baseline,channels,stop,sample_times=times,recording_plan=plan,
                                    state_observer=monitor.observe)
        return state,{'wall_seconds':time.perf_counter()-begin,
            'integration_wall_seconds':runner.integration_metrics['integration_wall_seconds']-before['integration_wall_seconds'],
            'solver_calls':runner.integration_metrics['solver_calls']-before['solver_calls'],
            'nfev':runner.integration_metrics['nfev']-before['nfev'],'observations':monitor.count,
            'recorded_samples':len(result.outputs_with_units['time'].values)}
    optimized=None
    if args.compare_schema_cache:
        from retina_sim import engine
        original_helper=engine._state_schema_hash
        if getattr(runner.model,'_engine_state_schema_identity',(None,))[0] is not runner.model.state_specs:
            raise AssertionError('Actual schema was not certified immutable; no benchmark cache assumption')
        # Retain the original mathematical/identity operation as the local
        # baseline; this hook is restored before any optimized measurement.
        engine._state_schema_hash=lambda model:engine.canonical_hash(model.state_specs)
        try:first,plain=run_chunk()
        finally:engine._state_schema_hash=original_helper
        optimized_state,optimized=run_chunk()
        print('SHORT_PROFILE_CACHE_COMPARISON',json.dumps({'uncached':plain,'cached':optimized}),flush=True)
    else:first,plain=run_chunk()
    print('SHORT_PROFILE_PLAIN_CHUNK',json.dumps(plain),flush=True)
    profiler=cProfile.Profile();profiler.enable()
    second,profiled=run_chunk();profiler.disable();profiler.dump_stats(str(output/'one_ms.pstats'))
    arrays_exact=all(np.array_equal(a,b) for a,b in zip(first.ordered_state_arrays,second.ordered_state_arrays))
    auxiliary_exact=(first.time_seconds==second.time_seconds and first.component_tick_indices==second.component_tick_indices
        and first.sample_index==second.sample_index and first.spike_states==second.spike_states
        and first.delay_states==second.delay_states and first.link_delay_states==second.link_delay_states)
    if not arrays_exact or not auxiliary_exact:raise AssertionError('Repeated exact-state profiling chunk differs')
    if optimized is not None:
        if (not all(np.array_equal(a,b) for a,b in zip(first.ordered_state_arrays,optimized_state.ordered_state_arrays))
            or first.time_seconds!=optimized_state.time_seconds
            or first.grid_origin_seconds!=optimized_state.grid_origin_seconds
            or first.component_tick_indices!=optimized_state.component_tick_indices
            or first.sample_index!=optimized_state.sample_index or first.spike_states!=optimized_state.spike_states
            or first.delay_states!=optimized_state.delay_states or first.link_delay_states!=optimized_state.link_delay_states
            or first.input_position!=optimized_state.input_position):
            raise AssertionError('Schema-cache optimization changed states/clocks/RNG/history/input identity')
    def label(key):
        filename,line,name=key;p=Path(filename)
        try:filename=p.resolve().relative_to(ROOT).as_posix()
        except (ValueError,OSError):filename=p.name
        return filename+':'+str(line)+':'+name
    stats=pstats.Stats(profiler);rows=[]
    for key,(primitive,total,self_seconds,cumulative,callers) in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:65]:
        rows.append({'function':label(key),'primitive_calls':primitive,'total_calls':total,
            'self_seconds':self_seconds,'cumulative_seconds':cumulative,
            'callers':[{'function':label(k),'raw_cProfile_tuple':v} for k,v in callers.items()]})
    tick=time.perf_counter()
    store.save(output/'timed_complete_checkpoint.json',baseline,report=None,origin_seconds=0.)
    save_seconds=time.perf_counter()-tick
    result={'schema':'prepared_background_short_profile_v1','UTC':datetime.now(timezone.utc).isoformat(),
        'scope':'One exact existing-model1ms background chunk, repeated only for cProfile; no new optical/physiological condition',
        'source_review_commit':'3333811d2dd737cacbb30bf746b19ad0ebdfb0e0','model_hash':runner.model.model_hash,
        'checkpoint_strict_reuse':True,'checkpoint_source':checkpoint.relative_to(ROOT).as_posix(),
        'source_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'state_scalars':sum(s.shape[0] for s in runner.model.state_specs),'population_counts':{k:len(v.cells.cell_id) for k,v in compiled.populations.items()},
        'links':[{**{key:getattr(link.specification,key) for key in ('source_component','source_port','target_component','target_port')},
                  'shape':list(link.matrix.shape),'edges':link.matrix.nnz} for link in compiled.links],
        'setup':setup,'unprofiled_chunk':plain,'cached_unprofiled_chunk':optimized,'profiled_chunk':profiled,
        'schema_cache_comparison':args.compare_schema_cache,
        'schema_cache_speedup_same_state_unprofiled':None if optimized is None else plain['wall_seconds']/optimized['wall_seconds'],
        'engine_source_sha256':hashlib.sha256((ROOT/'retina_sim/engine.py').read_bytes()).hexdigest(),
        'profiler_overhead_not_a_throughput_estimate':True,
        'repeated_state_arrays_exact':arrays_exact,'repeated_auxiliary_clocks_RNG_history_exact':auxiliary_exact,
        'single_complete_checkpoint_save_seconds':save_seconds,'top_cumulative_functions':rows,
        'physiological_validation':False,'simulation_seconds_per_chunk':.001}
    (output/'profile.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('SHORT_PROFILE_COMPLETE',json.dumps({k:result[k] for k in ('model_hash','unprofiled_chunk','profiled_chunk','single_complete_checkpoint_save_seconds','repeated_state_arrays_exact','repeated_auxiliary_clocks_RNG_history_exact')}),flush=True)
    print('TOP_PROFILE_FUNCTIONS',json.dumps([{k:v for k,v in row.items() if k!='callers'} for row in rows[:12]]),flush=True)
if __name__=='__main__':main()
