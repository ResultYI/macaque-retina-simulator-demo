"""Bounded equivalent execution timing; never prepares a new background.

prepared: one cached real model,1ms background arms and2ms existing-image arms.
fixture: same interfaces on the existing small full-layer software fixture.
index: actual frozen native IDs only, no state allocation or neural simulation.
"""
from pathlib import Path
from dataclasses import replace
import argparse,gc,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
FROZEN_COMPILER=None
if '--frozen-v7-compiler' in sys.argv:
    from tools.benchmark_compiler_snapshot import use_v7_compiler
    FROZEN_COMPILER=use_v7_compiler()
from retina_sim import engine
from retina_sim.background import BackgroundPolicy,_WindowMonitor,constant_background_channels
from retina_sim.background_advance import StateOnlyBackgroundRunner
from retina_sim.background_checkpoint import MemoryGuard,BackgroundCheckpointStore
from retina_sim.contracts import SolverConfig,canonical_hash
from retina_sim.recording import RecordingPlan
from retina_sim.recording_indices import IndexedRecordingPlan
from tools.disk3mm_response_check import aligned_response_times,observation_path


def available_windows_commit_bytes():
    """Read additional commit capacity, distinct from available physical RAM."""
    import ctypes
    from ctypes import wintypes
    class Status(ctypes.Structure):
        _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD)]+[(key,ctypes.c_ulonglong) for key in
            ('ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]
    status=Status();status.dwLength=ctypes.sizeof(status)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(Status)];kernel.GlobalMemoryStatusEx.restype=wintypes.BOOL
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):raise ctypes.WinError(ctypes.get_last_error())
    return int(status.ullAvailPageFile)


def equal_states(a,b):
    if len(a.ordered_state_arrays)!=len(b.ordered_state_arrays):return False
    if not all(np.array_equal(x,y) for x,y in zip(a.ordered_state_arrays,b.ordered_state_arrays)):return False
    if not all(getattr(a.dynamic,key)==getattr(b.dynamic,key) for key in ('state_spec_hash','graph_hash','parameter_hash',
        'model_id','model_version','model_hash','input_interpolation','solver_config','evidence_tag','optional_rng_state')):return False
    return all(getattr(a,key)==getattr(b,key) for key in ('assembly_hash','time_seconds','grid_origin_seconds',
        'sample_index','component_tick_indices','spike_states','delay_states','link_delay_states','input_position'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('prepared','fixture','index'),default='prepared')
    parser.add_argument('--output',required=True)
    parser.add_argument('--export-legacy-bridge',action='store_true',help='After A timing, export the original-GPU short case for sequential B comparison')
    parser.add_argument('--frozen-v7-compiler',action='store_true',help='Execute pinned old compiler bytes to preserve the old checkpoint/capture identity')
    args=parser.parse_args();output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():raise ValueError('Use a fresh project output directory')
    output.mkdir();started=time.perf_counter();setup={}
    def save(value):(output/'result.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    if args.scope=='index':
        from retina_sim.disk_circuit import GlobalDiskCircuit
        config=json.loads((ROOT/'configurations/atlas_human_borrowed_disk3mm_floor_input_v2.json').read_bytes())
        circuit=GlobalDiskCircuit(ROOT,config['global_graph']);table=circuit.table('cones')
        original=ROOT/'reports/human_disk3mm_floor_transition_v7/transition'
        with np.load(original/'input.npz',allow_pickle=False) as data:rows=data['native_rows']
        ids=[]
        for first in range(0,len(rows),8192):ids.extend(table.rows(rows[first:first+8192]).global_cell_ids)
        ids=tuple(ids);response=json.loads((original/'matched_response/response_check.json').read_bytes())
        plan=RecordingPlan(cell_ids=frozenset(response['recorded_cell_ids']));fast=IndexedRecordingPlan(plan)
        tick=time.perf_counter();expected=plan.indices(ids);compile_start=time.perf_counter()
        np.testing.assert_array_equal(fast.indices(ids),expected);compile_seconds=time.perf_counter()-compile_start
        iterations=32;tick=time.perf_counter()
        for _ in range(iterations):old=plan.indices(ids)
        old_seconds=time.perf_counter()-tick;tick=time.perf_counter()
        for _ in range(iterations):new=fast.indices(ids)
        new_seconds=time.perf_counter()-tick;np.testing.assert_array_equal(old,new)
        result={'status':'ACTUAL_FROZEN_ID_INDEX_EQUIVALENCE_PASS','scope':'ID-index lookup only; no neural step or whole-run speed claim',
            'native_ID_count':len(ids),'selected_native_ID_count':len(new),'lookup_repetitions':iterations,
            'legacy_total_seconds':old_seconds,'cached_total_seconds':new_seconds,'one_time_compile_seconds':compile_seconds,
            'index_order_dtype_values_equal':True,'procedure':fast.procedure_identity,'wall_seconds':time.perf_counter()-started}
        save(result);print(json.dumps(result,indent=2),flush=True);return
    guard=None
    if args.scope=='prepared':
        guard=MemoryGuard(max_rss_bytes=14336*1024**2,min_available_bytes=8192*1024**2);guard.check()
        # Measured real worker peak was6.0-6.5GiB. Keep a conservative7GiB
        # construction/load allowance in addition to the unchanged8GiB reserve.
        minimum_before_build=15*1024**3
        minimum_commit_before_build=7*1024**3
        available_commit=available_windows_commit_bytes()
        if (guard.last['available_physical_bytes']<minimum_before_build or available_commit<minimum_commit_before_build):
            result={'status':'MEMORY_ADMISSION_BLOCKED_NO_MODEL_ALLOCATED','scope':args.scope,
                'minimum_available_before_build_bytes':minimum_before_build,'resource_measurement':guard.last,
                'available_Windows_commit_bytes':available_commit,'minimum_commit_before_build_bytes':minimum_commit_before_build,
                'reason':'Keep the original8GiB physical reserve plus conservative7GiB single-worker working allowance; also require at least that much Windows commit capacity. Historical RSS peak6.062GiB, historical private-commit peak unmeasured. No numerical criteria relaxed.',
                'saved_functional_state':'reports/human_disk3mm_functional_validation_v2_empty_sum/central/matched_response/background_checkpoint.json'}
            save(result);print(json.dumps(result,indent=2),flush=True);return
        from retina_platform.config import RunConfig
        from retina_sim.atlas_population_network import atlas_population_network
        from retina_sim.execution_backends import make_runner
        original=ROOT/'reports/human_disk3mm_floor_transition_v7/transition'
        manifest=json.loads((original/'manifest.json').read_bytes());config=RunConfig.from_dict(manifest['configuration'])
        construction=replace(config,native_background_values=None,frontend_signature='');tick=time.perf_counter()
        print('REAL_SHORT_PAIR_ASSEMBLY_BEGIN',flush=True)
        compiled,_,audit=atlas_population_network(ROOT,duration=.0001,light_context=construction.light_context(),
            owner_bounds_mm=config.owner_bounds(),halo_mm=.04,max_spatial_cells=config.max_spatial_cells,
            seed=config.seed,execution_profile=config.execution_profile,configuration_path=config.circuit_configuration_path)
        setup['assembly_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        print('REAL_SHORT_PAIR_ASSEMBLY_COMPLETE',setup['assembly_seconds'],flush=True)
        if not guard.check():raise MemoryError('Original reserve reached after assembly')
        runner=make_runner(compiled,config.neural_integration_backend);solver=SolverConfig(**audit['resolved_execution_profile']['solver'])
        reference=runner.initialize(solver);light=config.light_context();policy=config.background_policy()
        store=BackgroundCheckpointStore(ROOT,runner,reference,light,policy)
        save_json_binding=output/'current_background_binding.json'
        save_json_binding.write_text(json.dumps(store.binding,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        with (original/'matched_response/background_checkpoint.json').open('rb') as stream:prefix=stream.read(1024**2).decode('utf-8')
        key='"binding":';offset=prefix.index(key)+len(key);expected_binding,_=json.JSONDecoder().raw_decode(prefix[offset:])
        differences={key:{'expected':expected_binding.get(key),'actual':store.binding.get(key)}
            for key in set(expected_binding)|set(store.binding) if expected_binding.get(key)!=store.binding.get(key)}
        (output/'background_binding_differences.json').write_text(json.dumps(differences,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        print('REAL_SHORT_PAIR_BACKGROUND_BINDING_DIFFERENCE_KEYS',list(differences),flush=True)
        setup['runner_and_binding_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        print('REAL_SHORT_PAIR_RUNNER_AND_BINDING_COMPLETE',setup['runner_and_binding_seconds'],flush=True)
        if not guard.check():raise MemoryError('Original reserve reached before strict restore')
        baseline,saved=store.load(original/'matched_response/background_checkpoint.json',require_converged=True)
        setup['strict_converged_restore_seconds']=time.perf_counter()-tick;del saved
        print('REAL_SHORT_PAIR_STRICT_PREPARED_RESTORE_COMPLETE',baseline.time_seconds,setup['strict_converged_restore_seconds'],flush=True)
        if not guard.check():raise MemoryError('Original reserve reached after strict restore')
        from retina_sim.disk_input_packet import load_global_input
        packet=load_global_input(original/'input.npz',compiled,light,reference_id=compiled.global_disk_circuit.record['reference_id'])
        captured=packet.channels
        target,chosen,bc,_=observation_path(compiled,audit)
        trace_plan=RecordingPlan(states=tuple(name+'.terminal_release' for name in bc),cell_ids=frozenset(chosen),max_bytes=64*1024**2)
    else:
        from examples.full_network_fixture import full_fixture
        from retina_sim.hybrid import HybridRunner
        from retina_sim.runtime_profiles import LightContext
        tick=time.perf_counter();compiled,captured,_=full_fixture(ROOT)
        runner=HybridRunner(compiled,compute_backend='numba');solver=SolverConfig(rtol=1e-6,atol=1e-9,max_step=.0001,
            input_interpolation=next(iter(captured.values())).interpolation)
        baseline=runner.initialize(solver);reference=baseline;light=LightContext();policy=BackgroundPolicy()
        setup['fixture_build_and_compile_seconds']=time.perf_counter()-tick
        trace_plan=RecordingPlan(states=(),cell_ids=frozenset(('cones:0','cones:1','RGC_midget_ON:0')),max_bytes=64*1024**2)
    start=baseline.time_seconds;duration=.001;stop=start+duration
    channels=constant_background_channels(compiled,light,stop+.01,interpolation=solver.input_interpolation)
    times,_=aligned_response_times(runner,baseline.grid_origin_seconds,start,duration)
    fast=StateOnlyBackgroundRunner(runner);empty=RecordingPlan(ports=(),states=(),cell_ids=())
    def run_background(target):
        monitor=_WindowMonitor(target,reference,policy);points=[];before=dict(runner.integration_metrics)
        def observe(t,y,bundle):
            if guard is not None and not guard.check():raise MemoryError('Original memory reserve reached')
            points.append(t);monitor.observe(t,y,bundle)
        tick=time.perf_counter();state,result=target.advance(baseline,channels,stop,sample_times=times,
            recording_plan=empty,state_observer=observe);wall=time.perf_counter()-tick
        metrics={key:runner.integration_metrics[key]-before[key] for key in ('solver_calls','nfev','integration_wall_seconds')}
        return state,dict(wall_seconds=wall,observations=monitor.count,**metrics),monitor.report(duration),points
    warm=run_background(runner);del warm;gc.collect()
    print('REAL_SHORT_PAIR_BACKGROUND_WARMUP_COMPLETE',flush=True)
    saved_helper=engine._state_schema_hash
    engine._state_schema_hash=lambda model:engine.canonical_hash(model.state_specs)
    try:uncached,uncached_metrics,uncached_monitor,uncached_points=run_background(runner)
    finally:engine._state_schema_hash=saved_helper
    cached,cached_metrics,cached_monitor,cached_points=run_background(runner)
    optimized,optimized_metrics,optimized_monitor,optimized_points=run_background(fast)
    print('REAL_SHORT_PAIR_BACKGROUND_ARMS_COMPLETE',json.dumps({'uncached':uncached_metrics,'cached':cached_metrics,'state_only':optimized_metrics}),flush=True)
    if not (equal_states(uncached,cached) and equal_states(cached,optimized)
        and uncached_monitor==cached_monitor==optimized_monitor and uncached_points==cached_points==optimized_points
        and uncached_metrics['nfev']==cached_metrics['nfev']==optimized_metrics['nfev']
        and uncached_metrics['solver_calls']==cached_metrics['solver_calls']==optimized_metrics['solver_calls']):
        raise AssertionError('Equivalent background procedure comparison failed')
    save({'status':'BACKGROUND_EQUIVALENCE_PASS_TRACE_PENDING','scope':args.scope,
        'legacy_uncached_schema':uncached_metrics,'legacy_cached_schema':cached_metrics,
        'state_only_cached_schema':optimized_metrics,'complete_state_monitor_points_RNG_events_histories_equal':True})
    del uncached,cached,optimized;gc.collect()
    trace_duration=.002;trace_stop=start+trace_duration;trace_times,_=aligned_response_times(runner,baseline.grid_origin_seconds,start,trace_duration)
    stimulus={}
    for name,channel in captured.items():
        value=channel.at(.00005)
        stimulus[name]=replace(channel,start_seconds=start,stop_seconds=trace_stop,evaluator=lambda t,value=value:value,
            breakpoints_seconds=(),input_contract=dict(channel.input_contract,static_expected_rate_repeated_for_short_execution_benchmark=True))
    indexed=IndexedRecordingPlan(trace_plan);trace=[]
    for plan in (trace_plan,indexed):
        points=[];before=dict(runner.integration_metrics)
        def observe(t,y,bundle):
            if guard is not None and not guard.check():raise MemoryError('Original memory reserve reached')
            points.append(t)
        tick=time.perf_counter();state,result=runner.advance(baseline,stimulus,trace_stop,sample_times=trace_times,
            recording_plan=plan,state_observer=observe)
        trace.append((state,result,{'wall_seconds':time.perf_counter()-tick,'observer_points':points,
            'nfev':runner.integration_metrics['nfev']-before['nfev']}))
    left,right=trace
    if not equal_states(left[0],right[0]) or canonical_hash(left[1].metadata)!=canonical_hash(right[1].metadata):
        raise AssertionError('Trace index optimization changed state or metadata')
    for key,q in left[1].outputs_with_units.items():
        other=right[1].outputs_with_units[key]
        if ((q.unit,q.meaning,q.evidence_tag,q.coordinate_frame)!=(other.unit,other.meaning,other.evidence_tag,other.coordinate_frame)
            or not np.array_equal(q.values,other.values)):raise AssertionError('Trace output/units/order changed')
    if left[2]['observer_points']!=right[2]['observer_points'] or left[2]['nfev']!=right[2]['nfev']:
        raise AssertionError('Trace numerical schedule changed')
    result={'status':'SAME_STATE_BACKGROUND_AND_TRACE_EXECUTION_EQUIVALENCE_PASS','scope':args.scope,
        'frozen_compiler':FROZEN_COMPILER,
        'model_hash':runner.model.model_hash,'state_scalars':sum(a.size for a in baseline.ordered_state_arrays),'setup':setup,
        'new_background_preparation_performed':False,'background_duration_seconds':duration,
        'background':{'legacy_uncached_schema':uncached_metrics,'legacy_cached_schema':cached_metrics,'state_only_cached_schema':optimized_metrics,
            'state_only_incremental_speedup':cached_metrics['wall_seconds']/optimized_metrics['wall_seconds'],
            'cumulative_speedup_from_uncached_legacy':uncached_metrics['wall_seconds']/optimized_metrics['wall_seconds'],
            'complete_state_monitor_points_RNG_events_histories_equal':True},
        'recorded_inference':{'duration_seconds':trace_duration,'legacy':left[2],'indexed':right[2],
            'speedup':left[2]['wall_seconds']/right[2]['wall_seconds'],'complete_state_outputs_units_ID_order_metadata_equal':True},
        'procedures':{'background':fast.procedure_identity,'recording':indexed.procedure_identity},
        'scope_limit':'One warm short matched pair/triple; no whole-background extrapolation. Fixture numbers are not real large-graph inference numbers.',
        'historical_large_graph_phase_evidence':{'fresh_v6_background_seconds':15460.81949960001,
            'v7_cached_full_delivery_seconds':668.282,'single_strict_restore_profile_seconds':55.72577530000126,
            'different_phase_boundaries_do_not_divide_as_a_pure_speedup':True},
        'wall_seconds':time.perf_counter()-started}
    save(result);print(json.dumps({key:value for key,value in result.items() if key!='procedures'},indent=2),flush=True)
    if args.export_legacy_bridge:
        if args.scope!='prepared':raise ValueError('Real legacy bridge export requires the prepared frozen graph scope')
        from tools.benchmark_real_bridge import run_legacy_bridge
        result['legacy_bridge_case']=run_legacy_bridge(runner,reference,captured,audit,config,output/'legacy_bridge_case',guard)
        save(result)

if __name__=='__main__':main()
