"""Compare the single completed formal cache with its preserved exact input baseline."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import atomic_json
from retina_sim.group_time_storage import load_bundle
from retina_sim.group_time_monitor import frozen_budget,_strip_variable_aux
from retina_sim.contracts import canonical_hash
BASE=ROOT/'reports/CELL6_COMPILED_INNER_V1'
FORMAL=BASE/'formal_g1';CACHE=FORMAL/'cache/g1'
OLD=ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1/runtime_monitor_repair_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    control=FORMAL/'capture/g1_process_handoffs_guard_v2'
    resource=read(control/'RESOURCE.json')
    if resource['status']!='PASS' or read(control/'TERMINAL.json')['status']!='EXITED_0':raise RuntimeError('Formal controller not complete')
    current=read(CACHE/'manifest.json');original=read(OLD/'cache/g1/manifest.json')
    if current['status']!='COMPLETE_COMPILED_BACKEND_CELL6_CACHE' or current['samples']!=71001:raise RuntimeError('Formal cache incomplete')
    new_input=read(CACHE/'INPUT_AND_MODEL_IDENTITY.json');old_input=read(OLD/'cache/g1/INPUT_AND_MODEL_IDENTITY.json')
    if new_input['input_payload_sha256']!=old_input['input_payload_sha256'] or new_input['target']!=old_input['target']:raise RuntimeError('Input or target changed')
    target=[]
    with np.load(OLD/'cache/g1/endogenous.npz',allow_pickle=False) as a,np.load(CACHE/'endogenous.npz',allow_pickle=False) as b:
        if not np.array_equal(a['time_source_s'],b['time_source_s']):raise RuntimeError('Full time grid differs')
        for name,unit in [('V','V'),('gE','S'),('gI','S')]:
            budget=frozen_budget(np.array([a[name][0]]),unit)[2][0];error=float(np.max(abs(a[name]-b[name])))
            target.append(dict(field=name,nodes=len(a[name]),max_abs_error=error,original_budget=float(budget),passed=bool(error<=budget)))
        rate_error=float(np.max(abs(a['expected_rate_base_Hz']-b['expected_rate_base_Hz'])))
    from tools.cell6_halo_context_v1 import load
    assembly,seed=load()
    new_cp=read(CACHE/'LATEST_CHECKPOINT.json')['directory'];old_cp=read(OLD/'cache/g1/LATEST_CHECKPOINT.json')['directory']
    new=load_bundle(CACHE/new_cp,expected_kind='validation_evidence')['payload']
    old=load_bundle(OLD/'cache/g1'/old_cp,expected_kind='validation_evidence')['payload']
    if new['controller']['time_seconds']!=7.1 or old['controller']['time_seconds']!=7.1:raise RuntimeError('Endpoint differs')
    units={(b.population,b.mechanism,b.name):b.unit for b in (*assembly.state_blocks,*assembly.probability_blocks,*assembly.auxiliary_z1_blocks)}
    endpoint=[]
    for key,a in old['inner'].items():
        budget=frozen_budget(seed['states'][key],units[key])[2];error=abs(new['inner'][key]-a)
        endpoint.append(dict(field=list(key),rows=len(a),max_budget_ratio=float(np.max(error/budget,initial=0)),passed=bool(np.all(error<=budget))))
    histories=canonical_hash(_strip_variable_aux(old['spike_states']))==canonical_hash(_strip_variable_aux(new['spike_states']))
    rates=[]
    for pop,fields in old['spike_states'].items():
        a=np.asarray(fields['conditional_rate_Hz']);b=np.asarray(new['spike_states'][pop]['conditional_rate_Hz'])
        pa=seed['inner_parameters']['populations'][pop];Bv=frozen_budget(seed['states'][pop,'RGC_spike_history','V'],'V')[2]
        allowed=pa['alpha']/pa['beta']*Bv+8192*np.finfo(float).eps*np.maximum(abs(a),1.)
        rates.append(dict(population=pop,max_error_Hz=float(np.max(abs(a-b),initial=0)),max_original_budget_ratio=float(np.max(abs(a-b)/allowed,initial=0)),passed=bool(np.all(abs(a-b)<=allowed))))
    events=canonical_hash(read(CACHE/'GROUP_OBJECT_EVENTS.json'))==canonical_hash(read(OLD/'cache/g1/GROUP_OBJECT_EVENTS.json'))
    native_equal=canonical_hash(new['native'])==canonical_hash(old['native'])
    receipts=read(control/'SEGMENT_RECEIPTS.json');backend=[]
    for receipt in receipts:
        p=Path(receipt['capture']);evidence=read(p/'COMPILED_EXECUTION.json')
        if read(p/'DEBUG_TERMINAL.json')['native_failure_captured']:raise RuntimeError('Native failure in formal cache')
        backend.append(dict(index=receipt['index'],source_stop=receipt['source_stop'],**evidence))
    before=read(OLD/'capture/g1_process_handoffs_v1/RESOURCE.json');a=original['neural_metrics'];b=current['neural_metrics']
    old_receipts=read(OLD/'capture/g1_process_handoffs_v1/SEGMENT_RECEIPTS.json')
    old_successful_workers=sum(v['resource']['wall_seconds'] for v in old_receipts)
    new_successful_workers=sum(v['resource']['wall_seconds'] for v in receipts)
    failed_attempt=read(FORMAL/'capture/g1_part01_v1/TERMINAL.json')['wall_seconds']
    from datetime import datetime
    first_start=datetime.fromisoformat(read(FORMAL/'capture/g1_part00_v1/CONTROLLER.json')['utc'].replace('Z','+00:00')).timestamp()
    interrupted_end_to_end=(control/'TERMINAL.json').stat().st_mtime-first_start
    times=dict(reference_successful_workers_seconds=old_successful_workers,compiled_successful_workers_seconds=new_successful_workers,
        successful_worker_speedup=old_successful_workers/new_successful_workers,
        compiled_continuation_controller_seconds=resource['wall_seconds'],successful_prefix_worker_seconds=receipts[0]['resource']['wall_seconds'],
        failed_worker_seconds=failed_attempt,interrupted_end_to_end_seconds=interrupted_end_to_end,
        end_to_end_includes_failure_and_repair_pause=True,reference_total_wall_seconds=before['wall_seconds'],compiled_prefix_plus_continuation_seconds=receipts[0]['resource']['wall_seconds']+resource['wall_seconds'],
        prefix_plus_continuation_speedup=before['wall_seconds']/(receipts[0]['resource']['wall_seconds']+resource['wall_seconds']),reference_native_seconds=a['native_wall_seconds'],compiled_native_seconds=b['native_wall_seconds'],
        reference_inner_seconds=a['inner_wall_seconds'],compiled_inner_seconds=b['inner_wall_seconds'],
        inner_speedup=a['inner_wall_seconds']/b['inner_wall_seconds'],reference_snapshot_seconds=a['snapshot_wall_seconds'],compiled_snapshot_seconds=b['snapshot_wall_seconds'],
        source='Historical full same-input g1 baseline vs new formal run; same15 segment schedule. Primary comparison sums successful workers consistently; failure and repair pause listed separately, not simultaneous performance sampling')
    passed=all(v['passed'] for v in target+endpoint+rates) and histories and events and native_equal and current['physical_gate_passed']
    report=dict(status='FORMAL_CACHE_VALIDATED' if passed else 'FORMAL_CACHE_COMPARISON_FAILED',passed=passed,
        source_group=1,target=new_input['target'],input_program_exactly_equal=True,scientific_parameters_or_topology_changed=False,
        source_simulated_seconds=7.1,target_full_grid_comparison=target,inner_endpoint_comparison=endpoint,
        native_endpoint_and_retained_intervals_bitwise_equal=native_equal,exact_PCG_and_discrete_history=histories,exact_all_group_object_events=events,
        auxiliary_rate_original_budget=rates,original_expected_rate_max_abs_error_Hz=rate_error,
        expected_rate_error_scope='Reported absolute error; no invented standalone Hz budget; state and propagated endpoint rate checks separate',
        error_scope='50ms all-state501-node paired curves;7.1s target3 ports all71001 nodes and inner all-state endpoint; not all-state all-time comparison',
        timings=times,segments=len(receipts),compiled_runtime=backend,
        total_compiled_rhs_calls=sum(v['compiled_rhs_calls'] for v in backend),
        total_construction_and_first_rhs_seconds=sum(v['construction_and_first_rhs_seconds'] for v in backend),
        python_continuous_rhs_fallback_calls=sum(v['python_direct_continuous_rhs_fallback_calls'] for v in backend),
        cold_definition='Actual per-instance plan construction plus first RHS call, included in full controller wall;not just dispatcher lazy-registration time',
        peak_RSS_bytes=resource['peak_rss_bytes'],original_numerical_thresholds_unchanged=True,
        formal_cache_is_verification_not_new_cell_coverage=True,defensive_BC_constant_validation_guard_used=True,native_corruption_root_cause_proven=False,FIT_READY=False,background_qualified=False,production_adopted=False)
    atomic_json(BASE/'FORMAL_RESULT.json',report)
    if not passed:raise RuntimeError('Formal cache violates original comparison contracts')
    print(json.dumps(dict(status=report['status'],timings=times,compiled_calls=report['total_compiled_rhs_calls'])))


if __name__=='__main__':main()
