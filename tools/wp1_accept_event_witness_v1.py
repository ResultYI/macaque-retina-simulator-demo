"""Supplement, never overwrite, the preserved strict-bitwise event comparison."""
from pathlib import Path
import hashlib,json
ROOT=Path(r'D:\macaque retinal simulator')
BASE=ROOT/'reports/CELL6_COMPILED_INNER_V1'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    original=read(BASE/'FORMAL_RESULT.json');scope=read(BASE/'FORMAL_EVENT_SCOPE.json');windows=[]
    if not read(BASE/'MONITOR_V3_REGRESSION.json')['passed']:raise RuntimeError('Original statistic equivalence not established')
    for index in range(5):
        folder=BASE/('event_witness_v3' if index==4 else 'event_witness_v2')/f'window{index}'
        result=read(folder/'WITNESS_RESULT.json');terminal=read(folder/'TERMINAL.json');debug=read(folder/'DEBUG_TERMINAL.json')
        if not result['passed'] or terminal['status']!='EXITED_0' or debug['native_failure_captured']:raise RuntimeError('Original witness gate not passed')
        if index==4 and result['monitor_statistic_repair']['calls']<=0:raise RuntimeError('Repair statistic kernel not exercised')
        windows.append(dict(index=index,folder=str(folder),result_sha256=sha(folder/'WITNESS_RESULT.json'),bins=result['bins'],
            source_worker_sha256=result.get('source_worker_sha256',sha(ROOT/'tools/wp1_event_witness_v2.py')),
            controller_wall_seconds=terminal['wall_seconds'],maximum_original_rate_or_probability_budget_ratio=max(x['max_budget_ratio'] for x in result['numeric_checks'])))
    wanted={int(round(x['reference']['bin_start_seconds']*1000)) for x in scope['changes']}
    if {v for w in windows for v in w['bins']}!=wanted:raise RuntimeError('Changed event coverage incomplete')
    numerical=all(x['passed'] for x in original['target_full_grid_comparison']+original['inner_endpoint_comparison']+original['auxiliary_rate_original_budget'])
    passed=numerical and original['native_endpoint_and_retained_intervals_bitwise_equal'] and original['exact_PCG_and_discrete_history'] and scope['identities_order_and_bins_exact']
    result=dict(status='ORIGINAL_NUMERICAL_AND_EVENT_CONTRACT_PASSED_WITH_RECOVERED_WITNESSES' if passed else 'FAILED',passed=passed,
        original_formal_result_sha256=sha(BASE/'FORMAL_RESULT.json'),original_strict_bitwise_failure_preserved=True,
        event_scope_sha256=sha(BASE/'FORMAL_EVENT_SCOPE.json'),monitor_v3_regression_sha256=sha(BASE/'MONITOR_V3_REGRESSION.json'),
        windows=windows,changed_timestamps_qualified=scope['changed_timestamp_count'],
        rule='Unchanged GroupTimeComparator._draw: same uniform/order/history/RNG/decision, original propagated rate and probability budgets, original lambda-derived event-time intervals',
        exact_replay_of_original_cached_event_values=True,bitwise_reference_vs_candidate_event_time_equality_claimed=False,
        no_sampler_equation_or_numerical_threshold_changed=True,
        remaining_scope='50ms all-state all-node comparison;7.1s targetV/gE/gI all71001 nodes plus all-inner endpoint, PCG/history and all event identities/bins;changed event times carry original sampler witnesses. Not all-state all-time7.1s tight proof.',
        native_corruption_root_cause_proven=False,FIT_READY=False,background_qualified=False,production_adopted=False)
    (BASE/'FORMAL_ACCEPTANCE_SUPPLEMENT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    if not passed:raise RuntimeError('Formal supplement failed')
    print(json.dumps(dict(status=result['status'],changed_events=result['changed_timestamps_qualified'],diagnostic_controller_seconds=sum(w['controller_wall_seconds'] for w in windows))))


if __name__=='__main__':main()
