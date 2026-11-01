"""Describe saved event differences without changing any acceptance threshold."""
import json
import math
from pathlib import Path

ROOT = Path(r'D:\macaque retinal simulator\reports')
OUT = ROOT / 'CELL6_COMPILED_INNER_V1'
OLD = ROOT / 'CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1/runtime_monitor_repair_v1/cache/g1'
NEW = OUT / 'formal_g1/cache/g1'


def main():
    a = json.loads((OLD / 'GROUP_OBJECT_EVENTS.json').read_text())
    b = json.loads((NEW / 'GROUP_OBJECT_EVENTS.json').read_text())
    identities = len(a) == len(b) and all(
        {k: v for k, v in x.items() if k != 'time_seconds'} ==
        {k: v for k, v in y.items() if k != 'time_seconds'}
        for x, y in zip(a, b)
    )
    changes = [dict(index=i, reference=x, compiled=y,
                    absolute_difference_seconds=abs(x['time_seconds'] - y['time_seconds']),
                    difference_in_reference_ULP=abs(x['time_seconds'] - y['time_seconds']) / math.ulp(x['time_seconds']))
               for i, (x, y) in enumerate(zip(a, b)) if x['time_seconds'] != y['time_seconds']]
    result = dict(status='INCONCLUSIVE_ORIGINAL_EVENT_TIME_WITNESS_MISSING',
                  reference_count=len(a), compiled_count=len(b),
                  identities_order_and_bins_exact=identities,
                  bitwise_event_times_equal=not changes,
                  changed_timestamp_count=len(changes),
                  max_absolute_difference_seconds=max((x['absolute_difference_seconds'] for x in changes), default=0.),
                  max_reference_ULP_difference=max((x['difference_in_reference_ULP'] for x in changes), default=0.),
                  original_contract='GroupTimeComparator._draw: original uniform/rate witness and lambda-derived interval; no new time tolerance introduced',
                  evidence_gap='Formal caches retain event timestamps and endpoint PCG/history, not every-bin original uniform/rate witness. Exact endpoint PCG does not substitute for those witnesses.',
                  no_model_rerun=True, no_acceptance_threshold_changed=True,
                  original_failed_comparison_preserved='FORMAL_RESULT.json', changes=changes)
    (OUT / 'FORMAL_EVENT_SCOPE.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k != 'changes'}))


if __name__ == '__main__':
    main()
