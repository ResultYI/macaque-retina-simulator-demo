"""Recheck the archived failure against the original equations and a tighter oracle."""
from pathlib import Path
import argparse
import json
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retina_sim.group_inner_dynamics import GroupInnerDynamics
from retina_sim.group_time_storage import load_bundle
from retina_sim.spatial_group_model import GroupModelAssembly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--oracle', action='store_true')
    args = parser.parse_args()
    started = perf_counter()
    evidence = ROOT/'reports/solver_scc_radau_20261025_v1/feasibility_r5/first_failed_numeric_witness'
    bundle = load_bundle(evidence, expected_kind='validation_evidence')
    witness = bundle['payload']
    protocol = json.loads((ROOT/'configurations/numerical_equivalence/group_scc_radau_fp64_v1.json').read_text(encoding='utf8'))
    seed = load_bundle(ROOT/protocol['construction_seed'], expected_kind='source_construction_seed')['payload']
    assembly = GroupModelAssembly(ROOT/protocol['group_artifact'])
    model = GroupInnerDynamics(assembly, seed['inner_parameters'])
    time = witness['reference']['time_seconds']
    states = {mode: {k: witness[mode]['states'][k] for k in model.state_keys}
              for mode in ('reference', 'radau')}
    original = {mode: model.rhs(time, state) for mode, state in states.items()}
    report = dict(scope='Archived full-graph witness; no acceptance criteria changed',
                  witness_manifest_hash=bundle['manifest_hash'], stamp=witness['comparison']['stamp'],
                  issues=witness['comparison']['issues'], contacts={})
    for component in ('AII_FMB_glycine', 'AII_DB1_glycine'):
        key = ('AII_AC', component, 'a')
        tau = model.cp(component, 'tau_rise', positive=True)
        row = witness['comparison']['per_cell']['rhs.'+':'.join(key)]
        state_row = witness['comparison']['per_cell']['states.'+':'.join(key)]
        ratio = np.abs(row['difference'])/row['allowance']
        worst = int(np.argmax(ratio))
        delta = states['radau'][key]-states['reference'][key]
        rate_difference = (original['radau'][key]+states['radau'][key]/tau
                           -original['reference'][key]-states['reference'][key]/tau)
        report['contacts'][component] = dict(
            count=len(delta), first_rhs_difference=float(abs(row['difference'][0])),
            first_rhs_allowance=float(row['allowance'][0]), first_rhs_ratio=float(ratio[0]),
            worst_rhs_index=worst, worst_rhs_ratio=float(ratio[worst]),
            first_state_difference=float(delta[0]),
            maximum_state_budget_ratio=float(np.max(abs(state_row['difference'])/state_row['allowance'])),
            tau_rise_first_seconds=float(tau[0]),
            predicted_first_rhs_from_state=float(-delta[0]/tau[0]),
            maximum_source_rate_difference=float(np.max(abs(rate_difference))),
            saved_rhs_reproduction_max_abs={mode:float(np.max(abs(original[mode][key]-witness[mode]['rhs'][key])))
                                            for mode in states})
    if args.oracle:
        from retina_sim.group_time_runner import GroupTimeArm
        arm = GroupTimeArm(assembly, seed, protocol, mode='reference')
        arm.initial_commit()
        begin, end = protocol['start_seconds'], protocol['start_seconds']+1e-4
        oracle_started = perf_counter()
        oracle = arm.inner.advance_reference(arm.inner_state, begin, end, rtol=2e-12, atol=2e-14)
        exact = oracle.evaluate(time, witness['reference']['side'])
        report['tighter_original_DOP853_oracle'] = dict(rtol=2e-12, atol=2e-14,
            wall_seconds=perf_counter()-oracle_started, metrics=oracle.metrics, contacts={})
        for component in report['contacts']:
            key = ('AII_AC', component, 'a')
            report['tighter_original_DOP853_oracle']['contacts'][component] = {
                mode:dict(first_state_error=float(states[mode][key][0]-exact[key][0]),
                          maximum_state_error=float(np.max(abs(states[mode][key]-exact[key]))))
                for mode in states}
    report['wall_seconds'] = perf_counter()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
