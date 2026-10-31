"""Original6305-state coupled inner assay with native RGC sampling and clocks."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.parameter_system import data
from retina_sim.rgc_functional_replay import RGCFunctionalReplay
from retina_sim.rgc_functional_v1 import RGCFunctionalV1


def dump(path, value):
    path.write_text(json.dumps(data(value), indent=2, allow_nan=False) + '\n', encoding='utf8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--precision', choices=['standard', 'tight'], required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    layer = RGCFunctionalV1(ROOT, 'configurations/research/RGC_FUNCTIONAL_V1_candidate.json')
    replay, m, bc = RGCFunctionalReplay(layer), layer.model, layer.ac.bc
    state, auxiliary = layer.initial(), layer.initial_spike_states()
    saved, law = {f: [] for f in layer.fields}, {p: [] for p in layer.populations}
    pre_rates, counts, events, continuous = {}, {}, {p: [] for p in layer.populations}, []
    tight = args.precision == 'tight'
    rtol, atol, step = (1e-10, 1e-13, .000025) if tight else (1e-6, 1e-9, .0001)
    nfev = 0

    def capture(index, cone):
        observed = layer.observe(bc.origin + index * .0001, state, cone)
        for f, value in observed.items():
            saved[f].append(value)
        continuous.append(replay.from_state(state))
        for pop, value in replay.law_from_state(state).items():
            law[pop].append(value)

    for index in range(9000):
        if time.perf_counter() - start > 820:
            raise TimeoutError('Original coupled process exceeded820s internal guard')
        cone, rod, _, _ = bc.assay_inputs(index)
        tick = index % 10 == 0
        if tick:
            before = layer.observe(bc.origin + index * .0001, state, cone)
        state, following, drawn = layer.commit(index, state, cone, rod, auxiliary if tick else None)
        if tick:
            auxiliary = following
            for pop in layer.populations:
                pre_rates.setdefault(pop, []).append(before[pop + '__expected_binary_count_rate'])
                counts.setdefault(pop, []).append(auxiliary[pop]['last_bin_counts'])
                events[pop] += drawn[pop]
            capture(index, cone)
        left, right = bc.origin + index * .0001, bc.origin + (index + 1) * .0001
        sol = solve_ivp(layer.rhs, (left, right), m.pack(state), method='DOP853',
            rtol=rtol, atol=atol, max_step=step, first_step=min(step, right - left), t_eval=[right])
        if not sol.success:
            raise ContractError(sol.message)
        state = m.unpack(sol.y[:, -1])
        nfev += sol.nfev
        if (index + 1) % 1000 == 0:
            print(json.dumps(dict(precision=args.precision, completed_ms=(index + 1) / 10,
                                  wall_seconds=time.perf_counter() - start)), flush=True)
    cone, rod, _, _ = bc.assay_inputs(9000)
    state = bc.commit(9000, state, cone, rod)
    capture(9000, cone)
    packet = dict(time_seconds=np.arange(901) * .001, continuous_states=np.asarray(continuous).T,
                  **{f: np.asarray(v).T for f, v in saved.items()})
    for pop in layer.populations:
        packet[pop + '__history_law_after_tick'] = np.asarray(law[pop]).transpose(1, 0, 2)
        packet[pop + '__expected_bin_count_rate'] = np.asarray(pre_rates[pop]).T
        packet[pop + '__sampled_bin_counts'] = np.asarray(counts[pop]).T
        packet[pop + '__event_time_seconds'] = np.asarray([e['time_seconds'] - bc.origin for e in events[pop]])
        packet[pop + '__event_group_index'] = np.asarray([e['cell_index'] for e in events[pop]], dtype=np.int64)
    np.savez_compressed(out / 'RESPONSE.npz', **packet)
    flat = m.pack(state)
    np.savez_compressed(out / 'STATE_END.npz', inner_flat=flat)
    old = ROOT / 'reports/layerwise_physiology/BC_functional_v1' / args.precision / (
        '07500_09000' if tight else '06000_09000') / 'CONTINUATION_END.json'
    old_meta = json.loads(old.read_bytes())
    metrics = dict(precision=args.precision, RGC_identity=layer.identity,
        source_inner_identity=m.identity, source_BC_identity=bc.identity,
        full_inner_nstate=m.nstate, original_events_and_RNG=True, nfev=nfev,
        rtol=rtol, atol=atol, max_step=step, wall_seconds=time.perf_counter() - start,
        final_inner_sha256=canonical_hash(flat),
        exact_final_inner_hash_matches_original_BC=canonical_hash(flat) == old_meta['inner_flat_sha256'],
        all_original_inner_states_evolved=True, physical_gap_changed=False,
        upstream_Cone_HC_held_original_source_background=True,
        endpoint_has_no_sampled_future_bin=True, scientific_parameters_changed=False,
        continuation_scope='Inner experiment only; not complete native/global checkpoint', FIT_READY=False)
    dump(out / 'METRICS.json', metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
