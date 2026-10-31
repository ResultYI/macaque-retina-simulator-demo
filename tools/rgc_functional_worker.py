"""Compute the declared retained-structure RGC assays, never fit coefficients."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_inner_dynamics import RGC_MECH, key
from retina_sim.parameter_system import data
from retina_sim.rgc_functional_replay import RGCFunctionalReplay
from retina_sim.rgc_functional_v1 import RGCFunctionalV1


def dump(path, value):
    path.write_text(json.dumps(data(value), indent=2, allow_nan=False) + '\n', encoding='utf8')


def source_witness(layer, replay, state, label):
    m = layer.model
    raw = m.outputs(layer.origin, state, layer.ac.bc.background)
    release = {p: raw[p, 'effective_release'] for p in layer.ac.bc.populations[:-1]}
    y, laws = replay.from_state(state), replay.law_from_state(state)
    original = layer.observe(layer.origin, state, layer.ac.bc.background)
    factored = layer.observe_ports(replay.raw(y, release, laws))
    difference = {f: float(np.max(abs(original[f] - factored[f]), initial=0)) for f in original}
    rhs = m.rhs(layer.origin, state, layer.ac.bc.background)
    expected = np.concatenate([*[rhs[k] for k in layer.ac.state_keys],
                               *[rhs[key(p, RGC_MECH, 'V')] for p in layer.populations]])
    actual = replay.rhs(y, release)
    error = float(np.max(abs(expected - actual), initial=0))
    if any(difference.values()) or error:
        raise ContractError('Factored original algebra differs: ' + label)
    return dict(label=label, all50_readouts_bitwise_equal=True, rhs_max_abs_error=error,
                source_V={p: raw[p, 'voltage'].tolist() for p in layer.populations})


def run_case(layer, replay, cache, case, out, deadline):
    if (out / (case + '.npz')).exists():
        raise FileExistsError('Preserve completed assay: ' + case)
    tight = case == 'assay_tight'
    control = case == 'no_inhibition_standard'
    background = case == 'background_standard'
    rtol, atol, step = (1e-11, 1e-14, .00025) if tight else (1e-8, 1e-11, .001)
    t = cache['time_seconds']
    forcing = {p: cache[p + '__effective_release'] for p in layer.ac.bc.populations[:-1]}
    if background:
        forcing = {p: np.repeat(v[:, :1], len(t), axis=1) for p, v in forcing.items()}
    if case == 's_channel_only_standard':
        forcing = {p: v if p == 'S_ON_BC' else np.repeat(v[:, :1], len(t), axis=1)
                   for p, v in forcing.items()}
    y, laws = replay.initial()
    auxiliary = layer.initial_spike_states()
    saved = {f: [] for f in layer.fields}
    continuous, law_after, expected, counts = [], {p: [] for p in layer.populations}, {}, {}
    events = {p: [] for p in layer.populations}
    nfev = 0
    started = time.perf_counter()
    for i, node in enumerate(t):
        if time.perf_counter() > deadline:
            raise TimeoutError('Owned worker exceeded its800s internal guard')
        release = {p: v[:, i] for p, v in forcing.items()}
        before = replay.raw(y, release, laws, remove_inhibition=control)
        if i < len(t) - 1:
            auxiliary, drawn = replay.sample(layer.origin + node, y, auxiliary)
            for p in layer.populations:
                expected.setdefault(p, []).append(before[p, 'expected_rate_Hz'])
                counts.setdefault(p, []).append(auxiliary[p]['last_bin_counts'])
                events[p] += drawn[p]
        laws = replay.transition(y, laws)
        observed = layer.observe_ports(replay.raw(y, release, laws, remove_inhibition=control))
        for f, value in observed.items():
            saved[f].append(value)
        continuous.append(y.copy())
        for p in layer.populations:
            law_after[p].append(laws[p].copy())
        if i == len(t) - 1:
            break
        def rhs(q, value):
            w = (q - node) / .001
            r = {p: (1 - w) * v[:, i] + w * v[:, i + 1] for p, v in forcing.items()}
            return replay.rhs(value, r, remove_inhibition=control)
        sol = solve_ivp(rhs, (node, t[i + 1]), y, method='DOP853', rtol=rtol,
            atol=atol, max_step=step, first_step=min(step, t[i + 1] - node), t_eval=[t[i + 1]])
        if not sol.success:
            raise ContractError(sol.message)
        y = sol.y[:, -1]
        nfev += sol.nfev
    packet = dict(time_seconds=t, continuous_states=np.asarray(continuous).T,
        **{f: np.asarray(v).T for f, v in saved.items()})
    for p in layer.populations:
        packet[p + '__history_law_after_tick'] = np.asarray(law_after[p]).transpose(1, 0, 2)
        packet[p + '__expected_bin_count_rate'] = np.asarray(expected[p]).T
        packet[p + '__sampled_bin_counts'] = np.asarray(counts[p]).T
        packet[p + '__event_time_seconds'] = np.asarray([e['time_seconds'] - layer.origin for e in events[p]])
        packet[p + '__event_group_index'] = np.asarray([e['cell_index'] for e in events[p]], dtype=np.int64)
    np.savez_compressed(out / (case + '.npz'), **packet)
    dump(out / (case + '_RUN.json'), dict(case=case, wall_seconds=time.perf_counter() - started,
        nfev=nfev, rtol=rtol, atol=atol, max_step=step, nodes=len(t), bins=len(t) - 1,
        continuous_state_values=replay.nstate, original_AC_equations=True,
        original_RGC_equations=True, native_sampler=True, same_saved_initial_RNG=True))
    print(json.dumps(dict(case=case, nfev=nfev, wall_seconds=time.perf_counter() - started)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--s-probe', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if not (out / 'PROTOCOL.json').is_file():
        raise ContractError('Predeclared fresh protocol required')
    deadline = time.perf_counter() + 800
    layer = RGCFunctionalV1(ROOT, 'configurations/research/RGC_FUNCTIONAL_V1_candidate.json')
    replay = RGCFunctionalReplay(layer)
    path = ROOT / 'reports/layerwise_physiology/BC_functional_v1/standard_ALL_BC_RESPONSE.npz'
    cache = dict(np.load(path, allow_pickle=False))
    if not np.allclose(cache['time_seconds'], np.arange(901) * .001, rtol=0, atol=1e-12):
        raise ContractError('Original901 native1ms nodes required')
    if args.s_probe:
        if not (out / 'S_TYPE_PROBE_PROTOCOL.json').is_file():
            raise ContractError('Predeclared S-type functional condition required')
        run_case(layer, replay, cache, 's_channel_only_standard', out, deadline)
        return
    initial = layer.initial()
    witnesses = [source_witness(layer, replay, initial, 'actual_joint_initial_state')]
    original_final = ROOT / 'reports/layerwise_physiology/BC_functional_v1/standard/06000_09000'
    flat = np.load(original_final / 'STATE_END.npz')['inner_flat']
    meta = json.loads((original_final / 'CONTINUATION_END.json').read_bytes())
    if canonical_hash(flat) != meta['inner_flat_sha256'] or meta['layer_identity'] != layer.ac.bc.identity:
        raise ContractError('Frozen900ms inner continuation differs')
    final_state = layer.model.unpack(flat)
    witnesses.append(source_witness(layer, replay, final_state, 'saved_original_full_inner_900ms'))
    dump(out / 'SOURCE_ALGEBRA_WITNESSES.json', witnesses)
    np.savez_compressed(out / 'SOURCE_900ms_ENDPOINT.npz',
        **layer.observe(layer.origin + .9, final_state, layer.ac.bc.background))
    dump(out / 'IDENTITY.json', dict(RGC_identity=layer.identity, AC_identity=layer.ac.identity,
        BC_identity=layer.ac.bc.identity, cache_path=str(path.relative_to(ROOT)),
        cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        types=layer.types, routes=layer.routes, fields=layer.fields,
        continuous_replay_states=replay.nstate, full_original_inner_states=layer.model.nstate,
        scope='Conditional original-equation BC cache replay; no whole-network trajectory equivalence claim'))
    dump(out / 'PARAMETERS.json', dict(
        populations={p: layer.model.p[p] for p in layer.populations},
        route_gains={r.name: layer.model.parameters['route_gains'][r.name] for r in replay.routes},
        source_provenance=layer.model.parameters.get('provenance'),
        source_identity=layer.model.parameters.get('identity'),
        units=dict(C='F/member', gL='S/member', EL='V', EE='V', EI='V',
                   alpha='Hz', mu='V', beta='V', spike_history_kernel='V', spike_interval='s'),
        interpretation='Inherited source-bound effective engineering parameters; no new fit or physiological calibration',
        multiplicity='Physical totals multiply target members once; all spike/rate outputs remain per sampled group object',
        no_Cell6_fit_or_target_PSTH_used=True))
    for case in json.loads((out / 'PROTOCOL.json').read_bytes())['cases']:
        run_case(layer, replay, cache, case, out, deadline)
    dump(out / 'INITIALIZATION_CHECK.json', dict(
        initial_RGC_voltage_derivative_V_s={p: layer.observe(layer.origin, initial, layer.ac.bc.background)[p + '__voltage_derivative'].tolist() for p in layer.populations},
        original_stationary_probability_law_retained=True,
        sampled_history_and_rng_reused=True, finite_record_spike_stationarity_claimed=False,
        upstream_seed_reinitialized=False, background_qualified=False, FIT_READY=False))


if __name__ == '__main__':
    main()
