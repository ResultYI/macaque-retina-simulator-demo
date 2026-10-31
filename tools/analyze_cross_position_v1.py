"""Two-position functional coverage from original coupled responses; no model run."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.cross_position_context_v1 import read, write
from retina_sim.group_time_monitor import frozen_budget
from retina_sim.group_time_storage import load_bundle
from retina_sim.spatial_group_graph import digest


def csv_write(path, rows):
    with path.open('x', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--runs', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args(); prepared = args.prepared.resolve()
    base = args.runs.resolve() if args.runs else prepared / 'dynamic_24g_v1'
    out = args.output.resolve() if args.output else base / 'analysis'
    out.mkdir(exist_ok=False)
    identity = read(base / 'standard/IDENTITY.json')
    tight_id = read(base / 'tight/IDENTITY.json')
    if identity['RGC_identity'] != tight_id['RGC_identity']: raise ValueError('Paired model identities differ')
    std = dict(np.load(base / 'standard/RESPONSE.npz', allow_pickle=False))
    tight = dict(np.load(base / 'tight/RESPONSE.npz', allow_pickle=False))
    t = std['time_seconds']; expected = np.arange(901) * .001
    if not np.array_equal(t, expected) or not np.array_equal(t, tight['time_seconds']):
        raise ValueError('Original 901-node native grid required')
    short = dict(np.load(base / 'short6/RESPONSE.npz', allow_pickle=False))
    old_root = ROOT / 'reports/layerwise_physiology/RGC_functional_v1_laptop_20261029'
    old = dict(np.load(old_root / 'source_coupled_v1/standard/RESPONSE.npz', allow_pickle=False))
    if not np.array_equal(np.rint(old['time_seconds'] / .001).astype(int), np.arange(901)):
        raise ValueError('Saved4.73 native event indices differ')
    old_id = read(old_root / 'IDENTITY.json')
    old_rows = {r['type']: r for r in read(old_root / 'source_coupled_v1/analysis/TYPE_COMPLETION.json')}
    numerical = []; budgets = {}
    for f, contract in identity['fields'].items():
        initial = std[f][:, 0]; unit = contract['unit']
        budget = (frozen_budget(std[contract['population'] + '__voltage'][:, 0], 'V')[2] / .0001
                  if unit == 'V/s' else frozen_budget(initial, 's^-1' if unit == 'Hz' else unit)[2])
        error = np.max(abs(std[f] - tight[f]), axis=1); budgets[f] = budget
        numerical.append(dict(field=f, unit=unit, maximum_absolute_error=float(error.max()),
            maximum_error_over_frozen_budget=float(np.max(error / budget)), passed=bool(np.all(error <= budget))))
    ends = [dict(np.load(base / name / 'STATE_END.npz', allow_pickle=False)) for name in ['standard', 'tight']]
    end_rows = []
    seed = load_bundle(prepared / 'background_24g_v3/seed', expected_kind='source_construction_seed')['payload']
    for f, contract in identity['state_fields'].items():
        key = tuple(f.split('__'))
        budget = frozen_budget(seed['states'][key], contract['unit'])[2]
        error = abs(ends[0][f] - ends[1][f])
        end_rows.append(dict(field=f, unit=contract['unit'], maximum_absolute_error=float(error.max(initial=0)),
            maximum_error_over_frozen_budget=float(np.max(error / budget, initial=0)), passed=bool(np.all(error <= budget))))
    rows = []; curves = []; parameters = []; total_events = 0
    old_seed = load_bundle(ROOT / 'reports/cone_aperture_eccentricity/HC_new_v1/seed', expected_kind='source_construction_seed')['payload']
    first, second = (t >= .1) & (t < .2), (t >= .4) & (t < .5)
    for pop, info in identity['types'].items():
        w = np.asarray(info['represented_members'], float)
        mean = lambda a: np.average(a, weights=w, axis=0)
        get = lambda data, f: data[pop + '__' + f]
        V = get(std, 'voltage'); rate = get(std, 'expected_binary_count_rate')
        delta = V - V[:, :1]; dv_first = delta[:, first].mean(axis=1); dv_second = delta[:, second].mean(axis=1)
        b = budgets[pop + '__voltage']
        numerical_passed = all(x['passed'] for x in numerical if x['field'].startswith(pop + '__'))
        causal = bool(np.all(abs(delta[:, t < .05]) <= b[:, None]))
        background = bool(np.all(abs(get(short, 'voltage') - get(short, 'voltage')[:, :1]) <= b[:, None]))
        probability_error = max(float(np.max(abs(get(z, 'history_law_after_tick').sum(2) - 1))) for z in [std, tight])
        sampled = get(std, 'sampled_bin_counts'); times = get(std, 'event_time_seconds')
        sampling = bool(sampled.shape == (info['groups'], 900) and np.isin(sampled, [0, 1]).all()
            and len(times) == int(sampled.sum()) and np.all((times >= 0) & (times < .9)) and probability_error <= 1e-12)
        resolved = bool(np.ptp(mean(rate)) > max(5 * np.max(abs(rate - get(tight, 'expected_binary_count_rate'))),
            np.finfo(float).eps * max(1., float(rate.max())) * 8192))
        mixed = pop == 'small_bistratified_S_ON'
        sign = 1 if pop.endswith('_ON') else -1
        polarity = None if mixed else bool(np.all(sign * dv_first > b) and np.all(sign * dv_second < -b))
        inhibition = bool(np.all(get(std, 'inhibitory_conductance') >= 0)
                          and np.all(get(std, 'inhibitory_current') <= 1e-25)
                          and np.any(get(std, 'inhibitory_current') < 0))
        passed = all([numerical_passed, causal, background, sampling, resolved, inhibition, polarity is not False])
        status = ('MIXED_INPUT_COVERED_S_ON_POLARITY_UNTESTED' if mixed else 'CONDITIONAL_FUNCTIONAL_COVERAGE_COMPLETE') if passed else 'FUNCTIONAL_GAP_RECORDED'
        total_events += int(sampled.sum())
        row = dict(type=pop, actual_eccentricity_deg=.06431027659945468, groups=info['groups'],
            represented_members=info['total_represented_members'], status=status,
            baseline_V_mV=float(mean(V)[0] * 1000), baseline_source_rate_Hz=float(mean(rate)[0]),
            common_decrease_deltaV_mV=float(mean(delta)[first].mean() * 1000),
            common_increase_deltaV_mV=float(mean(delta)[second].mean() * 1000),
            late_rod_deltaV_mV=float(mean(delta)[800:900].mean() * 1000),
            source_rate_min_Hz=float(mean(rate).min()), source_rate_max_Hz=float(mean(rate).max()),
            rate_dynamic_resolved=resolved, numerical_passed=numerical_passed, causal_passed=causal,
            background_short6ms_passed=background, preprobe_50ms_stationary=causal,
            polarity_passed=polarity, S_ON_spectral_polarity_not_tested=mixed,
            inhibition_current_sign_passed=inhibition, inhibitory_removal_dynamic_control_run=False,
            sampling_passed=sampling, sampled_object_events=int(sampled.sum()),
            expected_object_events=float(get(std, 'expected_bin_count_rate').sum() * .001),
            identical_standard_tight_binary_draws=bool(np.array_equal(sampled, get(tight, 'sampled_bin_counts'))),
            maximum_probability_mass_error=probability_error,
            baseline_gE_per_member_nS=float(get(std, 'excitatory_conductance')[:, 0].sum() / w.sum() * 1e9),
            baseline_gI_per_member_nS=float(get(std, 'inhibitory_conductance')[:, 0].sum() / w.sum() * 1e9))
        rows.append(row)
        previous = old_rows[pop]
        for position, z, members in [('near', std, w), ('reference473', old, np.asarray(old_id['types'][pop]['represented_members'], float))]:
            avg = lambda a: np.average(a, weights=members, axis=0)
            v = get(z, 'voltage'); rr = get(z, 'expected_binary_count_rate')
            vm, rm = avg(v), avg(rr)
            eg, ig = avg(get(z, 'excitatory_conductance')), avg(get(z, 'inhibitory_conductance'))
            ep = get(z, 'excitatory_conductance').sum(0) / members.sum()
            ip = get(z, 'inhibitory_conductance').sum(0) / members.sum()
            for j, time_s in enumerate(t):
                curves.append(dict(position=position, type=pop, time_seconds=float(time_s),
                    V_mV=float(vm[j] * 1000), deltaV_local_background_mV=float((vm[j] - vm[0]) * 1000),
                    source_rate_Hz=float(rm[j]), delta_rate_local_background_Hz=float(rm[j] - rm[0]),
                    mean_group_gE_nS=float(eg[j] * 1e9), mean_group_gI_nS=float(ig[j] * 1e9),
                    gE_per_represented_member_nS=float(ep[j] * 1e9),
                    gI_per_represented_member_nS=float(ip[j] * 1e9)))
        for name in ['C', 'gL', 'EL', 'EE', 'EI', 'alpha', 'mu', 'beta', 'spike_history_kernel', 'spike_interval', 'initial_spike_history']:
            current = seed['inner_parameters']['populations'][pop][name]
            prior = old_seed['inner_parameters']['populations'][pop][name]
            parameters.append(dict(type=pop, parameter=name, identical_existing_value=bool(np.array_equal(np.asarray(current), np.asarray(prior))),
                current_value=json.dumps(np.asarray(current).tolist()), eccentricity_function_introduced=False))
    coordinates = []
    artifact = ROOT / 'reports/spatial_group_graph_v02_20261025_v1/artifact_r2'
    for position, source in [('near', identity), ('reference473', old_id)]:
        for pop, info in source['types'].items():
            folder = artifact / 'populations' / pop
            all_ids = np.load(folder / 'group_ids.npy', allow_pickle=False).astype(str)
            lookup = {key: i for i, key in enumerate(all_ids)}
            selected = np.array([lookup[str(key)] for key in info['group_ids']])
            xy = np.load(folder / 'centroid_mm.npy', allow_pickle=False)[selected] / .223
            radii = np.linalg.norm(xy, axis=1)
            coordinates.append(dict(position=position, type=pop, groups=len(selected),
                patch_center_eccentricity_deg=.06431027659945468 if position == 'near' else 4.73,
                actual_group_ecc_min_deg=float(radii.min()), actual_group_ecc_max_deg=float(radii.max()),
                member_weighted_group_ecc_mean_deg=float(np.average(radii, weights=info['represented_members'])),
                coordinates='Original frozen group RF centroids; not individual fine-cell reconstruction'))
    comparison = []
    for row in rows:
        pop = row['type']; old_row = old_rows[pop]
        comparison.append(dict(type=pop, near_deg=row['actual_eccentricity_deg'], reference_deg=4.73,
            near_groups=row['groups'], reference_groups=old_row['groups'], near_members=row['represented_members'],
            reference_members=old_row['represented_members'], near_status=row['status'], reference_status=old_row['status'],
            near_V_mV=row['baseline_V_mV'], reference_V_mV=old_row['baseline_V_mV'],
            near_rate_Hz=row['baseline_source_rate_Hz'], reference_rate_Hz=old_row['baseline_source_rate_Hz'],
            near_release_down_deltaV_mV=row['common_decrease_deltaV_mV'], reference_release_down_deltaV_mV=old_row['common_decrease_deltaV_mV'],
            near_release_up_deltaV_mV=row['common_increase_deltaV_mV'], reference_release_up_deltaV_mV=old_row['common_increase_deltaV_mV'],
            near_events=row['sampled_object_events'], reference_events=old_row['actual_sampled_object_events'],
            interpretation='Relative local-background layer probes; no calibrated eccentricity dependence inferred'))
    csv_write(out / 'NEAR_TYPE_COMPLETION.csv', rows); write(out / 'NEAR_TYPE_COMPLETION.json', rows)
    csv_write(out / 'TWO_POSITION_FUNCTIONAL_COVERAGE.csv', comparison)
    csv_write(out / 'ALL_POSITION_TYPE_TIME_COURSES.csv', curves)
    csv_write(out / 'NUMERICAL_REFERENCE.csv', numerical)
    csv_write(out / 'FULL_INNER_ENDPOINT_REFERENCE.csv', end_rows)
    csv_write(out / 'RGC_PARAMETER_INHERITANCE.csv', parameters)
    csv_write(out / 'RGC_COORDINATE_COVERAGE.csv', coordinates)
    result = dict(status='TWO_POSITION_RECORDINGS_COMPLETE_WITH_DECLARED_FUNCTIONAL_LIMITS',
        near_actual_eccentricity_deg=.06431027659945468, reference_deg=4.73,
        full_inner_nstate=identity['full_inner_nstate'], types=len(rows), groups=sum(r['groups'] for r in rows),
        actual_sampled_object_events=total_events, native_grid_complete=True,
        maximum_numerical_error_over_frozen_budget=max(r['maximum_error_over_frozen_budget'] for r in numerical),
        full_inner_endpoint_maximum_error_over_budget=max(r['maximum_error_over_frozen_budget'] for r in end_rows),
        numerical_passed=all(r['passed'] for r in numerical + end_rows),
        inherited_RGC_parameters_equal=all(r['identical_existing_value'] for r in parameters),
        functional_gaps=[r['type'] for r in rows if r['status'] == 'FUNCTIONAL_GAP_RECORDED'],
        S_bistratified_separate_S_ON_polarity_verified=False, blue_yellow_opponency_verified=False,
        no_inhibition_control='Actual negative inhibitory currents and original algebra only; no extra suppression trajectory',
        background_scope='Independent fixedpoint, first-event check, measured6ms and preprobe50ms; no global physiology qualification',
        reference_results_recomputed=False, reference_final_SHA256=digest(old_root / 'FINAL_RESULT.json'),
        distinct_input_identities_disclosed=True, same_physical_background_calibration_established=False,
        eccentricity_function_fitted=False, complete_global_checkpoint=False,
        FIT_READY=False, background_qualified=False, production_adopted=False)
    write(out / 'RESULT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
