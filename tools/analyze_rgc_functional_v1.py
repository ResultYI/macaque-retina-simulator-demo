"""Analyze retained RGC functional assays without another model construction."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.group_time_monitor import frozen_budget


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def write_csv(path, rows):
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def native_grid(actual):
    expected = np.arange(901) * .001
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError('Complete finite901-node original grid required')
    if not np.array_equal(np.rint(actual / .001).astype(np.int64), np.arange(901)):
        raise ValueError('Missing, duplicate or shifted native event index')
    error = float(np.max(abs(actual - expected)))
    bound = float(4 * np.spacing(1.0))
    if error > bound:
        raise ValueError('Actual time shift exceeds FP64 roundoff; no resampling permitted')
    return dict(nodes=901, integer_native_bin_indices_exact=True,
                max_roundoff_seconds=error, roundoff_bound_seconds=bound, resampled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coupled', action='store_true')
    args = parser.parse_args()
    base = args.output.resolve()
    out = base / 'source_coupled_v1/analysis' if args.coupled else base
    if args.coupled:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Preserve existing nonempty analysis directory')
        out.mkdir(exist_ok=True)
    if (out / 'RESULT.json').exists():
        raise FileExistsError('Preserve completed analysis')
    identity = json.loads((base / 'IDENTITY.json').read_bytes())
    paths = {name: base / (name + '.npz') for name in
             ('assay_standard', 'assay_tight', 'background_standard',
              'no_inhibition_standard', 's_channel_only_standard')}
    if args.coupled:
        paths.update({f'assay_{p}': base / 'source_coupled_v1' / p / 'RESPONSE.npz'
                      for p in ('standard', 'tight')})
    cases = {name: dict(np.load(path, allow_pickle=False)) for name, path in paths.items()}
    assay, tight, bg, removed, selective = (cases[n] for n in cases)
    t = assay['time_seconds']
    grids = {name: native_grid(z['time_seconds']) for name, z in cases.items()}
    dump(out / 'SOURCE_TIME_GRID_ALIGNMENT.json', grids)
    pair = []
    budgets = {}
    for field, contract in identity['fields'].items():
        initial = bg[field][:, 0].astype(np.float64)
        unit = contract['unit']
        if unit == 'V/s':
            b = frozen_budget(bg[contract['population'] + '__voltage'][:, 0], 'V')[2] / .0001
        else:
            b = frozen_budget(initial, 's^-1' if unit == 'Hz' else unit)[2]
        error = np.max(abs(assay[field] - tight[field]), axis=1)
        budgets[field] = b
        pair.append(dict(field=field, unit=unit, max_abs_error=float(error.max()),
                         max_error_over_budget=float(np.max(error / b)), passed=bool(np.all(error <= b))))
    write_csv(out / 'NUMERICAL_REFERENCE.csv', pair)
    endpoint = dict(np.load(base / 'SOURCE_900ms_ENDPOINT.npz'))
    source_end = [dict(field=f, max_abs_error=float(np.max(abs(assay[f][:, -1] - v))),
                       max_error_over_budget=float(np.max(abs(assay[f][:, -1] - v) / budgets[f])))
                  for f, v in endpoint.items()]
    write_csv(out / 'SOURCE_ENDPOINT_COMPARISON.csv', source_end)
    rows, curves = [], []
    for pop, info in identity['types'].items():
        w = np.asarray(info['represented_members'], dtype=float)
        mean = lambda value: np.average(value, axis=0, weights=w)
        get = lambda z, field: z[pop + '__' + field]
        voltage = get(assay, 'voltage')
        delta = voltage - get(bg, 'voltage')
        rate = get(assay, 'expected_binary_count_rate')
        d_rate = rate - get(bg, 'expected_binary_count_rate')
        first, second = (t >= .1) & (t < .2), (t >= .4) & (t < .5)
        b = budgets[pop + '__voltage']
        causal = bool(np.all(abs(delta[:, t < .05]) <= b[:, None]))
        background = bool(np.all(abs(get(bg, 'voltage') - get(bg, 'voltage')[:, :1]) <= b[:, None]))
        inhibition = bool(np.all(get(removed, 'voltage') - voltage >= -b[:, None]))
        group_first, group_second = delta[:, first].mean(axis=1), delta[:, second].mean(axis=1)
        selective_delta = get(selective, 'voltage') - get(bg, 'voltage')
        selective_rate = get(selective, 'expected_binary_count_rate') - get(bg, 'expected_binary_count_rate')
        if pop == 'small_bistratified_S_ON':
            polarity = bool(np.all(selective_delta[:, first].mean(axis=1) > b)
                            and np.all(selective_rate[:, first].mean(axis=1) > 0))
            polarity_scope = 'Selective S_ON_BC terminal input; common-input net response is mixed and OFF-like'
        else:
            sign = 1 if pop.endswith('_ON') else -1
            polarity = bool(np.all(sign * group_first > b) and np.all(sign * group_second < -b))
            polarity_scope = 'Native Cone release decrease/increase; not optical contrast calibration'
        max_error = max(r['max_error_over_budget'] for r in pair if r['field'].startswith(pop + '__'))
        amplitude = float(np.ptp(mean(rate)))
        resolution = float(np.max(abs(get(assay, 'expected_binary_count_rate') - get(tight, 'expected_binary_count_rate'))))
        resolved = amplitude > max(5 * resolution, np.finfo(float).eps * max(1., float(rate.max())) * 8192)
        probability_error = max(float(np.max(abs(z[pop + '__history_law_after_tick'].sum(axis=2) - 1)))
                                for z in cases.values())
        sampled = get(assay, 'sampled_bin_counts')
        event_times = get(assay, 'event_time_seconds')
        sampling = (sampled.shape == (info['groups'], 900) and np.isin(sampled, (0, 1)).all()
                    and len(event_times) == int(sampled.sum())
                    and np.all((event_times >= 0) & (event_times < .9)) and probability_error <= 1e-12)
        exact_draws = bool(np.array_equal(sampled, get(tight, 'sampled_bin_counts')))
        passed = all((causal, background, inhibition, polarity, resolved, sampling, max_error <= 1))
        rows.append(dict(type=pop, groups=info['groups'], represented_members=info['total_represented_members'],
            status='CONDITIONAL_FUNCTIONAL_V1_COMPLETE' if passed else 'INCOMPLETE',
            baseline_V_mV=float(mean(voltage)[0] * 1000), baseline_source_rate_Hz=float(mean(rate)[0]),
            common_decrease_deltaV_mV=float(mean(delta) [first].mean() * 1000),
            common_increase_deltaV_mV=float(mean(delta)[second].mean() * 1000),
            source_rate_min_Hz=float(mean(rate).min()), source_rate_max_Hz=float(mean(rate).max()),
            selective_S_deltaV_mV=float(mean(selective_delta)[first].mean() * 1000),
            selective_S_delta_rate_Hz=float(mean(selective_rate)[first].mean()),
            maximum_inhibition_removal_deltaV_mV=float(np.max(mean(get(removed, 'voltage') - voltage)) * 1000),
            max_numerical_error_over_budget=max_error, max_probability_mass_error=probability_error,
            expected_total_sampled_object_events=float(get(assay, 'expected_bin_count_rate').sum() * .001),
            actual_sampled_object_events=int(sampled.sum()), same_standard_tight_draws=exact_draws,
            polarity_passed=polarity, polarity_scope=polarity_scope, causal_passed=causal,
            constant_background_passed=background, inhibition_control_passed=inhibition,
            finite_resolved_rate_dynamic=bool(resolved), native_sampling_passed=bool(sampling),
            physiological_calibration='NOT_COMPLETED', cross_eccentricity='NOT_COMPLETED'))
        vavg, ravg, davg = mean(voltage), mean(rate), mean(delta)
        for j, node in enumerate(t):
            curves.append(dict(type=pop, time_seconds=float(node), V_V=float(vavg[j]),
                deltaV_from_background_V=float(davg[j]), source_post_tick_rate_Hz=float(ravg[j]),
                gE_S=float(mean(get(assay, 'excitatory_conductance'))[j]),
                gI_S=float(mean(get(assay, 'inhibitory_conductance'))[j]),
                IE_A=float(mean(get(assay, 'excitatory_current'))[j]),
                II_A=float(mean(get(assay, 'inhibitory_current'))[j]),
                no_inhibition_V_V=float(mean(get(removed, 'voltage'))[j]),
                selective_S_V_V=float(mean(get(selective, 'voltage'))[j]),
                selective_S_rate_Hz=float(mean(get(selective, 'expected_binary_count_rate'))[j])))
    write_csv(out / 'TYPE_COMPLETION.csv', rows)
    write_csv(out / 'ALL_TYPE_MEAN_TIME_COURSES.csv', curves)
    dump(out / 'TYPE_COMPLETION.json', rows)
    result = dict(status='RGC_FUNCTIONAL_V1_LOCAL_CONDITIONAL_STAGE_COMPLETE'
        if all(r['status'] != 'INCOMPLETE' for r in rows) else 'RGC_FUNCTIONAL_V1_INCOMPLETE',
        RGC_identity=identity['RGC_identity'], types=len(rows), groups=sum(r['groups'] for r in rows),
        declared_fields=len(identity['fields']), numerical_reference_fields=len(pair),
        max_numerical_error_over_budget=max(r['max_error_over_budget'] for r in pair),
        numerical_passed=all(r['passed'] for r in pair),
        source_900ms_endpoint_max_error_over_budget=max(r['max_error_over_budget'] for r in source_end),
        conditional_replay_scope=not args.coupled,
        original_coupled_inner_assay=args.coupled,
        whole_retina_trajectory_equivalence_claimed=False,
        all_types_functional_completed=all(r['status'] != 'INCOMPLETE' for r in rows),
        ordinary_input='Original6305-state coupled inner with held original Cone/HC source background'
            if args.coupled else 'Original cached BC group effective release with1ms linear interpolation',
        AII='Existing indirect chemical/electrical influence retained in coupled BC cache; no direct edge added',
        S_scope='S pathway passes selective BC input; common-input mixed response is OFF-like; no optical spectral validation',
        spike_scope='Strict-history Bernoulli group-object events; no reset action potential or independent fine-cell spikes',
        parameter_scope='Frozen inherited source-bound effective parameters; no fit, rescaling, or target PSTH used',
        physiologically_calibrated=False, production_adopted=False, background_qualified=False, FIT_READY=False,
        next_step='Return to predeclared cross-eccentricity calibration; retain the local conditional scope and S mixed-input limitation')
    if args.coupled:
        result['source_coupled_exact_endpoint_hash_matches'] = {
            p: json.loads((base / 'source_coupled_v1' / p / 'METRICS.json').read_bytes())[
                'exact_final_inner_hash_matches_original_BC'] for p in ('standard', 'tight')}
        result['controls_scope'] = 'Background, inhibition removal and selective S-terminal controls use explicitly conditional driven equations'
        result['prior_replay_qualification'] = '../../RESULT.json; preserved with its source-endpoint allocation exceedance'
    dump(out / 'RESULT.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
