"""One bounded shared candidate vector on actual source repeats/native caches.

No full retinal simulation or parameter writeback occurs here.  Missing native cache,
background identity, or equivalence prevents the real-data optimization.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import hashlib
import json
import sys
import time
import threading
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.functional_fit_math import (MembraneParameters, binary_history_from_past,
    candidate_voltage, background_voltage, native_voltage_equivalence, teacher_forced_nll,
    teacher_probability, free_probability, constant_nll, psth_metrics,
    prepare_event_conductance, candidate_voltage_events)
from retina_sim.functional_fit_math import stimulus_covariance_metrics
from retina_sim.natural_movie_supervision import file_sha

BASE = ROOT/'reports/cross_layer_functional_fit_20261021_v1/regional_v2'


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def path_from(value, relative_to):
    path = Path(value)
    return path if path.is_absolute() else relative_to/path


def supervision_report(binding, admitted_cases):
    def describe(cases):
        return {split:dict(cells=sum(c['split']==split for c in cases),
                          bins=sum(c['source_supervision']['scored_trial_bins'] for c in cases if c['split']==split),
                          observed_spikes=sum(c['source_supervision']['scored_observed_spikes'] for c in cases if c['split']==split))
                for split in ('training', 'development_validation')}
    ids = {c['case_id'] for c in admitted_cases}
    admitted = [c for c in binding['cases'] if c['case_id'] in ids]
    if len(admitted) != len(ids):
        raise ValueError('Protocol admits an unknown source case')
    return dict(cases=len(admitted), admitted_cases=len(admitted), available_source_cases=len(binding['cases']),
        splits=describe(admitted), available_source_splits=describe(binding['cases']),
        inactive_source_cases=[c['case_id'] for c in binding['cases'] if c['case_id'] not in ids],
        source_supervision=binding['output_arrays'], observations='Actual raw source binary repeats, no synthetic targets')


def verify_source_interface_and_training_null(binding, admitted_cases):
    """Read actual prepared labels; only training estimates the common null p."""
    descriptor = binding['output_arrays']
    path = ROOT/descriptor['path']
    if file_sha(path) != descriptor['sha256']:
        raise ValueError('Prepared real-source supervision identity differs')
    training, checked = {}, []
    admitted_ids = {c['case_id'] for c in admitted_cases}
    with np.load(path, allow_pickle=False) as data:
        for case in binding['cases']:
            keys = case['source_supervision']
            y, past, mask = (np.asarray(data[keys[k]]) for k in ('counts_key','prehistory_key','score_mask_key'))
            if (y.shape != tuple(keys['shape']) or past.shape != (len(y),3) or mask.shape != (y.shape[1],) or
                    not np.isin(y,(0,1)).all() or not np.isin(past,(0,1)).all()):
                raise ValueError('Actual binary source label/history dimensions differ')
            lo, hi = case['score_bins_half_open']
            if not np.array_equal(np.flatnonzero(mask),np.arange(lo,hi)):
                raise ValueError('Prepared source scoring interval differs')
            scored = y[:,mask].copy()
            if scored.size != keys['scored_trial_bins'] or int(scored.sum()) != keys['scored_observed_spikes']:
                raise ValueError('Prepared actual source event counts differ')
            checked.append(dict(case_id=case['case_id'], admitted_to_main_fit=case['case_id'] in admitted_ids,
                                scored_trial_bins=scored.size, observed_spikes=int(scored.sum())))
            if case['split'] == 'training' and case['case_id'] in admitted_ids:
                training.setdefault(cell_family(case['target_component']),[]).append(scored)
    p = sum(int(y.sum()) for rows in training.values() for y in rows)/sum(y.size for rows in training.values() for y in rows)
    def group(family, rows):
        started = time.thread_time()
        return dict(family=family, NLL_nats=float(sum(constant_nll(p,y) for y in rows)),
                    scored_trial_bins=sum(y.size for y in rows), observed_spikes=sum(int(y.sum()) for y in rows),
                    thread_name=threading.current_thread().name, thread_ident=threading.get_ident(),
                    CPU_seconds=time.thread_time()-started)
    with ThreadPoolExecutor(max_workers=min(3,len(training)), thread_name_prefix='source-null-family') as pool:
        jobs = {family:pool.submit(group,family,rows) for family,rows in training.items()}
        contributions = [jobs[family].result() for family in training]
    return dict(actual_source_npz_read_and_verified=True, checked_cases=checked,
        training_constant_probability=p, training_constant_rate_Hz=p/.001,
        training_null_NLL_nats=float(sum(r['NLL_nats'] for r in contributions)),
        training_null_NLL_nats_per_bin=float(sum(r['NLL_nats'] for r in contributions)/sum(r['scored_trial_bins'] for r in contributions)),
        family_contributions=contributions, validation_null_not_scored=True,
        physiological_prediction=False, mechanism_multiplier_objective_evaluations=0)


@dataclass
class Case:
    metadata: dict
    parameters: MembraneParameters
    t: np.ndarray
    ge: np.ndarray
    gi: np.ndarray
    native_v: np.ndarray
    bg_ge: float
    bg_gi: float
    y: np.ndarray
    h: np.ndarray
    indices: np.ndarray
    cache_sha256: str
    stimulus_projection: np.ndarray | None = None
    stimulus_projection_definition: str | None = None
    prepared_events: tuple | None = None

    def voltage(self, multipliers):
        index = self.metadata['excitation_multiplier_index']
        excitation = float(multipliers[index]) if index >= 0 else 1.
        inhibition = float(multipliers[self.metadata['inhibition_multiplier_index']])
        if self.prepared_events is not None:
            voltage = candidate_voltage_events(self.prepared_events, self.parameters,
                excitation=excitation, inhibition=inhibition,
                background_gE_S=self.bg_ge, background_gI_S=self.bg_gi)
        else:
            voltage = candidate_voltage(self.t, self.ge, self.gi, self.parameters,
                excitation=excitation, inhibition=inhibition,
                background_gE_S=self.bg_ge, background_gI_S=self.bg_gi)
        return voltage, background_voltage(self.parameters, self.bg_ge, self.bg_gi, excitation, inhibition)

    def loss(self, multipliers):
        voltage, _ = self.voltage(multipliers)
        return teacher_forced_nll(voltage[::10][:-1], self.y, self.h, self.indices, self.parameters)

    @property
    def scored_bins(self):
        return self.y.shape[0]*len(self.indices)


def load_cases(protocol, binding, protocol_parent, protocol_sha256):
    descriptor = binding['output_arrays']
    source = ROOT/descriptor['path']
    if file_sha(source) != descriptor['sha256']:
        raise ValueError('Frozen actual source-supervision identity differs')
    bound_cases = {c['case_id']:c for c in binding['cases']}
    cases = []
    with np.load(source, allow_pickle=False) as observed:
        for selected in protocol['cases']:
            original = bound_cases[selected['case_id']]
            for k in ('source_record_id', 'source_cell_key', 'split', 'score_window_seconds', 'target_cell_id'):
                if k in selected and selected[k] != original[k]:
                    raise ValueError('Protocol changed frozen source grouping or target identity: '+k)
            meta = dict(original, **selected)
            # Explicit field semantics: native_cache_directory is project-root
            # relative; the old relative_directory field is protocol-relative.
            directory = (path_from(meta['native_cache_directory'], ROOT) if 'native_cache_directory' in meta
                         else path_from(meta.get('relative_directory', meta['case_id']), protocol_parent))
            path = directory/'downstream_cache.npz'
            if not path.is_file():
                raise FileNotFoundError('No completed real native cache: '+str(path))
            result = read(directory/'cache_result.json')
            if result.get('status') not in ('COMPLETE_NATIVE_CACHE', 'COMPLETE_REGIONAL_NATIVE_CACHE'):
                raise ValueError('Partial native cache cannot train: '+meta['case_id'])
            if result.get('cache_sha256') != file_sha(path):
                raise ValueError('Native cache identity differs')
            for key in ('case_id', 'target_component', 'target_cell_id'):
                if result.get(key) != meta[key]:
                    raise ValueError('Native cache source/region/cell binding differs or is absent: '+key)
            if result.get('input_protocol_sha256') != protocol_sha256:
                raise ValueError('Native cache input protocol differs from the fixed regional protocol')
            params = MembraneParameters.from_dict(meta.get('parameters', result.get('parameters', {})))
            bg = result.get('background', meta.get('background', {}))
            if (not bg.get('is_declared_constant_background', False) or
                    'gE_S' not in bg or 'gI_S' not in bg):
                raise ValueError('Actual constant-background gE/gI identity missing; first movie frame is not background')
            interpolation = result.get('conductance_interpolation', protocol.get('numerical', {}).get('conductance_interpolation'))
            supported_interpolations = ('piecewise_linear_100us',
                'piecewise_linear_100us_with_exact_original_AC_ZOH_event_boundaries')
            if interpolation not in supported_interpolations:
                raise ValueError('Native conductance representation must be explicitly declared')
            with np.load(path, allow_pickle=False) as cached:
                t, ge, gi, v = (np.asarray(cached[k], dtype=float).copy() for k in
                                ('relative_time', 'gE_S', 'gI_S', 'V_native_V'))
                projection = (np.asarray(cached['stimulus_projection_1ms'], dtype=float).copy()
                              if 'stimulus_projection_1ms' in cached else None)
            prepared_events = None
            if result.get('event_cache'):
                event_path = directory/result['event_cache']
                if event_path.resolve().parent != directory.resolve():
                    raise ValueError('Native event cache must remain beside its conductance cache')
                if (not result.get('event_times_preserve_duplicate_pre_post_observations', False) or
                        result.get('event_cache_sha256') != file_sha(event_path)):
                    raise ValueError('Actual pre/post event order or event cache identity differs')
                with np.load(event_path, allow_pickle=False) as events:
                    prepared_events = prepare_event_conductance(t, ge, gi,
                        events['relative_time'], events['gE_S'], events['gI_S'])
            if ('exact_original_AC_ZOH_event_boundaries' in
                    protocol.get('numerical', {}).get('conductance_interpolation', '') and prepared_events is None):
                raise ValueError('The fixed protocol requires the actual AC event cache; no linear fallback')
            projection_definition = result.get('stimulus_projection_definition')
            if projection is not None and not isinstance(projection_definition, str):
                raise ValueError('A source-stimulus projection needs a fixed input definition; no inferred polarity')
            keys = meta['source_supervision']
            y = np.array(observed[keys['counts_key']], copy=True)
            past = np.asarray(observed[keys['prehistory_key']])
            mask = np.asarray(observed[keys['score_mask_key']])
            if (y.shape != tuple(keys['shape']) or not np.isin(y, (0, 1)).all() or
                    past.shape != (len(y), 3) or mask.shape != (y.shape[1],) or
                    not np.isfinite(t).all() or abs(t[-1]-y.shape[1]*.001) > 1e-10 or len(t) != y.shape[1]*10+1):
                raise ValueError('Real source bins and native cached duration differ')
            h = np.asarray([binary_history_from_past(row, p, params.history_kernel_V) for row, p in zip(y, past)])
            indices = np.flatnonzero(mask)
            lo, hi = meta['score_bins_half_open']
            if not np.array_equal(indices, np.arange(lo, hi)):
                raise ValueError('Scoring mask changed from the frozen source selection')
            cases.append(Case(meta, params, t, ge, gi, v, float(bg['gE_S']), float(bg['gI_S']),
                              y, h, indices, result['cache_sha256'], projection, projection_definition,
                              prepared_events))
    return cases


class Budget:
    def __init__(self, maximum_evaluations=600, wall_seconds=2700., CPU_seconds=7200.):
        self.started, self.cpu = time.monotonic(), time.process_time()
        self.maximum = min(600, int(maximum_evaluations))
        self.wall_seconds, self.cpu_seconds = min(2700., wall_seconds), min(7200., CPU_seconds)
        self.n = 0

    def check(self, reserve=0):
        if self.n+reserve >= self.maximum or time.monotonic()-self.started >= self.wall_seconds or time.process_time()-self.cpu >= self.cpu_seconds:
            raise RuntimeError('Declared whole-fit objective/wall/CPU budget reached')


class Objective:
    def __init__(self, training, active, budget):
        self.training, self.active, self.budget = training, np.asarray(active, dtype=int), budget
        self.memo, self.rows = {}, []
        self.groups = {name:[c for c in training if cell_family(c.metadata['target_component']) == name]
                       for name in ('midget', 'parasol', 'S')}
        self.groups = {k:v for k,v in self.groups.items() if v}
        self.executor = ThreadPoolExecutor(max_workers=min(3, len(self.groups)), thread_name_prefix='functional-cell-family')
        self.worker_receipts = []

    def close(self):
        self.executor.shutdown(wait=True)

    @staticmethod
    def group_contribution(family, cases, multipliers):
        started = time.thread_time()
        nll = float(sum(c.loss(multipliers) for c in cases))
        return dict(family=family, NLL_nats=nll, scored_trial_bins=sum(c.scored_bins for c in cases),
                    thread_name=threading.current_thread().name, thread_ident=threading.get_ident(),
                    CPU_seconds=time.thread_time()-started, shared_candidate=multipliers.tolist())

    def multipliers(self, x):
        result = np.ones(3)
        result[self.active] = np.exp(x)
        return result

    def __call__(self, x, phase='refinement'):
        x = np.asarray(x, dtype=float)
        key = tuple(x)
        if key in self.memo:
            return self.memo[key]
        self.budget.check(reserve=4)
        s = self.multipliers(x)
        pending = {family:self.executor.submit(self.group_contribution, family, cases, s)
                   for family, cases in self.groups.items()}
        contributions = [pending[family].result() for family in self.groups]
        self.worker_receipts.extend(dict(evaluation=self.budget.n+1, **r) for r in contributions)
        nll = float(sum(r['NLL_nats'] for r in contributions))
        bins = sum(r['scored_trial_bins'] for r in contributions)
        penalty = .01*float(np.log(s) @ np.log(s))
        value = nll/bins+penalty
        self.budget.n += 1
        self.memo[key] = value
        self.rows.append(dict(evaluation=self.budget.n, phase=phase,
                              midget_excitation=s[0], parasol_excitation=s[1], shared_AC_inhibition=s[2],
                              total_training_NLL_nats=nll, scored_trial_bins=bins,
                              mean_training_NLL_nats_per_bin=nll/bins, regularizer=penalty, objective=value,
                              source_distance_squared=float(np.sum((np.log(s)/np.log(2))**2)),
                              **{family+'_NLL_nats':sum(r['NLL_nats'] for r in contributions if r['family']==family)
                                 for family in ('midget','parasol','S')}))
        return value


def cell_family(component):
    if component.startswith('midget'):
        return 'midget'
    if component.startswith('parasol'):
        return 'parasol'
    if component == 'small_bistratified_S_ON':
        return 'S'
    raise ValueError('Unsupported supervised RGC family: '+component)


def _score_group(cases, multipliers, training_constant_p, family):
    started = time.thread_time()
    rows, curves = {}, {}
    for case in cases:
        voltage, vbg = case.voltage(multipliers)
        v = voltage[::10][:-1]
        nll = teacher_forced_nll(v, case.y, case.h, case.indices, case.parameters)
        expected = free_probability(v, case.parameters, background_V=vbg)
        metrics, trace = psth_metrics(case.y, expected, case.indices)
        measured = case.y[:, case.indices]
        row = dict(case_id=case.metadata['case_id'], split=case.metadata['split'],
                   target_component=case.metadata['target_component'], source_cell_key=case.metadata['source_cell_key'],
                   NLL_nats=nll, scored_bins=case.scored_bins, observed_spikes=int(measured.sum()),
                   mean_NLL_nats_per_bin=nll/case.scored_bins,
                   training_constant_NLL_nats=constant_nll(training_constant_p, measured),
                   candidate_background_voltage_V=vbg, teacher_forced_expected_rate_Hz=float(teacher_probability(v, case.h, case.parameters)[:, case.indices].mean()/.001),
                   **metrics)
        row['stimulus_covariance'] = (dict(stimulus_covariance_metrics(case.y, expected, case.stimulus_projection, case.indices),
                                          input_projection_definition=case.stimulus_projection_definition)
            if case.stimulus_projection is not None else dict(status='NOT_AVAILABLE: no frozen source-stimulus projection supplied',
                                                              physiological_ON_OFF_classification=False))
        rows[case.metadata['case_id']] = row
        for key, value in trace.items():
            curves[case.metadata['case_id']+'_'+key] = value
        curves[case.metadata['case_id']+'_voltage_bin_start_V'] = v
        curves[case.metadata['case_id']+'_free_probability'] = expected
    receipt = dict(family=family, cells=len(cases), shared_candidate=multipliers.tolist(),
                   thread_name=threading.current_thread().name, thread_ident=threading.get_ident(),
                   CPU_seconds=time.thread_time()-started)
    return rows, curves, receipt


def score(cases, multipliers, training_constant_p, prefix, output_dir):
    groups = {family:[c for c in cases if cell_family(c.metadata['target_component']) == family]
              for family in ('midget', 'parasol', 'S')}
    groups = {family:members for family,members in groups.items() if members}
    rows, curves, receipts = {}, {}, []
    with ThreadPoolExecutor(max_workers=min(3,len(groups)), thread_name_prefix='functional-score-family') as executor:
        pending = {family:executor.submit(_score_group, members, multipliers, training_constant_p, family)
                   for family,members in groups.items()}
        for family in groups:
            row, trace, receipt = pending[family].result()
            rows.update(row);curves.update(trace);receipts.append(receipt)
    np.savez_compressed(output_dir/(prefix+'_curves.npz'), **curves)
    save(output_dir/(prefix+'_parallel.json'), dict(stage=prefix, receipts=receipts,
         numerical_kernels_release_GIL=True, candidate_identical_across_families=True,
         total_worker_CPU_seconds=sum(r['CPU_seconds'] for r in receipts)))
    return rows


def pooled_summary(rows):
    """Exposure pooling; per-cell modulation/error rows are retained separately."""
    values = list(rows.values())
    bins = sum(r['scored_bins'] for r in values)
    spikes = sum(r['observed_spikes'] for r in values)
    weights = np.asarray([r['scored_bins'] for r in values], dtype=float)
    observed_rate = spikes/(bins*.001)
    predicted_rate = float(np.average([r['predicted_rate_Hz'] for r in values], weights=weights))
    return dict(cells=len(values), scored_trial_bins=bins, observed_spikes=spikes,
        NLL_nats=float(sum(r['NLL_nats'] for r in values)),
        mean_NLL_nats_per_bin=float(sum(r['NLL_nats'] for r in values)/bins),
        observed_rate_Hz=observed_rate, predicted_rate_Hz=predicted_rate,
        signed_rate_error_Hz=predicted_rate-observed_rate,
        absolute_pooled_rate_error_Hz=abs(predicted_rate-observed_rate),
        exposure_weighted_per_cell_absolute_rate_error_Hz=float(np.average([r['absolute_rate_error_Hz'] for r in values], weights=weights)),
        exposure_weighted_PSTH_RMSE_Hz=float(np.sqrt(np.average([r['PSTH_RMSE_Hz']**2 for r in values], weights=weights))),
        exposure_weighted_per_cell_modulation_std_error_Hz=float(np.average([r['modulation_std_absolute_error_Hz'] for r in values], weights=weights)),
        note='Pooled rate cancellation does not erase the complete per-cell errors; event peaks/polarity remain N/A for the natural movie')


def run(args):
    started = time.monotonic()
    output = args.output_dir
    output.mkdir(exist_ok=True, parents=True)
    if (output/'result.json').exists() and not args.check_inputs_only:
        raise ValueError('Preserved real fitting result exists; do not silently repeat or overwrite')
    binding = read(args.bindings)
    protocol = read(args.protocol)
    base_report = supervision_report(binding, protocol['cases'])
    if args.check_inputs_only:
        actual = verify_source_interface_and_training_null(binding, protocol['cases'])
        report = dict(status='REAL_LABEL_INTERFACE_READY_NATIVE_CACHE_REQUIRED', **base_report,
                      actual_supervision_and_training_null=actual,
                      objective_evaluations=0, fitting_ran=False, main_configuration_changed=False,
                      required_cache_keys=['relative_time', 'gE_S', 'gI_S', 'V_native_V'],
                      required_cache_result_fields=['status=COMPLETE_NATIVE_CACHE', 'cache_sha256',
                            'case_id', 'target_component', 'target_cell_id', 'input_protocol_sha256',
                            'parameters:all_SI_membrane_output_history',
                            'background:{is_declared_constant_background:true,gE_S,gI_S}',
                            'conductance_interpolation=piecewise_linear_100us'],
                      driver_source_sha256=file_sha(Path(__file__)),
                      math_source_sha256=file_sha(ROOT/'retina_sim/functional_fit_math.py'))
        save(output/'source_interface.json', report)
        print(json.dumps(report, ensure_ascii=False))
        return
    protocol_hash = file_sha(args.protocol)
    if not protocol.get('fixed_before_candidate_predictions', False):
        raise ValueError('A protocol fixed before candidate predictions is required')
    if not protocol.get('parameters', {}).get('single_shared_candidate_vector', False):
        raise ValueError('One central shared candidate vector, not separately fitted AC vectors, is required')
    if args.case:
        if not args.native_equivalence_only:
            raise ValueError('A case subset is allowed only for numerical equivalence, never fitting')
        selected_ids = set(args.case)
        protocol = dict(protocol, cases=[c for c in protocol['cases'] if c['case_id'] in selected_ids])
        if selected_ids != {c['case_id'] for c in protocol['cases']}:
            raise ValueError('Unknown fixed source case for native equivalence')
    cases = load_cases(protocol, binding, args.protocol.parent, protocol_hash)
    equivalence = {}
    for c in cases:
        voltage, _ = c.voltage(np.ones(3))
        equivalence[c.metadata['case_id']] = native_voltage_equivalence(voltage, c.native_v)
        equivalence[c.metadata['case_id']]['conductance_representation'] = (
            'BC gE continuous linear native nodes plus event values; AC gI exact recorded ZOH pre/post events'
            if c.prepared_events is not None else 'piecewise_linear_100us')
    save(output/'native_input_equivalence.json', equivalence)
    if args.native_equivalence_only:
        report = dict(status='NATIVE_INPUT_EQUIVALENCE_PASS' if all(r['passed'] for r in equivalence.values())
                      else 'BLOCKED_NATIVE_INPUT_EQUIVALENCE', fitting_ran=False,
                      objective_evaluations=0, main_configuration_changed=False,
                      protocol_sha256=protocol_hash, equivalence=equivalence)
        save(output/'result.json', report)
        print(json.dumps(report, ensure_ascii=False))
        return
    if not all(r['passed'] for r in equivalence.values()):
        save(output/'result.json', dict(status='BLOCKED_NATIVE_INPUT_EQUIVALENCE', fitting_ran=False,
                                      objective_evaluations=0, main_configuration_changed=False,
                                      protocol_sha256=protocol_hash, equivalence=equivalence))
        return
    training = [c for c in cases if c.metadata['split']=='training']
    validation = [c for c in cases if c.metadata['split']=='development_validation']
    if not training or not validation:
        raise ValueError('Both predefined source groups must be present; no split repair from results')
    active = sorted({j for c in training for j in (c.metadata['excitation_multiplier_index'], c.metadata['inhibition_multiplier_index']) if j >= 0})
    configured = protocol['parameters'].get('active_indices', active)
    if list(configured) != active:
        raise ValueError('Requested active multiplier has no training coverage or changes the fixed scope')
    optimizer = protocol.get('optimizer', {})
    budget = Budget(min(optimizer.get('objective_evaluations_max',600), args.maximum_evaluations),
                    min(optimizer.get('wall_seconds',2700), args.wall_seconds),
                    min(optimizer.get('CPU_seconds',7200), args.cpu_seconds))
    objective = Objective(training, active, budget)
    main_hash = file_sha(ROOT/'configurations/retina_platform_v1.json')
    save(output/'frozen_fit_identity.json', dict(protocol_sha256=protocol_hash, source_bindings_sha256=file_sha(args.bindings),
        source_supervision=binding['output_arrays'], main_sha256=main_hash,
        caches={c.metadata['case_id']:c.cache_sha256 for c in cases},
        frozen_SI_parameters={c.metadata['case_id']:asdict(c.parameters) for c in cases},
        code_sha256={str(p.relative_to(ROOT)).replace('\\','/'):file_sha(p) for p in
                     (Path(__file__),ROOT/'retina_sim/functional_fit_math.py')},
        objective='sum_training_NLL/sum_training_trial_bins + .01*sum(log(s)^2), regularizer once',
        active_indices=active, all_other_multipliers=1., observations=base_report,
        bounds_kind='Engineering domain, not physiological confidence interval',
        candidate_selection='Training only; near minimum1e-4nat/bin, minimum source log-distance',
        validation_selection=False))
    bounds = [(np.log(.5), np.log(2.))]*len(active)
    grid = list(product(np.linspace(np.log(.5), np.log(2.),7), repeat=len(active)))
    stopped = None
    refinements = []
    try:
        for x in grid:
            objective(x, 'log_grid')
        starts = []
        for row in sorted(objective.rows, key=lambda r:r['objective']):
            x = np.log([row[['midget_excitation','parasol_excitation','shared_AC_inhibition'][j]] for j in active])
            if all(np.linalg.norm((x-s)/(2*np.log(2))) >= .2 for s in starts):
                starts.append(x)
            if len(starts) == 5:
                break
        for x in starts:
            result = minimize(objective, x, method='L-BFGS-B', bounds=bounds,
                              options=dict(ftol=1e-4, maxiter=40, maxfun=40))
            refinements.append(dict(x=result.x.tolist(), objective=float(result.fun), success=bool(result.success), message=str(result.message)))
    except RuntimeError as exc:
        stopped = str(exc)
    finally:
        objective.close()
    if not objective.rows:
        raise ValueError('No objective evaluation completed')
    best = min(r['objective'] for r in objective.rows)
    feasible = [r for r in objective.rows if r['objective'] <= best+1e-4]
    selected = min(feasible, key=lambda r:(r['source_distance_squared'], r['objective']))
    names = ('midget_excitation','parasol_excitation','shared_AC_inhibition')
    multipliers = np.asarray([selected[k] for k in names])
    minimum_row = min(objective.rows, key=lambda r:r['objective'])
    touching = lambda row:[names[j] for j in active if min(abs(row[names[j]]-.5),abs(row[names[j]]-2)) < 1e-8]
    selection = dict(selected_training_row=selected, minimum_training_objective=best,
        near_equivalent_engineering_tolerance_nats_per_bin=1e-4, feasible_evaluated_candidates=len(feasible),
        representative_selected_on_training_only=True, active_indices=active, multipliers=multipliers.tolist(),
        selected_touches_engineering_boundary=touching(selected),
        training_minimum_touches_engineering_boundary=touching(minimum_row),
        near_equivalent_evaluated_ranges={name:[min(r[name] for r in feasible), max(r[name] for r in feasible)] for name in names},
        identifiability='Evaluated engineering near-equivalence set, not a physiological confidence interval or unique optimum',
        refinements=refinements, budget_stop=stopped, selected_before_validation_UTC=datetime.now(timezone.utc).isoformat())
    save(output/'training_selection.json', selection)
    save(output/'parallel_worker_receipts.json', dict(maximum_workers=min(3, len(objective.groups)),
        grouped_by='cell family', numerical_kernels_release_GIL=True, regularizer_evaluated_only_in_central_objective=True,
        worker_CPU_seconds=float(sum(r['CPU_seconds'] for r in objective.worker_receipts)),
        actual_worker_thread_ids=sorted({r['thread_ident'] for r in objective.worker_receipts}),
        receipts=objective.worker_receipts))
    for name, rows in [('profile.csv',objective.rows),('near_equivalent_candidates.csv',feasible)]:
        with (output/name).open('w', encoding='utf8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    pnull = sum(int(c.y[:,c.indices].sum()) for c in training)/sum(c.scored_bins for c in training)
    scores = {}
    for split, rows in [('training',training), ('development_validation',validation)]:
        scores[split] = dict(frozen=score(rows,np.ones(3),pnull,split+'_frozen',output),
                             candidate=score(rows,multipliers,pnull,split+'_candidate',output))
    budget.n += 4
    for split in scores.values():
        for key, row in split['candidate'].items():
            base = split['frozen'][key]
            row['bits_per_observed_spike_over_frozen'] = (base['NLL_nats']-row['NLL_nats'])/(row['observed_spikes']*np.log(2)) if row['observed_spikes'] else None
            row['bits_per_observed_spike_over_training_constant'] = (row['training_constant_NLL_nats']-row['NLL_nats'])/(row['observed_spikes']*np.log(2)) if row['observed_spikes'] else None
            row['bits_denominator_same_scored_set_spikes'] = row['observed_spikes']
    save(output/'scores.json', scores)
    pooled = {split:{label:pooled_summary(rows) for label, rows in phases.items()} for split, phases in scores.items()}
    for phases in pooled.values():
        b, c = phases['frozen'], phases['candidate']
        c['NLL_improvement_fraction_over_frozen'] = (b['NLL_nats']-c['NLL_nats'])/b['NLL_nats']
        c['PSTH_RMSE_improvement_fraction_over_frozen'] = ((b['exposure_weighted_PSTH_RMSE_Hz']-c['exposure_weighted_PSTH_RMSE_Hz'])/b['exposure_weighted_PSTH_RMSE_Hz']
            if b['exposure_weighted_PSTH_RMSE_Hz'] else None)
        c['bits_per_observed_spike_over_frozen'] = ((b['NLL_nats']-c['NLL_nats'])/(c['observed_spikes']*np.log(2))
            if c['observed_spikes'] else None)
    save(output/'pooled_scores.json', pooled)
    assert file_sha(args.protocol) == protocol_hash, 'Protocol changed during real fit'
    assert file_sha(ROOT/'configurations/retina_platform_v1.json') == main_hash, 'Main changed during real fit'
    save(output/'result.json', dict(status='CONDITIONAL_CANDIDATE_FIT_COMPLETE_NOT_ADOPTED', fitting_ran=True,
        objective_evaluations=budget.n, training_objective_evaluations=len(objective.rows),
        validation_representatives_scored=1, train_constant_probability=pnull,
        candidate=selection, score_path='scores.json', pooled_score_path='pooled_scores.json',
        pooled_scores=pooled, wall_seconds=time.monotonic()-started,
        CPU_seconds=time.process_time()-budget.cpu, main_configuration_changed=False, integrated=False,
        controlled_polarity_regression='REQUIRED_BEFORE_ADOPTION_NOT_INFERRED_FROM_NATURAL_MOVIE',
        free_initialization='Exact8state stationary law at candidate fixed-background voltage; not a whole-network spike-statistics stationarity claim',
        fixed_mechanisms_membrane_output_parameters=True, central_foveal_parameters_unchanged=True,
        physiological_optimum_claim=False, full_data_range_fitted_claim=False))
    print(json.dumps(dict(status='CONDITIONAL_CANDIDATE_FIT_COMPLETE_NOT_ADOPTED', multipliers=multipliers.tolist(), objective_evaluations=budget.n)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=BASE/'protocol.json')
    parser.add_argument('--bindings', type=Path, default=BASE/'source_bindings.json')
    parser.add_argument('--output-dir', type=Path, default=BASE/'shared_fit')
    parser.add_argument('--check-inputs-only', action='store_true')
    parser.add_argument('--native-equivalence-only', action='store_true')
    parser.add_argument('--case', action='append', help='Only for a same-input equivalence check, never a fit subset')
    parser.add_argument('--maximum-evaluations', type=int, default=600)
    parser.add_argument('--wall-seconds', type=float, default=2700.)
    parser.add_argument('--cpu-seconds', type=float, default=7200.)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save(args.output_dir/'execution_failure.json', dict(status='BLOCKED_OR_FAILED_NOT_ADOPTED',
             error_type=type(exc).__name__, error=str(exc), main_configuration_changed=False))
        raise


if __name__ == '__main__':
    main()
