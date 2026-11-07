"""Bounded Table 2 latency calibration using the existing homogeneous native RHS.

No published-fit teacher targets, spatial extrapolation or production mutation.
The geometry preflight is recorded independently from the temporal decision.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, brentq

from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.hc_joint_v2 import JointV2HCDensityProfile
from retina_sim.hc_teacher_calibration import (
    make_homogeneous_source, DerivedHomogeneousTeacher, sha256,
)
from retina_sim.vh2005 import SOURCE_VALUES
from tools.run_hc_minimal_real_diagnostic_v1 import peak_memory_bytes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02'
TARGET = ROOT/'data/source_data/h1_temporal_targets.csv'
PROTOCOL = dict(target='Smith 2001 Table 2 population summaries',
    target_sha256=sha256(TARGET), approximation='homogeneous A=B=[1]; not 1 mm spatial spot',
    input='native td; L_on=20*background; exact LED spectrum/ex vivo calibration not reproduced',
    train_background_td=[100, 10], heldout_background_td=[1], pulse_ms=[10, 100],
    optimized_parameter='tau_h_ms', bounds_ms=[10., 40.], source_tau_h_ms=20.,
    max_parameter_evaluations_including_baseline=16, optimizer='scipy bounded scalar',
    optimizer_xatol_ms=.01, method='DOP853', rtol=1e-8, atol=1e-10,
    max_step_seconds=.001, stop_seconds=.6, sample_seconds=.0001,
    peak='global hyperpolarizing minimum; dense-output derivative root within sample bracket',
    observation_delay_ms=2.82, delay_role='readout only; added once to peak latency',
    amplitude='native voltage change; ratios only; no conversion to mV',
    acceptance='train loss strictly decreases AND heldout loss does not increase AND hyperpolarizing polarity retained')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def geometry_preflight():
    p = JointV2HCDensityProfile(ROOT)
    result = dict(profile=str(p.profile_path.relative_to(ROOT)), profile_sha256=p.source_asset_hash,
        profile_identity=p.profile_hash, supported_radius_mm=p.support_radius_mm,
        requested_reference_mm=4., reference_radius_mm=None)
    try:
        result['reference_radius_mm'] = float(p.contact_radius_mm(4., 'H1'))
        result['status'] = 'REFERENCE_AVAILABLE'
    except ContractError as error:
        result.update(status='BLOCKED', reason=str(error))
    return result


def simulate(source, tau, row, protocol_hash):
    model = DerivedHomogeneousTeacher.create(source, tau, protocol_hash, source.model_hash)
    y0 = model.equilibrium
    bg = row['background_td']; edge = row['pulse_ms']*.001
    residual = float(np.max(abs(model.rhs(0., y0, bg))))
    if residual > 1e-8:
        raise ContractError('Background fixed point fails original 1e-8 residual gate')
    segments = []; y = y0.copy(); calls = 0
    for start, end, light in ((0., edge, bg*20.), (edge, PROTOCOL['stop_seconds'], bg)):
        sol = solve_ivp(lambda t, state: model.rhs(t, state, light), (start, end), y,
            method=PROTOCOL['method'], rtol=PROTOCOL['rtol'], atol=PROTOCOL['atol'],
            max_step=PROTOCOL['max_step_seconds'], dense_output=True)
        if not sol.success:
            raise ContractError(sol.message)
        segments.append(sol); y = sol.y[:, -1]; calls += sol.nfev
    times = np.arange(6001)*PROTOCOL['sample_seconds']
    before = times <= edge
    states = np.empty((10, len(times)))
    states[:, before] = segments[0].sol(times[before])
    states[:, ~before] = segments[1].sol(times[~before])
    if not np.isfinite(states).all() or np.any(states < 0):
        raise ContractError('Native state finite/nonnegative domain failed')
    delta = states[-1]-y0[-1]
    k = int(np.argmin(delta))
    if not 0 < k < len(times)-1:
        raise ContractError('Peak not captured inside observation window')
    def state_at(t):
        return segments[0 if t <= edge else 1].sol(t)
    def slope(t):
        return model.rhs(t, state_at(t), bg*20 if t < edge else bg)[-1]
    peak = brentq(slope, times[k-1], times[k+1], xtol=1e-12)
    amp = float(y0[-1]-state_at(peak)[-1])
    latency = peak*1000+PROTOCOL['observation_delay_ms']
    error = latency-row['peak_latency_ms']
    key = f"{int(bg)}td_{int(row['pulse_ms'])}ms"
    result = dict(condition=key, background_td=bg, pulse_ms=row['pulse_ms'],
        split='heldout' if bg == 1 else 'train', predicted_peak_latency_ms=latency,
        observed_peak_latency_ms=row['peak_latency_ms'], error_ms=error,
        standardized_error=error/row['peak_latency_sd_ms'],
        amplitude_native=amp, observed_amplitude_mV=row['amplitude_mV'],
        polarity_hyperpolarizing=bool(amp > 0 and np.min(delta) < 0),
        peak_captured=True, peak_time_without_delay_ms=peak*1000,
        final_delta_native=float(delta[-1]), background_rhs_max=residual,
        rhs_evaluations=calls, model_identity=model.candidate_hash)
    return result, dict(time_seconds=times, states=states, background_state=y0,
        observed_time_seconds=times+.00282, observed_h1_delta_native=delta)


def scores(rows):
    return {split: float(np.mean([r['standardized_error']**2 for r in rows if r['split'] == split]))
            for split in ('train', 'heldout')}


def accept(baseline, candidate):
    a, b = scores(baseline), scores(candidate)
    return (b['train'] < a['train'] and b['heldout'] <= a['heldout']
            and all(r['polarity_hyperpolarizing'] for r in baseline+candidate))


def amplitude_ratios(rows):
    for row in rows:
        ref = next(r for r in rows if r['pulse_ms'] == row['pulse_ms'] and r['background_td'] == 100)
        row['predicted_amplitude_ratio_to_100td'] = row['amplitude_native']/ref['amplitude_native']
        row['observed_amplitude_ratio_to_100td'] = row['observed_amplitude_mV']/ref['observed_amplitude_mV']


def main():
    started = datetime.now(timezone.utc).isoformat(); clock = time.perf_counter()
    metrics_path = OUT/'CALIBRATION_METRICS.json'
    if metrics_path.exists():
        raise ContractError('Existing calibration receipt: refuse duplicate fitting or overwrite')
    # Exclusive receipt also prevents a concurrent invocation from duplicating fits.
    with metrics_path.open('x', encoding='utf-8') as stream:
        json.dump(dict(status='RUNNING', started_utc=started, protocol=PROTOCOL), stream, indent=2)
    journal = dict(status='RUNNING', started_utc=started, protocol=PROTOCOL, evaluations=[])
    protected = [ROOT/p for p in ('retina_sim/vh2005.py', 'retina_sim/h1_h2_effective.py',
        'retina_sim/hc_biexponential.py', 'retina_sim/group_inner_dynamics.py',
        'retina_sim/assembly.py', 'configurations/research/cone_hc_biexponential_functional.json',
        'configurations/research/BC_PHYSIOLOGY_2026-11-02.json', 'sources/core_registry.json')]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    try:
        geometry = geometry_preflight(); journal['geometry_preflight'] = geometry
        with TARGET.open(encoding='utf-8', newline='') as stream:
            rows = [{k: v if k.startswith('source_') else float(v) for k, v in r.items()}
                    for r in csv.DictReader(stream)]
        if len(rows) != 6 or any(r['weber_contrast'] != 19 for r in rows):
            raise ContractError('Frozen six-condition target contract differs')
        sources = {bg: make_homogeneous_source(ROOT, bg) for bg in (100., 10., 1.)}
        protocol_hash = canonical_hash(PROTOCOL)
        traces = {}; cache = {}
        def evaluate(tau, subset, label):
            result = []
            for row in subset:
                key = (float(tau), row['background_td'], row['pulse_ms'])
                if key not in cache:
                    cache[key] = simulate(sources[key[1]], tau, row, protocol_hash)
                record, trace = cache[key]; result.append(dict(record))
                if label:
                    for name, value in trace.items():
                        traces[label+'_'+record['condition']+'_'+name] = value
            return result
        baseline = evaluate(20., rows, 'baseline')
        journal['baseline'] = dict(tau_h_ms=20., losses=scores(baseline), conditions=baseline)
        print(json.dumps(dict(stage='baseline', losses=scores(baseline))), flush=True)
        write(metrics_path, journal)
        training = [r for r in rows if r['background_td'] != 1]
        def objective(tau):
            if len(journal['evaluations']) >= 15:
                raise ContractError('Candidate evaluation budget exceeded')
            t = time.perf_counter(); result = evaluate(tau, training, None)
            loss = float(np.mean([r['standardized_error']**2 for r in result]))
            record = dict(index=len(journal['evaluations'])+1, tau_h_ms=float(tau),
                train_loss=loss, conditions=result, wall_seconds=time.perf_counter()-t)
            journal['evaluations'].append(record); write(metrics_path, journal)
            print(json.dumps({k:v for k,v in record.items() if k != 'conditions'}), flush=True)
            return loss
        optimum = minimize_scalar(objective, method='bounded', bounds=(10.,40.),
            options=dict(maxiter=15, xatol=.01))
        tau = float(optimum.x)
        candidate = evaluate(tau, rows, 'candidate')
        amplitude_ratios(baseline); amplitude_ratios(candidate)
        accepted = bool(accept(baseline, candidate))
        after = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
        if after != hashes:
            raise ContractError('Protected model/configuration changed during calibration')
        np.savez_compressed(OUT/'H1_TEMPORAL_TRAJECTORIES.npz', **traces,
            protocol_identity=np.array(protocol_hash), target_sha256=np.array(sha256(TARGET)))
        journal.update(status='COMPLETE', protocol_identity=protocol_hash,
            baseline=dict(tau_h_ms=20., losses=scores(baseline), conditions=baseline),
            candidate=dict(tau_h_ms=tau, losses=scores(candidate), conditions=candidate),
            optimizer=dict(success=bool(optimum.success), message=str(optimum.message), nfev=int(optimum.nfev)),
            decision='ACCEPTED' if accepted else 'TEMPORAL_CALIBRATION_NOT_ACCEPTED',
            decision_tau_h_ms=tau if accepted else 20., installed_tau_h_ms=20.,
            integration='BLOCKED' if geometry['status']=='BLOCKED' else 'NOT_EXECUTED',
            protected_sha256=hashes, protected_unchanged=True,
            implementation_sha256=sha256(Path(__file__)),
            reused_rhs_adapter_sha256=sha256(ROOT/'retina_sim/hc_teacher_calibration.py'),
            total_parameter_points=1+len(journal['evaluations']), condition_simulations=len(cache),
            rhs_evaluations=sum(value[0]['rhs_evaluations'] for value in cache.values()),
            ended_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-clock,
            peak_working_set_bytes=peak_memory_bytes())
        write(metrics_path, journal)
        print(json.dumps({k:journal[k] for k in ('status','decision','decision_tau_h_ms','wall_seconds','peak_working_set_bytes')}), flush=True)
    except Exception as error:
        journal.update(status='FAILED', error=repr(error), wall_seconds=time.perf_counter()-clock,
            peak_working_set_bytes=peak_memory_bytes())
        write(metrics_path, journal)
        raise


if __name__ == '__main__':
    main()
