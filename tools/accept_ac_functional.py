"""One short functional AC/AII acceptance on the existing BC full-chain input."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import tempfile
import time
import traceback

import numpy as np

from retina_sim.assembly import RetinaBuilder
from retina_sim.ac_vs_model import DirectACVSRHS, MECHANISM
from retina_sim.aii_functional import COMPONENTS, V_KEY, S_KEY, FunctionalACRunner
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_inner_dynamics import BC_MECH, MEMBRANE
from retina_sim.group_time_recording import GroupRecording
from retina_sim.spatial_group_graph import digest
from retina_sim.spatial_group_model import BC, RGC
from tools.accept_bc_physiology import protected, write
from tools.validate_h1_h2_effective import tick


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/ac_vs_redesign_v1'


def static_checks(model, arm):
    inner, state = arm.inner, arm.inner_state
    old = model.upstream.assembly.runtime_factories['inner'](model.upstream.assembly, model.upstream.seed['inner_parameters'])
    prior = {k:model.upstream.seed['states'][k] for k in old.state_keys}
    old_outputs, out = old.outputs(0., prior), inner.outputs(0., state)
    for pop in (*BC, *RGC, *inner.ac_populations, 'rod_BC', 'rods'):
        if canonical_hash(inner.p[pop]) != canonical_hash(old.p[pop]):
            raise ContractError('Protected population parameters changed: '+pop)
    for pop in ('cones', 'HC', 'rods', 'rod_BC'):
        for k,v in prior.items():
            if k[0] == pop and k[1] not in COMPONENTS: np.testing.assert_array_equal(v, state[k])
    np.testing.assert_array_equal(inner.gap.matrix.toarray(), old.gap.matrix.toarray())
    symmetry = inner.gap.matrix-inner.gap.matrix.T
    if np.max(abs(symmetry.data), initial=0.) > 1e-25: raise ContractError('Nonreciprocal gap graph')
    background = {}
    for c in COMPONENTS:
        np.testing.assert_allclose(out[c, 'conductance'], old_outputs[c, 'conductance'], rtol=1e-10, atol=1e-23)
        background[c] = dict(old_total_S=float(old_outputs[c, 'conductance'].sum()),
            new_total_S=float(out[c, 'conductance'].sum()),
            max_abs_difference_S=float(np.max(abs(out[c, 'conductance']-old_outputs[c, 'conductance']), initial=0.)),
            edges=len(inner.contacts[c]['pre']))
    base = inner.rhs(0., state)
    checks = {}
    changed = deepcopy(state); changed['rod_BC', MEMBRANE, 'release'] *= .9
    delta = inner.rhs(0., changed)[V_KEY]-base[V_KEY]
    if not np.min(delta) < 0 or np.max(delta) > 1e-12: raise ContractError('RBC excitation direction')
    checks['RBC_release_reduction_AII_dV_min_V_s'] = float(delta.min())
    changed = deepcopy(state); changed[S_KEY] *= .9
    d = inner.rhs(0., changed)
    for pop in ('midget_BC_OFF', 'diffuse_BC_OFF'):
        delta = d[pop, BC_MECH, 'V']-base[pop, BC_MECH, 'V']
        if not delta.max() > 0 or delta.min() < -1e-12: raise ContractError('OFF inhibition direction')
        checks[pop+'_disinhibition_max_V_s'] = float(delta.max())
    for pop in inner.gap_names:
        if pop == 'AII_AC': continue
        changed = deepcopy(state); changed[V_KEY] += .001
        forward = inner.rhs(0., changed)[pop, BC_MECH, 'V']-base[pop, BC_MECH, 'V']
        changed = deepcopy(state); changed[pop, BC_MECH, 'V'] += .001
        reverse = inner.rhs(0., changed)[V_KEY]-base[V_KEY]
        if forward.max() <= 0 or reverse.max() <= 0: raise ContractError('Bidirectional gap lost: '+pop)
        checks[pop+'_gap'] = dict(AII_to_BC_max_V_s=float(forward.max()), BC_to_AII_max_V_s=float(reverse.max()))
    section = inner.gap_slices['AII_AC']
    aii_gap = inner.gap.matrix[section, section].copy(); aii_gap.setdiag(0); aii_gap.eliminate_zeros()
    if not aii_gap.nnz: raise ContractError('AII-AII edges missing')
    row, column = aii_gap.nonzero(); source, target = int(column[0]), int(row[0])
    changed = deepcopy(state); changed[V_KEY][source] += .001
    delta = inner.rhs(0., changed)[V_KEY]-base[V_KEY]
    if delta[target] <= 0: raise ContractError('AII-AII current direction')
    checks['AII_AII'] = dict(directed_edges=aii_gap.nnz, source=source, target=target, target_delta_V_s=float(delta[target]))
    original_kinetics = inner._kinetics
    def forbidden(*args): raise ContractError('Old receptor path evaluated')
    inner._kinetics = forbidden
    try: inner.rhs(0., state)
    finally: inner._kinetics = original_kinetics
    aii_old = sum(old.sizes[k] for k in old.state_keys if k[0] == 'AII_AC' or k[1] == COMPONENTS[0])
    aii_new = sum(inner.sizes[k] for k in inner.state_keys if k[0] == 'AII_AC')
    return old, prior, dict(background_conductances=background, directionality=checks,
        total_inner_states=dict(old=old.nstate, new=inner.nstate), AII_related_states=dict(old=aii_old, new=aii_new),
        AII_groups=inner.counts['AII_AC'], AII_physical_members=int(inner.members['AII_AC'].sum()),
        no_old_receptor_evaluation=True, protected_population_parameters_unchanged=True,
        gap_graph_identical=True, gap_reciprocal=True, background=model.seed['functional_aii'])


def short_comparison(old, prior, new, state):
    result, traces = {}, {}
    for label, inner, initial in (('old', old, prior), ('new', new, state)):
        inner.compiled_rhs = DirectACVSRHS(inner)
        perturbed = deepcopy(initial)
        perturbed['rod_BC', MEMBRANE, 'V'] += .001
        for pop in ('midget_BC_ON', 'diffuse_BC_ON', 'S_ON_BC'):
            perturbed[pop, BC_MECH, 'V'] += .001
        started = time.perf_counter()
        advance = inner.advance_reference(perturbed, 0., .005, rtol=1e-6, atol=1e-9)
        elapsed = time.perf_counter()-started
        baseline, after = inner.outputs(0., initial), inner.outputs(.005, advance.state)
        result[label] = dict(wall_seconds=elapsed, nfev=advance.metrics['nfev'], state_scalars=inner.nstate,
            maximum_AII_voltage_change_V=float(np.max(abs(after['AII_AC', 'voltage']-baseline['AII_AC', 'voltage']))),
            maximum_OFF_inhibition_change_S={c:float(np.max(abs(after[c, 'conductance']-baseline[c, 'conductance']))) for c in COMPONENTS[1:]})
        times = np.linspace(0., .005, 11)
        traces[label+'__time_seconds'] = times
        for pop, port in (('AII_AC', 'voltage'), ('AII_FMB_glycine', 'conductance'),
                          ('AII_DB1_glycine', 'conductance'), ('rod_BC', 'release')):
            traces[label+'__'+pop+'__'+port] = np.column_stack([
                inner.outputs(t, advance.evaluate(t, 'right' if t == 0 else 'left'))[pop, port] for t in times])
    result.update(duration_seconds=.005, stimulus='fixed +1 mV initial RBC and three ON BC perturbation; held history activation',
        interpretation='one paired engineering timing sample; not a speed benchmark or frequency-response equivalence',
        new_over_old_wall_ratio=result['new']['wall_seconds']/result['old']['wall_seconds'])
    return result, traces


def recorder_check(assembly, snapshot, arm):
    # Exercise the existing disk writer and cursor restoration in owned scratch space.
    with tempfile.TemporaryDirectory(prefix='ac_recording_', dir=ROOT/'tmp') as temporary:
        path = Path(temporary)/'replay'
        recorder = GroupRecording(assembly, path, np.array([0., .0001]))
        recorder.record(0., arm.record_native_values(0., snapshot=snapshot), snapshot['inner_outputs'])
        cursor = recorder.checkpoint_payload()
        restored = GroupRecording.restore(assembly, path, cursor)
        assert restored.layout_identity == recorder.layout_identity
        with np.load(path/'windows/window_0000.npz') as stored:
            assert 'AII_AC.s' in stored and 'AII_AC.gE_RBC' in stored
            np.testing.assert_array_equal(stored['AII_AC.s'].ravel(), snapshot['ports']['AII_AC', 's'])
        return dict(fields=len(recorder.fields), restore_exact=True,
            units={f.key:f.unit for f in recorder.fields},
            identity={c['id']:{n:p['cell_identity_status'] for n,p in c['ports'].items()} for c in recorder.components})


def forward(model, arm):
    rows = [arm.snapshot(0., 'right', queue_node=True)]
    recorder = recorder_check(model.assembly, rows[0], arm)
    restored = None; max_restore = 0.; started = time.perf_counter()
    arm.initial_commit(); arm.ack(0.)
    try:
        for i in range(1, 501):
            _, right = tick(arm, i*.0001, i)
            if any(not np.isfinite(v).all() for s in ('states', 'ports') for v in right[s].values()):
                raise ContractError('Nonfinite functional full-chain state')
            gate = right['states'][S_KEY]
            if np.any(gate < 0) or np.any(gate > 1): raise ContractError('AII gate domain')
            if i % 10 == 0: rows.append(right)
            if i == 490:
                payload = arm.checkpoint_payload()
                restored = FunctionalACRunner.from_checkpoint(model.assembly, model.seed, model.protocol, payload)
                wrong = deepcopy(payload); wrong['controller'].pop('functional_aii')
                try: FunctionalACRunner.from_checkpoint(model.assembly, model.seed, model.protocol, wrong)
                except ContractError: pass
                else: raise ContractError('Legacy AII checkpoint accepted')
            elif i > 490:
                _, other = tick(restored, i*.0001, i)
                for section in ('states', 'rhs', 'ports'):
                    for k,v in right[section].items():
                        max_restore = max(max_restore, float(np.max(abs(v-other[section][k]), initial=0.)))
                if canonical_hash(arm.spike_states) != canonical_hash(restored.spike_states):
                    raise ContractError('RGC RNG continuity failed')
            if i % 100 == 0: print('Functional AC full chain', i/10, 'ms', round(time.perf_counter()-started, 2), 's wall', flush=True)
        if max_restore != 0: raise ContractError('Functional checkpoint continuation differs')
        arrays = {'time_seconds':np.array([r['time_seconds'] for r in rows])}
        for section in ('states', 'ports'):
            for k in rows[0][section]:
                if k[0] in ('AII_AC', *BC, *arm.inner.ac_populations, *RGC, *COMPONENTS):
                    arrays[section+'__'+'__'.join(k)] = np.column_stack([r[section][k] for r in rows])
        change = {n:{port:float(np.max(abs(rows[-1]['ports'][n, port]-rows[0]['ports'][n, port]), initial=0.))
                     for port in (('voltage', 's', 'gE_RBC') if n == 'AII_AC' else ('voltage',))}
                  for n in ('AII_AC', *BC, *RGC)}
        if change['AII_AC']['voltage'] <= 0 or change['AII_AC']['s'] <= 0:
            raise ContractError('AII failed cone-driven crossover response')
        return dict(duration_seconds=.05, wall_seconds=time.perf_counter()-started,
            checkpoint_max_abs_difference=max_restore, RGC_RNG_exact=True, old_checkpoint_rejected=True,
            finite=True, gate_in_bounds=True, endpoint_change=change, recording=recorder), arrays
    finally:
        if restored is not None: restored.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-failed', action='store_true')
    args = parser.parse_args()
    metrics = REPORT/'FUNCTIONAL_ACCEPTANCE.json'
    previous = None
    if metrics.exists():
        import json
        previous = json.loads(metrics.read_bytes())
        if not args.resume_failed or previous['status'] != 'FAILED_RETAINED':
            raise ContractError('Existing acceptance must not be repeated or overwritten')
    started = time.perf_counter()
    result = dict(status='RUNNING', started_utc=datetime.now(timezone.utc).isoformat())
    if previous: result['prior_attempts'] = previous.get('prior_attempts', [])+[{k:v for k,v in previous.items() if k != 'prior_attempts'}]
    before = protected(ROOT)
    for name in ('bc_independent.py', 'bc_physiology.py', 'group_parameter_binding.py', 'h2_priority.py', 'cone_hc_eccentric.py'):
        before['retina_sim/'+name] = digest(ROOT/'retina_sim'/name)
    arm = None
    try:
        model = RetinaBuilder.build_research_ac(ROOT)
        print('Functional AC built', round(time.perf_counter()-started, 2), 's', flush=True)
        result['model_identity'] = model.identity
        arm = model.runner()
        old, prior, result['static'] = static_checks(model, arm)
        if previous and previous.get('model_identity') == model.identity and 'comparison' in previous:
            result['comparison'] = previous['comparison']
            result['comparison_reused_from'] = previous['started_utc']
            comparison = {}
        else:
            result['comparison'], comparison = short_comparison(old, prior, arm.inner, arm.inner_state)
        result['fullchain'], arrays = forward(model, arm)
        arrays.update({'comparison__'+k:v for k,v in comparison.items()})
        for pop in ('AII_AC', *BC):
            layout = model.assembly.populations[pop]
            for field in ('group_ids', 'counts', 'member_offsets', 'member_global_rows', 'centroid_mm'):
                arrays['identity__'+pop+'__'+field] = getattr(layout, field)
        np.savez_compressed(REPORT/'FUNCTIONAL_STATES.npz', **arrays)
        write(REPORT/'FUNCTIONAL_PARAMETERS.json', dict(model_identity=model.identity,
            configuration=model.config, reference_input=model.upstream.seed['input_reference'],
            mapping=model.mapping, AII_parameters_SI=arm.inner.p['AII_AC'],
            source_provenance=model.upstream.seed['inner_parameters']['provenance'],
            source_AII_parameters=model.upstream.seed['inner_parameters']['populations']['AII_AC'],
            ordinary_AC_reused=True, physiological_calibration=False,
            state_units={'V':'V', 's':'dimensionless', 'conductance':'S', 'gap_current':'A', 'tau_release':'s'},
            qualification='inherited effective membrane and output gate; fixed-reference functional conductance mapping; not measured AII calibration'))
        after = {p:digest(ROOT/p) for p in before}
        if before != after: raise ContractError('Protected source changed during acceptance')
        result.update(status='AC_FUNCTIONAL_ENGINEERING_PASS', protected_sources=after,
            implementation_sha256={p:digest(ROOT/p) for p in ('retina_sim/aii_functional.py',
                'retina_sim/group_inner_dynamics.py', 'retina_sim/group_time_recording.py',
                'retina_sim/assembly.py', 'configurations/research/AC_functional.json')},
            FIT_READY=False, physiological_calibration=False, optimization_runs=0,
            whole_retina_qualified=False, engineering_ready_for_layered_calibration=True)
    except Exception as error:
        result.update(status='FAILED_RETAINED', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if arm is not None: arm.close()
        result.update(wall_seconds=time.perf_counter()-started, ended_utc=datetime.now(timezone.utc).isoformat())
        write(metrics, result)
    print(result['status'], result['comparison'], flush=True)


if __name__ == '__main__':
    main()
