"""Manufactured circuits test the reduction contract, not physiological calibration."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import root
from scipy.special import expit

from retina_sim.ac_vs_model import MECHANISM, LEGACY_S, equilibrium, DirectACVSRHS
from retina_sim.aii_functional import (COMPONENTS, V_KEY, S_KEY, FunctionalAIIAssembly,
    FunctionalAIIInner, reference_mapping, initialize_functional_background)
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_inner_dynamics import MEMBRANE, BC_MECH, AC_MECH, RGC_MECH
from retina_sim.spatial_group_model import BC, RGC, RecordingField
from test_ac_vs_model import new_fixture


def reference_fixture():
    _, _, old, state = new_fixture()
    # Include a reciprocal AII-AII edge as well as the existing AII-ON edge.
    gap = old.gap.matrix.tolil()
    gap[0, 0] += 2e-10; gap[1, 1] += 2e-10
    gap[0, 1] -= 2e-10; gap[1, 0] -= 2e-10
    gap = gap.tocsr()
    previous_matrix = old.assembly.matrix
    old.assembly.matrix = lambda p: gap if p == 'gap/laplacian_S.npz' else previous_matrix(p)
    old.gap.matrix = gap
    def ribbon(pop):
        mech = 'rod_membrane' if pop == 'rods' else MEMBRANE
        v = state[pop, mech, 'V']
        state[pop, mech, 'release'] = expit((v-old.par(pop, 'release_half'))/old.par(pop, 'release_slope'))
        if pop != 'rods':
            attempts = old.par(pop, 'rmax')*expit((v-old.par(pop, 'r_half'))/old.par(pop, 'r_slope'))
            state[pop, mech, 'x'] = 1/(1+old.par(pop, 'tau_recovery')*old.par(pop, 'U')*attempts)
            return old.par(pop, 'Npool')*old.par(pop, 'U')*attempts*state[pop, mech, 'x']
    state['rods', 'rod_membrane', 'V'] = old.par('rods', 'EL')-(old.rod_current(state)-old.par('rods', 'current_anchor'))/old.par('rods', 'gL')
    ribbon('rods')
    component = 'individual_rod_RBC_contacts'; c = old.contacts[component]
    m = expit(-old.cp(component, 'hill')*np.log(state['rods', 'rod_membrane', 'release'][c['pre']]/old.cp(component, 'K')))
    state['rods', component, 'm'] = m
    ge = np.bincount(c['post'], weights=c['coefficient']*m, minlength=old.counts['rod_BC'])
    state['rod_BC', MEMBRANE, 'V'] = (old.gL['rod_BC']*old.par('rod_BC', 'EL')+ge*old.par('rod_BC', 'EE'))/(old.gL['rod_BC']+ge)
    rate_rbc = ribbon('rod_BC')
    voltage_keys = [V_KEY]+[(p, BC_MECH, 'V') for p in BC]
    def balances(values):
        at = 0
        for k in voltage_keys:
            n = old.sizes[k]; state[k] = values[at:at+n].copy(); at += n
        rate_aii = ribbon('AII_AC')
        for component in COMPONENTS:
            c = old.contacts[component]
            rate = (rate_rbc if component == COMPONENTS[0] else rate_aii)[c['pre']]
            rise, decay, _ = old._kinetics(state, component)
            state[c['route'].pre, component, 'a'] = rate*rise
            state[c['route'].pre, component, 'b'] = rate*decay
        rhs = old.rhs(0., state)
        return np.concatenate([rhs[k] for k in voltage_keys])
    solution = root(balances, np.concatenate([state[k] for k in voltage_keys]), tol=1e-11)
    assert solution.success
    np.testing.assert_allclose(balances(solution.x), 0., atol=1e-10)
    for pop in BC:
        rel = expit((state[pop, BC_MECH, 'V']-old.par(pop, 'release_half'))/old.par(pop, 'release_slope'))
        state[pop, BC_MECH, 'terminal_release'] = rel
        state[pop, BC_MECH, 'A'] = old.par(pop, 'adaptation_scale')*rel
    out = old.outputs(0., state)
    state[LEGACY_S, AC_MECH, 'h'] = out[LEGACY_S, 'target_inhibition']
    for channel in old.channels:
        pop = channel['pool']; v, _, s = equilibrium(out[pop, 'gE'], old.p[pop])
        state[pop, MECHANISM, 'V'], state[pop, MECHANISM, 's'] = v, s
    out = old.outputs(0., state)
    for pop in RGC:
        g, ge, gi = old.gL[pop], out[pop, 'gE'], out[pop, 'gI']
        state[pop, RGC_MECH, 'V'] = (g*old.par(pop, 'EL')+ge*old.par(pop, 'EE')+gi*old.par(pop, 'EI'))/(g+ge+gi)
    a = old.assembly
    a.runtime_factories = {}; a.recording_fields = ()
    for name, layout in a.populations.items():
        layout.coordinate_role = 'soma'
    original = dict(states=state, inner_parameters=deepcopy(old.parameters), h1_h2_model={},
                    joint_background_initialization={})
    original['inner_parameters']['provenance'] = {}
    matrices, edges, mapping = reference_mapping(old, state)
    assembly = FunctionalAIIAssembly(a, matrices, edges, mapping)
    seed = initialize_functional_background(assembly, original)
    new = FunctionalAIIInner(assembly, seed['inner_parameters'])
    new.compiled_rhs = DirectACVSRHS(new)
    return old, state, new, seed['states']


def test_fixed_background_units_counts_and_unchanged_ac_bc_parameters():
    old, original, model, state = reference_fixture()
    assert set(k[2] for k in model.state_keys if k[0] == 'AII_AC') == {'V', 's'}
    assert not any(k[1] in COMPONENTS for k in model.state_keys)
    assert model.nstate == old.nstate-old.counts['AII_AC']-2*sum(len(old.contacts[c]['pre']) for c in COMPONENTS)
    np.testing.assert_array_equal(model.C['AII_AC'], [2e-11, 4e-11])
    for pop in (*BC, *old.ac_populations, *RGC, 'rods', 'rod_BC'):
        assert canonical_hash(model.p[pop]) == canonical_hash(old.p[pop])
    out, prior = model.outputs(0., state), old.outputs(0., original)
    for c in COMPONENTS:
        np.testing.assert_allclose(out[c, 'conductance'], prior[c, 'conductance'], rtol=1e-11, atol=1e-23)
        assert np.all(model.K[c].data >= 0)
    assert max(abs(v).max(initial=0.) for v in model.rhs(0., state).values()) < 1e-8


def test_rbc_gate_directly_drives_aii_and_aii_gate_inhibits_off():
    _, _, model, state = reference_fixture()
    base = model.rhs(0., state)
    changed = deepcopy(state); changed['rod_BC', MEMBRANE, 'release'] *= .9
    assert model.rhs(0., changed)[V_KEY][0] < base[V_KEY][0]
    changed = deepcopy(state); changed[S_KEY] *= .9
    for pop in ('midget_BC_OFF', 'diffuse_BC_OFF'):
        assert model.rhs(0., changed)[pop, BC_MECH, 'V'][0] > base[pop, BC_MECH, 'V'][0]
    before = model.outputs(0., state)
    changed = deepcopy(state); changed['rod_BC', MEMBRANE, 'x'] *= .5
    np.testing.assert_array_equal(model.outputs(0., changed)['AII_AC', 'gE_RBC'], before['AII_AC', 'gE_RBC'])


def test_bidirectional_gap_and_reciprocal_aii_no_double_count():
    _, _, model, state = reference_fixture()
    base = model.rhs(0., state)
    for source, target, row in ((V_KEY, ('midget_BC_ON', BC_MECH, 'V'), 0),
                                (('midget_BC_ON', BC_MECH, 'V'), V_KEY, 0),
                                (V_KEY, V_KEY, 1)):
        changed = deepcopy(state); changed[source][0] += .001
        assert model.rhs(0., changed)[target][row] > base[target][row]
        out = model.outputs(0., changed)
        assert abs(sum(v.sum() for k,v in out.items() if k[1] == 'gap_current')) < 1e-25
    np.testing.assert_array_equal(model.gap.matrix.toarray(), model.gap.matrix.T.toarray())


def test_gate_bounds_si_tau_and_no_legacy_receptor_evaluation():
    _, _, model, state = reference_fixture()
    def forbidden(*args): raise AssertionError('Old receptor kinetics evaluated')
    model._kinetics = forbidden
    for s in (0., 1.):
        state[S_KEY][:] = s
        derivative = model.rhs(0., state)[S_KEY]
        expected = (expit((state[V_KEY]+.05)/.005)-s)/.003
        np.testing.assert_allclose(derivative, expected)
        assert np.all(derivative >= 0) if s == 0 else np.all(derivative <= 0)
        # The runner packs derivatives too: gate domain checks belong to states.
        result = model.advance_reference(state, 0., .0002, rtol=1e-6, atol=1e-9)
        assert np.all((result.state[S_KEY] >= 0)&(result.state[S_KEY] <= 1))
    state[S_KEY][0] = 1.01
    with pytest.raises(ContractError, match='no clipping'): model.outputs(0., state)


def test_mapping_rejects_nonreference_and_zero_denominator():
    old, state, _, _ = reference_fixture()
    changed = deepcopy(state); changed[V_KEY][0] += .001
    with pytest.raises(ContractError, match='fixed reference'): reference_mapping(old, changed)
    # Force an exactly zero reference release while all source dynamics stay at rest.
    old.p['rod_BC']['release_half'] = 100.
    old._par_cache.clear()
    state['rod_BC', MEMBRANE, 'release'][:] = 0.
    with pytest.raises(ContractError, match='Blocked'): reference_mapping(old, state)


def test_recording_preserves_typed_hc_fine_selection_and_effective_pool_identity():
    old, _, model, _ = reference_fixture()
    parent = old.assembly
    parent.populations['HC'].fine_to_group = np.array([0, 1, 0])
    parent.recording_fields = (RecordingField('HC', 'outer_native', 'w', 'vH_native_voltage',
        'typed H1 auxiliary selection', np.array([0]), False, 'soma', np.array([0, 1, 2])),)
    assembly = FunctionalAIIAssembly(parent, model.K, model.edge_gain, model.assembly.aii_mapping)
    field = assembly.recording_fields[0]
    np.testing.assert_array_equal(field.original_fine_rows, [0, 2])
    np.testing.assert_array_equal(field.rows, [0, 0])
    assert assembly.recording_field_count == len({f.key for f in assembly.recording_fields})
    pools = {c['pool'] for c in old.channels}
    assert all(f.identity == 'effective_functional_pool' for f in assembly.recording_fields if f.population in pools)
