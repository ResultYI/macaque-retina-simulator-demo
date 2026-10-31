"""Manufactured software checks, never real-network functional acceptance."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from retina_sim.contracts import ContractError
from retina_sim.group_inner_dynamics import GroupInnerDynamics, RGC_MECH, key
from retina_sim.inner import HistoryRGC
from retina_sim.rgc_expectation import STATES
from retina_sim.rgc_functional_v1 import RGCFunctionalV1
from retina_sim.spatial_group_model import RGC


class ManufacturedInner:
    commit_ticks = GroupInnerDynamics.commit_ticks
    _commit_ticks = GroupInnerDynamics._commit_ticks
    _probability_tick_updates = GroupInnerDynamics._probability_tick_updates

    def __init__(self):
        self.counts = {p: 2 for p in RGC}
        self.C = {p: np.array([5e-11, 35e-11]) for p in RGC}
        self.gL = {p: np.array([2e-9, 14e-9]) for p in RGC}
        self.p = {p: dict(EL=-.06, EE=0., EI=-.07, alpha=50., mu=-.04,
                         beta=.005, spike_history_kernel=[-.005, -.002, -.001])
                  for p in RGC}
        self.state = {}
        for pop in RGC:
            self.state[key(pop, RGC_MECH, 'V')] = np.array([-.05, -.04])
            for i in range(8):
                self.state[key(pop, 'RGC_expected_history', 'history_probability_' + str(i))] = np.full(2, float(i == 0))
        self.state_keys = tuple(self.state)

    def par(self, pop, name, **kwargs):
        return np.full(2, self.p[pop][name])

    def validate(self, state):
        if set(state) != set(self.state_keys):
            raise ContractError('Manufactured state layout differs')

    def outputs(self, t, state, cone_release=None):
        out = {}
        for pop in RGC:
            V = state[key(pop, RGC_MECH, 'V')]
            pi = np.column_stack([state[key(pop, 'RGC_expected_history', 'history_probability_' + str(i))] for i in range(8)])
            hazard = self.par(pop, 'alpha')[:, None] * np.logaddexp(0.,
                (V[:, None] + (STATES @ np.array(self.p[pop]['spike_history_kernel']))[None, :]
                 - self.par(pop, 'mu')[:, None]) / self.par(pop, 'beta')[:, None])
            for name, value in dict(voltage=V, gE=np.array([1e-10, 7e-10]),
                                    gI=np.array([2e-10, 14e-10]),
                                    expected_hazard_Hz=(pi * hazard).sum(1),
                                    expected_rate_Hz=(pi * -np.expm1(-.001 * hazard)).sum(1) / .001).items():
                out[pop, name] = value
        return out


def fixture():
    layer = object.__new__(RGCFunctionalV1)
    layer.model = ManufacturedInner()
    layer.populations = tuple(RGC)
    layer.origin = 1.
    layer.fields = layer._fields()
    return layer, layer.model.state


def spike_states():
    result = {}
    for i, pop in enumerate(RGC):
        rng = np.random.Generator(np.random.PCG64(200 + i))
        result[pop] = dict(history=[[0, 0] for _ in range(3)],
                           last_bin_counts=[1, 0], rng_state=rng.bit_generator.state,
                           pending_events=[])
    return result


class RGCFunctionalChecks(unittest.TestCase):
    def test_unqualified_configuration_fails_before_upstream_construction(self):
        cfg = dict(schema='RGC-functional-effective-layer-1', types=list(RGC),
                   new_kinetic_mechanism=False, new_connections=False,
                   physiologically_calibrated=False, production_adopted=False,
                   background_qualified=False, FIT_READY=False)
        for changed in (dict(FIT_READY=True), dict(types=list(RGC[:-1])),
                        dict(new_connections=True), dict(physiologically_calibrated=True)):
            with self.subTest(changed=changed), patch('retina_sim.rgc_functional_v1.ACFunctionalV1') as upstream:
                with self.assertRaises(ContractError):
                    RGCFunctionalV1('.', dict(cfg, **changed))
                upstream.assert_not_called()

    def test_frozen_upstream_identity_mismatch_is_rejected(self):
        cfg = dict(schema='RGC-functional-effective-layer-1', types=list(RGC),
                   new_kinetic_mechanism=False, new_connections=False,
                   physiologically_calibrated=False, production_adopted=False,
                   background_qualified=False, FIT_READY=False,
                   AC_configuration='unused', AC_layer_identity='frozen')
        with patch('retina_sim.rgc_functional_v1.ACFunctionalV1', return_value=SimpleNamespace(identity='different')):
            with self.assertRaises(ContractError):
                RGCFunctionalV1('.', cfg)

    def test_all_five_types_and_physical_currents_match_native_membrane(self):
        layer, state = fixture()
        observed = layer.observe(1., state, None)
        self.assertEqual(len(observed), 50)
        for pop in RGC:
            raw = layer.model.outputs(1., state)
            proxy = SimpleNamespace(tag='numeric_fixture', C=layer.model.C[pop], gL=layer.model.gL[pop],
                                    EL=layer.model.par(pop, 'EL'), EE=layer.model.par(pop, 'EE'),
                                    EI=layer.model.par(pop, 'EI'))
            native = HistoryRGC.derivative(proxy, 1., [raw[pop, 'voltage']],
                dict(excitation=SimpleNamespace(values=raw[pop, 'gE']),
                     inhibition=SimpleNamespace(values=raw[pop, 'gI'])))[0].values
            np.testing.assert_array_equal(observed[pop + '__voltage_derivative'], native)
            self.assertTrue(np.all(observed[pop + '__excitatory_current'] > 0))
            self.assertTrue(np.all(observed[pop + '__inhibitory_current'] < 0))

    def test_hazard_and_binary_rate_remain_distinct_without_member_multiplier(self):
        layer, state = fixture()
        observed = layer.observe(1., state, None)
        for pop in RGC:
            raw = layer.model.outputs(1., state)
            np.testing.assert_array_equal(observed[pop + '__expected_binary_count_rate'], raw[pop, 'expected_rate_Hz'])
            self.assertTrue(np.all(observed[pop + '__expected_binary_count_rate'] < observed[pop + '__expected_hazard']))

    def test_original_sampler_preserves_rng_history_and_reproducibility(self):
        layer, state = fixture()
        aux = spike_states()
        before = deepcopy(aux)
        a = layer.model.commit_ticks(1.001, state, rgc_tick=True, spike_states=aux)
        b = layer.model.commit_ticks(1.001, state, rgc_tick=True, spike_states=aux)
        self.assertEqual(aux, before)
        self.assertEqual(a[1:], b[1:])
        for pop in RGC:
            self.assertEqual(a[1][pop]['history'][0], [1, 0])
            probabilities = np.column_stack([a[0][key(pop, 'RGC_expected_history', 'history_probability_' + str(i))] for i in range(8)])
            np.testing.assert_allclose(probabilities.sum(1), 1., rtol=0, atol=1e-15)
            self.assertLessEqual(len(a[2][pop]), 2)
            self.assertTrue(all(1.001 <= e['time_seconds'] < 1.002 for e in a[2][pop]))

    def test_commit_keeps_original_joint_clocks_and_common_snapshot(self):
        layer, state = fixture()
        calls = []
        layer.model.commit_ticks = lambda t, s, **kw: calls.append((t, s, kw))
        layer.commit(10, state, 'cone', 'rod', 'rng')
        self.assertIs(calls[0][1], state)
        self.assertEqual(calls[0][2], dict(cone_release='cone', rod_light='rod', rod_tick=True,
                                         bc_tick=True, rgc_tick=True, spike_states='rng'))
        layer.commit(11, state, 'cone', 'rod', 'rng')
        self.assertFalse(calls[1][2]['bc_tick'])
        self.assertFalse(calls[1][2]['rgc_tick'])
        self.assertIsNone(calls[1][2]['cone_release'])
        with self.assertRaises(ContractError):
            layer.commit(True, state, 'cone', 'rod', 'rng')

    def test_nonfinite_readout_fails_without_repair(self):
        layer, state = fixture()
        state[key(RGC[0], RGC_MECH, 'V')][0] = np.nan
        with np.errstate(invalid='ignore'), self.assertRaises(ContractError):
            layer.observe(1., state, None)

    def test_invalid_original_probability_law_is_rejected(self):
        layer, state = fixture()
        state[key(RGC[-1], 'RGC_expected_history', 'history_probability_0')][:] = .5
        with self.assertRaises(ContractError):
            layer.model.commit_ticks(1., state, rgc_tick=True, spike_states=spike_states())


if __name__ == '__main__':
    unittest.main()
