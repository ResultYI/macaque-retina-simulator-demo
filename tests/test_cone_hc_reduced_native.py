"""Small CPU mathematical/ownership checks, not full-model numerical acceptance."""
from types import SimpleNamespace as NS
import subprocess
import sys
import unittest

import numpy as np
from scipy import sparse
from scipy.linalg import expm

from retina_sim.contracts import ContractError, Quantity
from retina_sim.vh2005 import SOURCE_VALUES, NativeConeH1
from retina_sim.cone_hc_reduced_native import (
    AnalyticREAnchor, BASE_STEP_SECONDS, QuadraticNativeSegment,
    ReducedNative, ReducedNativeState, frozen_exponential,
)


def fixture(aux=(0, 2)):
    A = sparse.csr_matrix(np.array([[.2, .3, .5], [.5, .25, .25]], dtype=np.float64))
    B = sparse.csr_matrix(np.array([[.3, .7], [.6, .4], [.5, .5]], dtype=np.float64))
    units = ('td', 'td', 'vH_native_precursor', 'vH_native_current',
             'vH_native_voltage', 'vH_native_conductance', 'vH_native_voltage',
             'vH_native_voltage', 'vH_native_voltage', 'vH_native_voltage')
    model = NS(p=dict(SOURCE_VALUES), nc=3, nh=2, A=A, B=B,
               cones=NS(cells=NS(cell_id=('cone0', 'cone1', 'cone2'))),
               model_hash='small-source-fixture', tag='synthetic_fixture',
               state_specs=tuple(NS(unit=u, meaning='test_native_state') for u in units))
    model.signals = lambda blocks, context=None: NativeConeH1.signals(model, blocks, context)
    old = np.concatenate(tuple(np.array(a, dtype=np.float64) for a in (
        [90, 100, 110], [95, 96, 97], [4, 5, 6], [3, 4, 5],
        [18, 20, 22], [.25, .3, .35], [17, 19, 21], [100, 110, 120],
        [80, 90], [5, 8])))
    light = np.array([105, 90, 130], dtype=np.float64)
    native = ReducedNative(model, aux_indices=aux)
    return model, native, old, light


class AnalyticFilterTests(unittest.TestCase):
    def test_RE_matches_independent_matrix_exponential_and_no_extrapolation(self):
        _, _, old, light = fixture()
        anchor = AnalyticREAnchor(1., old[:3], old[3:6], light, 1.01, 7)
        tau_r, tau_e = SOURCE_VALUES['tau_R']/1000, SOURCE_VALUES['tau_E']/1000
        matrix = np.array([[-1/tau_r, 0., 1/tau_r],
                           [1/tau_e, -1/tau_e, 0.], [0., 0., 0.]])
        for delta in (0., .000031, .0064, .01):
            actual = anchor.at(1.+delta)
            operator = expm(matrix*((1.+delta)-1.))
            expected = operator @ np.stack((old[:3], old[3:6], light))
            np.testing.assert_allclose(actual[0], expected[0], rtol=2e-14, atol=2e-13)
            np.testing.assert_allclose(actual[1], expected[1], rtol=2e-14, atol=2e-13)
        with self.assertRaises(ContractError):
            anchor.at(np.nextafter(1.01, np.inf))
        with self.assertRaises(ContractError):
            anchor.at(np.nextafter(1., -np.inf))

    def test_real_boundary_refresh_preserves_memory_and_changes_drive(self):
        _, native, old, light = fixture()
        boundary = .000067
        state = native.migrate(old, time_seconds=0., light=light,
                               input_stop_seconds=boundary, input_epoch=2)
        advanced, segment = native.step(state)
        self.assertEqual(advanced.time_seconds, boundary)
        self.assertEqual(segment.input_epoch, 2)
        before = state.anchor.at(boundary)
        after = native.refresh_input(advanced, light*2, .006666666666666667, 3)
        for actual, expected in zip(after.anchor.at(boundary), before):
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(advanced.anchor.rhs(boundary)['R'], after.anchor.rhs(boundary)['R']))
        with self.assertRaisesRegex(ContractError, 'actual input boundary'):
            native.refresh_input(state, light*2, .02, 3)
        with self.assertRaisesRegex(ContractError, 'epoch'):
            native.refresh_input(advanced, light*2, .02, 4)
        with self.assertRaisesRegex(ContractError, 'refresh'):
            native.step(advanced)


class ExponentialAndDenseTests(unittest.TestCase):
    def test_zero_and_tiny_b_have_stable_limits(self):
        x = np.array([1., 2., 3.])
        a = np.array([4., 5., 6.])
        b = np.array([0., 1e-300, 7.])
        actual = frozen_exponential(x, a, b, .0001)
        np.testing.assert_array_equal(actual[:2], (x+.0001*a)[:2])
        expected = expm(np.array([[-7., 6.], [0., 0.]])*.0001) @ np.array([3., 1.])
        self.assertAlmostEqual(actual[2], expected[0], places=14)
        np.testing.assert_array_equal(frozen_exponential(x, a, 0., .0001), x+.0001*a)
        with self.assertRaises(ContractError):
            frozen_exponential(x, a, b, -.1)

    def test_quadratic_endpoints_midpoint_internal_ULP_and_projection(self):
        y0, f0, y1 = (np.array(v, dtype=np.float64) for v in
                       ([2., 3.], [4., 5.], [7., 11.]))
        segment = QuadraticNativeSegment(1., 1.0001, y0, f0, y1, 3)
        y0[0] = -100
        self.assertEqual(segment.at(1.)[0], 2.)
        np.testing.assert_array_equal(segment.at(segment.stop), y1)
        t = (segment.start+segment.stop)/2
        theta = (t-segment.start)/(segment.stop-segment.start)
        expected = segment.y0+theta*(segment.stop-segment.start)*segment.f0+theta**2*(segment.y1-segment.y0-(segment.stop-segment.start)*segment.f0)
        np.testing.assert_array_equal(segment.at(t), expected)
        for fraction in (.113, .731):
            time = segment.start+fraction*(segment.stop-segment.start)
            self.assertTrue(np.all(np.isfinite(segment.at(time))))
        self.assertTrue(np.all(np.isfinite(segment.at(np.nextafter(segment.start, segment.stop)))))
        np.testing.assert_array_equal(segment.project_rows(np.array([1])).at(t), segment.at(t)[[1]])
        with self.assertRaises(ContractError):
            segment.at(np.nextafter(segment.stop, np.inf))
        with self.assertRaises(ValueError):
            segment.y0[0] = 7


class ReducedEquationTests(unittest.TestCase):
    def test_migration_and_true_rhs_use_original_equations_and_aggregation(self):
        model, native, old, light = fixture()
        state = native.migrate(old, time_seconds=0., light=light, input_stop_seconds=.01)
        self.assertEqual(native.nstate, 5*3+3*2+2)
        self.assertEqual(native.aux_cell_ids, ('cone0', 'cone2'))
        np.testing.assert_array_equal(state.y[native.slices['w']], model.A @ old[21:24])
        np.testing.assert_array_equal(state.y[native.slices['z1_aux']], old[21:24][[0, 2]])
        cuts = np.cumsum((0,)+(3,)*8+(2,)*2)
        blocks = tuple(old[cuts[i]:cuts[i+1]] for i in range(10))
        reference = NativeConeH1.derivative(model, 0., blocks,
                    {'light': Quantity(light, 'td', 'native_retinal_illuminance', 'synthetic_fixture')})
        d = tuple(q.values for q in reference)
        actual = native.evaluate(state.y, 0., state.anchor)
        expected = np.concatenate((*d[2:7], model.A@d[7], d[8], d[9], d[7][[0, 2]]))
        np.testing.assert_allclose(actual.rhs, expected, rtol=2e-14, atol=3e-11)
        np.testing.assert_array_equal(actual.analytic_rhs['R'], d[0])
        np.testing.assert_array_equal(actual.analytic_rhs['E'], d[1])
        np.testing.assert_allclose(actual.outputs['release'], model.signals(blocks)[3], rtol=2e-15, atol=0)
        self.assertNotIn('z1', actual.outputs)

    def test_step_uses_two_coefficient_rounds_and_true_f0(self):
        _, native, old, light = fixture()
        state = native.migrate(old, time_seconds=0., light=light, input_stop_seconds=.01)
        before = state.y.copy()
        result, segment = native.step(state)
        self.assertEqual(result.time_seconds, BASE_STEP_SECONDS)
        self.assertEqual(native.statistics, {'steps': 1, 'coefficient_evaluations': 2})
        np.testing.assert_array_equal(state.y, before)
        np.testing.assert_array_equal(segment.f0, native.rhs(state.y, 0., state.anchor))
        np.testing.assert_array_equal(segment.at(result.time_seconds), result.y)
        # A real RHS is not the derivative of the quadratic interpolation.
        tm = (segment.start+segment.stop)/2
        actual_rhs = native.rhs(segment.at(tm), tm, segment.anchor)
        polynomial_rhs = segment.f0+2*.5*(segment.y1-segment.y0-(segment.stop-segment.start)*segment.f0)/(segment.stop-segment.start)
        self.assertGreater(float(np.max(np.abs(actual_rhs-polynomial_rhs))), 1e-8)

    def test_frozen_source_K_identity_domains_and_complete_old_state_required(self):
        model, native, old, light = fixture()
        state = native.migrate(old, time_seconds=0., light=light, input_stop_seconds=.01)
        with self.assertRaisesRegex(ContractError, 'selection/order'):
            native.require_aux_indices((2, 0))
        with self.assertRaisesRegex(ContractError, 'complete legal old'):
            native.migrate(old[:-1], time_seconds=0., light=light, input_stop_seconds=.01)
        with self.assertRaisesRegex(ContractError, 'source/auxiliary'):
            native.step(ReducedNativeState(0., state.y, state.anchor, 'different'))
        bad = state.y.copy()
        bad[native.slices['g_i']] = 0
        with self.assertRaisesRegex(ContractError, 'domain'):
            native.evaluate(bad, 0., state.anchor)
        model.p = dict(model.p, n_X=2.)
        with self.assertRaisesRegex(ContractError, 'source parameters'):
            ReducedNative(model)

    def test_topology_is_readonly_and_auxiliary_selection_is_optional(self):
        _, native, old, light = fixture(aux=())
        with self.assertRaises(ValueError):
            native.A.data[0] = 0
        self.assertEqual(native.sizes[-1], 0)
        state = native.migrate(old, time_seconds=0., light=light, input_stop_seconds=.01)
        result, _ = native.step(state)
        self.assertEqual(result.y[native.slices['z1_aux']].shape, (0,))

    def test_module_import_does_not_import_torch(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            "import sys; import retina_sim.cone_hc_reduced_native; assert 'torch' not in sys.modules"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
