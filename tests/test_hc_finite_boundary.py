import copy
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from retina_sim.contracts import ContractError
from retina_sim.hc_biexponential import condition_spatial_background
from retina_sim.hc_support_closure import require_finite_boundary_native
from retina_sim.h1_h2_effective import H1H2Native, joint_equilibrium
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.vh2005 import SOURCE_VALUES


def complete_exterior(nh, nc):
    return dict(A_missing_mass=np.zeros(nh), B_missing_mass=np.zeros(nc),
                A_release=np.zeros(nh), A_Vbar=np.zeros(nh), B_h=np.zeros(nc),
                evidence='complete declared manufactured parent')


class FiniteBoundary(unittest.TestCase):
    def setUp(self):
        self.A = sparse.csr_matrix([[.2, .5, .3], [.1, .2, .7]])
        self.B = sparse.csr_matrix([[.6, .4], [.3, .7], [.8, .2]])
        self.kw = dict(release0=np.array([.1, .2, .4]), Vbar0=np.array([20., 30., 40.]),
                       Vh0=np.array([4., 7.]), exterior=complete_exterior(2, 3),
                       background_identity='fixture-background')

    def bind(self, **kw):
        return condition_spatial_background(self.A, self.B, np.array([0, 2]), np.array([1]),
                                            **dict(self.kw, **kw))

    def test_affine_products_equal_background_plus_perturbation(self):
        b = self.bind(); c, h = b['cone_rows'], b['hc_rows']
        for values, key in ((self.kw['release0'], 'A_release'), (self.kw['Vbar0'], 'A_Vbar')):
            dynamic = values[c]+[.02, -.01]
            expected = (self.A@values)[h]+self.A[h][:, c]@(dynamic-values[c])
            np.testing.assert_allclose(b['A']@dynamic+b['boundary'][key], expected, rtol=2e-15)
        dynamic_h = self.kw['Vh0'][h]+.3
        expected = (self.B@self.kw['Vh0'])[c]+self.B[c][:, h]@(dynamic_h-self.kw['Vh0'][h])
        np.testing.assert_allclose(b['B']@dynamic_h+b['boundary']['B_h'], expected)
        np.testing.assert_allclose(b['A'].sum(1), .8)
        np.testing.assert_allclose(b['B'].sum(1).A.ravel(), [.4, .2])
        self.assertFalse(b['closed_loop_qualified'])

    def test_explicit_missing_source_is_retained_and_never_inferred_zero(self):
        self.A *= .8
        with self.assertRaisesRegex(ContractError, 'missing mass'): self.bind()
        ext = complete_exterior(2, 3)
        ext.update(A_missing_mass=np.full(2, .2), A_release=np.full(2, .03),
                   A_Vbar=np.full(2, 5.), evidence='declared source-exterior background approximation')
        b = self.bind(exterior=ext)
        np.testing.assert_allclose(b['mass_ledger']['A_missing_source_mass'], .2)
        np.testing.assert_allclose(b['A']@self.kw['Vbar0'][[0, 2]]+b['boundary']['A_Vbar'],
                                   (self.A@self.kw['Vbar0'])[1:]+5.)
        with self.assertRaises(ContractError): self.bind(exterior={})

    def test_invalid_rows_and_nonfinite_data_rejected(self):
        for rows in (np.array([0, 0]), np.array([-1]), np.array([3]), np.array([.2])):
            with self.assertRaises(ContractError):
                condition_spatial_background(self.A, self.B, rows, np.array([1]), **self.kw)
        with self.assertRaises(ContractError): self.bind(Vh0=np.array([np.nan, 7.]))
        with self.assertRaises(ContractError): self.bind(Vbar0=np.zeros(3))
        self.A.data[0] = -1
        with self.assertRaises(ContractError): self.bind()

    def test_boundary_identity_covers_background_and_binding(self):
        b = self.bind()
        self.assertEqual(b['identity'], self.bind()['identity'])
        self.assertNotEqual(b['identity'], self.bind(Vh0=np.array([4., 8.]))['identity'])
        self.kw['Vh0'][0] = 99.
        np.testing.assert_allclose(b['boundary']['B_h'], [2.4, 3.2])

    def test_finite_admission_is_separate_and_fail_closed(self):
        b = self.bind()
        kwargs = dict(dynamic_feedback_complete=np.ones(2, bool), dynamic_input_available=np.ones(2, bool),
                      background_max_rhs=0., background_tolerance=1e-8)
        result = require_finite_boundary_native(b, **kwargs)
        self.assertTrue(result['experimental_admission'])
        self.assertFalse(result['response_convergence_qualified'])
        for key, value in [('dynamic_feedback_complete', np.array([True, False])),
                           ('dynamic_input_available', np.array([True, False])),
                           ('background_max_rhs', 1e-7), ('background_max_rhs', np.nan)]:
            with self.assertRaises(ContractError): require_finite_boundary_native(b, **dict(kwargs, **{key: value}))
        missing = copy.deepcopy(b); missing['mass_ledger']['B_missing_source_mass'][0] = .1
        with self.assertRaises(ContractError): require_finite_boundary_native(missing, **kwargs)

    def test_native_background_and_120ms_restart_with_nonzero_exterior(self):
        p = SOURCE_VALUES
        pars = dict(tau_seconds=p['tau_h']/1000, input_scale=p['g_t'], feedback_scale=1.)
        light = np.full(3, 1000.)
        eq = joint_equilibrium(self.A, self.B, np.array([0, 1]), np.array([], int), light, pars)
        b = self.bind(release0=eq['release'], Vbar0=eq['cone']['Vbar'], Vh0=eq['h'])
        pop = lambda n: SimpleNamespace(group_count=n, group_ids=np.arange(n))
        cones = pop(2); cones.recording_auxiliary_group_rows = np.array([], int)
        cones.recording_auxiliary_fine_rows = np.array([], int)
        assembly = SimpleNamespace(populations={'cones': cones, 'HC': pop(1)},
            matrix=lambda path: b['A'] if 'cones_to_HC' in path else b['B'],
            artifact_sha256=b['identity'], h1_rows=np.array([0]), h2_rows=np.array([], int),
            h2_parameters=pars, layout={'fixture': True}, connection_identity={'identity': b['identity']})
        native = H1H2Native(assembly); native.boundary = b['boundary']
        ci, hi = b['cone_rows'], b['hc_rows']
        y0 = np.concatenate([eq['cone'][k][ci] for k in ('X', 'C', 'Vis', 'g_i', 'Vbar')]
                            +[eq[k][hi] for k in ('w', 'z2', 'h')]+[np.array([])])
        bg = AnalyticREAnchor(0., light[ci], light[ci], light[ci], .12, 0)
        self.assertLess(np.max(abs(native.rhs(y0, 0., bg))), 1e-8)
        with self.assertRaises(ContractError): native.boundary = b['boundary']
        anchor = AnalyticREAnchor(.02, light[ci], light[ci], light[ci]*[1., 1.01], .12, 1)
        def run(start, state, a, times):
            sol = solve_ivp(lambda t, y: native.rhs(y, t, a), (start, .12), state,
                            method='DOP853', rtol=1e-7, atol=1e-10, max_step=.0002, t_eval=times)
            self.assertTrue(sol.success)
            return sol.y
        full = run(.02, y0, anchor, [.06, .12])
        R, E = anchor.at(.06)
        resumed = run(.06, full[:, 0], AnalyticREAnchor(.06, R, E, anchor.light, .12, 1), [.12])
        np.testing.assert_allclose(resumed[:, -1], full[:, -1], rtol=0, atol=1e-7)


if __name__ == '__main__': unittest.main()
