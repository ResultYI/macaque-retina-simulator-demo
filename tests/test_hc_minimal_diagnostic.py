import unittest
import numpy as np
from scipy import sparse
from retina_sim.contracts import ContractError
from retina_sim.hc_minimal_diagnostic import bind_diagnostic, require_diagnostic_native


class DiagnosticPermission(unittest.TestCase):
    def test_only_registered_zero_feedback_is_admitted(self):
        args = dict(cone_parent_rows=np.array([11, 22]), registered_unsupported_rows=np.array([22]),
            A_held_mass=np.array([.5]), B_held_mass=np.zeros(2), exterior_release=.2,
            exterior_Vbar=20., exterior_h=30., source_identity='fixture')
        A, B = sparse.csr_matrix([[.2, .3]]), sparse.csr_matrix([[1.], [0.]])
        bound = bind_diagnostic(A, B, **args)
        self.assertEqual(bound['B'].nnz, 1)
        np.testing.assert_array_equal(bound['unsupported_feedback'], [False, True])
        for registry in (np.array([], int), np.array([11]), np.array([22, 22])):
            with self.assertRaises(ContractError):
                bind_diagnostic(A, B, **dict(args, registered_unsupported_rows=registry))
        off = bind_diagnostic(A, B, **args, feedback_off=True)
        self.assertEqual(off['B'].nnz, 0)
        np.testing.assert_array_equal(off['mass_ledger']['B_ablation_removed_mass'], [1, 0])
        good = dict(input_available=np.ones(2, bool), feedback_geometry_complete=np.ones(2, bool),
                    background_max_rhs=0.)
        self.assertTrue(require_diagnostic_native(bound, **good)['admitted'])
        with self.assertRaises(ContractError):
            require_diagnostic_native(bound, **dict(good, feedback_geometry_complete=np.array([True, False])))

    def test_conditioned_equilibrium_satisfies_release_and_h_equations(self):
        from retina_sim.hc_research_native import joint_equilibrium
        from retina_sim.vh2005 import SOURCE_VALUES as p
        A, B = sparse.csr_matrix([[.2, .3]]), sparse.csr_matrix([[1.], [0.]])
        boundary = dict(A_release=np.array([.1]), A_Vbar=np.array([10.]), B_h=np.zeros(2))
        eq = joint_equilibrium(A, B, np.array([0]), np.array([], int), np.full(2, 1000.),
            dict(input_scale=p['g_t'], feedback_scale=1.), boundary=boundary)
        np.testing.assert_allclose(eq['h'], p['g_t']*(A@eq['release']+boundary['A_release']), atol=1e-10)
        np.testing.assert_allclose(eq['w'], eq['h'], atol=1e-10)


if __name__ == '__main__': unittest.main()
