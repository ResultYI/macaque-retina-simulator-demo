import unittest
import numpy as np
from scipy import sparse
from retina_sim.contracts import ContractError
from retina_sim.hc_support_closure import native_dependency_closure, require_closed_native


class Closure(unittest.TestCase):
    def run_graph(self, A=None, B=None, **kw):
        A=sparse.csr_matrix([[.5,.5,0],[0,.5,.5]]) if A is None else A
        B=sparse.csr_matrix([[1,0],[0,1],[0,1]]) if B is None else B
        args=dict(hc_support_complete=np.ones(2,bool),cone_feedback_complete=np.ones(3,bool),cone_input_available=np.ones(3,bool))
        args.update(kw)
        return native_dependency_closure(A,B,np.array([0]),np.array([],int),**args)

    def test_multiple_hops_follow_effective_input_and_local_feedback(self):
        r=self.run_graph();require_closed_native(r)
        np.testing.assert_array_equal(r['cone_rows'],[0,1,2])
        np.testing.assert_array_equal(r['hc_rows'],[0,1])
        self.assertGreater(len(r['history']),1)

    def test_missing_remote_feedback_not_hidden_by_steady_background(self):
        B=sparse.csr_matrix([[1,0],[0,1],[0,0]])
        r=self.run_graph(B=B)
        np.testing.assert_array_equal(r['missing']['cone_nonunit_fine_row'],[2])
        with self.assertRaises(ContractError):require_closed_native(r)

    def test_input_and_geometry_certificates_required_for_all_ancestors(self):
        r=self.run_graph(hc_support_complete=np.array([True,False]),cone_input_available=np.array([True,True,False]))
        self.assertFalse(r['ready']);self.assertEqual(r['missing']['cone_input'].tolist(),[2])
        with self.assertRaises(ContractError):require_closed_native(r)

    def test_disconnected_unsupported_cells_are_not_selected(self):
        r=self.run_graph(A=sparse.csr_matrix([[1,0,0],[0,.5,.5]]),cone_input_available=np.array([True,False,False]))
        require_closed_native(r)
        np.testing.assert_array_equal(r['A'].toarray(),[[1.]])

    def test_cropped_mass_never_repaired(self):
        r=self.run_graph(A=sparse.csr_matrix([[.5,0,0],[0,.5,.5]]))
        self.assertEqual(float(r['A'].sum()),.5)
        with self.assertRaises(ContractError):require_closed_native(r)

    def test_identity_tracks_input_and_determinism(self):
        self.assertEqual(self.run_graph()['identity'],self.run_graph()['identity'])
        self.assertNotEqual(self.run_graph()['identity'],self.run_graph(cone_input_available=np.array([True,False,True]))['identity'])

    def test_bad_ids_and_negative_weights_rejected(self):
        with self.assertRaises(ContractError):self.run_graph(A=sparse.csr_matrix([[-1,2,0],[0,0,1]]))
        with self.assertRaises(ContractError):self.run_graph(cone_feedback_complete=np.ones(3))


if __name__=='__main__':unittest.main()
