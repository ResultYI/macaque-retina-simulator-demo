import unittest
import numpy as np
from scipy import sparse
from retina_sim.hc_group_projection import project_hc_contacts
from retina_sim.contracts import ContractError


class GroupProjection(unittest.TestCase):
    def build(self,raw=None,HC_map=None,HC_types=None):
        return project_hc_contacts(sparse.csr_matrix([[1,0,0],[0,1,1]]) if raw is None else raw,
            np.array(['L','L','S']),np.array(['H1','H2']) if HC_types is None else HC_types,
            np.array([0,0,1]),np.array([0,1]) if HC_map is None else HC_map,np.array([2,1]),np.array([1,1]))

    def test_partial_members_are_not_promoted_to_full_feedback(self):
        result=self.build()
        np.testing.assert_allclose(result['typed_B'].toarray(),[[.5,0],[0,1]],rtol=0,atol=0)
        np.testing.assert_allclose(result['cone_supported_member_fraction'],[.5,1],rtol=0,atol=0)
        # A constant original HC signal must not be amplified by group renormalization.
        np.testing.assert_allclose(result['typed_B']@np.array([10.,20.]),[5.,20.],rtol=0,atol=0)

    def test_original_H2_LM_inputs_survive_but_feedback_type_rule_stays(self):
        result=self.build()
        np.testing.assert_allclose(result['A'].toarray(),[[1,0],[.5,.5]],rtol=0,atol=0)
        self.assertEqual(result['typed_B'][0,1],0)
        self.assertEqual(result['typed_B'][1,0],0)

    def test_zero_contact_geometry_is_not_turned_into_a_fake_dynamic_cell(self):
        raw=sparse.csr_matrix([[1,0,0],[0,1,1],[0,0,0]])
        result=self.build(raw,np.array([0,1,-1]),np.array(['H1','H2','H1']))
        self.assertEqual(result['A'].shape,(2,2));self.assertEqual(result['omitted_zero_contact_geometry_HC'],1)
        with self.assertRaises(ContractError):
            self.build(sparse.csr_matrix([[1,0,0],[0,1,1],[1,0,0]]),np.array([0,1,-1]),np.array(['H1','H2','H1']))

    def test_crop_keeps_exterior_mass_without_rescaling(self):
        result=self.build();B=result['typed_B']
        local=B[:1,:1];held=B[:1,1:]
        self.assertEqual(float(local.sum()+held.sum()),.5)
        with self.assertRaises(ContractError):self.build(sparse.csr_matrix([[1,0,1],[0,1,1]]))


if __name__=='__main__':unittest.main()
