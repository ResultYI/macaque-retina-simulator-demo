import unittest
import numpy as np
from scipy import sparse
from retina_sim.contracts import ContractError
from retina_sim.hc_biexponential import (
    build_spatial_operators, project_spatial_operators, support_radius, exponential_tail)


def fixture():
    xy = np.array([[0.,0.], [.01,0.], [.05,0.], [.1,0.], [.015,0.]])
    hx = np.array([[0.,0.], [.01,0.]])
    ct = np.array(['L','M','L','M','S']); ht = np.array(['H1','H2'])
    raw = sparse.csr_matrix([[1,2,0,0,0], [1,0,0,0,1]])
    cfg = dict(schema='HC_biexponential_candidate_v1',coordinate_frame='OPL_pedicle_mm',
               H1=dict(lambda_s_um=10.,lambda_l_um=50.,w_s=.3,source='test',evidence_class='synthetic'),
               H2=None,tail_mass_budget=1e-4)
    return raw, xy, hx, ct, ht, cfg


class Biexponential(unittest.TestCase):
    def build(self, *, config=None, radius=1., coupling=False, raw=None):
        original, xy, hx, ct, ht, cfg = fixture()
        return build_spatial_operators(original if raw is None else raw,xy,hx,ct,ht,
            configuration=cfg if config is None else config,parent_disk_radius_mm=radius,
            explicit_hc_coupling=coupling)

    def test_exact_independent_component_formula_and_integral_weight(self):
        result=self.build(); short=np.exp(-np.array([0.,.01])/.01)*[1,2]
        long=np.exp(-np.array([0.,.01,.05,.1])/.05)
        expected=.7*long/long.sum();expected[:2]+=.3*short/short.sum()
        np.testing.assert_allclose(result['A'].toarray()[0,:4],expected,rtol=2e-15)
        self.assertEqual(result['A'][0,4],0)
        self.assertGreater(result['A'][0,3],0)
        self.assertEqual(result['raw_contacts'][0,3],0)
        np.testing.assert_allclose(result['A'].sum(1),1,rtol=0,atol=2e-15)

    def test_local_feedback_is_independent_and_type_restricted(self):
        r=self.build(); _,_,_,_,_,cfg=fixture();cfg['H1']['lambda_l_um']=80
        altered=self.build(config=cfg)
        np.testing.assert_array_equal(r['B'].toarray(),altered['B'].toarray())
        np.testing.assert_array_equal(r['B'].toarray(),[[1,0],[1,0],[0,0],[0,0],[0,1]])
        self.assertNotEqual(r['identity'],altered['identity'])

    def test_uniform_and_h2_legacy_are_exact(self):
        r=self.build();cfg=fixture()[-1];cfg['H1']=None
        old=self.build(config=cfg)
        np.testing.assert_array_equal(old['A'].toarray(),r['uniform_A'].toarray())
        np.testing.assert_array_equal(old['A'][1].toarray(),r['A'][1].toarray())
        self.assertEqual(r['identity'],self.build()['identity'])

    def test_h2_has_independent_explicit_parameters(self):
        cfg=fixture()[-1];cfg['H2']=dict(cfg['H1'],lambda_s_um=7.,lambda_l_um=30.,w_s=.7)
        r=self.build(config=cfg)
        self.assertGreater(r['A'][1,4],0);self.assertGreater(r['A'][1,3],0)
        self.assertEqual(r['B'][0,1],0)

    def test_tail_uses_area_not_point_amplitude(self):
        radius=support_radius(.3,1e-4)
        self.assertAlmostEqual(float(exponential_tail(radius,.3)),1e-4,places=15)
        self.assertGreater(radius,-.3*np.log(1e-4))
        self.assertGreater(radius,3.)

    def test_boundary_loss_is_reported_not_hidden(self):
        r=self.build(radius=.11)
        self.assertFalse(r['support_certified'][0])
        np.testing.assert_allclose(r['A'].sum(1),1)

    def test_empty_component_and_double_coupling_rejected(self):
        raw=fixture()[0].tolil();raw[0]=0
        with self.assertRaisesRegex(ContractError,'empty active'):self.build(raw=raw.tocsr())
        with self.assertRaisesRegex(ContractError,'mutually exclusive'):self.build(coupling=True)

    def test_invalid_parameters_geometry_weights_rejected(self):
        for key,value in [('lambda_s_um',0),('lambda_l_um',5),('w_s',1.1),('w_s',float('nan'))]:
            cfg=fixture()[-1];cfg['H1'][key]=value
            with self.assertRaises(ContractError):self.build(config=cfg)
        raw=fixture()[0].copy();raw.data[0]=-1
        with self.assertRaises(ContractError):self.build(raw=raw)
        with self.assertRaises(ContractError):self.build(radius=.01)

    def test_grouping_is_fine_member_mean_and_crop_loses_mass(self):
        r=self.build()
        p=project_spatial_operators(r,np.array([0,0,1,1,2]),np.array([0,1]),np.array([2,2,1]),np.array([1,1]))
        np.testing.assert_allclose(p['A'].toarray()[0],
            [r['A'][0,:2].sum(),r['A'][0,2:4].sum(),0])
        np.testing.assert_array_equal(p['B'].toarray(),[[1,0],[0,0],[0,1]])
        self.assertLess(float(p['A'][0,:1].sum()),1)
        with self.assertRaisesRegex(ContractError,'unrepresented'):
            project_spatial_operators(r,np.array([0,0,-1,-1,1]),np.array([0,1]),np.array([2,1]),np.array([1,1]))

    def test_complete_feedback_native_control_has_expected_surround_sign(self):
        from tools.validate_hc_biexponential_v1 import native, integrate
        cfg=fixture()[-1]
        r=build_spatial_operators(sparse.csr_matrix([[1,1,0],[1,0,1]]),
            np.array([[0.,0.],[.03,0.],[.02,0.]]),np.zeros((2,2)),
            np.array(['L','M','S']),np.array(['H1','H2']),configuration=cfg,parent_disk_radius_mm=1.)
        light=np.full(3,1000.);stimulus=light*np.array([1.,1.01,1.])
        deltas=[]
        for B in (r['B'],sparse.csr_matrix(r['B'].shape)):
            n,y,error=native(r['A'],B,'manufactured_complete_feedback_control',light)
            self.assertLess(error,1e-8)
            _,release,_=integrate(n,y,light,stimulus)
            deltas.append(float(release[0,-1]-release[0,0]))
        self.assertGreater(deltas[0],0)
        self.assertLess(abs(deltas[1]),1e-10)


if __name__ == '__main__':
    unittest.main()
