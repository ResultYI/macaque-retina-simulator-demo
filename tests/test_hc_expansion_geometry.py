import unittest
from pathlib import Path
import numpy as np
from retina_sim.hc_expansion_geometry import bounded_pedicles,native_td_step
from retina_sim.henle_bridge import ScheinBoundedBridge
from retina_sim.contracts import Quantity,ContractError


class Expansion(unittest.TestCase):
    def test_batch_matches_existing_rule_across_meridians_and_periphery(self):
        b=ScheinBoundedBridge(Path(__file__).resolve().parents[1])
        theta=np.r_[np.arange(4)*np.pi/2,np.linspace(-3.1,3.1,35)]
        radii=np.array([.01,.1,.5,1,2,3,4.601,5.5,6.])
        xy=np.concatenate([np.c_[np.cos(theta),np.sin(theta)]*r for r in radii])
        xy=xy[np.linalg.norm(xy,axis=1)<=6]
        expected=b.apply(Quantity(xy,'mm','cone_inner_segment_position','test','anatomical_retina')).values
        np.testing.assert_allclose(bounded_pedicles(b,xy),expected,rtol=0,atol=2e-15)

    def test_invalid_domain_is_not_extrapolated(self):
        b=ScheinBoundedBridge(Path(__file__).resolve().parents[1])
        for xy in ([[0,0]],[[6.1,0]],[[float('nan'),0]]):
            with self.assertRaises(ContractError):bounded_pedicles(b,xy)

    def test_all_halo_cones_have_input_and_on_off_times_are_exact(self):
        xy=np.array([[0.,0.],[.2,0.],[4.,0.]])
        cfg=dict(center_mm=[0,0],inner_radius_mm=.1,outer_radius_mm=.3,contrast=.01,onset_seconds=.1,offset_seconds=.2)
        np.testing.assert_array_equal(native_td_step(xy,1000,time_seconds=.1,**cfg),[1000,1010,1000])
        np.testing.assert_array_equal(native_td_step(xy,1000,time_seconds=.2,**cfg),[1000]*3)


if __name__=='__main__':unittest.main()
