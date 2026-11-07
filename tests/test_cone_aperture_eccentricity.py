import unittest
from unittest.mock import patch
import numpy as np
from retina_sim.contracts import ContractError
from retina_sim.cone_aperture_eccentricity import resolve_apertures
from retina_sim.native_cone_apertures import PROFILE

class NativeApertureBindingTests(unittest.TestCase):
    def setUp(self):
        self.xy=np.c_[np.arange(8)*.003,np.zeros(8)]
    def test_selected_cones_inherit_complete_support_without_rounding(self):
        diameters=np.linspace(1.12345,4.67891,8);areas=diameters*1e-12
        with patch('retina_sim.cone_aperture_eccentricity.native_apertures',return_value=(diameters,areas)) as native:
            result=resolve_apertures(['a','b'],['S','L'],self.xy[[6,1]],PROFILE,support_positions_mm=self.xy)
        np.testing.assert_array_equal(native.call_args.args[0],self.xy)
        np.testing.assert_array_equal(result['reference_diameter_um'],diameters[[6,1]])
        np.testing.assert_array_equal(result['zone_diameter_um'],diameters[[6,1]])
        np.testing.assert_array_equal(result['effective_area_m2'],areas[[6,1]])
        np.testing.assert_array_equal(result['positions_mm'],self.xy[[6,1]])
    def test_custom_profiles_and_missing_support_cones_are_rejected(self):
        for profile,xy in [({'schema':'macaque-cone-aperture-eccentricity-1'},self.xy[:1]),(PROFILE,np.array([[.001,0.]]))]:
            with self.assertRaises(ContractError):
                resolve_apertures(['a'],['M'],xy,profile,support_positions_mm=self.xy)
    def test_duplicate_identity_is_rejected_before_native_generation(self):
        with self.assertRaises(ContractError):
            resolve_apertures(['a','a'],['L','M'],self.xy[:2],PROFILE,support_positions_mm=self.xy)

if __name__=='__main__':unittest.main()
