"""Source interface regression only; no biological response fixtures."""
import unittest
import numpy as np
from retina_sim.gin_source_binding_v1 import source_display_lms,admission_blockers,require_neural_binding
from retina_sim.natural_movie_supervision import frame_index_for_ticks


class BindingTests(unittest.TestCase):
    def test_source_gamma_and_channel_order_keep_256_denominator(self):
        black=source_display_lms(np.zeros((1,3),dtype=np.uint8))[0]
        expected=[2.74*.01451+3.4*.005123+1.34*.02612,
                  1.21*(1.06*.01451+3.58*.005123+2.07*.02612),.212*.01451+8.28*.005123+285*.02612]
        np.testing.assert_allclose(black,expected,rtol=0,atol=2e-15)
        red=source_display_lms(np.array([[255,0,0]],dtype=np.uint8))[0]
        np.testing.assert_allclose((red-black)[1]/(red-black)[0],1.21*1.06/2.74,rtol=1e-14)
        with self.assertRaises(ValueError):source_display_lms(np.ones((3,3),dtype=float))

    def test_declared_movie_clock_is_integer_150hz_not_container_fps(self):
        ticks=np.array([0,66,67,133,134,9999,10000])
        np.testing.assert_array_equal(frame_index_for_ticks(ticks),[751,751,752,752,753,900,901])

    def test_true_eccentricity_and_relative_lms_alone_cannot_admit_network(self):
        binding=dict(schema='gin-real-cell-source-binding-v1',source_cell_key='TEST_ONLY',source_type='M off',
                     mapped_model_type='parasol_RGC_OFF',movie_sha256='fixture',source_eccentricity_deg=3.88,
                     source_and_movie_pins_verified=True)
        self.assertIn('display_to_native_contract_closed',admission_blockers(binding))
        self.assertIn('azimuth_contract_accepted',admission_blockers(binding))
        with self.assertRaisesRegex(ValueError,'NEURAL_FORWARD_NOT_ADMITTED'):require_neural_binding(binding)


if __name__=='__main__':unittest.main()
