"""Small checks of the authorized reference and complete-source projection."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.hc_eccentric import spatial_parameters, temporal_decision, precompute_crop, project_members
from retina_sim.hc_joint_v2 import JointV2HCDensityProfile


ROOT = Path(__file__).resolve().parents[1]


class EccentricContract(unittest.TestCase):
    def test_reference_and_current_calibration(self):
        cfg = json.loads((ROOT/'configurations/research/cone_hc_eccentric.json').read_bytes())
        spatial = json.loads((ROOT/cfg['spatial_configuration']).read_bytes())
        p, record, _ = spatial_parameters(ROOT, [[.2, 0], [.8, 0], [.4, 0]], ['H1', 'H1', 'H2'], spatial)
        self.assertGreater(record['reference_contact_radius_mm'], 0)
        self.assertNotEqual(p['lambda_long_mm'][0], p['lambda_long_mm'][1])
        np.testing.assert_allclose(p['lambda_short_mm'][:2], p['contact_radius_mm'][:2]/np.log(10))
        np.testing.assert_allclose(p['lambda_long_mm'][:2], .3*p['contact_radius_mm'][:2]/record['reference_contact_radius_mm'])
        with self.assertRaises(ContractError):
            JointV2HCDensityProfile(ROOT).contact_radius_mm(4., 'H1')
        decision = temporal_decision(ROOT, cfg['calibration'])
        self.assertEqual(decision['decision'], 'ACCEPTED')
        self.assertAlmostEqual(decision['tau_h_ms'], 30.865749126454666)

    def test_crop_preserves_full_denominator_and_H2(self):
        xy = np.array([[0, 0], [.1, 0], [.2, 0], [.3, 0]])
        types = np.array(['L', 'M', 'S', 'L'])
        anchors = np.array([[0., 0.], [.1, 0.], [.2, 0.]])
        short = np.array([.02, .03, 0.]); long = np.array([.1, .15, 0.])
        parameters = dict(lambda_short_mm=short, lambda_long_mm=long,
                          support_radius_mm=11.75637*long, eccentricity_mm=np.array([0, .1, .2]))
        uniform = sparse.csr_matrix([[.5, .5, 0], [.5, .5, 0], [.2, .3, .4]])
        args = (xy, types, anchors, np.array(['H1', 'H1', 'H2']), np.array([0, 1, 2]), uniform, np.array([0., 0., .1]))
        result = precompute_crop(*args, parameters=parameters, envelope_center=[0, 0], envelope_radius_mm=2.)
        distances = np.array([0., .1, .3])
        a = np.exp(-distances/.02); b = np.exp(-distances/.1)
        expected = .15*a/a.sum()+.85*b/b.sum()
        np.testing.assert_allclose(result['A'][0].toarray()[0], [expected[0], expected[1], 0])
        self.assertAlmostEqual(result['held_mass'][0], expected[2])
        np.testing.assert_array_equal(result['A'][2].toarray(), uniform[2].toarray())
        grouped, held = project_members(result['A'], result['held_mass'], np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([2, 1]), np.array([2, 1]))
        self.assertAlmostEqual(grouped[0, 0], result['A'][:2, :2].sum()/2)
        np.testing.assert_allclose(np.asarray(grouped.sum(1)).ravel()+held, 1.)
        with self.assertRaises(ContractError):
            precompute_crop(*args, parameters=parameters, envelope_center=[0, 0], envelope_radius_mm=.5)


if __name__ == '__main__':
    unittest.main()
