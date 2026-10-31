import unittest
import numpy as np
from scipy.spatial import cKDTree
from retina_sim.geometry_anchored_pooling import anchored_radius, complete_d8, select_row


class GeometryPooling(unittest.TestCase):
    def test_log_interpolation_preserves_decreasing_anchors_and_edges(self):
        np.testing.assert_allclose(anchored_radius([0, 1, 2, 3, 4], 1, 3, .04, .01), [.04, .04, .02, .01, .01], rtol=1e-14)

    def test_d8_uses_complete_source_and_not_old_eight_edges(self):
        tree = cKDTree(np.c_[np.arange(12)*.001, np.zeros(12)])
        d, valid = complete_d8(tree, [[0, 0]], 1)
        self.assertTrue(valid[0]); self.assertAlmostEqual(d[0], .007)
        indices, weights, status = select_row(tree, [0, 0], .011, .6, 1)
        self.assertEqual(len(indices), 12); self.assertEqual(status, 'ENGINEERING_CANDIDATE_VALID')
        self.assertAlmostEqual(weights.sum(), .6)

    def test_boundary_or_empty_support_does_not_invent_partner(self):
        tree = cKDTree(np.array([[0., 0.]]))
        self.assertIsNone(select_row(tree, [.9, 0], .2, 1, 1)[0])
        self.assertIsNone(select_row(tree, [.5, 0], .1, 1, 1)[0])
        self.assertFalse(complete_d8(tree, [[0, 0]], 1)[1][0])


if __name__ == '__main__': unittest.main()
