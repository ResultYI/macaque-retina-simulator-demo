"""Use the actual owned dense type returned by ACVSInner's CPU reference."""
import unittest
import numpy as np
from scipy.integrate import DOP853
from retina_sim.continuous_state import Dop853Dense
from tools.run_external_first_frame_v1 import segment_midpoint


class ExternalFirstFrameAdapterTests(unittest.TestCase):
    def test_owned_dense_midpoint_and_derivative(self):
        solver=DOP853(lambda t,y:-y,0.,np.array([1.]),.0001,rtol=1e-6,atol=1e-9)
        solver.step()
        curve=Dop853Dense.from_scipy(solver.dense_output(),np.array([0]))
        t=segment_midpoint(curve)
        self.assertGreater(t,curve.t0)
        self.assertLess(t,curve.t1)
        np.testing.assert_allclose(curve.representation_derivative(t,'right'),-curve.evaluate(t,'right'),atol=1e-9,rtol=1e-6)


if __name__=='__main__':unittest.main()
