import unittest
import numpy as np
from scipy import sparse
from retina_sim.hc_timevarying_exterior import FixedFeedbackCone, DrivenExteriorH1
from retina_sim.hc_minimal_diagnostic import bind_diagnostic
from retina_sim.hc_research_native import joint_equilibrium
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from tools.run_hc_minimal_real_diagnostic_v1 import native_model


class ExteriorResponse(unittest.TestCase):
    def test_cache_has_native_state_response_and_valid_background(self):
        c=FixedFeedbackCone(28.,member_identity='test')
        self.assertEqual(c.at(0.),c.at(.02))
        self.assertGreater(abs(c.at(.12)[0]-c.at(0.)[0]),1e-6)
        self.assertGreater(abs(c.at(.12)[1]-c.at(0.)[1]),1e-6)
        self.assertTrue(np.isfinite(c.solution.y).all())

    def test_zero_exterior_mass_preserves_rhs_and_nonzero_mass_drives_pool(self):
        cache=FixedFeedbackCone(28.,member_identity='test')
        binding=bind_diagnostic(sparse.csr_matrix([[.4]]),sparse.csr_matrix([[1.]]),
            cone_parent_rows=np.array([1]),registered_unsupported_rows=np.array([],int),
            A_held_mass=np.array([.6]),B_held_mass=np.array([0.]),
            exterior_release=cache.background_release,exterior_Vbar=cache.y0[4],exterior_h=28.,source_identity='test')
        base,pars=native_model(binding,np.array([2]),np.array([1]))
        n=DrivenExteriorH1(base.assembly);n.boundary=binding['boundary'];n.attach_exterior(cache,np.zeros(1))
        light=np.array([1000.]);eq=joint_equilibrium(base.A,base.B,np.array([0]),np.array([],int),light,pars,boundary=base.boundary)
        y=np.concatenate([eq['cone'][k] for k in ('X','C','Vis','g_i','Vbar')]+[eq['w'],eq['z2'],eq['h'],np.array([])])
        anchor=AnalyticREAnchor(0.,light,light,light,.12,0)
        np.testing.assert_array_equal(base.rhs(y,.06,anchor),n.rhs(y,.06,anchor))
        driven=DrivenExteriorH1(base.assembly);driven.boundary=binding['boundary'];driven.attach_exterior(cache,np.array([.6]))
        self.assertAlmostEqual(float(driven.boundary_at(.12)['A_release'][0]),.6*cache.at(.12)[0],places=15)
        self.assertGreater(np.max(abs(driven.rhs(y,.06,anchor)-base.rhs(y,.06,anchor))),1e-6)


if __name__=='__main__':unittest.main()
