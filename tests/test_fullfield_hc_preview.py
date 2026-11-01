import numpy as np
from scipy import sparse

from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.h1_h2_effective import joint_equilibrium
from retina_sim.hc_lateral_candidate import LateralHCCandidate
from tools.run_fullfield_hc_preview import SparseFullFieldNative, make_assembly, member_mean


def test_sparse_complete_network_adapter_matches_existing_lateral_rhs():
    A=sparse.csr_matrix([[.5,.5,0],[.2,.8,0],[.3,.2,.5],[.2,.3,.5]],dtype=float)
    B=sparse.csr_matrix([[.4,.6,0,0],[.5,.5,0,0],[0,0,.5,.5]],dtype=float)
    assembly=make_assembly(A,B,np.array([2,3,1]),np.array([3,2,1,2]),
        np.array(['L','M','S']),np.array(['H1','H1','H2','H2']),dict(test='equation_equivalence'))
    original=LateralHCCandidate(assembly,coupling_rate_per_second=50.,S_efficacy_ratio=1.)
    full=SparseFullFieldNative(assembly)
    light=np.array([90.,100.,110.])
    eq=joint_equilibrium(A,B,assembly.h1_rows,np.array([],dtype=int),light,assembly.h2_parameters)
    y=np.concatenate([eq['cone'][name] for name in ('X','C','Vis','g_i','Vbar')]
                     +[eq['w'],eq['z2'],eq['h'],np.array([],dtype=float)])
    anchor=AnalyticREAnchor(0.,light,light,light*1.1,.04,1)
    for model in (original,full):
        model.boundary=dict(A_release=np.zeros(4),A_Vbar=np.zeros(4),B_h=np.zeros(3))
    np.testing.assert_array_equal(original.Qfull.toarray(),full.Qfull.toarray())
    np.testing.assert_allclose(original.rhs(y,.01,anchor),full.rhs(y,.01,anchor),rtol=1e-14,atol=1e-11)


def test_partial_optical_support_retains_the_whole_member_denominator():
    # One of four members is driven; the three exterior members stay at 100.
    actual=member_mean(np.array([200.,50.]),np.array([0,1]),np.array([4,1,2]))
    np.testing.assert_array_equal(actual,np.array([125.,50.,100.]))
