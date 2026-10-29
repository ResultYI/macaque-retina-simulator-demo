"""Mathematical contracts of the isolated new coupling/selectivity provider."""
import numpy as np
import pytest
from scipy import sparse
from retina_sim.contracts import ContractError
from retina_sim.hc_lateral_candidate import homotypic_operator,preference_operator

def test_homotypic_diffusion_conserves_capacity_and_dissipates():
    A=sparse.csr_matrix([[1.,0.,0.],[.5,.5,0.],[0.,.5,.5],[0.,0.,1.]])
    capacity=np.array([1.,3.,2.,5.])
    Q,overlap,_=homotypic_operator(A,capacity)
    assert np.allclose(Q@np.ones(4),0.,atol=1e-14)
    assert np.allclose(capacity@Q,0.,atol=1e-14)
    h=np.array([4.,1.,3.,2.]);dh=-Q@h
    assert np.dot(capacity*h,dh)<0
    assert dh[np.argmax(h)]<=0 and dh[np.argmin(h)]>=0
    assert np.allclose(overlap.toarray(),overlap.toarray().T)
    assert np.max(Q.diagonal())<=1.+1e-14

def test_preference_preserves_complete_DC_LM_ratio_and_partial_rows():
    A=sparse.csr_matrix([[.2,.3,.5],[.1,.2,.2],[0.,.7,.3]])
    B,selected=preference_operator(A,np.array(['L','M','S']),2.)
    assert selected.tolist()==[True,False,True]
    assert np.allclose(B.sum(1),A.sum(1),rtol=0,atol=1e-15)
    assert np.array_equal(B[1].toarray(),A[1].toarray())
    assert B[0,0]/B[0,1]==pytest.approx(A[0,0]/A[0,1])
    assert (B[0,2]/B[0,0])/(A[0,2]/A[0,0])==pytest.approx(2.)
    assert not B.data.flags.writeable

def test_preference_does_not_invent_missing_S_contacts_or_accept_wrong_direction():
    A=sparse.csr_matrix([[.4,.6,0.]])
    B,selected=preference_operator(A,np.array(['L','M','S']),2.)
    assert not selected.any() and np.array_equal(B.toarray(),A.toarray())
    with pytest.raises(ContractError):preference_operator(A,np.array(['L','M','S']),.9)
