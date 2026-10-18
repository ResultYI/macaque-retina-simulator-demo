"""Effective functional association invariants; no anatomical validation claim."""
import numpy as np
import pytest
from retina_sim.contracts import ContractError
from retina_sim.disk_hc_functional import functional_projection

ANCHORS={'H1':[3.7,7.5],'H2':[2.2,3.2],'project_representative_radii_mm':[.75,5.5]}

def test_existing_typed_partners_reciprocity_and_finite_support():
    cones=np.c_[np.full(7,.3),np.arange(7)*.001];before=cones.copy();labels=np.array(['L','M','S','L','M','S','L'])
    hc=np.array([[.25,-.02],[.27,.03],[.33,-.03],[.35,.02],[.28,-.04],[.32,.04],[.37,.0]])
    types=np.array(['H1']*4+['H2']*3)
    matrix,audit=functional_projection(cones,labels,hc,types,np.ones(7,bool),1.,ANCHORS)
    np.testing.assert_array_equal(cones,before)
    assert matrix.shape==(7,7) and np.all(np.diff(matrix.indptr)>0)
    assert matrix[labels=='S',:4].nnz==0
    assert audit['owner_complete'] and audit['anatomical_contact_claim'] is False
    assert audit['subtypes']['H1']['actual_associations']==19  # Existing floor(total+.5), not bankers rounding.
    assert audit['subtypes']['H2']['actual_associations']==15
    A=matrix.T.tocsr();A=A.multiply(1/np.asarray(A.sum(1)).ravel()[:,None]).tocsr()
    B=matrix.multiply(1/np.asarray(matrix.sum(1)).ravel()[:,None]).tocsr()
    np.testing.assert_allclose(np.asarray(A.sum(1)).ravel(),1)
    np.testing.assert_allclose(np.asarray(B.sum(1)).ravel(),1)
    assert ((A.T!=0)!=(B!=0)).nnz==0

def test_owner_support_must_be_inside_supplied_canonical_tissue():
    hc=np.array([[.25,-.02],[.27,.03],[.33,-.03],[.35,.02],[.28,-.04],[.32,.04],[.37,.0]])
    with pytest.raises(ContractError,match='exceeds certified'):
        functional_projection(np.array([[.499,0.]]),np.array(['L']),hc,np.array(['H1']*4+['H2']*3),np.array([True]),.5,ANCHORS)
