import numpy as np
import pytest
from scipy import sparse
from retina_sim.contracts import ContractError
from retina_sim.s_on_connectivity_candidate import one_secondary_S_input

def fixture():return sparse.csr_matrix([[1.,0.,0.],[0.,1.,0.]])

def test_one_edge_converges_and_diverges_without_changing_uniform_gain_or_other_rows():
    W=fixture();before=W.copy();V,d=one_secondary_S_input(W,['S','S','L'],[1,1,1],[1,1],target_row=0,secondary_column=1,secondary_fraction=.25)
    np.testing.assert_array_equal(W.toarray(),before.toarray())
    np.testing.assert_array_equal(V.toarray(),[[.75,.25,0.],[0.,1.,0.]])
    np.testing.assert_array_equal(V@np.ones(3),[1.,1.])
    assert V[:,1].nnz==2 and V[0].nnz==2 and not d['production_adopted']

def test_rejects_LM_input_and_unresolved_group_assignment():
    for kwargs in [dict(cone_types=['S','L','M'],cone_members=[1,1,1],BC_members=[1,1]),
                   dict(cone_types=['S','S','M'],cone_members=[1,2,1],BC_members=[1,1]),
                   dict(cone_types=['S','S','M'],cone_members=[1,1,1],BC_members=[2,1])]:
        with pytest.raises(ContractError):one_secondary_S_input(fixture(),target_row=0,secondary_column=1,secondary_fraction=.25,**kwargs)
