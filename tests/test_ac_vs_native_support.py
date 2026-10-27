"""Directed selection checks without any model construction or advance."""
import numpy as np
import pytest
from retina_sim.contracts import ContractError
from tools.validate_ac_vs_native_support import merge_selection,compare_arrays


def test_native_override_preserves_inner_contacts_and_core_and_owns_arrays():
    a={k:np.array(v,dtype=np.int64) for k,v in {'population__cones':[1,3],'population__HC':[2],
        'population__BC':[5,9],'contact__glycine':[2,8],'core__cones':[3],'core__HC':[2],'core__BC':[5]}.items()}
    b={k:v.copy() for k,v in a.items()};b['population__cones']=np.array([1,2,3,7]);b['population__HC']=np.array([2,6]);b['population__BC']=np.array([5,6,9]);b['contact__glycine']=np.array([1,2,8])
    result=merge_selection(a,b)
    for k in a:
        np.testing.assert_array_equal(result[k],b[k] if k in ['population__cones','population__HC'] else a[k])
        assert not np.shares_memory(result[k],a[k]) and not result[k].flags.writeable
    np.testing.assert_array_equal(a['population__cones'],[1,3])


def test_selection_rejects_duplicate_global_rows_or_lost_core():
    a={'population__cones':np.array([1,2]),'population__HC':np.array([1]),'core__cones':np.array([2])}
    b={k:v.copy() for k,v in a.items()};b['population__cones']=np.array([1,1,2])
    with pytest.raises(ContractError,match='Unique sorted'):merge_selection(a,b)
    b['population__cones']=np.array([1,3])
    with pytest.raises(ContractError,match='Core escaped'):merge_selection(a,b)


def test_original_near_zero_history_budget_still_fails_same_point():
    budget=dict(voltage_V=1e-5,conductance_S=1e-11,expected_rate_Hz=.1,general_atol=1e-8,general_rtol=.001)
    x=np.array([[7.666217606894232e-6]]);y=np.array([[7.710230418844777e-6]])
    norm=compare_arrays(x,y,'dimensionless',budget)
    assert norm[0,0]==pytest.approx(2.485163146364949)
    assert compare_arrays(np.array([0.]),np.array([2e-5]),'V',budget)[0]==2
