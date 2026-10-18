"""A legal absent central S-ON owner cohort must remain absent."""
import numpy as np
import pytest
from retina_sim.atlas_runtime import _owner_ids
from retina_sim.contracts import ContractError

@pytest.mark.parametrize('coordinates',[[],np.empty((0,2))])
def test_empty_rf_owner_cohort_has_no_fabricated_target(coordinates):
    table={'global_cell_ids':[],'coordinates':{'rf':{'values':coordinates}}}
    assert _owner_ids(table,(-.01,.01,-.01,.01))==()

def test_nonempty_owner_selection_and_shape_guard():
    table={'global_cell_ids':['inside','outside','upper_edge'],
        'coordinates':{'rf':{'values':[[0.,0.],[.02,0.],[.01,0.]]}}}
    assert _owner_ids(table,(-.01,.01,-.01,.01))==('inside',)
    malformed={'global_cell_ids':['missing'],'coordinates':{'rf':{'values':[]}}}
    with pytest.raises(ContractError,match='N-by-2'):_owner_ids(malformed,(-.01,.01,-.01,.01))
