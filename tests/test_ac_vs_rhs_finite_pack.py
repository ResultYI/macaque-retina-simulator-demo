"""Finite-scan elimination without changing equations, layouts or ownership."""
from types import MethodType
import numpy as np
import pytest
from retina_sim.ac_vs_rhs_readouts import RequiredOutputsACVSRHS
from retina_sim.ac_vs_rhs_finite_pack import FinitePackedACVSRHS
from retina_sim.contracts import ContractError
from tests.test_ac_vs_model import new_fixture


def test_exact_owned_rhs_and_one_redundant_finite_scan_removed(monkeypatch):
    _,_,model,state=new_fixture();old=RequiredOutputsACVSRHS(model);new=FinitePackedACVSRHS(model);y=model.pack(state)
    old.rhs_flat(0.,y);new.rhs_flat(0.,y)
    original=np.isfinite;counts=[0]
    def counted(*args,**kwargs):counts[0]+=1;return original(*args,**kwargs)
    monkeypatch.setattr(np,'isfinite',counted)
    expected=old.rhs_flat(0.,y);old_count=counts[0];counts[0]=0;actual=new.rhs_flat(0.,y)
    assert old_count-counts[0]==len(model.state_keys)
    assert actual.tobytes()==expected.tobytes() and not np.shares_memory(actual,y)
    saved=actual.copy();y2=y.copy();y2[model.slices['parasol_RGC_ON','RGC_spike_history','V']]+=.001
    assert new.rhs_flat(.001,y2).tobytes()==old.rhs_flat(.001,y2).tobytes()
    assert actual.tobytes()==saved.tobytes()
    y2[0]=np.inf
    for engine in [old,new]:
        with pytest.raises(ContractError):engine.rhs_flat(0.,y2)


@pytest.mark.parametrize('bad',['nonfinite','layout'])
def test_original_final_finite_and_pack_layout_guards_remain(bad):
    _,_,model,state=new_fixture();old=RequiredOutputsACVSRHS(model);new=FinitePackedACVSRHS(model);original=model.ac_rhs
    key=('cross_OFF_ON_parasol_AC_v1','OrdinaryAmacrineVoltageGateSI','s')
    # Use actual registered key, not a new state.
    key=next(k for k in model.state_keys if k[0]=='cross_OFF_ON_parasol_AC_v1' and k[-1]=='s')
    def broken(self,current,outputs,result):
        original(current,outputs,result);result[key]=np.full(self.sizes[key],np.nan) if bad=='nonfinite' else np.ones(self.sizes[key]+1)
    model.ac_rhs=MethodType(broken,model)
    for engine in [old,new]:
        with pytest.raises(ContractError):engine.rhs_flat(0.,model.pack(state))
