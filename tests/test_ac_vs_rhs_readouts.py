"""Only the new RHS view and unchanged observation contracts."""
from copy import deepcopy
import numpy as np
import pytest
from retina_sim.ac_vs_rhs_readouts import RequiredOutputsACVSRHS,contract,install
from retina_sim.ac_vs_model import DirectACVSRHS,MECHANISM
from retina_sim.contracts import ContractError,canonical_hash
from tests.test_ac_vs_model import new_fixture


def test_rhs_exact_without_derived_observation_evaluation_or_cross_call_cache(monkeypatch):
    _,_,model,state=new_fixture();original=DirectACVSRHS(model);optimized=RequiredOutputsACVSRHS(model)
    assert original.identity!=optimized.identity
    outputs=model.outputs;calls=[]
    def counted(*a,**kw):calls.append(1);return outputs(*a,**kw)
    monkeypatch.setattr(model,'outputs',counted)
    for j in range(3):
        current=deepcopy(state);current['cross_OFF_ON_parasol_AC_v1',MECHANISM,'s'][:]=.2+.1*j
        model.p['diffuse_BC_ON']['tau_release']=.003+.001*j
        release=np.full(model.counts['cones'],.2+.01*j);before=canonical_hash(current);y=model.pack(current)
        calls.clear();expected=original.rhs_flat(.001*j,y,release);assert len(calls)==1
        calls.clear();actual=optimized.rhs_flat(.001*j,y,release);assert not calls
        assert actual.tobytes()==expected.tobytes() and canonical_hash(current)==before
        model.observation_witness(.001*j,current,release,lambda:np.full(model.counts['rods'],100.))
        assert len(calls)==1


def test_rhs_preserves_invalid_state_input_and_parameter_rejection():
    _,_,model,state=new_fixture();old=DirectACVSRHS(model);new=RequiredOutputsACVSRHS(model)
    for broken in ['state','input','parameter']:
        current=deepcopy(state);release=np.full(model.counts['cones'],.2)
        if broken=='state':current['cross_OFF_ON_parasol_AC_v1',MECHANISM,'s'][:]=2.
        elif broken=='input':release[:]=np.nan
        else:model.p['parasol_RGC_ON']['beta']=-1.
        for engine in [old,new]:
            with pytest.raises(ContractError):engine.rhs_flat(0.,model.pack(current),release)


def test_backend_requires_explicit_identity_and_only_changes_instance_factory():
    _,_,model,_=new_fixture();a=model.assembly
    a.runtime_factories=dict(inner=type(model),compiled=DirectACVSRHS)
    old=a.runtime_factories;identity=contract();bad=dict(identity,implementation_sha256='wrong')
    with pytest.raises(ContractError):install(a,bad)
    result=install(a,identity)
    assert old['compiled'] is DirectACVSRHS and a.runtime_factories['compiled'] is RequiredOutputsACVSRHS
    assert result['no_cross_call_state_cache']
