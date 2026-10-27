"""Verify that the new knob changes actual native integration, not outputs."""
from types import SimpleNamespace
import numpy as np
import pytest
import retina_sim.native_multirate as native
from retina_sim.vh2005 import SOURCE_VALUES
from retina_sim.contracts import ContractError
from tools.validate_ac_vs_native_temporal import native_cap


def test_native_half_cap_changes_real_accepted_steps_and_keeps_inner_independent():
    counts=[];widths=[]
    for cap in [.0001,.00005]:
        with native_cap(cap):
            curve,metrics=native._solve_fast(lambda t,y:np.ones_like(y),0.,np.ones(2),.0005,rtol=1e-6,atol=1e-9,ops=native._Ops())
            counts.append(metrics['steps']);widths.append(max(s.t1-s.t0 for s in curve.segments))
            assert all(s.t1<=s.t0+cap for s in curve.segments)
            np.testing.assert_allclose(curve.at(.0005),np.full(2,1.0005),rtol=0,atol=2e-15)
    assert counts[1]>counts[0] and widths[1]<widths[0]
    assert native.MAX_STEP==.0001


def test_numerical_identity_binds_native_cap_and_context_restores_after_failure():
    physics=SimpleNamespace(p=SOURCE_VALUES,torch=None,device=None,nstate=1,source_version='same-equations',aux_indices=())
    with native_cap(.0001):first=native.NativeMultirateBackend(physics,rtol=1e-6,atol=1e-9).reference_identity
    with native_cap(.00005):second=native.NativeMultirateBackend(physics,rtol=1e-6,atol=1e-9).reference_identity
    assert first!=second
    with pytest.raises(RuntimeError):
        with native_cap(.00005):raise RuntimeError('intentional synthetic interruption')
    assert native.MAX_STEP==.0001
    with pytest.raises(ContractError):
        with native_cap(.000025):pass
