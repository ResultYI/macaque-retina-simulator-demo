from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
from retina_sim.contracts import ContractError
from retina_sim.rgc_expectation import exact_history_rates


def model():
    n=SimpleNamespace(alpha=np.array([150.]),mu=np.array([-.070]),beta=np.array([.005]),
        history_kernel=np.array([-.012,-.008,-.004]),dt=.001,
        deterministic_state_independent_of_spikes=True)
    n.intensity=lambda V,h:n.alpha*np.logaddexp(0.,(V+n.history_kernel@h-n.mu)/n.beta)
    return n


def test_exact_three_history_rates_match_independent_path_enumeration():
    n=model();t=np.arange(9)*.001;V=np.array([[-.065,-.062,-.055,-.068,-.060,-.063,-.066,-.058,-.064]])
    h=np.array([[1],[0],[1]]);aux=dict(history=h.tolist(),last_bin_counts=[1])
    r=exact_history_rates(n,V,t,aux)
    for j in range(1,8):
        expected_hazard=expected_count=mass=0.
        for sequence in product((0,1),repeat=j-1):
            past=[1,1,0];prob=1.
            for k,y in enumerate(sequence,start=1):
                lam=float(n.intensity(V[:,k],np.array(past)[:,None])[0]);p=-np.expm1(-lam*n.dt)
                prob*=p if y else 1-p;past=[y,*past[:2]]
            lam=float(n.intensity(V[:,j],np.array(past)[:,None])[0])
            mass+=prob;expected_hazard+=prob*lam;expected_count+=prob*(-np.expm1(-lam*n.dt))/n.dt
        assert mass==pytest.approx(1.,abs=2e-15)
        assert r['mean_history_hazard_Hz'][0,j]==pytest.approx(expected_hazard,abs=2e-12)
        assert r['expected_binary_count_rate_Hz'][0,j]==pytest.approx(expected_count,abs=2e-12)
    assert r['expected_binary_count_rate_Hz'][0,0]==1000.
    assert r['zero_history_hazard_Hz'].shape==(1,8)
    assert r['mean_history_hazard_Hz'][0,0]<r['zero_history_hazard_Hz'][0,0]
    assert r['maximum_probability_mass_error']<1e-14


@pytest.mark.parametrize('edit',[lambda n,t,a:setattr(n,'dt',.002),
    lambda n,t,a:t.__setitem__(2,.0025),lambda n,t,a:a.update(last_bin_counts=[2]),
    lambda n,t,a:setattr(n,'deterministic_state_independent_of_spikes',False)])
def test_exact_analysis_refuses_unverified_or_interpolated_contract(edit):
    n=model();t=np.arange(4)*.001;aux=dict(history=[[0],[0],[0]],last_bin_counts=[0]);edit(n,t,aux)
    with pytest.raises(ContractError):exact_history_rates(n,np.full((1,4),-.065),t,aux)
