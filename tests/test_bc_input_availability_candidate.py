import numpy as np
import pytest
from scipy.integrate import solve_ivp
from retina_sim.contracts import ContractError
from retina_sim.bc_input_availability_candidate import OFFInputAvailabilityCandidate


def candidate(bg):
    return OFFInputAvailabilityCandidate(bg,alpha=.6,tau_recovery_seconds=.1,evidence='unit_test_only')


def test_piecewise_input_matches_closed_form_and_recovers():
    model=candidate(np.array([.2,.4]))
    state=model.initial()
    for u,duration in [(2.,.5),(0.,.3),(1.,1.)]:
        rate=(1.+model.alpha*u)/model.tau_recovery_seconds
        exact=1./(1.+model.alpha*u)+(state-1./(1.+model.alpha*u))*np.exp(-rate*duration)
        sol=solve_ivp(lambda t,d:model.rhs(d,u*model.background),(0.,duration),state,rtol=1e-10,atol=1e-12)
        assert sol.success
        np.testing.assert_allclose(sol.y[:,-1],exact,rtol=1e-9,atol=1e-10)
        assert np.all((sol.y>=0)&(sol.y<=1))
        state=sol.y[:,-1]


def test_adapted_background_preserved_without_changing_reference():
    bg=np.array([.24,.28]);reference=np.array([.27,.26]);model=candidate(bg)
    effective=model.effective_drive(model.initial(),bg)
    np.testing.assert_allclose(effective-reference,bg-reference,atol=1e-16)
    np.testing.assert_allclose(model.rhs(model.initial(),bg),0.,atol=1e-14)


def test_rejects_implicit_chemical_conversion_and_invalid_domains():
    model=candidate(np.array([.25]))
    with pytest.raises(ContractError):model.normalized_drive(np.array([-.1]))
    with pytest.raises(ContractError):model.normalized_drive(np.array([.1,.2]))
    with pytest.raises(ContractError):model.effective_drive(np.array([1.2]),np.array([.25]))
    with pytest.raises(ContractError):candidate(np.array([0.]))
