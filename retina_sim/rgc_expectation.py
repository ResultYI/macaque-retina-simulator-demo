"""Exact conditional three-bin history analysis on native recorded RGC voltage.

This is analysis of the retained sampler, not a replacement neural mechanism.
The prepared first bin is already drawn. Its count remains fixed; its hazard
still reports the intensity before that observed draw. The terminal voltage
sample is excluded because its following bin is outside the observation.
"""
from itertools import product
import numpy as np
from .contracts import ContractError

STATES = np.asarray(list(product((0, 1), repeat=3)), dtype=int)
INDEX = {tuple(s): j for j, s in enumerate(STATES)}
NEXT0 = np.array([INDEX[(0, s[0], s[1])] for s in STATES])
NEXT1 = np.array([INDEX[(1, s[0], s[1])] for s in STATES])


def stationary_history_rates(native, voltage, *, cold_start_bins=1250):
    """Exact constant-V stationary law of the unchanged binary three-tap sampler.

    This does not initialize or sample a simulated cell. A stationary Markov
    law is separate from finite-record spike stationarity statistics.
    """
    V=np.asarray(voltage,float)
    if (V.ndim!=1 or not np.isfinite(V).all() or np.shape(native.history_kernel)!=(3,) or
            native.dt!=.001 or not native.deterministic_state_independent_of_spikes):
        raise ContractError('Stationary analysis requires finite per-cell V and the retained binary sampler')
    alpha,mu,beta=(np.asarray(getattr(native,k),float) for k in ('alpha','mu','beta'))
    if any(x.shape!=V.shape for x in (alpha,mu,beta)) or np.any(alpha<=0) or np.any(beta<=0):
        raise ContractError('Stationary per-cell parameter mismatch')
    lam=alpha[:,None]*np.logaddexp(0.,(V[:,None]+(STATES@native.history_kernel)[None,:]-mu[:,None])/beta[:,None])
    prob=-np.expm1(-native.dt*lam);laws=[];errors=[];cold_errors=[]
    for p in prob:
        transition=np.zeros((8,8))
        transition[np.arange(8),NEXT0]+=1-p
        transition[np.arange(8),NEXT1]+=p
        matrix=transition.T-np.eye(8);matrix[-1]=1.;rhs=np.zeros(8);rhs[-1]=1.
        pi=np.linalg.solve(matrix,rhs)
        if pi.min() < -1e-13:raise ContractError('Negative stationary history probability')
        error=float(np.max(abs(pi@transition-pi)))
        if error>1e-12 or abs(pi.sum()-1)>1e-12:raise ContractError('Stationary probability conservation failed')
        cold=np.zeros(8);cold[INDEX[(0,0,0)]]=1.
        cold=cold@np.linalg.matrix_power(transition,int(cold_start_bins))
        laws.append(pi);errors.append(error);cold_errors.append(.5*float(abs(cold-pi).sum()))
    laws=np.asarray(laws)
    return dict(history_probabilities=laws,mean_history_hazard_Hz=(laws*lam).sum(axis=1),
        expected_binary_count_rate_Hz=(laws*prob).sum(axis=1)/native.dt,
        zero_history_hazard_Hz=alpha*np.logaddexp(0.,(V-mu)/beta),
        maximum_stationary_residual=max(errors,default=0.),
        cold_start_bins=int(cold_start_bins),maximum_cold_start_total_variation=max(cold_errors,default=0.),
        scope='Exact frozen-V binary-history law; does not establish finite-record spike statistics')


def validate_sampler(compiled, component_id):
    component = compiled.components[component_id]
    native = component.native_model
    if (component.mechanism != 'HistoryConductanceRGC' or
            not native.deterministic_state_independent_of_spikes or
            np.shape(native.history_kernel) != (3,) or native.dt != .001):
        raise ContractError('Exact analysis requires the retained spike-independent three-bin 1ms RGC')
    # The declared mechanism exports graded V and conductances only. Check the
    # actual graph as well, rather than inferring independence from its name.
    if any(l.source_component == component_id and l.source_port not in ('voltage', 'gE', 'gI')
           for l in compiled.spec.links):
        raise ContractError('A spike-dependent graph output invalidates conditional voltage analysis')
    return native


def exact_history_rates(native, voltage, time_seconds, prepared_spike_state):
    """Return lambda0, E[lambda], E[count]/dt on an exact native bin-start grid.

    Expectation conditions on the saved preparation history and first draw.
    It is not a stationary-history estimate. No interpolation is performed.
    """
    V = np.asarray(voltage, dtype=float)
    t = np.asarray(time_seconds, dtype=float)
    if (V.ndim != 2 or V.shape[1] != len(t) or len(t) < 2 or
            not np.isfinite(V).all() or not np.isfinite(t).all() or
            not np.allclose(np.diff(t), native.dt, atol=1e-12, rtol=0) or
            np.shape(native.history_kernel) != (3,) or native.dt != .001 or
            not native.deterministic_state_independent_of_spikes):
        raise ContractError('Finite native 1ms bin-start voltage plus endpoint required; no sparse interpolation')
    n, nt = V.shape
    h = np.asarray(prepared_spike_state['history'])
    first = np.asarray(prepared_spike_state['last_bin_counts'])
    if (h.shape != (3, n) or first.shape != (n,) or
            not np.isin(h, (0, 1)).all() or not np.isin(first, (0, 1)).all()):
        raise ContractError('Exact analysis requires binary saved history and an already sampled first bin')
    alpha, mu, beta = (np.asarray(getattr(native, key), float) for key in ('alpha', 'mu', 'beta'))
    if (any(x.shape != (n,) or not np.isfinite(x).all() for x in (alpha, mu, beta)) or
            np.any(alpha <= 0) or np.any(beta <= 0) or not np.isfinite(native.history_kernel).all()):
        raise ContractError('Per-cell output parameter shape differs from recorded voltage')
    lam0 = alpha[:, None] * np.logaddexp(0., (V[:, :-1] - mu[:, None]) / beta[:, None])
    mean_hazard = np.empty_like(lam0)
    mean_count = np.empty_like(lam0)
    mean_hazard[:, 0] = native.intensity(V[:, 0], h)
    mean_count[:, 0] = first / native.dt
    pi = np.zeros((n, 8))
    for cell in range(n):
        pi[cell, INDEX[(int(first[cell]), int(h[0, cell]), int(h[1, cell]))]] = 1.
    max_mass_error = 0.
    for j in range(1, nt - 1):
        lam = alpha[:, None] * np.logaddexp(0.,
            (V[:, j, None] + (STATES @ native.history_kernel)[None, :] - mu[:, None]) / beta[:, None])
        p = -np.expm1(-native.dt * lam)
        mean_hazard[:, j] = (pi * lam).sum(axis=1)
        mean_count[:, j] = (pi * p).sum(axis=1) / native.dt
        nxt = np.zeros_like(pi)
        for k in range(8):
            nxt[:, NEXT0[k]] += pi[:, k] * (1 - p[:, k])
            nxt[:, NEXT1[k]] += pi[:, k] * p[:, k]
        pi = nxt
        max_mass_error = max(max_mass_error, float(np.max(np.abs(pi.sum(axis=1) - 1))))
    if max_mass_error > 1e-12 or not all(np.isfinite(x).all() for x in (lam0, mean_hazard, mean_count)):
        raise ContractError('History probability conservation or finite-value check failed')
    return dict(zero_history_hazard_Hz=lam0, mean_history_hazard_Hz=mean_hazard,
                expected_binary_count_rate_Hz=mean_count,
                maximum_probability_mass_error=max_mass_error)
