"""Conditional triangular BC replay -> original AC -> original RGC equations.

No AII state is removed from its upstream computation: its coupled contribution
is already contained in the saved BC release. This is not a full-network replay.
"""
from types import SimpleNamespace

import numpy as np

from .contracts import ContractError
from .group_inner_dynamics import RGC_MECH, key
from .inner import HistoryRGC
from .rgc_expectation import STATES


class RGCFunctionalReplay:
    def __init__(self, layer):
        self.layer = layer
        self.ac = layer.ac
        self.model = layer.model
        self.populations = layer.populations
        self.routes = tuple(r for r in self.model.routes if r.post in self.populations)
        if any(r.pre in self.populations for r in self.model.routes):
            raise ContractError('RGC feedback prevents triangular factoring')
        allowed = set(self.ac.pools) | set(self.ac.bc.populations[:-1])
        if any(r.pre not in allowed for r in self.routes):
            raise ContractError('Unsupported upstream route; no invented AII direct input')
        self.slices = {}
        at = self.ac.nordinary
        for pop in self.populations:
            self.slices[pop] = slice(at, at + self.model.counts[pop])
            at += self.model.counts[pop]
        self.nstate = at

    def initial(self):
        state = self.layer.initial()
        return self.from_state(state), self.law_from_state(state)

    def from_state(self, state):
        return np.concatenate([*[state[k] for k in self.ac.state_keys],
                               *[state[key(p, RGC_MECH, 'V')] for p in self.populations]])

    def law_from_state(self, state):
        return {p: np.column_stack([state[key(p, 'RGC_expected_history',
                    'history_probability_' + str(i))] for i in range(8)])
                for p in self.populations}

    def conductances(self, y, release):
        ordinary = self.ac.ordinary_outputs(y[:self.ac.nordinary], release)
        ge = {p: np.zeros(self.model.counts[p]) for p in self.populations}
        gi = {p: np.zeros(self.model.counts[p]) for p in self.populations}
        for route in self.routes:
            if route.pre in self.ac.pools:
                port = 'output_activity' if route.pre == 'local_feedforward_AC' else 'inhibition_gate'
                activation = ordinary[route.pre + '__' + port]
                target = gi
            else:
                activation = release[route.pre]
                target = ge
            target[route.post] += (self.model.members[route.post]
                * self.model.parameters['route_gains'][route.name]
                * (self.model.mean[route.name] @ activation))
        return ge, gi

    def rhs(self, y, release, *, remove_inhibition=False):
        ge, gi = self.conductances(y, release)
        result = [self.ac.ordinary_rhs(y[:self.ac.nordinary], release)]
        for pop in self.populations:
            V = y[self.slices[pop]]
            inhibitory = np.zeros_like(gi[pop]) if remove_inhibition else gi[pop]
            result.append((self.model.gL[pop] * (self.model.par(pop, 'EL') - V)
                + ge[pop] * (self.model.par(pop, 'EE') - V)
                + inhibitory * (self.model.par(pop, 'EI') - V)) / self.model.C[pop])
        return np.concatenate(result)

    def raw(self, y, release, laws, *, remove_inhibition=False):
        ge, gi = self.conductances(y, release)
        result = {}
        for pop in self.populations:
            V = y[self.slices[pop]]
            pi = laws[pop]
            hazard = self.model.par(pop, 'alpha')[:, None] * np.logaddexp(0.,
                (V[:, None] + (STATES @ np.asarray(self.model.p[pop]['spike_history_kernel']))[None, :]
                 - self.model.par(pop, 'mu')[:, None]) / self.model.par(pop, 'beta')[:, None])
            for name, value in dict(voltage=V, gE=ge[pop],
                    gI=np.zeros_like(gi[pop]) if remove_inhibition else gi[pop],
                    expected_hazard_Hz=(pi * hazard).sum(axis=1),
                    expected_rate_Hz=(pi * -np.expm1(-.001 * hazard)).sum(axis=1) / .001).items():
                result[pop, name] = value
        return result

    def transition(self, y, laws):
        state = {key(p, RGC_MECH, 'V'): y[self.slices[p]] for p in self.populations}
        for pop in self.populations:
            state.update({key(pop, 'RGC_expected_history', 'history_probability_' + str(i)):
                          laws[pop][:, i] for i in range(8)})
        updates = self.model._probability_tick_updates(state)
        return self.law_from_state(updates)

    def sample(self, t, y, auxiliary):
        following, events = {}, {}
        for pop in self.populations:
            alpha, mu, beta = (self.model.par(pop, n) for n in ('alpha', 'mu', 'beta'))
            kernel = np.asarray(self.model.p[pop]['spike_history_kernel'])
            proxy = SimpleNamespace(n=self.model.counts[pop], dt=.001, history_kernel=kernel,
                intensity=lambda v, h, a=alpha, m=mu, b=beta, k=kernel:
                    a * np.logaddexp(0., (v + k @ h - m) / b))
            following[pop], events[pop] = HistoryRGC.sample_spikes(
                proxy, t, y[self.slices[pop]], auxiliary[pop])
        return following, events
