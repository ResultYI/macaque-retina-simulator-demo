"""Native Cone transduction under explicitly fixed exterior HC feedback."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit
from .contracts import ContractError, canonical_hash
from .h1_h2_effective import cone_fixed_point
from .hc_research_native import ConditionedH1H2Native as H1H2Native
from .cone_hc_reduced_native import AnalyticREAnchor
from .vh2005 import SOURCE_VALUES


class FixedFeedbackCone:
    """One exact common-drive state, mapped back to each immutable source member.

    All mapped cells have identical light, initial state and held feedback.
    No spatial coordinates or individual A coefficients are approximated.
    """
    def __init__(self, feedback, *, member_identity, tightened=False):
        p = SOURCE_VALUES
        if not np.isfinite(feedback) or feedback < 0 or not member_identity:
            raise ContractError('Explicit finite exterior feedback and members required')
        self.feedback = float(feedback)
        eq = cone_fixed_point(np.array([1000.]))
        self.y0 = np.array([eq[k][0] for k in ('X', 'C', 'Vis', 'g_i', 'Vbar')])
        self.anchor = AnalyticREAnchor(.02, np.array([1000.]), np.array([1000.]), np.array([1010.]), .12, 1)
        self.solver = dict(method='DOP853', rtol=1e-9 if tightened else 1e-7,
                           atol=1e-12 if tightened else 1e-10,
                           max_step=.0001 if tightened else .0002)
        self.identity = canonical_hash(dict(feedback=self.feedback, y0=self.y0,
            parameters=p, members=member_identity, solver=self.solver,
            input=dict(background=1000., on=1010., onset=.02, offset=.12)))
        self.solution = solve_ivp(self.rhs, (.02, .12), self.y0, dense_output=True, **self.solver)
        if not self.solution.success: raise ContractError(self.solution.message)
        self.background_release = float(expit((self.y0[2]-self.feedback-p['V_k'])/p['V_n'])
                                         /(self.y0[4]/p['V_I'])**p['mu'])

    def rhs(self, t, y):
        X, C, V, gi, vb = y; p = SOURCE_VALUES
        if not np.isfinite(y).all() or min(X, C, V) < 0 or min(gi, vb) <= 0:
            raise ContractError('Exterior Cone left native state domain')
        _, E = self.anchor.at(t)
        return 1000*np.array([1/(1+(p['a_C']*C)**p['n_C'])-(p['c_beta']+p['k_beta']*E[0])*X,
            (X-C)/p['tau_C'], (X/gi-V)/p['tau_m'],
            (p['a_is']*V**p['gamma']-gi)/p['tau_is'], (V-vb)/p['tau_a']])

    def at(self, t):
        if not np.isfinite(t) or not 0 <= t <= .12:
            raise ContractError('Exterior cache time outside frozen input')
        y = self.y0 if t <= .02 else self.solution.sol(t)
        if not np.isfinite(y).all() or np.any(y < 0) or min(y[3:]) <= 0:
            raise ContractError('Invalid exterior dense state')
        p = SOURCE_VALUES
        release = expit((y[2]-self.feedback-p['V_k'])/p['V_n'])/(y[4]/p['V_I'])**p['mu']
        return float(release), float(y[4])


class DrivenExteriorH1(H1H2Native):
    def attach_exterior(self, cache, stimulated_mass):
        mass = np.asarray(stimulated_mass, float)
        if (mass.shape != (self.nh,) or not np.isfinite(mass).all() or np.any(mass < 0)
                or np.any(mass > 1+2e-14) or hasattr(self, 'exterior_cache')):
            raise ContractError('Explicit immutable exterior mass required')
        self.exterior_cache = cache
        self.stimulated_exterior_mass = mass.copy(); self.stimulated_exterior_mass.setflags(write=False)
        self.source_version = canonical_hash(dict(native=self.source_version, exterior=cache.identity, mass=mass))

    def boundary_at(self, t):
        base = self.boundary
        if base is None or not hasattr(self, 'exterior_cache'): return base
        release, vb = self.exterior_cache.at(t)
        return dict(A_release=base['A_release']+self.stimulated_exterior_mass*(release-self.exterior_cache.background_release),
                    A_Vbar=base['A_Vbar']+self.stimulated_exterior_mass*(vb-self.exterior_cache.y0[4]),
                    B_h=base['B_h'])
