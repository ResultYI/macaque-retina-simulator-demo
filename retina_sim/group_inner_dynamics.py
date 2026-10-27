"""Actual CPU equations and accepted-time operators for the v0.2 inner graph.

Parameters and initial conditions are supplied by a source-bound binder. This
module never invents a baseline, initializes a retina or constructs fine cells.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import math

import numpy as np
from scipy import sparse
from scipy.integrate import DOP853
from scipy.special import expit

from .contracts import ContractError, canonical_hash
from .inner import HistoryRGC
from .rgc_expectation import STATES, NEXT0, NEXT1
from .spatial_group_model import BC, AC, RGC
from .spatial_group_graph import digest


RODS = 'Chen2024PrimateRodCurrent'
MEMBRANE = 'ReceptorConductanceMembrane'
BC_MECH = 'HistoryBipolarTerminal'
RGC_MECH = 'RGC_spike_history'
AC_MECH = 'DynamicAmacrineInhibition'
CONTACTS = ('individual_rod_RBC_contacts', 'RBC_AII_AMPA', 'AII_FMB_glycine', 'AII_DB1_glycine')


def key(population, mechanism, state):
    return population, mechanism, state


def _array(value, n, name, *, positive=False, nonnegative=False):
    try:
        result = np.broadcast_to(np.asarray(value, dtype=np.float64), (n,)).copy()
    except (ValueError, TypeError) as exc:
        raise ContractError('Inner parameter shape differs: '+name) from exc
    if (not np.isfinite(result).all() or positive and np.any(result <= 0)
            or nonnegative and np.any(result < 0)):
        raise ContractError('Inner parameter domain differs: '+name)
    result.flags.writeable = False
    return result


@dataclass
class InnerAdvance:
    model: object
    t0: float
    t1: float
    state: dict
    evaluator: object
    representations: dict
    metrics: dict
    accepted_times: tuple = ()
    dense_segments: tuple = ()

    @property
    def available_until(self):
        return self.t1

    def evaluate(self, t, side='right'):
        if (not math.isfinite(t) or not self.t0 <= t <= self.t1 or side not in ('left', 'right')
                or t == self.t0 and side != 'right' or t == self.t1 and side != 'left'):
            raise ContractError('Inner query outside accepted interval or on wrong side')
        return self.evaluator(float(t), side)

    def rhs_at(self, t, side='right'):
        return self.model.rhs(t, self.evaluate(t, side))

    def representation_derivative(self, t, side='right'):
        self.evaluate(t, side)  # Same accepted interval/side guard.
        if not self.representations:
            raise ContractError('This joint reference does not export statewise representation derivatives')
        result = {k: np.zeros_like(value) for k, value in self.state.items()}
        for k, entries in self.representations.items():
            for rows, representation in entries:
                selected = representation._segment(t, side) if hasattr(representation, '_segment') else representation
                result[k][rows] = selected.representation_derivative(t, side)
        return result


class GroupInnerDynamics:
    """Physical-total membranes, original per-contact gates and original clocks.

    ``parameters['populations']`` contains per-original-member physical/profile
    quantities. C and gL are multiplied by group member counts here, exactly
    once. Contact coefficient arrays already contain physical grouped totals.
    """
    def __init__(self, assembly, parameters):
        self.assembly = assembly
        self.parameters = parameters
        self.ac_populations = getattr(assembly, 'ac_populations', AC)
        if set(parameters) - {'populations', 'contacts', 'route_gains', 'rod_phototransduction', 'identity', 'provenance'}:
            raise ContractError('Unknown inner parameter bundle fields')
        self.counts = {n: p.group_count for n, p in assembly.populations.items()}
        self.members = {n: np.asarray(p.counts, dtype=np.float64) for n, p in assembly.populations.items()}
        self.blocks = tuple(b for b in (*assembly.state_blocks, *assembly.probability_blocks)
                            if b.population not in ('cones', 'HC'))
        self.state_keys = tuple(key(b.population, b.mechanism, b.name) for b in self.blocks)
        self.sizes = {k: b.count for k, b in zip(self.state_keys, self.blocks)}
        if len(self.sizes) != len(self.blocks):
            raise ContractError('Duplicate inner state identity')
        self.slices, offset = {}, 0
        for k, b in zip(self.state_keys, self.blocks):
            self.slices[k] = slice(offset, offset+b.count)
            offset += b.count
        self.nstate = offset
        self.compiled_rhs = None
        self.p = {}
        for name in ('rods', 'rod_BC', 'AII_AC', *BC, *self.ac_populations, *RGC):
            if name not in parameters['populations']:
                raise ContractError('Missing source-bound population parameters: '+name)
            self.p[name] = dict(parameters['populations'][name])
        self._par_cache, self._contact_parameter_cache = {}, {}
        self.C, self.gL = {}, {}
        for name in ('rods', 'rod_BC', 'AII_AC', *BC, *RGC):
            self.C[name] = self.par(name, 'C', positive=True)*self.members[name]
            self.gL[name] = self.par(name, 'gL', positive=True)*self.members[name]
            for field, actual in (('C', self.C[name]), ('gL', self.gL[name])):
                relative = f'populations/{name}/{field}_total.npy'
                if relative in assembly.manifest['files']:
                    if not np.array_equal(actual, assembly.array(relative)):
                        raise ContractError('Physical membrane totals differ from frozen group artifact: '+name+'.'+field)
        self.routes = tuple(assembly.routes.values())
        self.mean = {r.name: assembly.matrix('routes/'+r.name+'/mean_weights.npz')
                     for r in self.routes if r.kind == 'inner_chemical'}
        self.contacts = {}
        for route in self.routes:
            if not route.contact_component:
                continue
            c = route.contact_component
            prefix = 'routes/'+route.name+'/'
            post = np.asarray(assembly.array(prefix+'contact_post_group.npy'), dtype=np.int64)
            pre = np.asarray(assembly.array(prefix+'contact_pre_group.npy'), dtype=np.int64)
            matrix = assembly.matrix(prefix+'contact_coefficient_S.npz')
            coefficient = np.asarray(matrix[post, pre]).reshape(-1)
            self.contacts[c] = dict(route=route, pre=pre, post=post, coefficient=coefficient,
                                    parameters=dict(parameters['contacts'][c]))
        if set(self.contacts) != set(CONTACTS):
            raise ContractError('All four grouped contact mechanisms required')
        self.gap_names = tuple(assembly.manifest['gap']['population_order'])
        self.gap = assembly.matrix('gap/laplacian_S.npz')
        self.gap_slices, at = {}, 0
        for name in self.gap_names:
            self.gap_slices[name] = slice(at, at+self.counts[name]); at += self.counts[name]
        if self.gap.shape != (at, at):
            raise ContractError('Inner gap layout differs')
        self.rod = dict(parameters['rod_phototransduction'])
        required = ('timeStep', 'sigma', 'gamma', 'eta', 'phi', 'beta', 'k', 'n', 'kGC', 'm', 'q', 'smax')
        if (any(k not in self.rod for k in required) or self.rod['timeStep'] != .0001
                or any(not math.isfinite(float(self.rod[k])) for k in required)):
            raise ContractError('Complete unchanged rod author coefficients/100us clock required')
        for name in BC:
            if np.shape(self.p[name].get('kernel')) != (3,) or float(self.p[name].get('sample_interval', 0)) != .001:
                raise ContractError('Original three-tap 1ms BC history required')
            if type(self.p[name].get('adaptation_enabled')) is not bool:
                raise ContractError('Explicit BC adaptation enabled binding required')
        for name in RGC:
            if np.shape(self.p[name].get('spike_history_kernel')) != (3,) or self.p[name].get('spike_interval') != .001:
                raise ContractError('Original three-bin/1ms RGC event contract required')
        self.clock_metadata = dict(rod_author_seconds=.0001, BC_history_seconds=.001,
            RGC_history_and_sampling_seconds=.001, input_sampling='caller-provided original author interval rate',
            event_snapshot='same pre-event state; original rod order, BC history, then RGC PCG64',
            probability_history='unconditional binary-history law from explicitly supplied initial distribution')
        self.identity = canonical_hash(dict(schema='retipath-group-inner-dynamics-1',
            artifact=assembly.artifact_sha256, parameter_identity=parameters.get('identity'),
            parameters=parameters, state_keys=self.state_keys, clocks=self.clock_metadata,
            implementation_sha256=digest(Path(__file__)),
            physical_totals='count*C,count*gL; summed contact coefficients; original total gap Laplacian'))

    def par(self, pop, name, *, positive=False, nonnegative=False):
        cache_key = pop, name, positive, nonnegative
        if cache_key in self._par_cache:
            return self._par_cache[cache_key]
        if name not in self.p[pop]:
            raise ContractError('Missing source-bound parameter '+pop+'.'+name)
        result = _array(self.p[pop][name], self.counts[pop], pop+'.'+name,
                        positive=positive, nonnegative=nonnegative)
        self._par_cache[cache_key] = result
        return result

    def cp(self, component, name, *, positive=False):
        cache_key = component, name, positive
        if cache_key in self._contact_parameter_cache:
            return self._contact_parameter_cache[cache_key]
        c = self.contacts[component]
        if name not in c['parameters']:
            raise ContractError('Missing source-bound contact parameter '+component+'.'+name)
        result = _array(c['parameters'][name], len(c['pre']), component+'.'+name, positive=positive)
        self._contact_parameter_cache[cache_key] = result
        return result

    def validate(self, state):
        if set(state) != set(self.state_keys):
            raise ContractError('Complete unchanged inner state layout required')
        for k in self.state_keys:
            a = np.asarray(state[k])
            if a.dtype != np.float64 or a.shape != (self.sizes[k],) or not np.isfinite(a).all():
                raise ContractError('Finite original-layout FP64 inner state required: '+str(k))

    def pack(self, state):
        self.validate(state)
        return np.concatenate([state[k] for k in self.state_keys])

    def unpack(self, packed, *, copy=True):
        y = np.asarray(packed)
        if y.dtype != np.float64 or y.shape != (self.nstate,) or not np.isfinite(y).all():
            raise ContractError('Finite FP64 packed inner state required')
        return {k: y[s].copy() if copy else y[s] for k, s in self.slices.items()}

    def _mem(self, state, pop, name):
        return state[key(pop, 'rod_membrane' if pop == 'rods' else MEMBRANE, name)]

    def _bc(self, state, pop, name):
        return state[key(pop, BC_MECH, name)]

    def _contact(self, state, component, name):
        c = self.contacts[component]
        return state[key(c['route'].pre, component, name)]

    def _kinetics(self, state, component):
        rise = self.cp(component, 'tau_rise', positive=True)
        if component.startswith('AII_'):
            c = self.contacts[component]
            voltage = self._bc(state, c['route'].post, 'V')[c['post']]
            lo, hi = self.cp(component, 'decay_minus100', positive=True), self.cp(component, 'decay_0', positive=True)
            # This clamp is the retained original source interpolation, not a
            # numerical repair or clipping of the computed physiological state.
            decay = lo+(hi-lo)*np.minimum(1., np.maximum(0., (voltage+.1)/.1))
        else:
            decay = self.cp(component, 'tau_decay', positive=True)
        if np.any(decay <= rise):
            raise ContractError('Physical contact decay must exceed rise')
        peak_time = np.log(decay/rise)/(1/rise-1/decay)
        normalization = np.exp(-peak_time/decay)-np.exp(-peak_time/rise)
        return rise, decay, normalization

    def outputs(self, t, state, cone_release=None):
        self.validate(state)
        out = {}
        for pop in ('rods', 'rod_BC', 'AII_AC'):
            V, rel = self._mem(state, pop, 'V'), self._mem(state, pop, 'release')
            out[pop, 'voltage'], out[pop, 'release'], out[pop, 'activation'] = V, rel, rel
            if pop != 'rods':
                x = self._mem(state, pop, 'x')
                out[pop, 'event_rate'] = self.par(pop, 'Npool', positive=True)*self.par(pop, 'U', positive=True)*self.par(pop, 'rmax', positive=True)*expit((V-self.par(pop, 'r_half'))/self.par(pop, 'r_slope', positive=True))*x
                if np.any(out[pop, 'event_rate'] < 0):
                    raise ContractError('Negative current ribbon event rate')
        for pop in BC:
            rel, A = self._bc(state, pop, 'terminal_release'), self._bc(state, pop, 'A')
            gain = 1./(1.+A**6) if self.p[pop]['adaptation_enabled'] else np.ones_like(A)
            out[pop, 'voltage'] = self._bc(state, pop, 'V')
            out[pop, 'activation'] = out[pop, 'effective_release'] = rel*gain
            out[pop, 'unadapted_release'], out[pop, 'adaptation_activity'] = rel, A
            out[pop, 'generator_activation'] = self._bc(state, pop, 'activation')
        for component, c in self.contacts.items():
            if component == 'individual_rod_RBC_contacts':
                conductance = c['coefficient']*self._contact(state, component, 'm')
            else:
                _, _, norm = self._kinetics(state, component)
                conductance = c['coefficient']*(self._contact(state, component, 'b')-self._contact(state, component, 'a'))/norm
            if np.any(conductance < -1e-18) or not np.isfinite(conductance).all():
                raise ContractError('Negative/nonfinite physical contact conductance; no clipping')
            out[component, 'contact_conductance'] = conductance
            out[component, 'conductance'] = np.bincount(c['post'], weights=conductance, minlength=self.counts[c['route'].post])
        values = np.concatenate([self._mem(state, n, 'V') if n == 'AII_AC' else self._bc(state, n, 'V') for n in self.gap_names])
        currents = -(self.gap@values)
        for pop, section in self.gap_slices.items():
            out[pop, 'gap_current'] = currents[section]
        for pop in BC:
            if cone_release is not None:
                r = next(r for r in self.routes if r.pre == 'cones' and r.post == pop)
                value = np.asarray(cone_release, dtype=np.float64)
                if value.shape != (self.counts['cones'],) or not np.isfinite(value).all() or np.any(value < 0):
                    raise ContractError('Current original Cone release required at BC commit')
                out[pop, 'drive'] = self.mean[r.name]@value
        self.ac_outputs(state, out)
        for pop in RGC:
            ge, gi = np.zeros(self.counts[pop]), np.zeros(self.counts[pop])
            for r in self.routes:
                if r.post != pop:
                    continue
                if r.name not in self.parameters['route_gains']:
                    raise ContractError('Missing original RGC receptor gain: '+r.name)
                gain = _array(self.parameters['route_gains'][r.name], self.counts[pop], r.name, nonnegative=True)
                source = out[r.pre, 'activation']
                value = self.members[pop]*gain*(self.mean[r.name]@source)
                if r.pre in self.ac_populations:
                    gi += value
                else:
                    ge += value
            if np.any(ge < 0) or np.any(gi < 0):
                raise ContractError('Negative RGC conductance')
            out[pop, 'gE'], out[pop, 'gI'] = ge, gi
            out[pop, 'voltage'] = state[key(pop, RGC_MECH, 'V')]
        return out

    def ac_outputs(self, state, out):
        """Legacy AC contract; independently versioned models override this hook."""
        for pop in self.ac_populations:
            h = state[key(pop, AC_MECH, 'h')]
            out[pop, 'inhibition_state'] = h
            out[pop, 'activation'] = self.par(pop, 'output_scale', positive=True)*h
            routes = [r for r in self.routes if r.post == pop]
            drive = sum((self.mean[r.name]@out[r.pre, 'effective_release'] for r in routes), start=np.zeros(self.counts[pop]))
            out[pop, 'drive'] = drive
            out[pop, 'target_inhibition'] = expit(self.par(pop, 'input_slope', positive=True)*(drive-self.par(pop, 'reference_input')))

    def ac_rhs(self, state, out, result):
        for pop in self.ac_populations:
            h = state[key(pop, AC_MECH, 'h')]
            if np.any(h < -1e-10) or np.any(h > 1+1e-10):
                raise ContractError('Original generic AC invariant violated')
            result[key(pop, AC_MECH, 'h')] = (out[pop, 'target_inhibition']-h)/self.par(pop, 'tau_A_seconds', positive=True)

    def rod_current(self, state):
        return -self.rod['k']*state[key('rods', RODS, 'G')]**self.rod['n']*1e-12

    def rhs(self, t, state, cone_release=None, rod_current=None):
        out = self.outputs(t, state, cone_release)
        result = {k: np.zeros_like(state[k]) for k in self.state_keys}
        for pop in ('rods', 'rod_BC', 'AII_AC'):
            V, rel = self._mem(state, pop, 'V'), self._mem(state, pop, 'release')
            current = self.gL[pop]*(self.par(pop, 'EL')-V)+out.get((pop, 'gap_current'), 0.)
            if pop == 'rods':
                source_current = self.rod_current(state) if rod_current is None else _array(rod_current, self.counts[pop], 'rod_current')
                current -= self.members[pop]*(source_current-self.par(pop, 'current_anchor'))
            else:
                component = 'individual_rod_RBC_contacts' if pop == 'rod_BC' else 'RBC_AII_AMPA'
                current += out[component, 'conductance']*(self.par(pop, 'EE')-V)
                attempts = self.par(pop, 'rmax', positive=True)*expit((V-self.par(pop, 'r_half'))/self.par(pop, 'r_slope', positive=True))
                x = self._mem(state, pop, 'x')
                result[key(pop, MEMBRANE, 'x')] = (1-x)/self.par(pop, 'tau_recovery', positive=True)-self.par(pop, 'U', positive=True)*attempts*x
            mech = 'rod_membrane' if pop == 'rods' else MEMBRANE
            result[key(pop, mech, 'V')] = current/self.C[pop]
            result[key(pop, mech, 'release')] = (expit((V-self.par(pop, 'release_half'))/self.par(pop, 'release_slope', positive=True))-rel)/self.par(pop, 'tau_release', positive=True)
        for pop in BC:
            V, rel, A = (self._bc(state, pop, name) for name in ('V', 'terminal_release', 'A'))
            target = self.par(pop, 'Vref')+self.par(pop, 'native_voltage_gain', positive=True)*(self._bc(state, pop, 'activation')-self.par(pop, 'native_reference'))
            current = self.gL[pop]*(target-V)+out.get((pop, 'gap_current'), 0.)
            if pop.endswith('_OFF'):
                component = 'AII_FMB_glycine' if pop == 'midget_BC_OFF' else 'AII_DB1_glycine'
                current += out[component, 'conductance']*(self.par(pop, 'EI')-V)
            result[key(pop, BC_MECH, 'V')] = current/self.C[pop]
            result[key(pop, BC_MECH, 'terminal_release')] = (expit((V-self.par(pop, 'release_half'))/self.par(pop, 'release_slope', positive=True))-rel)/self.par(pop, 'tau_release', positive=True)
            if self.p[pop]['adaptation_enabled']:
                scale = self.par(pop, 'adaptation_scale', nonnegative=True)
                if np.any(scale > .75) or np.any(A < -1e-10) or np.any(A > scale+1e-10):
                    raise ContractError('Original BC adaptation invariant violated')
                result[key(pop, BC_MECH, 'A')] = (scale*rel-A)/self.par(pop, 'tau_a_seconds', positive=True)
        self.ac_rhs(state, out, result)
        for pop in RGC:
            V = state[key(pop, RGC_MECH, 'V')]
            result[key(pop, RGC_MECH, 'V')] = (self.gL[pop]*(self.par(pop, 'EL')-V)+out[pop, 'gE']*(self.par(pop, 'EE')-V)+out[pop, 'gI']*(self.par(pop, 'EI')-V))/self.C[pop]
        for component, c in self.contacts.items():
            if component == 'individual_rod_RBC_contacts':
                drive = out['rods', 'release'][c['pre']]
                if np.any(drive < 0):
                    raise ContractError('Negative original glutamate drive')
                target = expit(-self.cp(component, 'hill', positive=True)*np.log(np.maximum(drive, np.finfo(float).tiny)/self.cp(component, 'K', positive=True)))
                result[key(c['route'].pre, component, 'm')] = (target-self._contact(state, component, 'm'))/self.cp(component, 'tau', positive=True)
            else:
                rate = out[c['route'].pre, 'event_rate'][c['pre']]
                rise, decay, _ = self._kinetics(state, component)
                for name, tau in (('a', rise), ('b', decay)):
                    result[key(c['route'].pre, component, name)] = -self._contact(state, component, name)/tau+rate
        if any(not np.isfinite(a).all() for a in result.values()):
            raise ContractError('Nonfinite original inner RHS')
        return result

    def _probability_tick_updates(self, state):
        updates = {}
        for pop in RGC:
            V = state[key(pop, RGC_MECH, 'V')]
            alpha, mu, beta = self.par(pop, 'alpha', positive=True), self.par(pop, 'mu'), self.par(pop, 'beta', positive=True)
            kernel = np.asarray(self.p[pop]['spike_history_kernel'], dtype=np.float64)
            pi = np.column_stack([state[key(pop, 'RGC_expected_history', 'history_probability_'+str(i))] for i in range(8)])
            if np.any(pi < -1e-13) or np.max(np.abs(pi.sum(axis=1)-1), initial=0.) > 1e-12:
                raise ContractError('Original history probability law invalid')
            intensity = alpha[:, None]*np.logaddexp(0., (V[:, None]+(STATES@kernel)[None, :]-mu[:, None])/beta[:, None])
            probability = -np.expm1(-.001*intensity)
            nxt = np.zeros_like(pi)
            for i in range(8):
                nxt[:, NEXT0[i]] += pi[:, i]*(1-probability[:, i])
                nxt[:, NEXT1[i]] += pi[:, i]*probability[:, i]
            updates.update({key(pop, 'RGC_expected_history', 'history_probability_'+str(i)): nxt[:, i] for i in range(8)})
        return updates

    def pure_transition_witness(self, state, cone_release, rod_light, *, time_seconds=0.):
        """Original frozen one-step maps for monitoring; never samples RNG.

        Every mechanism reads the same current snapshot. Rod's step is100us;
        BC history and expected RGC law each use their own original1ms step.
        """
        committed, _, _ = self.commit_ticks(time_seconds, state, cone_release=cone_release,
            rod_light=rod_light, rod_tick=True, bc_tick=True, rgc_tick=False)
        committed.update(self._probability_tick_updates(state))
        return committed

    def observation_witness(self, t, state, cone_release, rod_light_source):
        """Readouts and pure maps of one snapshot, with no cross-call cache.

        The rod map only writes new update arrays. Its BC readout therefore
        uses exactly the same pre-event state as the readout computed here.
        Only this combined operation may supply an already checked readout.
        The input provider is queried after outputs, as in the original caller.
        """
        started = perf_counter()
        out = self.outputs(t, state, cone_release)
        output_seconds = perf_counter()-started
        started = perf_counter()
        committed, _, _ = self._commit_ticks(t, state, cone_release=cone_release,
            rod_light=rod_light_source(), rod_tick=True, bc_tick=True, _outputs=out)
        committed.update(self._probability_tick_updates(state))
        return out, committed, dict(outputs_seconds=output_seconds,
                                   transition_seconds=perf_counter()-started)

    def commit_ticks(self, t, state, *, cone_release=None, rod_light=None, rod_tick=False,
                     bc_tick=False, rgc_tick=False, spike_states=None):
        """Construct every update from the same pre-event snapshot, then commit.

        Returned arrays/auxiliary state are privately owned. No caller RNG or
        pending-event state changes on a failed/rejected attempt.
        """
        return self._commit_ticks(t, state, cone_release=cone_release, rod_light=rod_light,
            rod_tick=rod_tick, bc_tick=bc_tick, rgc_tick=rgc_tick, spike_states=spike_states)

    def _commit_ticks(self, t, state, *, cone_release=None, rod_light=None, rod_tick=False,
                      bc_tick=False, rgc_tick=False, spike_states=None, _outputs=None,_auxiliary_copy=deepcopy):
        self.validate(state)
        if not math.isfinite(t) or any(type(v) is not bool for v in (rod_tick, bc_tick, rgc_tick)):
            raise ContractError('Explicit original event time and tick flags required')
        if bc_tick and cone_release is None:
            raise ContractError('BC event requires the real current Cone release readout')
        if rgc_tick and spike_states is None:
            raise ContractError('RGC event requires original sampled-object histories and PCG64 states')
        before = state
        updates = {}
        if rod_tick:
            rate = _array(rod_light, self.counts['rods'], 'departing rod light', nonnegative=True)
            p = self.rod
            R, P, G, C, S = (before[key('rods', RODS, k)] for k in ('R', 'P', 'G', 'C', 'S'))
            dt = p['timeStep']
            new_R = R+dt*(-p['sigma']*R)
            new_R = new_R+p['gamma']*(rate*dt)
            new_P = P+dt*(R+p['eta']-p['phi']*P)
            new_C = C+dt*(p['q']*p['k']*G**p['n']-p['beta']*C)
            new_S = p['smax']/(1+(new_C/p['kGC'])**p['m'])
            new_G = G+dt*(S-P*G)
            values = (new_R, new_P, new_G, new_C, new_S)
            if (any(not np.isfinite(a).all() for a in values) or np.any(new_R < 0)
                    or any(np.any(a <= 0) for a in values[1:])):
                raise ContractError('Rod source trajectory invalid; no skipped ticks or clipping')
            updates.update({key('rods', RODS, n): value for n, value in zip(('R', 'P', 'G', 'C', 'S'), values)})
        out = (self.outputs(t, before, cone_release) if _outputs is None else _outputs) if bc_tick else None
        if bc_tick:
            for pop in BC:
                history = np.vstack((out[pop, 'drive']-self.par(pop, 'baseline'),
                                     self._bc(before, pop, 'history_0'), self._bc(before, pop, 'history_1')))
                kernel = np.asarray(self.p[pop]['kernel'], dtype=np.float64)
                sign = -1. if pop.endswith('_ON') or pop == 'S_ON_BC' else 1.
                if kernel[np.argmax(np.abs(kernel))] <= 0:
                    raise ContractError('Original canonical dominant-positive BC kernel required')
                activation = np.logaddexp(0., self.par(pop, 'bias')+sign*(kernel@history))
                updates.update({key(pop, BC_MECH, 'history_'+str(i)): history[i] for i in range(3)})
                updates[key(pop, BC_MECH, 'activation')] = activation
        auxiliary = _auxiliary_copy(spike_states) if spike_states is not None else None
        events = {}
        if rgc_tick:
            updates.update(self._probability_tick_updates(before))
            for pop in RGC:
                V = before[key(pop, RGC_MECH, 'V')]
                alpha, mu, beta = self.par(pop, 'alpha', positive=True), self.par(pop, 'mu'), self.par(pop, 'beta', positive=True)
                kernel = np.asarray(self.p[pop]['spike_history_kernel'], dtype=np.float64)
                if auxiliary is not None:
                    if pop not in auxiliary:
                        raise ContractError('Original RGC spike state must cover every sampled object')
                    proxy = SimpleNamespace(n=self.counts[pop], dt=.001,
                        history_kernel=kernel, intensity=lambda v, h, aa=alpha, mm=mu, bb=beta, kk=kernel: aa*np.logaddexp(0., (v+kk@h-mm)/bb))
                    auxiliary[pop], events[pop] = HistoryRGC.sample_spikes(proxy, t, V, auxiliary[pop])
        committed = {k: np.array(updates.get(k, state[k]), dtype=np.float64, copy=True) for k in self.state_keys}
        self.validate(committed)
        return committed, auxiliary, events

    def advance_reference(self, state, t0, t1, *, rtol, atol):
        """Joint same-equation CPU reference on one real-event-free interval."""
        if not math.isfinite(t0) or not math.isfinite(t1) or t0 >= t1:
            raise ContractError('Increasing original event-free interval required')
        started = perf_counter()
        initial = self.pack(state)
        def derivative(t, y):
            return self.compiled_rhs(t, y) if self.compiled_rhs is not None else self.pack(self.rhs(t, self.unpack(y, copy=False)))
        bind = getattr(self.compiled_rhs, '_reference_interval_rhs', None)
        if bind is not None:
            derivative = bind(initial)
        solver = DOP853(derivative, t0,
                        initial, t1, rtol=rtol, atol=atol, max_step=.0001)
        dense, steps = [], 0
        while solver.status == 'running':
            message = solver.step()
            if solver.status == 'failed':
                raise ContractError('Joint inner DOP853 failed: '+str(message))
            dense.append(solver.dense_output()); steps += 1
        final = self.unpack(solver.y)
        def evaluate(t, side):
            index = next((i for i, s in enumerate(dense) if s.t_old <= t <= s.t), None)
            if index is None:
                raise ContractError('Reference dense query unavailable')
            return self.unpack(np.asarray(dense[index](t), dtype=np.float64))
        return InnerAdvance(self, float(t0), float(t1), final, evaluate, {},
            dict(method='joint_DOP853_reference', steps=steps, nfev=solver.nfev,
                 wall_seconds=perf_counter()-started, max_step_seconds=.0001,
                 compiled_inner_rhs=self.compiled_rhs is not None),
            accepted_times=tuple(sorted({float(t0),float(t1),*(float(s.t) for s in dense)})),
            dense_segments=tuple(dense))

    def advance_candidate(self, state, t0, t1, *, rtol, atol):
        """Dependency-ordered continuous solve on ONE real-event-free interval.

        Rod current is held until its next original author event. True gap and
        voltage-dependent glycine feedback remain joint DOP853 blocks; known
        feedforward branches use one continuous exponential/integrating-factor
        expression at endpoints and all internal queries.
        """
        from .continuous_state import (AcceptedSeries, Dop853Dense, ExponentialForced,
                                       AffineMembraneState)
        from .semianalytic_time import fit_drive, exponential_error_bound
        from .multirate_contract import forcing_allowance
        self.validate(state)
        if (not math.isfinite(t0) or not math.isfinite(t1) or t0 >= t1
                or t1-t0 > .001 or not math.isfinite(rtol) or rtol <= 0
                or not np.isscalar(atol) or not math.isfinite(atol) or atol <= 0):
            raise ContractError('Candidate requires original scalar tolerances and event-free interval <=1ms')
        began = perf_counter()
        base = {k: value.copy() for k, value in state.items()}
        pieces = {k: [] for k in self.state_keys}
        metrics = dict(method='dependency_ordered_semianalytic_inner_candidate', nfev=0,
            fast_steps=0, fast_blocks=0, forcing_source_evaluations=0,
            forcing_wall_seconds=0., fast_wall_seconds=0., quadrature_wall_seconds=0.,
            forcing_error_max_original_scale=0., quadrature_error_max_original_scale=0.,
            max_step_seconds=.0001, compiled_inner_rhs=self.compiled_rhs is not None,
            compiled_fast_instruction_counts=[],
            local_error_shares=dict(time=.25, forcing=.25, coupling=.25, quadrature=.25))

        def side(t):
            return 'left' if t == t1 else 'right'

        def lookup(k, t, which=None):
            value = base[k].copy()
            for rows, representation in pieces[k]:
                value[rows] = representation.evaluate(t, side(t) if which is None else which)
            return value

        def install(k, representation, rows=None):
            rows = np.arange(self.sizes[k]) if rows is None else np.asarray(rows, dtype=np.int64)
            if any(np.intersect1d(rows, old_rows).size for old_rows, _ in pieces[k]):
                raise ContractError('Candidate attempted to replace an accepted state-row representation')
            pieces[k].append((rows, representation))

        def scale(k):
            return float(atol)+float(rtol)*np.abs(base[k])

        def linear(k, decay, source):
            started = perf_counter()
            decay = np.broadcast_to(np.asarray(decay, dtype=np.float64), base[k].shape)
            q_budget = forcing_allowance(scale(k)*.25, decay, t1-t0)
            approximation = fit_drive(source, t0, t1, forcing_budget=q_budget, unit='original_state_unit/s')
            estimated = exponential_error_bound(decay, t1-t0, approximation.estimated_max_error)
            normalized = float(np.max(estimated/scale(k), initial=0.))
            if normalized > .25:
                raise ContractError('Inner forcing estimate exceeds its non-transferable original-scale quarter')
            metrics['forcing_error_max_original_scale'] = max(metrics['forcing_error_max_original_scale'], normalized)
            anchor = base[k].copy()
            representations = []
            for drive in approximation.segments:
                represented = ExponentialForced(t0=drive.t0, t1=drive.t1,
                    state_ids=np.arange(len(anchor)), y0=anchor,
                    coefficients=drive.coefficients, decay=decay)
                anchor = represented.evaluate(drive.t1, 'left')
                representations.append(represented)
            install(k, AcceptedSeries(representations))
            metrics['forcing_source_evaluations'] += approximation.source_evaluations
            metrics['forcing_wall_seconds'] += perf_counter()-started

        def constant_affine(k, decay, target, rows=None):
            selected = np.arange(self.sizes[k]) if rows is None else np.asarray(rows, dtype=np.int64)
            if not len(selected):
                return
            coefficients = np.zeros((8, len(selected)))
            d = np.broadcast_to(np.asarray(decay, dtype=np.float64), base[k].shape)[selected]
            value = np.broadcast_to(np.asarray(target, dtype=np.float64), base[k].shape)[selected]
            coefficients[0] = d*value
            install(k, ExponentialForced(t0=t0, t1=t1, state_ids=selected,
                y0=base[k][selected], coefficients=coefficients, decay=d), selected)

        def joint(specs, derivative):
            specs = [(k, np.asarray(rows, dtype=np.int64)) for k, rows in specs if len(rows)]
            if not specs:
                return
            started = perf_counter()
            slices, offset = [], 0
            for k, rows in specs:
                slices.append(slice(offset, offset+len(rows))); offset += len(rows)
            initial = np.concatenate([base[k][rows] for k, rows in specs])
            compiled = self.compiled_rhs
            if compiled is not None and hasattr(compiled, 'for_keys'):
                compiled = compiled.for_keys(tuple(k for k, _ in specs))
                metrics['compiled_fast_instruction_counts'].append(dict(compiled.metrics))
            required = getattr(compiled, 'required_state_keys', self.state_keys)
            packed_base = self.pack(base) if compiled is not None else None
            def fun(t, y):
                local = {}
                for (k, rows), section in zip(specs, slices):
                    value = lookup(k, t)
                    value[rows] = y[section]
                    local[k] = value
                if compiled is None:
                    values = derivative(t, local)
                else:
                    # The full compiled original RHS remains an independent
                    # execution path. Only derivatives of this dependency-
                    # closed fast block are consumed; unrelated held/future
                    # derivatives cannot feed it under the audited partition.
                    packed = packed_base.copy()
                    for k in required:
                        packed[self.slices[k]] = local[k] if k in local else lookup(k, t)
                    values = self.unpack(compiled(t, packed), copy=False)
                answer = np.concatenate([values[k][rows] for k, rows in specs])
                if not np.isfinite(answer).all():
                    raise ContractError('Nonfinite original fast inner equation')
                return answer
            solver = DOP853(fun, t0, initial, t1, rtol=rtol*.25, atol=atol*.25, max_step=.0001)
            accepted = [[] for _ in specs]
            while solver.status == 'running':
                message = solver.step()
                if solver.status == 'failed':
                    raise ContractError('Candidate joint inner block failed: '+str(message))
                dense = solver.dense_output()
                for i, (((_, rows), section)) in enumerate(zip(specs, slices)):
                    accepted[i].append(Dop853Dense(t0=dense.t_old, t1=dense.t,
                        state_ids=rows, y0=dense.y_old[section], coefficients=dense.F[:, section]))
                metrics['fast_steps'] += 1
            for (k, rows), segments in zip(specs, accepted):
                install(k, AcceptedSeries(segments), rows)
            metrics['nfev'] += solver.nfev
            metrics['fast_blocks'] += 1
            metrics['fast_wall_seconds'] += perf_counter()-started

        def event_rate(pop, t, local=None):
            local = {} if local is None else local
            vk, xk = key(pop, MEMBRANE, 'V'), key(pop, MEMBRANE, 'x')
            V = local[vk] if vk in local else lookup(vk, t)
            x = local[xk] if xk in local else lookup(xk, t)
            return self.par(pop, 'Npool', positive=True)*self.par(pop, 'U', positive=True)*self.par(pop, 'rmax', positive=True)*expit((V-self.par(pop, 'r_half'))/self.par(pop, 'r_slope', positive=True))*x

        def pooled_contact(component, t, *, current=None):
            c = self.contacts[component]
            if component == 'individual_rod_RBC_contacts':
                conductance = c['coefficient']*lookup(key(c['route'].pre, component, 'm'), t)
            else:
                a = lookup(key(c['route'].pre, component, 'a'), t)
                b = lookup(key(c['route'].pre, component, 'b'), t)
                if current is None:
                    rise = self.cp(component, 'tau_rise', positive=True)
                    decay = self.cp(component, 'tau_decay', positive=True)
                else:
                    rise, decay = current
                tp = np.log(decay/rise)/(1/rise-1/decay)
                norm = np.exp(-tp/decay)-np.exp(-tp/rise)
                conductance = c['coefficient']*(b-a)/norm
            if np.any(conductance < -1e-18):
                raise ContractError('Negative continuous chemical conductance; no clipping')
            return np.bincount(c['post'], weights=conductance, minlength=self.counts[c['route'].post])

        # Author R/P/G/C/S stay held. Rod current and fixed leak give an exact
        # membrane solution over this one author interval.
        vk = key('rods', 'rod_membrane', 'V')
        current = self.rod_current(base)
        target = self.par('rods', 'EL')-(current-self.par('rods', 'current_anchor'))/self.par('rods', 'gL', positive=True)
        constant_affine(vk, self.gL['rods']/self.C['rods'], target)
        rk = key('rods', 'rod_membrane', 'release')
        tau = self.par('rods', 'tau_release', positive=True)
        linear(rk, 1/tau, lambda t, s: expit((lookup(vk, t, s)-self.par('rods', 'release_half'))/self.par('rods', 'release_slope', positive=True))/tau)
        c = self.contacts['individual_rod_RBC_contacts']
        mk = key('rods', 'individual_rod_RBC_contacts', 'm')
        tau_m = self.cp('individual_rod_RBC_contacts', 'tau', positive=True)
        def rod_gate(t, s):
            drive = lookup(rk, t, s)[c['pre']]
            return expit(-self.cp('individual_rod_RBC_contacts', 'hill', positive=True)*np.log(np.maximum(drive, np.finfo(float).tiny)/self.cp('individual_rod_RBC_contacts', 'K', positive=True)))/tau_m
        linear(mk, 1/tau_m, rod_gate)

        # RBC V/release/resource remain a coupled source equation block.
        rbc_keys = [key('rod_BC', MEMBRANE, n) for n in ('V', 'release', 'x')]
        def rbc_rhs(t, local):
            V, rel, x = (local[k] for k in rbc_keys)
            current = self.gL['rod_BC']*(self.par('rod_BC', 'EL')-V)+pooled_contact('individual_rod_RBC_contacts', t)*(self.par('rod_BC', 'EE')-V)
            attempts = self.par('rod_BC', 'rmax', positive=True)*expit((V-self.par('rod_BC', 'r_half'))/self.par('rod_BC', 'r_slope', positive=True))
            return {rbc_keys[0]: current/self.C['rod_BC'],
                rbc_keys[1]: (expit((V-self.par('rod_BC', 'release_half'))/self.par('rod_BC', 'release_slope', positive=True))-rel)/self.par('rod_BC', 'tau_release', positive=True),
                rbc_keys[2]: (1-x)/self.par('rod_BC', 'tau_recovery', positive=True)-self.par('rod_BC', 'U', positive=True)*attempts*x}
        joint([(k, np.arange(self.sizes[k])) for k in rbc_keys], rbc_rhs)
        c = self.contacts['RBC_AII_AMPA']
        for name, tau_name in (('a', 'tau_rise'), ('b', 'tau_decay')):
            tau = self.cp('RBC_AII_AMPA', tau_name, positive=True)
            linear(key('rod_BC', 'RBC_AII_AMPA', name), 1/tau,
                   lambda t, s, rows=c['pre']: event_rate('rod_BC', t)[rows])

        # Isolated BC membrane rows have no gap or glycine return. Their held
        # generator is fixed until the next actual BC tick, so affine is exact.
        coupled_masks = {}
        for pop in BC:
            if pop in self.gap_slices:
                lap = self.gap[self.gap_slices[pop], :]
                mask = np.asarray(lap.getnnz(axis=1)).ravel() > 0
            else:
                component = 'AII_FMB_glycine' if pop == 'midget_BC_OFF' else 'AII_DB1_glycine'
                cc = self.contacts[component]
                mask = np.bincount(cc['post'], weights=cc['coefficient'], minlength=self.counts[pop]) > 0
            coupled_masks[pop] = mask
            bk = key(pop, BC_MECH, 'V')
            target = self.par(pop, 'Vref')+self.par(pop, 'native_voltage_gain', positive=True)*(base[key(pop, BC_MECH, 'activation')]-self.par(pop, 'native_reference'))
            constant_affine(bk, self.gL[pop]/self.C[pop], target, np.flatnonzero(~mask))

        on = [n for n in self.gap_names if n != 'AII_AC']
        aii_keys = [key('AII_AC', MEMBRANE, n) for n in ('V', 'release', 'x')]
        on_keys = [key(n, BC_MECH, 'V') for n in on]
        def aii_rhs(t, local):
            V, rel, x = (local[k] for k in aii_keys)
            voltages = np.concatenate([V if n == 'AII_AC' else local.get(key(n, BC_MECH, 'V'), lookup(key(n, BC_MECH, 'V'), t)) for n in self.gap_names])
            gap_current = -(self.gap@voltages)
            current = self.gL['AII_AC']*(self.par('AII_AC', 'EL')-V)+gap_current[self.gap_slices['AII_AC']]+pooled_contact('RBC_AII_AMPA', t)*(self.par('AII_AC', 'EE')-V)
            attempts = self.par('AII_AC', 'rmax', positive=True)*expit((V-self.par('AII_AC', 'r_half'))/self.par('AII_AC', 'r_slope', positive=True))
            result = {aii_keys[0]: current/self.C['AII_AC'],
                aii_keys[1]: (expit((V-self.par('AII_AC', 'release_half'))/self.par('AII_AC', 'release_slope', positive=True))-rel)/self.par('AII_AC', 'tau_release', positive=True),
                aii_keys[2]: (1-x)/self.par('AII_AC', 'tau_recovery', positive=True)-self.par('AII_AC', 'U', positive=True)*attempts*x}
            for pop in on:
                k = key(pop, BC_MECH, 'V'); value = local.get(k, lookup(k, t))
                target = self.par(pop, 'Vref')+self.par(pop, 'native_voltage_gain', positive=True)*(base[key(pop, BC_MECH, 'activation')]-self.par(pop, 'native_reference'))
                result[k] = (self.gL[pop]*(target-value)+gap_current[self.gap_slices[pop]])/self.C[pop]
            return result
        joint([(k, np.arange(self.sizes[k])) for k in aii_keys]+
              [(k, np.flatnonzero(coupled_masks[k[0]])) for k in on_keys], aii_rhs)

        for component in ('AII_FMB_glycine', 'AII_DB1_glycine'):
            c = self.contacts[component]; pop = c['route'].post
            ak, bk = key('AII_AC', component, 'a'), key('AII_AC', component, 'b')
            rise = self.cp(component, 'tau_rise', positive=True)
            linear(ak, 1/rise, lambda t, s, rows=c['pre']: event_rate('AII_AC', t)[rows])
            vk = key(pop, BC_MECH, 'V')
            def gly_rhs(t, local, cc=c, pp=pop, comp=component, aa=ak, bb=bk, vv=vk, rr=rise):
                V = local.get(vv, lookup(vv, t)); b = local[bb]
                decay = self.cp(comp, 'decay_minus100', positive=True)+(self.cp(comp, 'decay_0', positive=True)-self.cp(comp, 'decay_minus100', positive=True))*np.minimum(1., np.maximum(0., (V[cc['post']]+.1)/.1))
                tp = np.log(decay/rr)/(1/rr-1/decay)
                norm = np.exp(-tp/decay)-np.exp(-tp/rr)
                g = cc['coefficient']*(b-lookup(aa, t))/norm
                if np.any(g < -1e-18):
                    raise ContractError('Negative current glycine conductance; no clipping')
                pooled = np.bincount(cc['post'], weights=g, minlength=self.counts[pp])
                target = self.par(pp, 'Vref')+self.par(pp, 'native_voltage_gain', positive=True)*(base[key(pp, BC_MECH, 'activation')]-self.par(pp, 'native_reference'))
                return {bb: -b/decay+event_rate('AII_AC', t)[cc['pre']],
                    vv: (self.gL[pp]*(target-V)+pooled*(self.par(pp, 'EI')-V))/self.C[pp]}
            joint([(bk, np.arange(self.sizes[bk])), (vk, np.flatnonzero(coupled_masks[pop]))], gly_rhs)

        for pop in BC:
            vk, rk, ak = (key(pop, BC_MECH, n) for n in ('V', 'terminal_release', 'A'))
            tau = self.par(pop, 'tau_release', positive=True)
            linear(rk, 1/tau, lambda t, s, pp=pop, kk=vk, tt=tau: expit((lookup(kk, t, s)-self.par(pp, 'release_half'))/self.par(pp, 'release_slope', positive=True))/tt)
            if self.p[pop]['adaptation_enabled']:
                tau_a = self.par(pop, 'tau_a_seconds', positive=True)
                linear(ak, 1/tau_a, lambda t, s, pp=pop, kk=rk, tt=tau_a: self.par(pp, 'adaptation_scale', nonnegative=True)*lookup(kk, t, s)/tt)

        def effective(pop, t, which):
            release = lookup(key(pop, BC_MECH, 'terminal_release'), t, which)
            return release/(1+lookup(key(pop, BC_MECH, 'A'), t, which)**6) if self.p[pop]['adaptation_enabled'] else release

        for pop in AC:
            incoming = [r for r in self.routes if r.post == pop]
            tau = self.par(pop, 'tau_A_seconds', positive=True)
            def ac_drive(t, s, pp=pop, routes=incoming, tt=tau):
                drive = sum((self.mean[r.name]@effective(r.pre, t, s) for r in routes), start=np.zeros(self.counts[pp]))
                return expit(self.par(pp, 'input_slope', positive=True)*(drive-self.par(pp, 'reference_input')))/tt
            linear(key(pop, AC_MECH, 'h'), 1/tau, ac_drive)

        for pop in RGC:
            vk = key(pop, RGC_MECH, 'V')
            routes = [r for r in self.routes if r.post == pop]
            def membrane_coefficients(t, s, pp=pop, incoming=routes):
                ge, gi = np.zeros(self.counts[pp]), np.zeros(self.counts[pp])
                for r in incoming:
                    activation = self.par(r.pre, 'output_scale', positive=True)*lookup(key(r.pre, AC_MECH, 'h'), t, s) if r.pre in AC else effective(r.pre, t, s)
                    gain = _array(self.parameters['route_gains'][r.name], self.counts[pp], r.name, nonnegative=True)
                    value = self.members[pp]*gain*(self.mean[r.name]@activation)
                    if r.pre in AC: gi += value
                    else: ge += value
                return ((self.gL[pp]*self.par(pp, 'EL')+ge*self.par(pp, 'EE')+gi*self.par(pp, 'EI'))/self.C[pp],
                        (self.gL[pp]+ge+gi)/self.C[pp])
            # Positive physical conductances imply V stays within its initial
            # value and reversal envelope; this bounds coefficient forcing in V.
            envelope = np.maximum.reduce([np.abs(base[vk]), np.abs(self.par(pop, 'EL')),
                                           np.abs(self.par(pop, 'EE')), np.abs(self.par(pop, 'EI'))])
            y_allowance = scale(vk)*.25
            q_budget = y_allowance/(t1-t0)
            started = perf_counter()
            a_fit = fit_drive(lambda t, s: membrane_coefficients(t, s)[0], t0, t1,
                forcing_budget=q_budget*.5, unit='V/s')
            b_fit = fit_drive(lambda t, s: membrane_coefficients(t, s)[1], t0, t1,
                forcing_budget=q_budget*.5/np.maximum(envelope, np.finfo(float).tiny), unit='1/s')
            estimated = (t1-t0)*(a_fit.estimated_max_error+envelope*b_fit.estimated_max_error)
            normalized = float(np.max(estimated/scale(vk), initial=0.))
            if normalized > .25:
                raise ContractError('RGC coefficient forcing estimate exceeds its original-scale quarter')
            metrics['forcing_error_max_original_scale'] = max(metrics['forcing_error_max_original_scale'], normalized)
            metrics['forcing_source_evaluations'] += a_fit.source_evaluations+b_fit.source_evaluations
            metrics['forcing_wall_seconds'] += perf_counter()-started
            # Fits may choose different intervals; refit their polynomials on
            # the common exact breakpoints, without querying the real drive anew.
            boundaries = sorted({t0, t1, *(d.t0 for d in a_fit.segments), *(d.t1 for d in a_fit.segments),
                                 *(d.t0 for d in b_fit.segments), *(d.t1 for d in b_fit.segments)})
            def coefficient_piece(fit, start, stop):
                drive = next(d for d in fit.segments if d.t0 <= start < stop <= d.t1)
                x = .5*(1-np.cos(np.arange(8)*np.pi/7))
                times = start+(stop-start)*x
                values = np.stack([drive.evaluate(t, 'left' if t == drive.t1 else 'right') for t in times])
                actual_x = (times-start)/(stop-start)
                return np.linalg.solve(np.vander(actual_x, 8, increasing=True), values)
            anchor, represented = base[vk].copy(), []
            quadrature_error = np.zeros_like(anchor)
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                representation = AffineMembraneState(t0=start, t1=stop, state_ids=np.arange(len(anchor)), y0=anchor,
                    a_coefficients=coefficient_piece(a_fit, start, stop), b_coefficients=coefficient_piece(b_fit, start, stop),
                    quadrature_budget=scale(vk)*.25*((stop-start)/(t1-t0)), state_unit='V')
                started = perf_counter()
                anchor, error = representation.evaluate_with_error(stop, 'left')
                quadrature_error += error
                metrics['quadrature_wall_seconds'] += perf_counter()-started
                represented.append(representation)
            normalized = float(np.max(quadrature_error/scale(vk), initial=0.))
            if normalized > .25:
                raise ContractError('RGC quadrature estimate exceeds its non-transferable original-scale quarter')
            metrics['quadrature_error_max_original_scale'] = max(metrics['quadrature_error_max_original_scale'], normalized)
            install(vk, AcceptedSeries(represented))

        def evaluate(t, which):
            return {k: lookup(k, t, which) for k in self.state_keys}
        final = evaluate(t1, 'left')
        self.validate(final)
        if self.compiled_rhs is None:
            self.rhs(t1, final)
        else:
            self.compiled_rhs(t1, self.pack(final))
        # This is the full original equation/domain check, never the derivative
        # of the represented forcing. NumPy rhs_at remains independently usable.
        metrics['wall_seconds'] = perf_counter()-began
        metrics['other_wall_seconds'] = metrics['wall_seconds']-metrics['fast_wall_seconds']-metrics['forcing_wall_seconds']-metrics['quadrature_wall_seconds']
        metrics['representation_segments'] = sum(len(series.representations) for entries in pieces.values() for _, series in entries if hasattr(series, 'representations'))
        accepted_times={float(t0),float(t1)}
        for entries in pieces.values():
            for _, series in entries:
                for representation in getattr(series,'representations',(series,)):
                    accepted_times.update((float(representation.t0),float(representation.t1)))
        return InnerAdvance(self, float(t0), float(t1), final, evaluate,
                            {k: tuple(v) for k, v in pieces.items() if v}, metrics,
                            accepted_times=tuple(sorted(accepted_times)))
