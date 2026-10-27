"""Candidate time policy and transactional controls, without a model runner.

The four local error shares use ONE caller-supplied original tolerance scale.
This module does not select parameters, initialize a retina or authorize B.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np

from .contracts import ContractError, canonical_hash


PARTS = ('time', 'forcing', 'coupling', 'quadrature')


def forcing_allowance(state_allowance, decay, duration, *, initial_error=0.):
    """Convert a y-unit allocated share to a q-unit uniform driver allowance.

    Valid at every internal time for a constant state-error allowance. An
    upstream initial error already exceeding the share is rejected, not erased.
    """
    allowed, rate, initial = np.broadcast_arrays(np.asarray(state_allowance, dtype=np.float64),
        np.asarray(decay, dtype=np.float64), np.asarray(initial_error, dtype=np.float64))
    if (not math.isfinite(duration) or duration <= 0 or not np.isfinite(allowed).all()
            or not np.isfinite(rate).all() or not np.isfinite(initial).all()
            or np.any(allowed <= 0) or np.any(rate < 0) or np.any(initial < 0) or np.any(initial > allowed)):
        raise ContractError('Positive state allowance and duration with admissible propagated initial error required')
    with np.errstate(over='raise', invalid='raise'):
        try:
            exponent = -rate*duration
        except FloatingPointError as exc:
            raise ContractError('Decay-time product outside FP64 domain') from exc
    gain = np.full(allowed.shape, duration, dtype=np.float64)
    np.divide(-np.expm1(exponent), rate, out=gain, where=rate > 0)
    available = (allowed-initial)-np.expm1(exponent)*initial
    result = available/gain
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ContractError('No positive driver approximation allowance remains')
    return result


@dataclass(frozen=True)
class MultiratePolicy:
    fast_max_step_seconds: float = .0001
    initial_slow_window_seconds: float = .0005
    max_slow_window_seconds: float = .001
    minimum_coupling_window_seconds: float = .0001
    max_corrections: int = 3
    polynomial_degree: int = 7
    gauss_orders: tuple = (8, 16)

    def __post_init__(self):
        if (self.fast_max_step_seconds != .0001 or self.initial_slow_window_seconds != .0005
                or self.max_slow_window_seconds != .001 or self.minimum_coupling_window_seconds != .0001
                or type(self.max_corrections) is not int or self.max_corrections != 3
                or type(self.polynomial_degree) is not int or self.polynomial_degree != 7
                or self.gauss_orders != (8, 16)):
            raise ContractError('This candidate contract fixes the supplied v0.2 policy; no hidden retuning')

    def describe(self):
        return dict(schema='retipath-multirate-policy-1', classification='B',
            default_adopted=False, numerical_acceptance=False, policy=asdict(self),
            algorithm='DOP853 fast + forced exponential + integrating factor + waveform coupling',
            local_error_shares={k: .25 for k in PARTS},
            norm='WRMS per explicitly active block; held states excluded, no budget borrowing',
            fallback='At 100us coupling failure, joint Vis/gi DOP853 for that interval only; domain/acceptance failures stop',
            input_boundaries='exact original frame times, outgoing left input then incoming right input',
            observation_delay_seconds=.00282,
            observation_interval_seconds=.0005,
            monitor_rhs='full original equation, never representation derivative',
            existing_optimizations_not_counted_as_new=['analytic R/E', 'w=A z1', 'necessary-node bridge', 'native not restarted by unrelated rod ticks'])


class LocalErrorAllocation:
    """Explicit non-transferable quarters of the existing local error scale.

    The caller supplies the actual frozen block tolerances, not SolverConfig's
    unrelated defaults. This local estimator is not an A/B or long-time proof.
    """
    def __init__(self, y0, y1, *, rtol, atol, active):
        a, b = np.asarray(y0, dtype=np.float64), np.asarray(y1, dtype=np.float64)
        mask = np.asarray(active)
        at = np.broadcast_to(np.asarray(atol, dtype=np.float64), a.shape)
        if (a.ndim != 1 or b.shape != a.shape or mask.shape != a.shape or mask.dtype.kind != 'b'
                or not np.isfinite(a).all() or not np.isfinite(b).all()
                or not math.isfinite(rtol) or rtol <= 0 or not np.isfinite(at).all() or np.any(at <= 0)):
            raise ContractError('Original positive tolerances, finite state and explicit active mask required')
        with np.errstate(over='ignore', invalid='ignore'):
            self.scale = at+rtol*np.maximum(np.abs(a), np.abs(b))
        if not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ContractError('Computed original tolerance scale must remain finite and positive')
        self.scale.setflags(write=False)
        self.active = np.array(mask, copy=True)
        self.active.setflags(write=False)

    def allowance(self, part):
        if part not in PARTS:
            raise ContractError('Unknown local error allocation')
        return self.scale*.25

    def assess(self, errors):
        if set(errors) != set(PARTS):
            raise ContractError('All four error sources must be accounted for explicitly')
        norms = {}
        for part in PARTS:
            e = np.asarray(errors[part], dtype=np.float64)
            if e.shape != self.scale.shape or not np.isfinite(e).all():
                raise ContractError('Finite original-layout error vector required: '+part)
            z = e[self.active]/self.scale[self.active]
            norms[part] = float(np.linalg.norm(z)/math.sqrt(len(z))) if len(z) else 0.
        return dict(schema='retipath-local-error-allocation-1', normalized_error=norms,
            quarter_satisfied={k: v <= .25 for k, v in norms.items()},
            accepted=all(v <= .25 for v in norms.values()),
            total_triangle_estimate=sum(norms.values()),
            scope='local finite-estimator decision only; original frozen A/B budgets still required')


@dataclass(frozen=True)
class CalendarEvent:
    time_seconds: float
    kind: str
    affected_blocks: tuple
    event_id: str

    def __post_init__(self):
        if (not math.isfinite(self.time_seconds) or not self.kind or not self.event_id
                or not isinstance(self.affected_blocks, tuple) or not self.affected_blocks
                or any(not isinstance(x, str) or not x for x in self.affected_blocks)):
            raise ContractError('Exact finite event time and affected block identities required')


class BlockEventCalendar:
    """Explicit real events, not every observation nor a global 100us grid.

    Event times come from the original clock/input manifests. This object never
    snaps neighboring floats together or dictates model commit order.
    """
    def __init__(self, events):
        self.events = tuple(sorted(events, key=lambda e: (e.time_seconds, e.event_id)))
        if any(not isinstance(e, CalendarEvent) for e in self.events):
            raise ContractError('Typed original events required')
        if len({e.event_id for e in self.events}) != len(self.events):
            raise ContractError('Unique original event IDs required')

    def next_boundary(self, block, start, stop):
        if not all(math.isfinite(x) for x in (start, stop)) or start >= stop:
            raise ContractError('Finite increasing candidate interval required')
        return min((e.time_seconds for e in self.events
                    if block in e.affected_blocks and start < e.time_seconds <= stop), default=stop)

    def at(self, time_seconds):
        if not math.isfinite(time_seconds):
            raise ContractError('Finite exact event query required')
        return tuple(e for e in self.events if e.time_seconds == time_seconds)


class PCG64Transaction:
    """Tentative event draws are private until an accepted numerical interval.

    Only the original sampler owns ordering, updates and pending-event meaning;
    this helper prevents a rejected attempt from consuming its RNG state.
    """
    def __init__(self, generator):
        if not isinstance(generator, np.random.Generator) or type(generator.bit_generator) is not np.random.PCG64:
            raise ContractError('Original PCG64 generator required; no seed substitution')
        self.original = generator
        self.before = deepcopy(generator.bit_generator.state)
        self.rng = np.random.Generator(np.random.PCG64(0))
        self.rng.bit_generator.state = deepcopy(self.before)
        self.pending_events = []
        self.closed = False

    def stage(self, event):
        if self.closed:
            raise ContractError('Event transaction is already closed')
        self.pending_events.append(deepcopy(event))

    def commit(self):
        if self.closed or canonical_hash(self.original.bit_generator.state) != canonical_hash(self.before):
            raise ContractError('Event transaction closed or original RNG advanced concurrently')
        self.original.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        self.closed = True
        return tuple(deepcopy(self.pending_events))

    def reject(self):
        if self.closed:
            raise ContractError('Event transaction is already closed')
        self.closed = True
        self.pending_events.clear()
