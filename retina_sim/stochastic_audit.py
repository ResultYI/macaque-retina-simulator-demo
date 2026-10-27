"""CPU-only witnesses for the existing HistoryRGC Bernoulli sampling path.

The formal method is called once. Uniforms are replayed using a separate PCG64
loaded from a deep copy of the incoming state. No formal RNG, comparator, rate,
event time, pending queue, or mechanism is replaced. This is evidence capture;
probability-error intervals and numerical acceptance are deliberately external.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
import hashlib
import inspect
import threading
import time

import numpy as np

from .contracts import ContractError, canonical_hash
from .inner import HistoryRGC

_PATCH_LOCK = threading.Lock()


def _array(value, dtype=np.float64):
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _float_bits(value):
    return int(np.float64(value).view(np.uint64))


def bernoulli_comparator(uniforms, probabilities):
    """QA replay of the literal strict u<p; no clipping/rounding or tolerance."""
    u, p = np.asarray(uniforms), np.asarray(probabilities)
    if u.dtype != np.float64 or p.dtype != np.float64 or u.shape != p.shape:
        raise ContractError('Stochastic witness requires matching FP64 uniform/probability vectors')
    if (not np.all(np.isfinite(u)) or not np.all(np.isfinite(p))
            or np.any(u < 0.) or np.any(u >= 1.) or np.any(p < 0.) or np.any(p > 1.)):
        raise ContractError('Stochastic witness uniform/probability domain violated')
    return (u < p).astype(np.int64)


def _events_bits(events):
    """Preserve event order and distinguish every float bit, including +/-0."""
    return tuple((tuple(sorted(event)), tuple((key, ('float64_bits', _float_bits(value))
        if isinstance(value, (float, np.floating)) else deepcopy(value))
        for key, value in sorted(event.items()))) for event in events)


@dataclass(frozen=True)
class SpikeWitness:
    arm: str
    component_id: str
    cell_ids: tuple
    t: float
    dt: float
    V: np.ndarray
    effective_history: np.ndarray
    rate_Hz: np.ndarray
    p: np.ndarray
    u: np.ndarray
    counts: np.ndarray
    replay_counts: np.ndarray
    events: tuple
    rng_before: dict
    rng_after: dict
    replay_rng_after: dict
    draw_count: int
    aux_before: dict
    aux_after: dict
    replay_verified: bool
    capture_wall_seconds: float
    acceptance_status: str = 'WITNESS_ONLY_PROBABILITY_BUDGET_NOT_EVALUATED'


def _strict_payload(witness):
    # Continuous V/rate/p values remain witnesses for the external probability
    # budget. Event times, integer histories and queue stamps have no waiver.
    return dict(component_id=witness.component_id, cell_ids=witness.cell_ids,
        t_bits=_float_bits(witness.t), dt_bits=_float_bits(witness.dt),
        rng_before=witness.rng_before, rng_after=witness.rng_after,
        draw_count=witness.draw_count, u_bits=witness.u.view(np.uint64),
        counts=witness.counts,
        history_before=witness.aux_before['history'],
        previous_counts=witness.aux_before['last_bin_counts'],
        history_after=witness.aux_after['history'],
        formal_counts=witness.aux_after['last_bin_counts'],
        events=_events_bits(witness.events),
        pending_before=_events_bits(witness.aux_before.get('pending_events', [])),
        pending_after=_events_bits(witness.aux_after.get('pending_events', [])),
        observation_scheme=witness.aux_after.get('observation_scheme'),
        replay_verified=witness.replay_verified)


def compare_witnesses(left, right):
    """Strict causal/discrete comparison only; never numerical acceptance."""
    failures = []
    if len(left) != len(right):
        failures.append(dict(field='sample_call_count', left=len(left), right=len(right)))
    for index, (a, b) in enumerate(zip(left, right)):
        aa, bb = _strict_payload(a), _strict_payload(b)
        for field in aa:
            if canonical_hash(aa[field]) != canonical_hash(bb[field]):
                failures.append(dict(index=index, component_id=a.component_id,
                    t=a.t, t_bits_hex=f'{_float_bits(a.t):016x}', field=field))
        if not a.replay_verified or not b.replay_verified:
            failures.append(dict(index=index, field='replay_does_not_match_formal_sample'))
    return dict(strict_equal=not failures, failures=failures,
        compared_calls=min(len(left), len(right)),
        numerical_or_probability_budget_evaluated=False,
        acceptance_status='STRICT_DISCRETE_WITNESSES_ONLY_NOT_BACKEND_ACCEPTANCE',
        event_time_tolerance_seconds=None)


class StochasticAuditSession:
    """Temporarily wrap compiled HistoryRGC instances and collect per-arm bins.

    ``integration_audit`` may be an AuditSession; its current arm is inherited
    whenever this session has no explicit label. ``witnesses[label]`` is a list
    of SpikeWitness objects. Call ``clear`` between streamed chunks. All labels
    are supported; no CPU/GPU backend is selected or imported by this module.
    """
    def __init__(self, compiled, *, integration_audit=None, active_label=None):
        self.compiled = (compiled,) if hasattr(compiled, 'components') else tuple(compiled)
        self.integration_audit = integration_audit
        self.active_label = active_label
        self.witnesses = {}
        self.errors = []
        self._restore = []
        self._entered = False
        self.source_sha256 = hashlib.sha256(inspect.getsource(HistoryRGC.sample_spikes).encode()).hexdigest()

    def __enter__(self):
        if self._entered or not _PATCH_LOCK.acquire(blocking=False):
            raise ContractError('Stochastic audit contexts cannot overlap')
        try:
            self._owner_thread = threading.get_ident()
            seen = {}
            for compiled in self.compiled:
                for cid, component in compiled.components.items():
                    model = component.native_model
                    if not hasattr(model, 'sample_spikes'):
                        continue
                    if type(model) is not HistoryRGC:
                        raise ContractError('Stochastic replay is implemented only for the original HistoryRGC')
                    if id(model) in seen:
                        if seen[id(model)] != cid:
                            raise ContractError('Shared stochastic model has ambiguous component identity')
                        continue
                    seen[id(model)] = cid
                    original = model.sample_spikes
                    existed = 'sample_spikes' in vars(model)
                    own_value = vars(model).get('sample_spikes')
                    self._restore.append((model, existed, own_value))
                    model.sample_spikes = self._wrapper(cid, model, original)
            self._entered = True
            return self
        except BaseException:
            self._restore_methods()
            _PATCH_LOCK.release()
            raise

    def _restore_methods(self):
        for model, existed, own_value in reversed(self._restore):
            if existed:
                model.sample_spikes = own_value
            else:
                del model.sample_spikes
        self._restore.clear()

    def __exit__(self, exc_type, exc, tb):
        try:
            self._restore_methods()
        finally:
            self._entered = False
            _PATCH_LOCK.release()

    @contextmanager
    def arm(self, label):
        if not isinstance(label, str) or not label:
            raise ContractError('Stochastic audit arm requires a nonempty label')
        previous = self.active_label
        self.active_label = label
        try:
            yield self
        finally:
            self.active_label = previous

    def clear(self, arm=None):
        if arm is None:
            self.witnesses.clear()
            self.errors.clear()
        else:
            self.witnesses.pop(arm, None)
            self.errors[:] = [error for error in self.errors if error['arm'] != arm]

    def strict_compare(self, left_arm, right_arm):
        result = compare_witnesses(self.witnesses.get(left_arm, []), self.witnesses.get(right_arm, []))
        errors = [error for error in self.errors if error['arm'] in (left_arm, right_arm)]
        if errors:
            result['strict_equal'] = False
            result['failures'].append(dict(field='formal_sampling_exception', errors=errors))
        return result

    def _wrapper(self, cid, model, original):
        @wraps(original)
        def sample(t, V, aux):
            label = self.active_label
            if label is None and self.integration_audit is not None:
                label = self.integration_audit.active_label
            if (not self._entered or threading.get_ident() != self._owner_thread
                    or not isinstance(label, str) or not label):
                raise ContractError('Active stochastic audit arm and owning thread required')
            before, voltage = deepcopy(aux), _array(V)
            # The original formal call is exactly once and receives its exact
            # original arguments. Every subsequent operation owns QA copies.
            try:
                updated, events = original(t, V, aux)
            except BaseException as error:
                self.errors.append(dict(arm=label, component_id=cid, t=float(t),
                    error_type=type(error).__name__, reason=str(error)))
                raise
            tick = time.perf_counter()
            if before['rng_state'].get('bit_generator') != 'PCG64':
                raise ContractError('Stochastic witness requires the original explicit PCG64 state')
            replay = np.random.Generator(np.random.PCG64())
            replay.bit_generator.state = deepcopy(before['rng_state'])
            uniforms = replay.random(model.n)
            if not len(model.history_kernel):
                history = np.empty((0, model.n))
            elif before['last_bin_counts'] is None:
                history = np.asarray(before['history'])
            else:
                history = np.vstack((np.asarray(before['last_bin_counts'])[None, :],
                    np.asarray(before['history'])[:-1]))
            rate = model.intensity(voltage, history)
            probability = -np.expm1(-rate*model.dt)
            replay_counts = bernoulli_comparator(uniforms, probability)
            after = deepcopy(updated)
            verified = (canonical_hash(replay.bit_generator.state) == canonical_hash(after['rng_state'])
                and np.array_equal(replay_counts, after['last_bin_counts'])
                and np.array_equal(np.asarray(rate).view(np.uint64),
                    np.asarray(after['conditional_rate_Hz'], dtype=np.float64).view(np.uint64)))
            witness = SpikeWitness(label, cid, tuple(model.state_specs[0].cell_ids),
                float(t), float(model.dt), voltage, _array(history), _array(rate),
                _array(probability), _array(uniforms), _array(after['last_bin_counts'], np.int64),
                _array(replay_counts, np.int64), tuple(deepcopy(events)), before['rng_state'],
                after['rng_state'], deepcopy(replay.bit_generator.state), model.n,
                before, after, bool(verified), time.perf_counter()-tick)
            self.witnesses.setdefault(label, []).append(witness)
            return updated, events
        return sample
