"""Accepted FP64 continuous packets; no simulator, solver or device execution.

The checkpoint here contains a representation only. A runner checkpoint must
also bind its controller, histories, events, RNG, input cursor and consumers.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Callable, ClassVar

import numpy as np

from .contracts import ContractError

PACKET_SCHEMA = 'retipath-accepted-continuous-packet-1'
CHECKPOINT_SCHEMA = 'retipath-continuous-representation-checkpoint-1'


def _immutable(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _vector(value, name, size=None):
    a = np.asarray(value)
    if (a.dtype != np.float64 or a.ndim != 1 or not len(a)
            or (size is not None and a.shape != (size,)) or not np.isfinite(a).all()):
        raise ContractError('Finite FP64 vector required: '+name)
    return _immutable(a)


def _ids(value):
    a = np.asarray(value)
    if a.ndim != 1 or not len(a) or a.dtype.kind not in 'iuUS':
        raise ContractError('Distinct integer or string state IDs required')
    if a.dtype.kind in 'iu':
        if a.dtype.kind == 'u' and np.any(a > np.iinfo(np.int64).max):
            raise ContractError('State ID outside int64 range')
        a = a.astype(np.int64)
    elif a.dtype.kind == 'S':
        a = a.astype('U')
    if len(np.unique(a)) != len(a):
        raise ContractError('Duplicate state IDs')
    return _immutable(a)


def _digest_array(array):
    h = hashlib.sha256()
    h.update(array.dtype.str.encode())
    h.update(json.dumps(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


@dataclass(frozen=True, kw_only=True)
class AcceptedRepresentation:
    t0: float
    t1: float
    state_ids: np.ndarray
    epoch: int = 0
    accepted: bool = True
    original_rhs: Callable | None = field(default=None, repr=False, compare=False)
    original_rhs_id: str | None = None
    _representation_id: str = field(default='', init=False, repr=False)
    _id_order: np.ndarray = field(default=None, init=False, repr=False)
    _id_sorted: np.ndarray = field(default=None, init=False, repr=False)
    kind: ClassVar[str]

    def __post_init__(self):
        if (not isinstance(self.t0, Real) or not isinstance(self.t1, Real)
                or isinstance(self.t0, (bool, np.bool_)) or isinstance(self.t1, (bool, np.bool_))
                or not math.isfinite(self.t0) or not math.isfinite(self.t1) or self.t1 <= self.t0):
            raise ContractError('Finite increasing representation interval required')
        if (isinstance(self.epoch, (bool, np.bool_)) or not isinstance(self.epoch, (int, np.integer))
                or self.epoch < 0 or type(self.accepted) is not bool):
            raise ContractError('Nonnegative epoch and explicit acceptance flag required')
        if self.original_rhs_id is not None and (not isinstance(self.original_rhs_id, str) or not self.original_rhs_id):
            raise ContractError('Nonempty original RHS binding identity required')
        if self.original_rhs is not None and (not callable(self.original_rhs) or self.original_rhs_id is None):
            raise ContractError('Original RHS callback requires its equation binding identity')
        ids = _ids(self.state_ids)
        order = np.argsort(ids)
        object.__setattr__(self, 't0', float(self.t0))
        object.__setattr__(self, 't1', float(self.t1))
        object.__setattr__(self, 'epoch', int(self.epoch))
        object.__setattr__(self, 'state_ids', ids)
        object.__setattr__(self, '_id_order', _immutable(order))
        object.__setattr__(self, '_id_sorted', _immutable(ids[order]))

    def _seal(self):
        payload = self._header()
        payload['arrays'] = {name: _digest_array(value) for name, value in self._arrays().items()}
        object.__setattr__(self, '_representation_id', hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest())

    def _header(self):
        return dict(schema=PACKET_SCHEMA, kind=self.kind, t0=self.t0, t1=self.t1,
            t0_hex=self.t0.hex(), t1_hex=self.t1.hex(), epoch=self.epoch, dtype='float64',
            side=dict(t0='right', t1='left', interior='either'), accepted=self.accepted,
            original_rhs_id=self.original_rhs_id)

    def _arrays(self):
        return dict(state_ids=self.state_ids)

    @property
    def representation_id(self):
        return self._representation_id

    @property
    def available_until(self):
        if not self.accepted:
            raise ContractError('An unaccepted prediction has no available time')
        return self.t1

    def _query(self, t, side, state_ids):
        if not self.accepted:
            raise ContractError('Unaccepted prediction is not queryable')
        if (not isinstance(t, Real) or isinstance(t, (bool, np.bool_)) or not math.isfinite(t)
                or t < self.t0 or t > self.t1):
            raise ContractError('Query outside accepted representation interval')
        if (side not in ('left', 'right') or (t == self.t0 and side != 'right')
                or (t == self.t1 and side != 'left')):
            raise ContractError('Wrong input side at the representation boundary')
        if state_ids is None:
            return slice(None)
        chosen = _ids(state_ids)
        if chosen.dtype.kind != self.state_ids.dtype.kind:
            raise ContractError('Requested state ID type differs')
        locations = np.searchsorted(self._id_sorted, chosen)
        if np.any(locations >= len(self.state_ids)) or not np.array_equal(self._id_sorted[locations], chosen):
            raise ContractError('Requested state ID is absent from this packet')
        return self._id_order[locations]

    def evaluate(self, t, side, state_ids=None):
        rows = self._query(t, side, state_ids)
        return self._value_selected(float(t), rows).copy()

    def _value_selected(self, t, rows):
        return self._value(t)[rows]

    def representation_derivative(self, t, side, state_ids=None):
        """Derivative of the represented waveform; never a full-RHS monitor."""
        rows = self._query(t, side, state_ids)
        return self._derivative(float(t))[rows].copy()

    def rhs_at(self, t, side, state_ids=None):
        """Call the original equation binding at the accepted current state."""
        rows = self._query(t, side, state_ids)
        if self.original_rhs is None:
            raise ContractError('Full RHS requires a bound original-equation callback; representation derivative is insufficient')
        result = _vector(self.original_rhs(float(t), side, self._value(float(t)).copy()),
                         'original RHS', len(self.state_ids))
        return result[rows].copy()

    def packet(self):
        return dict(metadata=dict(self._header(), representation_id=self.representation_id),
                    arrays=self._arrays())


@dataclass(frozen=True, kw_only=True)
class Dop853Dense(AcceptedRepresentation):
    y0: np.ndarray
    coefficients: np.ndarray
    kind: ClassVar[str] = 'dop853_dense'

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'y0', _vector(self.y0, 'DOP853 anchor', len(self.state_ids)))
        coefficients = np.asarray(self.coefficients)
        if coefficients.dtype != np.float64 or coefficients.shape != (7, len(self.y0)) or not np.isfinite(coefficients).all():
            raise ContractError('DOP853 continuous F coefficients require FP64 (7, state_count)')
        object.__setattr__(self, 'coefficients', _immutable(coefficients))
        self._seal()

    @classmethod
    def from_scipy(cls, dense, state_ids, **metadata):
        """Copy one accepted SciPy DOP853 dense segment, not an OdeSolution."""
        return cls(t0=float(dense.t_old), t1=float(dense.t), state_ids=state_ids,
                   y0=np.asarray(dense.y_old), coefficients=np.asarray(dense.F), **metadata)

    def _arrays(self):
        return dict(super()._arrays(), y0=self.y0, coefficients=self.coefficients)

    def _pair(self, t):
        x = (t-self.t0)/(self.t1-self.t0)
        y, dy = np.zeros_like(self.y0), np.zeros_like(self.y0)
        for i, coefficient in enumerate(self.coefficients[::-1]):
            y = y+coefficient
            scale, slope = (x, 1.) if i % 2 == 0 else (1.-x, -1.)
            dy = dy*scale+y*slope
            y = y*scale
        return y+self.y0, dy/(self.t1-self.t0)

    def _value(self, t):
        return self._pair(t)[0]

    def _derivative(self, t):
        return self._pair(t)[1]


@dataclass(frozen=True, kw_only=True)
class QuinticHermite(AcceptedRepresentation):
    """Two-halfstep Radau anchors with original physical-time RHS slopes.

    The midpoint is t0 + (t1-t0)/2. This reconstructs one accepted polynomial;
    it does not establish an error bound relative to the exact ODE solution.
    """
    y0: np.ndarray
    f0: np.ndarray
    ym: np.ndarray
    fm: np.ndarray
    y1: np.ndarray
    f1: np.ndarray
    kind: ClassVar[str] = 'quintic_hermite'

    def __post_init__(self):
        super().__post_init__()
        h = self.t1-self.t0
        if not math.isfinite(h) or not self.t0 < self.t0+h/2 < self.t1:
            raise ContractError('Finite Hermite duration and representable interior midpoint required')
        for name in ('y0', 'f0', 'ym', 'fm', 'y1', 'f1'):
            object.__setattr__(self, name, _vector(getattr(self, name),
                'quintic Hermite '+name, len(self.state_ids)))
        with np.errstate(over='ignore', invalid='ignore'):
            finite = np.isfinite(self._coefficient_array(slice(None))).all()
        if not finite:
            raise ContractError('Quintic Hermite coefficients exceed the finite FP64 domain')
        self._seal()

    def _arrays(self):
        return dict(super()._arrays(), **{name: getattr(self, name)
            for name in ('y0', 'f0', 'ym', 'fm', 'y1', 'f1')})

    def _coefficient_array(self, rows):
        h = self.t1-self.t0
        y0, s0 = self.y0[rows], h*self.f0[rows]
        a = self.ym[rows]-y0-s0/2
        b = h*self.fm[rows]-s0
        c = self.y1[rows]-y0-s0
        d = h*self.f1[rows]-s0
        return np.stack((y0, s0, 16*a-8*b+7*c-d,
            -32*a+32*b-34*c+5*d, 16*a-40*b+52*c-8*d,
            16*b-24*c+4*d))

    @property
    def coefficients(self):
        """Owned ascending power coefficients in x=(t-t0)/(t1-t0)."""
        return _immutable(self._coefficient_array(slice(None)))

    def _pair(self, t, rows=slice(None)):
        h = self.t1-self.t0
        x = (t-self.t0)/h
        l0, lm, l1 = (2*x-3)*x+1, 4*x*(1-x), (2*x-1)*x
        q0, qm, q1 = l0*l0, lm*lm, l1*l1
        dq0, dqm, dq1 = 2*l0*(4*x-3), 2*lm*(4-8*x), 2*l1*(4*x-1)
        y0, ym, y1 = self.y0[rows], self.ym[rows], self.y1[rows]
        s0, sm, s1 = h*self.f0[rows], h*self.fm[rows], h*self.f1[rows]
        value = y0+qm*(ym-y0)+(7-6*x)*q1*(y1-y0)
        value = value+x*q0*s0+(x-.5)*qm*sm+(x-1)*q1*s1
        derivative = dqm*(ym-y0)+((7-6*x)*dq1-6*q1)*(y1-y0)
        derivative = derivative+(q0+x*dq0)*s0+(qm+(x-.5)*dqm)*sm+(q1+(x-1)*dq1)*s1
        return value, derivative/h

    def _value(self, t):
        return self._pair(t)[0]

    def _value_selected(self, t, rows):
        return self._pair(t, rows)[0]

    def _derivative(self, t):
        return self._pair(t)[1]


@dataclass(frozen=True, kw_only=True)
class ExponentialForced(AcceptedRepresentation):
    y0: np.ndarray
    coefficients: np.ndarray
    decay: np.ndarray
    kind: ClassVar[str] = 'exponential_forced'

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'y0', _vector(self.y0, 'filter anchor', len(self.state_ids)))
        decay = np.asarray(self.decay)
        if decay.dtype != np.float64 or decay.shape not in ((), self.y0.shape) or not np.isfinite(decay).all() or np.any(decay < 0):
            raise ContractError('Finite nonnegative FP64 scalar or per-state decay required')
        coefficients = np.asarray(self.coefficients)
        if coefficients.dtype != np.float64 or coefficients.shape != (8, len(self.y0)) or not np.isfinite(coefficients).all():
            raise ContractError('Normalized degree-seven FP64 drive coefficients required')
        object.__setattr__(self, 'decay', _immutable(np.broadcast_to(decay, self.y0.shape)))
        object.__setattr__(self, 'coefficients', _immutable(coefficients))
        self._seal()

    def _arrays(self):
        return dict(super()._arrays(), y0=self.y0, coefficients=self.coefficients, decay=self.decay)

    @classmethod
    def from_constant_forcing(cls, *, y0, forcing, decay, **metadata):
        """Exact constant-drive y'=forcing-decay*y expression, up to roundoff."""
        forcing = _vector(forcing, 'constant forcing')
        coefficients = np.zeros((8, len(forcing)), dtype=np.float64)
        coefficients[0] = forcing
        return cls(y0=y0, coefficients=coefficients, decay=decay, **metadata)

    def _value(self, t):
        from .semianalytic_time import exponential_forced_value
        return exponential_forced_value(self.y0, self.decay, self.coefficients,
                                        self.t1-self.t0, t-self.t0)

    def _derivative(self, t):
        q = np.zeros_like(self.y0)
        x = (t-self.t0)/(self.t1-self.t0)
        for coefficient in self.coefficients[::-1]:
            q = q*x+coefficient
        return q-self.decay*self._value(t)


@dataclass(frozen=True, kw_only=True)
class DiscreteHold(AcceptedRepresentation):
    y0: np.ndarray
    kind: ClassVar[str] = 'discrete_hold'

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'y0', _vector(self.y0, 'held state', len(self.state_ids)))
        self._seal()

    def _arrays(self):
        return dict(super()._arrays(), y0=self.y0)

    def _value(self, t):
        return self.y0

    def _derivative(self, t):
        return np.zeros_like(self.y0)


@dataclass(frozen=True, kw_only=True)
class AffineMembraneState(AcceptedRepresentation):
    """Accepted CPU-inner integrating-factor waveform with an explicit budget.

    a has state_unit/s, b has s^-1 and quadrature_budget has state_unit.
    The represented a/b are not substituted for a full original-RHS monitor.
    """
    y0: np.ndarray
    a_coefficients: np.ndarray
    b_coefficients: np.ndarray
    quadrature_budget: np.ndarray
    state_unit: str
    max_depth: int = 20
    _membrane: object = field(default=None, init=False, repr=False, compare=False)
    kind: ClassVar[str] = 'integrating_factor'

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'y0', _vector(self.y0, 'affine membrane anchor', len(self.state_ids)))
        for name in ('a_coefficients', 'b_coefficients'):
            coefficients = np.asarray(getattr(self, name))
            if coefficients.dtype != np.float64 or coefficients.shape != (8, len(self.y0)) or not np.isfinite(coefficients).all():
                raise ContractError('Normalized degree-seven FP64 affine coefficients required: '+name)
            object.__setattr__(self, name, _immutable(coefficients))
        budget = np.asarray(self.quadrature_budget)
        if (budget.dtype != np.float64 or budget.shape not in ((), self.y0.shape)
                or not np.isfinite(budget).all() or np.any(budget <= 0)
                or not isinstance(self.state_unit, str) or not self.state_unit
                or type(self.max_depth) is not int or self.max_depth < 0):
            raise ContractError('Positive FP64 state-unit quadrature budget, explicit state unit and subdivision limit required')
        object.__setattr__(self, 'quadrature_budget', _immutable(np.broadcast_to(budget, self.y0.shape)))
        from .semianalytic_time import AffineMembrane, PolynomialDrive
        object.__setattr__(self, '_membrane', AffineMembrane(self.t0, self.t1, self.y0,
            PolynomialDrive(self.t0, self.t1, self.a_coefficients, self.state_unit+'/s'),
            PolynomialDrive(self.t0, self.t1, self.b_coefficients, 's^-1'),
            self.quadrature_budget, self.state_unit, self.max_depth))
        self._seal()

    def _header(self):
        return dict(super()._header(), state_unit=self.state_unit, quadrature_max_depth=self.max_depth)

    def _arrays(self):
        return dict(super()._arrays(), y0=self.y0, a_coefficients=self.a_coefficients,
                    b_coefficients=self.b_coefficients, quadrature_budget=self.quadrature_budget)

    def _value(self, t):
        return self._membrane.evaluate(t, 'left' if t == self.t1 else 'right')

    def _value_selected(self, t, rows):
        selected = None if isinstance(rows, slice) else rows
        return self._membrane.evaluate(t, 'left' if t == self.t1 else 'right', selected)

    def evaluate_with_error(self, t, side, state_ids=None):
        """Return the quadrature estimate; not a full waveform error proof."""
        rows = self._query(t, side, state_ids)
        selected = None if isinstance(rows, slice) else rows
        return self._membrane.evaluate_with_error(float(t), side, selected)

    def _derivative(self, t):
        from .semianalytic_time import polynomial_value
        x = (t-self.t0)/(self.t1-self.t0)
        return polynomial_value(self.a_coefficients, x)-polynomial_value(self.b_coefficients, x)*self._value(t)


class AcceptedSeries:
    """Continuous query selection across exact accepted interval boundaries."""
    def __init__(self, representations):
        self.representations = tuple(representations)
        if not self.representations:
            raise ContractError('Nonempty accepted interval sequence required')
        for i, segment in enumerate(self.representations):
            if not isinstance(segment, AcceptedRepresentation) or not segment.accepted:
                raise ContractError('Only accepted continuous packets enter a query series')
            if i:
                previous = self.representations[i-1]
                if previous.t1 != segment.t0 or not np.array_equal(previous.state_ids, segment.state_ids):
                    raise ContractError('Exact contiguous intervals and identical ordered state IDs required')
        self._stops = np.array([s.t1 for s in self.representations])

    @property
    def available_until(self):
        return self.representations[-1].available_until

    @property
    def representation_id(self):
        return hashlib.sha256(''.join(s.representation_id for s in self.representations).encode()).hexdigest()

    def _segment(self, t, side):
        if not isinstance(t, Real) or not math.isfinite(t) or side not in ('left', 'right'):
            raise ContractError('Finite scalar query and explicit input side required')
        i = int(np.searchsorted(self._stops, t, side='left' if side == 'left' else 'right'))
        if i >= len(self.representations):
            i = len(self.representations)-1
        return self.representations[i]

    def evaluate(self, t, side, state_ids=None):
        return self._segment(t, side).evaluate(t, side, state_ids)

    def rhs_at(self, t, side, state_ids=None):
        return self._segment(t, side).rhs_at(t, side, state_ids)


def from_packet(packet, *, original_rhs=None, original_rhs_id=None):
    """Rebind only an explicitly matching original RHS, never pickle callbacks."""
    if not isinstance(packet, dict) or set(packet) != {'metadata', 'arrays'}:
        raise ContractError('Explicit continuous packet metadata and typed arrays required')
    meta, arrays = packet['metadata'], packet['arrays']
    required = {'schema', 'kind', 't0', 't1', 't0_hex', 't1_hex', 'epoch', 'dtype',
                'side', 'accepted', 'original_rhs_id', 'representation_id'}
    if isinstance(meta, dict) and meta.get('kind') == 'integrating_factor':
        required.update(('state_unit', 'quadrature_max_depth'))
    if (not isinstance(meta, dict) or set(meta) != required or meta['schema'] != PACKET_SCHEMA
            or meta['dtype'] != 'float64' or meta['t0_hex'] != float(meta['t0']).hex()
            or meta['t1_hex'] != float(meta['t1']).hex()
            or meta['side'] != dict(t0='right', t1='left', interior='either')):
        raise ContractError('Continuous packet schema/boundary identity differs')
    classes = {'dop853_dense': Dop853Dense, 'quintic_hermite': QuinticHermite,
               'exponential_forced': ExponentialForced, 'discrete_hold': DiscreteHold,
               'integrating_factor': AffineMembraneState}
    expected = {'state_ids', 'y0'} | ({'coefficients'} if meta['kind'] in ('dop853_dense', 'exponential_forced') else set())
    if meta['kind'] == 'quintic_hermite':
        expected.update(('f0', 'ym', 'fm', 'y1', 'f1'))
    if meta['kind'] == 'exponential_forced':
        expected.add('decay')
    if meta['kind'] == 'integrating_factor':
        expected.update(('a_coefficients', 'b_coefficients', 'quadrature_budget'))
    if meta['kind'] not in classes or not isinstance(arrays, dict) or set(arrays) != expected:
        raise ContractError('Continuous packet representation layout differs')
    if original_rhs is not None and original_rhs_id != meta['original_rhs_id']:
        raise ContractError('Restored original RHS binding identity differs')
    if original_rhs is None and original_rhs_id is not None:
        raise ContractError('RHS identity alone cannot restore an equation callback')
    extra = dict(state_unit=meta['state_unit'], max_depth=meta['quadrature_max_depth']) if meta['kind'] == 'integrating_factor' else {}
    result = classes[meta['kind']](t0=meta['t0'], t1=meta['t1'], epoch=meta['epoch'],
        accepted=meta['accepted'], original_rhs=original_rhs, original_rhs_id=meta['original_rhs_id'], **arrays, **extra)
    if result.representation_id != meta['representation_id']:
        raise ContractError('Continuous packet array identity differs')
    return result


def save_representation(path, representation):
    """New-directory representation checkpoint; no full-run restore claim."""
    path = Path(path)
    if path.exists() or not path.parent.is_dir() or not isinstance(representation, AcceptedRepresentation) or not representation.accepted:
        raise ContractError('Fresh representation checkpoint directory and accepted packet required')
    path.mkdir()
    packet = representation.packet()
    archive = path/'arrays.npz'
    np.savez(archive, **packet['arrays'])
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = dict(schema=CHECKPOINT_SCHEMA, scope='representation_only_not_runner_checkpoint',
        full_runner_restore_supported=False, metadata=packet['metadata'],
        arrays=dict(path='arrays.npz', bytes=archive.stat().st_size, sha256=digest))
    (path/'manifest.json').write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return receipt


def load_representation(path, *, original_rhs=None, original_rhs_id=None):
    path = Path(path)
    receipt = json.loads((path/'manifest.json').read_text(encoding='utf8'))
    if (receipt.get('schema') != CHECKPOINT_SCHEMA or receipt.get('scope') != 'representation_only_not_runner_checkpoint'
            or receipt.get('full_runner_restore_supported') is not False or receipt.get('arrays', {}).get('path') != 'arrays.npz'):
        raise ContractError('Explicit new representation-only checkpoint required')
    archive = path/'arrays.npz'
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if archive.stat().st_size != receipt['arrays']['bytes'] or digest != receipt['arrays']['sha256']:
        raise ContractError('Representation checkpoint array payload differs')
    with np.load(archive, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    return from_packet(dict(metadata=receipt['metadata'], arrays=arrays),
                       original_rhs=original_rhs, original_rhs_id=original_rhs_id)
