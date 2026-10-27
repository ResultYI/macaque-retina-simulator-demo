"""Opt-in native Cone/HC multirate candidate; no device work on import.

The equation object supplies the unchanged source parameters and A/B operators.
The old exponential-midpoint step is never called. CUDA vectors and dense/forced
coefficients stay on their original device; only scalar control tests return.
"""
from dataclasses import dataclass, field, replace
from math import comb, factorial
from pathlib import Path
from time import perf_counter
import hashlib
import math

import numpy as np
from scipy.integrate import DOP853
from scipy.integrate._ivp.rk import SAFETY, MIN_FACTOR, MAX_FACTOR
from numpy.polynomial.legendre import leggauss

from .contracts import ContractError, canonical_hash
from .cone_hc_reduced_native import AnalyticREAnchor, STATE_NAMES, _owned, _signature
from .semianalytic_time import phi_sequence
from .vh2005 import SOURCE_VALUES


METHOD = 'cone-hc-native-dop853-forced-waveform-fp64-1'
INTERVAL_SCHEMA = 'retipath-native-multirate-accepted-interval-1'
MAX_STEP = .0001
_LOBATTO = .5*(1-np.cos(np.arange(8)*np.pi/7))
_CHECK = np.unique(np.r_[_LOBATTO, .5*(1+leggauss(9)[0])])


class _Ops:
    def __init__(self, torch=None, device=None):
        self.torch, self.device = torch, device

    def array(self, value):
        return np.asarray(value, dtype=np.float64) if self.torch is None else self.torch.as_tensor(value, dtype=self.torch.float64, device=self.device)

    def clone(self, value):
        if self.torch is None:
            a = np.ascontiguousarray(value)
            return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
        return value.clone()

    def cat(self, values):
        return np.concatenate(values) if self.torch is None else self.torch.cat(tuple(values))

    def stack(self, values):
        return np.stack(values) if self.torch is None else self.torch.stack(tuple(values))

    def zeros(self, shape):
        return np.zeros(shape, dtype=np.float64) if self.torch is None else self.torch.zeros(shape, dtype=self.torch.float64, device=self.device)

    def abs(self, value):
        return np.abs(value) if self.torch is None else value.abs()

    def maximum(self, a, b):
        return np.maximum(a, b) if self.torch is None else self.torch.maximum(a, b)

    def finite(self, *values):
        if self.torch is None:
            good = all(np.isfinite(v).all() for v in values)
        else:
            good = all(self.torch.isfinite(v).all().item() for v in values)
        if not good:
            raise ContractError('Nonfinite native multirate state, derivative or continuous coefficient')

    def any(self, value):
        return bool(np.any(value)) if self.torch is None else bool(value.any().item())

    def norm(self, value):
        return float(np.linalg.norm(value)/math.sqrt(value.size)) if self.torch is None else float(self.torch.linalg.vector_norm(value).item()/math.sqrt(value.numel()))

    def select(self, value, rows):
        if self.torch is None:
            return np.asarray(value)[..., np.asarray(rows, dtype=np.int64)].copy()
        chosen = self.torch.as_tensor(rows, dtype=self.torch.int64, device=self.device)
        return value.index_select(value.ndim-1, chosen)


def _polynomial(coefficients, x):
    value = coefficients[-1]*1.
    for coefficient in coefficients[-2::-1] if isinstance(coefficients, np.ndarray) else reversed(coefficients[:-1].unbind(0)):
        value = value*x+coefficient
    return value


def _shift(coefficients, alpha, beta, ops):
    """Exact polynomial basis change x=alpha+beta*u, up to FP64 roundoff."""
    transform = np.zeros((8, 8))
    for k in range(8):
        for j in range(k, 8):
            transform[k, j] = comb(j, k)*alpha**(j-k)*beta**k
    return ops.array(transform)@coefficients


def _dense_power_matrix():
    result = np.zeros((8, 7))
    for k in range(7):
        value = np.zeros(8)
        for i, j in enumerate(range(6, -1, -1)):
            value[0] += float(j == k)
            previous = value.copy()
            value[:] = 0.
            if i % 2:
                value += previous
                value[1:] -= previous[:-1]
            else:
                value[1:] += previous[:-1]
        result[:, k] = value
    return result


_DENSE_TO_POWER = _dense_power_matrix()
_POWER_TO_DENSE = np.linalg.inv(_DENSE_TO_POWER[1:])


@dataclass(frozen=True)
class DeviceRepresentation:
    """Owned device coefficients. Materialize CPU hashes only after D2H ready."""
    kind: str
    t0: float
    t1: float
    y0: object
    coefficients: object
    decay: float | None
    state_ids: object
    input_epoch: int


class _Dense:
    kind = 'dop853_dense'

    def __init__(self, t0, t1, y0, coefficients, ops):
        self.t0, self.t1, self.y0, self.coefficients, self.ops = t0, t1, ops.clone(y0), ops.clone(coefficients), ops
        self._binding = t0, t1
        self._versions = _signature(self.y0), _signature(self.coefficients)

    def _check(self):
        if self._binding != (self.t0, self.t1) or self._versions != (_signature(self.y0), _signature(self.coefficients)):
            raise ContractError('Accepted native dense representation was mutated')

    def at(self, t):
        self._check()
        if not self.t0 <= t <= self.t1:
            raise ContractError('Native dense query outside accepted interval')
        x = (t-self.t0)/(self.t1-self.t0)
        value = self.y0*0.
        rows = self.coefficients[::-1] if self.ops.torch is None else reversed(self.coefficients.unbind(0))
        for i, row in enumerate(rows):
            value = (value+row)*(x if i % 2 == 0 else 1-x)
        return value+self.y0

    def derivative(self, t):
        self._check()
        x = (t-self.t0)/(self.t1-self.t0)
        y, dy = self.y0*0., self.y0*0.
        rows = self.coefficients[::-1] if self.ops.torch is None else reversed(self.coefficients.unbind(0))
        for i, row in enumerate(rows):
            y = y+row
            scale, slope = (x, 1.) if i % 2 == 0 else (1-x, -1.)
            dy, y = dy*scale+y*slope, y*scale
        return dy/(self.t1-self.t0)

    def at_rows(self, t, rows):
        self._check()
        if not self.t0 <= t <= self.t1:
            raise ContractError('Selected native dense query outside accepted interval')
        x = (t-self.t0)/(self.t1-self.t0)
        y0 = self.ops.select(self.y0, rows)
        F = self.ops.select(self.coefficients, rows)
        value = y0*0.
        coefficients = F[::-1] if self.ops.torch is None else reversed(F.unbind(0))
        for i, row in enumerate(coefficients):
            value = (value+row)*(x if i % 2 == 0 else 1-x)
        return value+y0

    def restricted(self, lo, hi, rows, epoch):
        self._check()
        if not self.t0 <= lo < hi <= self.t1:
            raise ContractError('Dense restriction must stay in one accepted original step')
        selected = self.ops.select(self.coefficients, rows)
        power = self.ops.array(_DENSE_TO_POWER)@selected
        shifted = _shift(power, (lo-self.t0)/(self.t1-self.t0), (hi-lo)/(self.t1-self.t0), self.ops)
        F = self.ops.array(_POWER_TO_DENSE)@shifted[1:]
        return DeviceRepresentation(self.kind, lo, hi, self.ops.select(self.at(lo), rows), F, None,
                                    np.asarray(rows, dtype=np.int64).copy(), epoch)


class _Forced:
    kind = 'exponential_forced'

    def __init__(self, t0, t1, y0, coefficients, decay, ops):
        self.t0, self.t1, self.y0 = t0, t1, ops.clone(y0)
        self.coefficients, self.decay, self.ops = ops.clone(coefficients), float(decay), ops
        self._binding = t0, t1, self.decay
        self._versions = _signature(self.y0), _signature(self.coefficients)

    def _check(self):
        if self._binding != (self.t0, self.t1, self.decay) or self._versions != (_signature(self.y0), _signature(self.coefficients)):
            raise ContractError('Accepted native forced representation was mutated')

    def at(self, t):
        self._check()
        if not self.t0 <= t <= self.t1:
            raise ContractError('Native forced query outside accepted interval')
        s, H = t-self.t0, self.t1-self.t0
        if not s:
            return self.ops.clone(self.y0)
        phi = phi_sequence(-self.decay*s)
        value = math.exp(-self.decay*s)*self.y0
        power = 1.
        for j in range(8):
            value = value+self.coefficients[j]*(factorial(j)*s*power*phi[j])
            power *= s/H
        return value

    def derivative(self, t):
        self._check()
        return _polynomial(self.coefficients, (t-self.t0)/(self.t1-self.t0))-self.decay*self.at(t)

    def at_rows(self, t, rows):
        self._check()
        selected = _Forced(self.t0, self.t1, self.ops.select(self.y0, rows),
            self.ops.select(self.coefficients, rows), self.decay, self.ops)
        return selected.at(t)

    def restricted(self, lo, hi, rows, epoch):
        self._check()
        if not self.t0 <= lo < hi <= self.t1:
            raise ContractError('Forced restriction must stay in one original driving interval')
        q = _shift(self.ops.select(self.coefficients, rows), (lo-self.t0)/(self.t1-self.t0),
                   (hi-lo)/(self.t1-self.t0), self.ops)
        return DeviceRepresentation(self.kind, lo, hi, self.ops.select(self.at(lo), rows), q, self.decay,
                                    np.asarray(rows, dtype=np.int64).copy(), epoch)


class _Curve:
    def __init__(self, segments):
        self.segments = tuple(segments)
        if not self.segments or any(a.t1 != b.t0 for a, b in zip(self.segments, self.segments[1:])):
            raise ContractError('Native accepted waveform has a missing interval')
        self.t0, self.t1 = self.segments[0].t0, self.segments[-1].t1

    def segment(self, t, *, right=False):
        if not self.t0 <= t <= self.t1:
            raise ContractError('Native waveform query outside accepted availability')
        ends = [s.t1 for s in self.segments]
        i = min(int(np.searchsorted(ends, t, side='right' if right else 'left')), len(ends)-1)
        return self.segments[i]

    def at(self, t, rows=None):
        return self.segment(t).at(t) if rows is None else self.segment(t).at_rows(t, rows)

    def derivative(self, t):
        return self.segment(t).derivative(t)

    def cut(self, offset, size):
        segments = []
        for s in self.segments:
            value = s.coefficients[:, offset:offset+size]
            segments.append(_Dense(s.t0, s.t1, s.y0[offset:offset+size], value, s.ops))
        return _Curve(segments)


def _solve_fast(fun, t0, y0, t1, *, rtol, atol, ops):
    """Owned DOP853 dense steps on NumPy or the explicit torch device."""
    if not t0 < t1 or rtol < 100*np.finfo(float).eps:
        raise ContractError('Fast DOP853 needs a positive interval and unclamped quarter tolerance')
    if ops.torch is None:
        def checked_fun(t, y):
            with np.errstate(over='ignore', invalid='ignore'):
                scale = atol+rtol*np.abs(y)
            ops.finite(y, scale)
            value = fun(t, y)
            if np.asarray(value).dtype != np.float64 or value.shape != y.shape:
                raise ContractError('CPU native fast RHS changed FP64 layout')
            ops.finite(value)
            return value
        solver = DOP853(checked_fun, t0, np.array(y0, copy=True), t1, max_step=MAX_STEP, rtol=rtol, atol=atol)
        segments = []
        while solver.status == 'running':
            message = solver.step()
            if solver.status == 'failed':
                raise ContractError('Native fast DOP853 failed: '+str(message))
            d = solver.dense_output()
            ops.finite(d.y_old, d.F)
            segments.append(_Dense(float(d.t_old), float(d.t), d.y_old, d.F, ops))
        return _Curve(segments), dict(nfev=solver.nfev, steps=len(segments), rejected_attempts=None)
    torch = ops.torch
    n = y0.numel()
    A, B, E3, E5 = (ops.array(a) for a in (DOP853.A, DOP853.B, DOP853.E3, DOP853.E5))
    AE, DD, at = ops.array(DOP853.A_EXTRA), ops.array(DOP853.D), ops.array(atol)
    K = ops.zeros((16, n))
    y, t, nfev, rejects, segments = y0.clone(), float(t0), 0, 0, []

    def rhs(t, y):
        nonlocal nfev
        ops.finite(y, at+y.abs()*rtol)
        result = fun(float(t), y)
        nfev += 1
        if result.dtype != torch.float64 or result.device != y.device or result.shape != y.shape:
            raise ContractError('Device native fast RHS changed dtype/device/shape')
        ops.finite(result)
        return result

    f = rhs(t, y)
    scale = at+y.abs()*rtol
    d0, d1 = ops.norm(y/scale), ops.norm(f/scale)
    h0 = min(1e-6 if d0 < 1e-5 or d1 < 1e-5 else .01*d0/d1, t1-t0)
    f1 = rhs(t+h0, y+h0*f)
    d2 = ops.norm((f1-f)/scale)/h0
    h1 = max(1e-6, h0*1e-3) if d1 <= 1e-15 and d2 <= 1e-15 else (.01/max(d1, d2))**(1/8)
    h_abs = min(100*h0, h1, t1-t0, MAX_STEP)
    while t < t1:
        minimum = 10*abs(np.nextafter(t, np.inf)-t)
        h_abs, rejected = min(max(h_abs, minimum), MAX_STEP), False
        while True:
            if h_abs < minimum:
                raise ContractError('Device DOP853 step is below FP64 spacing')
            end = min(t+h_abs, t1)
            h = end-t
            h_abs = abs(h)
            K[0].copy_(f)
            for stage in range(1, 12):
                K[stage].copy_(rhs(t+DOP853.C[stage]*h, y+torch.mv(K[:stage].T, A[stage, :stage])*h))
            new = y+h*torch.mv(K[:12].T, B)
            fnew = rhs(end, new)
            K[12].copy_(fnew)
            scale = at+torch.maximum(y.abs(), new.abs())*rtol
            err5, err3 = torch.mv(K[:13].T, E5)/scale, torch.mv(K[:13].T, E3)/scale
            n5, n3 = torch.linalg.vector_norm(err5).item()**2, torch.linalg.vector_norm(err3).item()**2
            error = 0. if n5 == n3 == 0 else abs(h)*n5/math.sqrt((n5+.01*n3)*n)
            if not math.isfinite(error):
                raise ContractError('Nonfinite device DOP853 estimate')
            if error < 1:
                factor = MAX_FACTOR if not error else min(MAX_FACTOR, SAFETY*error**(-1/8))
                h_abs *= min(1., factor) if rejected else factor
                break
            h_abs *= max(MIN_FACTOR, SAFETY*error**(-1/8))
            rejected, rejects = True, rejects+1
        for stage in range(13, 16):
            K[stage].copy_(rhs(t+DOP853.C_EXTRA[stage-13]*h, y+torch.mv(K[:stage].T, AE[stage-13, :stage])*h))
        delta = new-y
        F = ops.zeros((7, n))
        F[0], F[1], F[2] = delta, h*K[0]-delta, 2*delta-h*(fnew+K[0])
        F[3:] = h*torch.mm(DD, K)
        ops.finite(F)
        segments.append(_Dense(t, end, y, F, ops))
        t, y, f = end, new, fnew
    return _Curve(segments), dict(nfev=nfev, steps=len(segments), rejected_attempts=rejects)


class _CouplingRetry(Exception):
    pass


@dataclass(frozen=True)
class NativeMultirateState:
    time_seconds: float
    y: object
    anchor: AnalyticREAnchor
    backend_id: str
    next_window_seconds: float = .0005
    _version: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (not math.isfinite(self.time_seconds) or not self.anchor.time_seconds <= self.time_seconds <= self.anchor.input_stop_seconds
                or not isinstance(self.backend_id, str) or not self.backend_id
                or not MAX_STEP <= self.next_window_seconds <= .001):
            raise ContractError('Native state time, source or slow controller proposal differs')
        if hasattr(self.y, 'detach'):
            if str(self.y.dtype) != 'torch.float64':
                raise ContractError('Native accepted state must remain FP64')
            owned = _owned(self.y)
        else:
            a = np.asarray(self.y)
            if a.dtype != np.float64:
                raise ContractError('Native accepted state must remain FP64')
            a = np.ascontiguousarray(a)
            owned = np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
        object.__setattr__(self, 'y', owned)
        object.__setattr__(self, '_version', _signature(self.y))

    def check_owned(self):
        self.anchor.check_owned()
        if self._version != _signature(self.y):
            raise ContractError('Accepted native state storage was mutated')


class AcceptedNativeInterval:
    def __init__(self, backend, anchor, curves, metrics):
        self.backend, self.anchor, self.curves, self.metrics = backend, anchor, curves, dict(metrics)
        self.start, self.stop = curves['X'].t0, curves['X'].t1
        self.input_epoch = anchor.input_epoch
        self.available_until = self.stop
        self.source_version = backend.reference_identity if self.metrics.get('solver_mode') == 'joint_reference' else backend.identity
        self.representation_id = canonical_hash(dict(method=METHOD, backend=self.source_version,
            t0=self.start, t1=self.stop, input_epoch=self.input_epoch))

    def evaluate(self, t, side='right', rows=None):
        if (not math.isfinite(t) or not self.start <= t <= self.stop or side not in ('left', 'right')
                or t == self.start and side != 'right' or t == self.stop and side != 'left'):
            raise ContractError('Wrong native accepted interval or input side')
        values = self.backend.ops.cat([self.curves[n].at(t) for n in STATE_NAMES])
        return values if rows is None else self.backend.ops.select(values, rows)

    def rhs_at(self, t, side='right', rows=None):
        values = self.evaluate(t, side)
        result = self.backend.native.evaluate(values, t, self.anchor).rhs
        return result if rows is None else self.backend.ops.select(result, rows)

    def outputs_at(self, t, side='right'):
        return self.backend.native.evaluate(self.evaluate(t, side), t, self.anchor).outputs

    def readout_chunks(self, cone_rows, hc_rows):
        """Selected-only device packets with exactly shared accepted intervals."""
        checked = []
        for rows, size in ((cone_rows, self.backend.native.nc), (hc_rows, self.backend.native.nh)):
            value = np.asarray(rows)
            if (value.ndim != 1 or value.dtype.kind not in 'iu' or np.any(value < 0)
                    or np.any(value >= size) or len(np.unique(value)) != len(value)):
                raise ContractError('Readout chunks require distinct existing integer native rows')
            checked.append(value.astype(np.int64, copy=True))
        cone_rows, hc_rows = checked
        if not len(cone_rows):
            raise ContractError('A native readout needs at least one declared cone row')
        names = ('Vis', 'Vbar', 'Vh')
        boundaries = sorted({self.start, self.stop, *(s.t1 for n in names for s in self.curves[n].segments)})
        for lo, hi in zip(boundaries, boundaries[1:]):
            packet = {}
            for name in names:
                rows = hc_rows if name == 'Vh' else cone_rows
                if len(rows) == 0:
                    packet[name] = None
                else:
                    s = self.curves[name].segment(lo, right=True)
                    packet[name] = s.restricted(lo, hi, rows, self.input_epoch)
            yield packet

    def payload(self):
        """Native component only; the owner must also save every consumer/RNG.

        Arrays retain their CPU/device ownership for the enclosing binary
        checkpoint encoder. This does not download all native state at a stage.
        """
        self.anchor.check_owned()
        curves = {}
        for name in STATE_NAMES:
            records = []
            for segment in self.curves[name].segments:
                segment._check()
                row = dict(kind=segment.kind, t0=segment.t0, t1=segment.t1,
                    t0_hex=float(segment.t0).hex(), t1_hex=float(segment.t1).hex(),
                    y0=segment.y0, coefficients=segment.coefficients)
                if isinstance(segment, _Forced):
                    row['decay'] = segment.decay
                records.append(row)
            curves[name] = records
        return dict(schema=INTERVAL_SCHEMA, scope='native_accepted_interval_not_complete_model_checkpoint',
            source_version=self.source_version, equation_source_version=self.backend.native.source_version,
            representation_id=self.representation_id, solver_mode='joint_reference' if self.source_version == self.backend.reference_identity else 'candidate',
            start=self.start, stop=self.stop, start_hex=float(self.start).hex(), stop_hex=float(self.stop).hex(),
            input_epoch=self.input_epoch, state_names=list(STATE_NAMES), state_sizes=list(self.backend.native.sizes),
            dtype='float64', anchor=dict(time_seconds=self.anchor.time_seconds, R=self.anchor.R, E=self.anchor.E,
                light=self.anchor.light, input_stop_seconds=self.anchor.input_stop_seconds, input_epoch=self.anchor.input_epoch),
            curves=curves, metrics=dict(self.metrics))

    @classmethod
    def from_payload(cls, backend, payload):
        """Restore exact original coefficients, never refit an accepted window."""
        fields = {'schema','scope','source_version','equation_source_version','representation_id','solver_mode',
            'start','stop','start_hex','stop_hex','input_epoch','state_names','state_sizes','dtype','anchor','curves','metrics'}
        if (not isinstance(payload, dict) or set(payload) != fields or payload['schema'] != INTERVAL_SCHEMA
                or payload['scope'] != 'native_accepted_interval_not_complete_model_checkpoint'
                or payload['equation_source_version'] != backend.native.source_version
                or payload['solver_mode'] not in ('candidate','joint_reference') or payload['dtype'] != 'float64'
                or payload['state_names'] != list(STATE_NAMES) or payload['state_sizes'] != list(backend.native.sizes)):
            raise ContractError('Native accepted interval schema/equation/layout differs; old checkpoints are incompatible')
        source = backend.reference_identity if payload['solver_mode'] == 'joint_reference' else backend.identity
        if payload['source_version'] != source:
            raise ContractError('Native accepted interval solver/source identity differs')
        start, stop = payload['start'], payload['stop']
        if (not isinstance(start, (int,float)) or not isinstance(stop, (int,float))
                or isinstance(start,bool) or isinstance(stop,bool) or not math.isfinite(start) or not math.isfinite(stop)
                or start >= stop or payload['start_hex'] != float(start).hex() or payload['stop_hex'] != float(stop).hex()
                or type(payload['input_epoch']) is not int or payload['input_epoch'] < 0):
            raise ContractError('Native accepted interval exact time/epoch differs')

        def array(value, shape):
            if hasattr(value, 'detach'):
                if backend.ops.torch is None or value.dtype != backend.ops.torch.float64 or value.device != backend.ops.device:
                    raise ContractError('Native accepted coefficient device/FP64 differs')
            elif np.asarray(value).dtype != np.float64:
                raise ContractError('Native accepted coefficient arrays must remain FP64')
            result = backend.native._array(value)
            if tuple(result.shape) != shape:
                raise ContractError('Native accepted coefficient shape differs')
            backend.ops.finite(result)
            return result

        raw_anchor = payload['anchor']
        if (not isinstance(raw_anchor,dict) or set(raw_anchor) != {'time_seconds','R','E','light','input_stop_seconds','input_epoch'}
                or raw_anchor['input_epoch'] != payload['input_epoch']):
            raise ContractError('Native accepted analytic anchor schema/epoch differs')
        anchor = AnalyticREAnchor(raw_anchor['time_seconds'], array(raw_anchor['R'],(backend.native.nc,)),
            array(raw_anchor['E'],(backend.native.nc,)), array(raw_anchor['light'],(backend.native.nc,)),
            raw_anchor['input_stop_seconds'], raw_anchor['input_epoch'])
        if not anchor.time_seconds <= start < stop <= anchor.input_stop_seconds:
            raise ContractError('Native accepted interval crosses its true input epoch')
        if not isinstance(payload['curves'],dict) or set(payload['curves']) != set(STATE_NAMES) or not isinstance(payload['metrics'],dict):
            raise ContractError('All native accepted state curves and metrics required')
        if payload['metrics'].get('solver_mode','candidate') != payload['solver_mode']:
            raise ContractError('Native accepted interval arm metadata differs')
        curves = {}
        for name, size in zip(STATE_NAMES, backend.native.sizes):
            records = payload['curves'][name]
            if not isinstance(records,list) or not records:
                raise ContractError('Native accepted curve lost an interval: '+name)
            segments = []
            for record in records:
                kind = record.get('kind') if isinstance(record,dict) else None
                expected = {'kind','t0','t1','t0_hex','t1_hex','y0','coefficients'}|({'decay'} if kind == 'exponential_forced' else set())
                if kind not in ('dop853_dense','exponential_forced') or set(record) != expected:
                    raise ContractError('Native accepted representation kind/schema differs')
                lo, hi = record['t0'], record['t1']
                if (not isinstance(lo,(int,float)) or not isinstance(hi,(int,float)) or isinstance(lo,bool) or isinstance(hi,bool)
                        or not math.isfinite(lo) or not math.isfinite(hi) or not start <= lo < hi <= stop
                        or record['t0_hex'] != float(lo).hex() or record['t1_hex'] != float(hi).hex()):
                    raise ContractError('Native accepted segment exact times differ')
                if kind == 'dop853_dense' and hi > lo+MAX_STEP:
                    raise ContractError('Native restored fast step exceeds the original 100us limit')
                if payload['solver_mode'] == 'joint_reference' and kind != 'dop853_dense':
                    raise ContractError('Joint reference cannot restore a candidate forced representation')
                y0 = array(record['y0'],(size,))
                coefficients = array(record['coefficients'],(7 if kind == 'dop853_dense' else 8,size))
                if kind == 'dop853_dense':
                    segment = _Dense(lo, hi, y0, coefficients, backend.ops)
                else:
                    decay = record['decay']
                    if not isinstance(decay,(float,int)) or isinstance(decay,bool) or not math.isfinite(decay) or decay < 0:
                        raise ContractError('Native restored fixed decay differs')
                    expected_decay = 1000/backend.native.p[{'g_i':'tau_is','Vbar':'tau_a','z1_aux':'tau_1'}.get(name,'tau_1')]
                    if name not in ('g_i','Vbar','z1_aux') or decay != expected_decay:
                        raise ContractError('Native restored forced mechanism/time constant differs')
                    segment = _Forced(lo, hi, y0, coefficients, decay, backend.ops)
                segments.append(segment)
            curve = _Curve(segments)
            if curve.t0 != start or curve.t1 != stop:
                raise ContractError('Native restored curve does not cover the complete accepted interval')
            curves[name] = curve
        interval = cls(backend, anchor, curves, payload['metrics'])
        if interval.source_version != source or interval.representation_id != payload['representation_id']:
            raise ContractError('Native restored representation identity differs')
        for t, side in ((start,'right'),(stop,'left')):
            backend.native._validate(interval.evaluate(t,side), anchor, t)
        return interval


@dataclass(frozen=True)
class NativeAdvanceResult:
    state: NativeMultirateState
    intervals: tuple
    metrics: dict


class NativeMultirateBackend:
    """Candidate native execution with original equations and explicit tolerances."""
    def __init__(self, native_physics, *, rtol, atol):
        if dict(native_physics.p) != SOURCE_VALUES:
            raise ContractError('Native multirate may not change source physiological parameters')
        self.native, self.ops = native_physics, _Ops(native_physics.torch, native_physics.device)
        a = np.asarray(atol, dtype=np.float64)
        if (not math.isfinite(rtol) or rtol*.25 < 100*np.finfo(float).eps
                or a.shape not in ((), (native_physics.nstate,)) or not np.isfinite(a).all() or np.any(a <= 0)):
            raise ContractError('Original explicit positive tolerances required without SciPy quarter clamping')
        self.rtol, self.atol = float(rtol), self.ops.array(np.broadcast_to(a, (native_physics.nstate,)).copy())
        self.identity = canonical_hash(dict(method=METHOD, source=native_physics.source_version,
            rtol=rtol, atol=a, max_step_seconds=MAX_STEP, slow_initial=.0005, slow_max=.001,
            corrections=3, shares=dict(time=.25, forcing=.25, coupling=.25, quadrature=.25),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        self.reference_identity = canonical_hash(dict(method='joint-native-reference-DOP853-fp64-100us-analyticRE-1',
            source=native_physics.source_version, rtol=rtol, atol=a, max_step_seconds=MAX_STEP,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        self.metrics = {}
        self._closed = False
        self._aux_B = None
        self._aux_hc_rows = np.empty(0, np.int64)
        self._prepare_aux_operator()

    def _prepare_aux_operator(self):
        """Exact necessary HC support; no full-vector aux release computation."""
        native, ops = self.native, self.ops
        if not len(native.aux_indices):
            return
        if ops.torch is None:
            from scipy import sparse
            chosen = native.B[np.asarray(native.aux_indices, dtype=np.int64)].tocsr()
            self._aux_hc_rows = np.unique(chosen.indices).astype(np.int64)
            self._aux_B = sparse.csr_matrix((chosen.data.copy(), np.searchsorted(self._aux_hc_rows, chosen.indices),
                chosen.indptr.copy()), shape=(len(native.aux_indices), len(self._aux_hc_rows)))
            return
        torch = ops.torch
        from .resident_cusparse import NativeCSRPlan
        if not isinstance(native.B, NativeCSRPlan):
            raise ContractError('CUDA native aux support requires the original retained CSR ALG2 plan')
        original = native.B.matrix
        rows = torch.as_tensor(native.aux_indices, dtype=torch.int64, device=ops.device)
        starts = original.crow_indices()[rows]
        counts = original.crow_indices()[rows+1]-starts
        crow = torch.cat((torch.zeros(1, dtype=torch.int64, device=ops.device), counts.cumsum(0)))
        entries = int(crow[-1].item())  # One setup scalar, never a state vector.
        positions = torch.repeat_interleave(starts, counts)+torch.arange(entries, dtype=torch.int64,
            device=ops.device)-torch.repeat_interleave(crow[:-1], counts)
        columns = original.col_indices().index_select(0, positions)
        self._aux_hc_rows, compact_columns = torch.unique(columns, sorted=True, return_inverse=True)
        matrix = torch.sparse_csr_tensor(crow, compact_columns, original.values().index_select(0, positions),
            size=(len(native.aux_indices), len(self._aux_hc_rows)), dtype=torch.float64, device=ops.device)
        if len(self._aux_hc_rows):
            self._aux_B = NativeCSRPlan(matrix, native.B.api, torch)

    def close(self):
        if self.ops.torch is not None and self._aux_B is not None:
            self._aux_B.close()
            self._aux_B = None
        self._closed = True

    def initial_state(self, y, anchor):
        """Bind supplied legal state; never generates a background or history."""
        if self._closed:
            raise ContractError('Native multirate backend is closed')
        y = self.native._array(y)
        self.native._validate(y, anchor, anchor.time_seconds)
        return NativeMultirateState(anchor.time_seconds, y, anchor, self.identity)

    def refresh_input(self, state, light, input_stop_seconds, input_epoch):
        self._guard(state)
        if state.time_seconds != state.anchor.input_stop_seconds or input_epoch != state.anchor.input_epoch+1:
            raise ContractError('Native input refresh requires exact real boundary and next epoch')
        R, E = state.anchor.at(state.time_seconds)
        anchor = AnalyticREAnchor(state.time_seconds, R, E, self.native._array(light), input_stop_seconds, input_epoch)
        self.native._validate(state.y, anchor, state.time_seconds)
        return NativeMultirateState(state.time_seconds, state.y, anchor, state.backend_id)

    def _guard(self, state):
        if self._closed or not isinstance(state, NativeMultirateState) or state.backend_id not in (self.identity, self.reference_identity):
            raise ContractError('Native multirate state/backend identity differs')
        state.check_owned()
        self.native._validate(state.y, state.anchor, state.time_seconds)

    def _atol(self, *names):
        return self.ops.cat([self.atol[self.native.slices[n]] for n in names])

    def _allowance(self, name, y0, y1):
        return .25*(self.atol[self.native.slices[name]]+self.rtol*self.ops.maximum(self.ops.abs(y0), self.ops.abs(y1)))

    def _fast(self, fun, lo, y0, hi, names):
        began = perf_counter()
        device_timer = self._device_timer()
        curve, metrics = _solve_fast(fun, lo, y0, hi, rtol=self.rtol*.25,
            atol=self._atol(*names)*.25, ops=self.ops)
        self.metrics['fast_cuda_seconds'] += self._finish_device_timer(device_timer)
        self.metrics['fast_solve_wall_seconds'] += perf_counter()-began
        self.metrics['fast_nfev'] += metrics['nfev']
        self.metrics['fast_steps'] += metrics['steps']
        return curve

    def _device_timer(self):
        if self.ops.torch is None or self.ops.device.type != 'cuda':
            return None
        torch = self.ops.torch
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(self.ops.device))
        return start, stop

    def _finish_device_timer(self, timer):
        if timer is None:
            return 0.
        start, stop = timer
        stop.record(self.ops.torch.cuda.current_stream(self.ops.device))
        stop.synchronize()
        return start.elapsed_time(stop)*.001

    def _forced(self, source, lo, hi, y0, decay, name):
        """Adaptive degree-seven forcing, with y-quarter converted to q units."""
        began = perf_counter()
        device_timer = self._device_timer()
        ops = self.ops
        # max(|y0|,|y1|) >= |y0|, so this conservative share never exceeds
        # the original accepted-step scale even for rapidly changing drives.
        allowed = self._allowance(name, y0, y0)
        gain = -math.expm1(-decay*(hi-lo))/decay if decay else hi-lo
        budget = allowed/gain
        accepted, maximum = [], ops.zeros(y0.shape)

        def fit(a, b):
            times = a+(b-a)*_LOBATTO
            times[0], times[-1] = a, b
            if np.any(np.diff(times) <= 0):
                raise ContractError('Native forcing nodes exhausted FP64 time precision')
            values = ops.stack([source(float(t)) for t in times])
            ops.finite(values)
            vander = np.polynomial.polynomial.polyvander((times-a)/(b-a), 7)
            self.metrics['forcing_source_evaluations'] += 8
            return ops.array(np.linalg.inv(vander))@values

        def visit(a, b, initial, depth):
            nonlocal maximum
            q, middle = fit(a, b), a+(b-a)/2
            if not a < middle < b:
                raise ContractError('Native forcing subdivision exhausted FP64 precision')
            halves = ((a, middle, fit(a, middle)), (middle, b, fit(middle, b)))
            error = ops.zeros(y0.shape)
            for c, d, half in halves:
                for x in _CHECK:
                    at = float(c+(d-c)*x)
                    if x == 0: at = c
                    if x == 1: at = d
                    truth = source(at)
                    full_value = _polynomial(q, (at-a)/(b-a))
                    half_value = _polynomial(half, (at-c)/(d-c))
                    error = ops.maximum(error, ops.maximum(ops.abs(full_value-truth), ops.abs(half_value-truth)))
                    error = ops.maximum(error, ops.abs(full_value-half_value))
                    self.metrics['forcing_source_evaluations'] += 1
            ops.finite(error)
            if not ops.any(error > budget):
                segment = _Forced(a, b, initial, q, decay, ops)
                accepted.append(segment)
                maximum = ops.maximum(maximum, gain*error)
                return segment.at(b)
            if depth >= 20:
                raise ContractError('Native forcing error allocation exhausted subdivision')
            self.metrics['forcing_subdivisions'] += 1
            at_mid = visit(a, middle, initial, depth+1)
            return visit(middle, b, at_mid, depth+1)

        visit(lo, hi, y0, 0)
        self.metrics['max_forcing_quarter_norm'] = max(self.metrics['max_forcing_quarter_norm'], ops.norm(maximum/allowed))
        self.metrics['forcing_cuda_seconds'] += self._finish_device_timer(device_timer)
        self.metrics['forcing_wall_seconds'] += perf_counter()-began
        return _Curve(accepted), maximum

    def _vis_gi(self, lo, hi, V0, g0, X):
        ops, p = self.ops, self.native.p
        decay = 1000/p['tau_is']
        q0 = 1000*p['a_is']*V0**p['gamma']/p['tau_is']
        predicted = _Curve([_Forced(lo, hi, g0, ops.stack([q0]+[q0*0.]*7), decay, ops)])
        points = sorted({lo, hi, *(lo+(hi-lo)*float(x) for x in _CHECK)})

        def solve(g):
            def fun(t, V):
                current_g = g.at(t)
                if ops.any(V < 0) or ops.any(current_g <= 0):
                    raise ContractError('Native Vis/gi domain violated; no clipping or hidden fallback')
                return 1000*(X.at(t)/current_g-V)/p['tau_m']
            return self._fast(fun, lo, V0, hi, ('Vis',))

        def update(V):
            def q(t):
                voltage = V.at(t)
                if ops.any(voltage < 0):
                    raise ContractError('Native gi driver has negative voltage')
                return 1000*p['a_is']*voltage**p['gamma']/p['tau_is']
            return self._forced(q, lo, hi, g0, decay, 'g_i')

        def consistent(V, g, recomputed):
            error = 0.
            for t in points:
                a = self._allowance('g_i', g0, g.at(t))
                error = max(error, ops.norm((g.at(t)-recomputed.at(t))/a))
                true_g_rhs = 1000*(p['a_is']*V.at(t)**p['gamma']-g.at(t))/p['tau_is']
                gain = -math.expm1(-decay*(hi-lo))/decay
                error = max(error, ops.norm(gain*(g.derivative(t)-true_g_rhs)/a))
                true_V_rhs = 1000*(X.at(t)/g.at(t)-V.at(t))/p['tau_m']
                av = self._allowance('Vis', V0, V.at(t))
                error = max(error, ops.norm((hi-lo)*(V.derivative(t)-true_V_rhs)/av))
            return error

        for correction in range(3):
            self.metrics['waveform_corrections'] += 1
            V = solve(predicted)
            g, _ = update(V)
            difference = max(ops.norm((g.at(t)-predicted.at(t))/self._allowance('g_i', g0, g.at(t))) for t in points)
            if difference <= 1.:
                final_V = solve(g)  # Last voltage explicitly uses last accepted gi.
                check_g, forcing_error = update(final_V)
                residual = consistent(final_V, g, check_g)
                self.metrics['max_attempted_coupling_quarter_norm'] = max(self.metrics['max_attempted_coupling_quarter_norm'], residual)
                if residual <= 1.:
                    self.metrics['max_coupling_quarter_norm'] = max(self.metrics['max_coupling_quarter_norm'], residual)
                    return final_V, g
                predicted = check_g
            else:
                predicted = g
        raise _CouplingRetry()

    def _window(self, state, stop, *, joint_fallback=False):
        ops, p, native = self.ops, self.native.p, self.native
        start, anchor = state.time_seconds, state.anchor
        initial = native.blocks(state.y)
        nc, nh = native.nc, native.nh
        curves = {}
        began = perf_counter()
        def xc_rhs(t, value):
            X, C = value[:nc], value[nc:]
            _, E = anchor.at(t)
            if ops.any(X < 0) or ops.any(C < 0):
                raise ContractError('Native X/C domain violated')
            return ops.cat((1000*(1/(1+(p['a_C']*C)**p['n_C'])-(p['c_beta']+p['k_beta']*E)*X),
                            1000*(X-C)/p['tau_C']))
        xc = self._fast(xc_rhs, start, ops.cat((initial['X'], initial['C'])), stop, ('X', 'C'))
        curves['X'], curves['C'] = xc.cut(0, nc), xc.cut(nc, nc)
        del xc
        self.metrics['XC_wall_seconds'] += perf_counter()-began
        began = perf_counter()
        before_fast, before_forcing = self.metrics['fast_solve_wall_seconds'], self.metrics['forcing_wall_seconds']
        try:
            if joint_fallback:
                def vg_rhs(t, value):
                    V, g = value[:nc], value[nc:]
                    if ops.any(V < 0) or ops.any(g <= 0):
                        raise ContractError('Joint fallback native Vis/gi domain violated')
                    return ops.cat((1000*(curves['X'].at(t)/g-V)/p['tau_m'],
                        1000*(p['a_is']*V**p['gamma']-g)/p['tau_is']))
                vg = self._fast(vg_rhs, start, ops.cat((initial['Vis'], initial['g_i'])), stop, ('Vis', 'g_i'))
                curves['Vis'], curves['g_i'] = vg.cut(0, nc), vg.cut(nc, nc)
                self.metrics['joint_Vis_gi_fallbacks'] += 1
            else:
                curves['Vis'], curves['g_i'] = self._vis_gi(start, stop, initial['Vis'], initial['g_i'], curves['X'])
        finally:
            elapsed = perf_counter()-began
            self.metrics['Vis_gi_wall_seconds'] += elapsed
            self.metrics['coupling_control_wall_seconds'] += max(0., elapsed-
                (self.metrics['fast_solve_wall_seconds']-before_fast)-(self.metrics['forcing_wall_seconds']-before_forcing))
        began = perf_counter()
        curves['Vbar'], _ = self._forced(lambda t:1000*curves['Vis'].at(t)/p['tau_a'], start, stop,
                                        initial['Vbar'], 1000/p['tau_a'], 'Vbar')
        projection_started = perf_counter()
        device_timer = self._device_timer()
        pooled = _Curve([_Forced(s.t0, s.t1, native._mv(native.A, s.y0),
            ops.stack([native._mv(native.A, row) for row in s.coefficients]), s.decay, ops)
            for s in curves['Vbar'].segments])
        self.metrics['basis_projection_cuda_seconds'] += self._finish_device_timer(device_timer)
        self.metrics['basis_projection_wall_seconds'] += perf_counter()-projection_started
        self.metrics['Vbar_wall_seconds'] += perf_counter()-began
        began = perf_counter()
        def hc_rhs(t, value):
            w, z2, h = value[:nh], value[nh:2*nh], value[2*nh:]
            V, Vbar, pooledVbar = curves['Vis'].at(t), curves['Vbar'].at(t), pooled.at(t)
            if ops.any(Vbar <= 0) or ops.any(pooledVbar <= 0):
                raise ContractError('Native HC dynamic p/adaptation domain violated')
            bh = native._mv(native.B, h)
            exp = np.exp if ops.torch is None else ops.torch.exp
            with np.errstate(over='ignore'):
                q = 1/(1+exp(-((V-bh-p['V_k'])/p['V_n'])))
            release = q/(Vbar/p['V_I'])**p['mu']
            aI = (pooledVbar/p['V_I'])**p['mu']
            return ops.cat((1000*(p['g_t']*native._mv(native.A, release)-w)/p['tau_1'],
                            1000*(w-z2)/(p['tau_2']*aI), 1000*(z2-h)/(p['tau_h']*aI)))
        hc = self._fast(hc_rhs, start, ops.cat((initial['w'], initial['z2'], initial['Vh'])), stop, ('w','z2','Vh'))
        curves['w'], curves['z2'], curves['Vh'] = hc.cut(0, nh), hc.cut(nh, nh), hc.cut(2*nh, nh)
        del hc
        self.metrics['HC_wall_seconds'] += perf_counter()-began
        began = perf_counter()
        if len(native.aux_indices):
            def aux_q(t):
                V, Vbar = curves['Vis'].at(t, native._aux), curves['Vbar'].at(t, native._aux)
                h = curves['Vh'].at(t, self._aux_hc_rows)
                bh = V*0. if self._aux_B is None else native._mv(self._aux_B, h)
                exp = np.exp if ops.torch is None else ops.torch.exp
                with np.errstate(over='ignore'):
                    release = 1/(1+exp(-((V-bh-p['V_k'])/p['V_n'])))/(Vbar/p['V_I'])**p['mu']
                return 1000*p['g_t']*release/p['tau_1']
            curves['z1_aux'], _ = self._forced(aux_q, start, stop, initial['z1_aux'], 1000/p['tau_1'], 'z1_aux')
        else:
            zero = ops.zeros((0,))
            curves['z1_aux'] = _Curve([_Forced(start, stop, zero, ops.zeros((8, 0)), 1000/p['tau_1'], ops)])
        self.metrics['auxiliary_wall_seconds'] += perf_counter()-began
        result = AcceptedNativeInterval(self, anchor, curves, dict(joint_fallback=joint_fallback))
        end = result.evaluate(stop, 'left')
        native._validate(end, anchor, stop)
        native._require_finite(native.evaluate(end, stop, anchor).rhs)
        return NativeMultirateState(stop, end, anchor, self.identity), result

    def advance_one(self, state, stop_limit):
        """Produce one accepted slow window, bounded by the real input end.

        Rod ticks and CPU sample requests do not become native restart points.
        The owner retains this frontier plus its unconsumed representations.
        """
        self._guard(state)
        if not math.isfinite(stop_limit) or stop_limit <= state.time_seconds:
            raise ContractError('Native one-window limit must be finite and in the future')
        return self.advance(state, min(stop_limit, state.anchor.input_stop_seconds), one_window=True)

    def advance(self, state, stop, *, on_accept=None, one_window=False):
        self._guard(state)
        if state.backend_id != self.identity:
            raise ContractError('Reference state cannot silently become a multirate candidate state')
        if not math.isfinite(stop) or not state.time_seconds < stop <= state.anchor.input_stop_seconds:
            raise ContractError('Native candidate advance must end at or before the real input boundary')
        self.metrics = dict(fast_nfev=0, fast_steps=0, forcing_source_evaluations=0, forcing_subdivisions=0,
            waveform_corrections=0, coupling_window_rejections=0, joint_Vis_gi_fallbacks=0,
            max_coupling_quarter_norm=0., XC_wall_seconds=0., Vis_gi_wall_seconds=0., Vbar_wall_seconds=0.,
            max_attempted_coupling_quarter_norm=0., max_forcing_quarter_norm=0.,
            HC_wall_seconds=0., auxiliary_wall_seconds=0., fast_solve_wall_seconds=0., forcing_wall_seconds=0.,
            coupling_control_wall_seconds=0., basis_projection_wall_seconds=0., fast_cuda_seconds=0.,
            forcing_cuda_seconds=0., basis_projection_cuda_seconds=0.)
        if on_accept is not None and not callable(on_accept):
            raise ContractError('Accepted interval consumer must be callable')
        if type(one_window) is not bool:
            raise ContractError('Native one-window scheduling choice must be explicit boolean')
        intervals, current, H = [], state, state.next_window_seconds
        if not math.isfinite(H) or not MAX_STEP <= H <= .001:
            raise ContractError('Native slow-window controller state differs from the approved limits')
        began = perf_counter()
        while current.time_seconds < stop:
            end = min(current.time_seconds+H, stop)
            try:
                next_state, interval = self._window(current, end)
            except _CouplingRetry:
                self.metrics['coupling_window_rejections'] += 1
                if end-current.time_seconds <= MAX_STEP:
                    next_state, interval = self._window(current, end, joint_fallback=True)
                else:
                    H = max(MAX_STEP, (end-current.time_seconds)/2)
                    continue
            H = min(.001, H*2)
            current = replace(next_state, next_window_seconds=H)
            if on_accept is None:
                intervals.append(interval)
            else:
                # A runner can consume/release each immutable window and keep
                # only its current/previous consumer-owned representations.
                on_accept(interval)
            if one_window:
                break
        self.metrics.update(wall_seconds=perf_counter()-began,
            execution='numpy_cpu_reference' if self.ops.torch is None else 'explicit_torch_device_native',
            complete_model_advanced=False, D2H_full_vectors_per_stage=0,
            timing_scope='host wall by native stage; CUDA caller must synchronize/event-time before speed claims')
        return NativeAdvanceResult(current, tuple(intervals), dict(self.metrics))

    def advance_joint_reference(self, state, stop, *, on_accept=None):
        """Same equations/state/input, full joint DOP853 at the original budget.

        This explicitly forks a supplied candidate initial state into a distinct
        reference identity. Subsequent reference states cannot enter advance().
        A caller should consume bounded windows to avoid retaining all dense
        coefficients for a long input frame or experiment.
        """
        self._guard(state)
        if not math.isfinite(stop) or not state.time_seconds < stop <= state.anchor.input_stop_seconds:
            raise ContractError('Joint native reference must stop at or before the real input boundary')
        if on_accept is not None and not callable(on_accept):
            raise ContractError('Reference accepted interval consumer must be callable')
        began = perf_counter()
        timer = self._device_timer()
        whole, counters = _solve_fast(lambda t, y:self.native.rhs(y, t, state.anchor), state.time_seconds,
            state.y, stop, rtol=self.rtol, atol=self.atol, ops=self.ops)
        device_seconds = self._finish_device_timer(timer)
        curves, offset = {}, 0
        for name, size in zip(STATE_NAMES, self.native.sizes):
            curves[name] = whole.cut(offset, size)
            offset += size
        interval = AcceptedNativeInterval(self, state.anchor, curves, dict(solver_mode='joint_reference'))
        end = interval.evaluate(stop, 'left')
        self.native._validate(end, state.anchor, stop)
        final = NativeMultirateState(stop, end, state.anchor, self.reference_identity, state.next_window_seconds)
        metrics = dict(counters, solver_mode='joint_reference', original_rtol=self.rtol,
            original_atol_preserved=True, wall_seconds=perf_counter()-began, native_cuda_seconds=device_seconds,
            D2H_full_vectors_per_stage=0, complete_model_advanced=False)
        if on_accept is not None:
            on_accept(interval)
        return NativeAdvanceResult(final, (interval,) if on_accept is None else (), metrics)
