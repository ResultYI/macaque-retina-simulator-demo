"""Lifecycle adapter for the structured native Radau candidate.

The equation provider, ordered state IDs, input epochs and external event clocks
are unchanged. Accepted intervals own their full state anchors; per-state curves
are views and selected readout packets own only their requested rows.
"""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import hashlib
import math

import numpy as np

from .contracts import ContractError, canonical_hash
from .cone_hc_reduced_native import AnalyticREAnchor, STATE_NAMES
from .group_native_radau import (GroupNativeRadau, NativeRadauSegment,
                                 PiecewiseNativeRadauDense, METHOD, MAX_STEP)
from .native_multirate import _Ops, NativeMultirateState, NativeAdvanceResult


INTERVAL_SCHEMA = 'retipath-group-native-radau-accepted-interval-1'
ANCHORS = ('y0', 'f0', 'ym', 'fm', 'y1', 'f1')
WINDOW_SECONDS = .0005


@dataclass(frozen=True)
class DeviceQuinticRepresentation:
    t0: float
    t1: float
    y0: object
    f0: object
    ym: object
    fm: object
    y1: object
    f1: object
    state_ids: object
    input_epoch: int
    kind: str = 'quintic_hermite'


def _basis_value(segment, rows, t, ops, *, derivative=False):
    x = segment._theta(t)
    y0, f0, ym, fm, y1, f1 = (getattr(segment, n)[rows] for n in ANCHORS)
    if t == segment.start:
        return ops.clone(f0 if derivative else y0)
    if t == segment.stop:
        return ops.clone(f1 if derivative else y1)
    h = segment.stop-segment.start
    if t == segment.start+h/2:
        return ops.clone(fm if derivative else ym)
    l0, lm, l1 = 2*x*x-3*x+1, 4*x*(1-x), 2*x*x-x
    if not derivative:
        return ((1+6*x)*l0*l0*y0+lm*lm*ym+(7-6*x)*l1*l1*y1
            + h*(x*l0*l0*f0+(x-.5)*lm*lm*fm+(x-1)*l1*l1*f1))
    d0, dm, d1 = 4*x-3, 4-8*x, 4*x-1
    return ((6*l0*l0+2*(1+6*x)*l0*d0)*y0+2*lm*dm*ym
        +(-6*l1*l1+2*(7-6*x)*l1*d1)*y1)/h + (
        (l0*l0+2*x*l0*d0)*f0+(lm*lm+2*(x-.5)*lm*dm)*fm+(l1*l1+2*(x-1)*l1*d1)*f1)


class _BlockSegment:
    """A state-block view of one owned full-state quintic, without anchor copies."""
    kind = 'quintic_hermite'

    def __init__(self, segment, rows, ops):
        self.parent, self.rows, self.ops = segment, rows, ops
        self.t0, self.t1 = segment.start, segment.stop

    def _check(self):
        self.parent._theta(self.t0)
        if (self.t0, self.t1) != (self.parent.start, self.parent.stop):
            raise ContractError('Native Radau curve view times were mutated')

    def at(self, t):
        self._check()
        return _basis_value(self.parent, self.rows, t, self.ops)

    def derivative(self, t):
        self._check()
        return _basis_value(self.parent, self.rows, t, self.ops, derivative=True)

    def at_rows(self, t, rows):
        return self.ops.select(self.at(t), rows)

    def readout(self, rows, epoch):
        self._check()
        arrays = tuple(self.ops.select(getattr(self.parent, name)[self.rows], rows) for name in ANCHORS)
        return DeviceQuinticRepresentation(self.t0, self.t1, *arrays,
            np.asarray(rows, dtype=np.int64).copy(), epoch)


class _CurveView:
    def __init__(self, segments, rows, ops):
        self.segments = tuple(_BlockSegment(segment, rows, ops) for segment in segments)
        self.t0, self.t1 = self.segments[0].t0, self.segments[-1].t1

    def segment(self, t, *, right=False):
        for segment in reversed(self.segments) if right else self.segments:
            if segment.t0 <= t <= segment.t1:
                return segment
        raise ContractError('Native Radau curve query outside accepted interval')

    def at(self, t, rows=None):
        segment = self.segment(t)
        return segment.at(t) if rows is None else segment.at_rows(t, rows)

    def derivative(self, t):
        return self.segment(t).derivative(t)


class AcceptedNativeRadauInterval:
    def __init__(self, backend, anchor, dense, metrics):
        if not isinstance(dense, PiecewiseNativeRadauDense) or not dense.segments:
            raise ContractError('Native Radau accepted interval needs owned quintic segments')
        segments = dense.segments
        if any(a.stop != b.start for a, b in zip(segments, segments[1:])):
            raise ContractError('Native Radau accepted interval lost a macro step')
        self.backend, self.anchor, self.dense = backend, anchor, dense
        self.start, self.stop = segments[0].start, segments[-1].stop
        if not anchor.time_seconds <= self.start < self.stop <= anchor.input_stop_seconds:
            raise ContractError('Native Radau accepted interval crosses its input epoch')
        if any(segment.anchor is not anchor for segment in segments):
            raise ContractError('Native Radau accepted segments must share their actual input anchor')
        self.input_epoch, self.available_until = anchor.input_epoch, self.stop
        self.source_version = backend.identity
        self.metrics = dict(metrics, solver_mode='candidate')
        self.representation_id = canonical_hash(dict(method=METHOD, backend=self.source_version,
            t0=self.start, t1=self.stop, input_epoch=self.input_epoch,
            macro_times=[(s.start, s.stop) for s in segments]))
        self.curves = {name: _CurveView(segments, backend.native.slices[name], backend.ops)
                       for name in STATE_NAMES}
        self._binding = (self.start, self.stop, self.source_version, self.representation_id,
                         self.input_epoch, id(self.backend), id(self.anchor), id(self.dense))

    def _check(self, t, side):
        self.anchor.check_owned()
        if self._binding != (self.start, self.stop, self.source_version, self.representation_id,
                self.input_epoch, id(self.backend), id(self.anchor), id(self.dense)):
            raise ContractError('Accepted native Radau interval identity was mutated')
        if (not math.isfinite(t) or not self.start <= t <= self.stop or side not in ('left', 'right')
                or t == self.start and side != 'right' or t == self.stop and side != 'left'):
            raise ContractError('Wrong native Radau accepted interval or input side')

    def evaluate(self, t, side='right', rows=None):
        self._check(t, side)
        if rows is None:
            return self.dense.at(t)
        # Index only the requested original coordinates before evaluating the basis.
        if self.backend.ops.torch is None:
            selected = np.asarray(rows, dtype=np.int64)
        else:
            selected = self.backend.ops.torch.as_tensor(rows, dtype=self.backend.ops.torch.int64,
                                                       device=self.backend.ops.device)
        return _basis_value(self.dense._segment(t), selected, t, self.backend.ops)

    def representation_derivative(self, t, side='right', rows=None):
        self._check(t, side)
        result = self.dense.representation_derivative(t)
        return result if rows is None else self.backend.ops.select(result, rows)

    def rhs_at(self, t, side='right', rows=None):
        result = self.backend.native.rhs(self.evaluate(t, side), t, self.anchor)
        return result if rows is None else self.backend.ops.select(result, rows)

    def outputs_at(self, t, side='right'):
        return self.backend.native.evaluate(self.evaluate(t, side), t, self.anchor).outputs

    def readout_chunks(self, cone_rows, hc_rows):
        checked = []
        for rows, size in ((cone_rows, self.backend.native.nc), (hc_rows, self.backend.native.nh)):
            value = np.asarray(rows)
            if (value.ndim != 1 or value.dtype.kind not in 'iu' or np.any(value < 0)
                    or np.any(value >= size) or len(np.unique(value)) != len(value)):
                raise ContractError('Radau readout requires distinct existing integer native rows')
            checked.append(value.astype(np.int64, copy=True))
        cone_rows, hc_rows = checked
        if not len(cone_rows):
            raise ContractError('Native Radau readout needs at least one declared cone row')
        self._check(self.start, 'right')
        for i in range(len(self.dense.segments)):
            yield {name: None if not len(rows) else self.curves[name].segments[i].readout(rows, self.input_epoch)
                   for name, rows in (('Vis', cone_rows), ('Vbar', cone_rows), ('Vh', hc_rows))}

    def payload(self):
        self._check(self.start, 'right')
        records = []
        for segment in self.dense.segments:
            segment._theta(segment.start)
            records.append(dict(kind='quintic_hermite', t0=segment.start, t1=segment.stop,
                t0_hex=float(segment.start).hex(), t1_hex=float(segment.stop).hex(),
                **{name: getattr(segment, name) for name in ANCHORS}))
        return dict(schema=INTERVAL_SCHEMA, scope='native_accepted_interval_not_complete_model_checkpoint',
            source_version=self.source_version, equation_source_version=self.backend.native.source_version,
            representation_id=self.representation_id, solver_mode='candidate',
            start=self.start, stop=self.stop, start_hex=float(self.start).hex(), stop_hex=float(self.stop).hex(),
            input_epoch=self.input_epoch, state_names=list(STATE_NAMES), state_sizes=list(self.backend.native.sizes),
            dtype='float64', anchor=dict(time_seconds=self.anchor.time_seconds, R=self.anchor.R, E=self.anchor.E,
                light=self.anchor.light, input_stop_seconds=self.anchor.input_stop_seconds,
                input_epoch=self.anchor.input_epoch), segments=records, metrics=dict(self.metrics))

    @classmethod
    def from_payload(cls, backend, payload):
        fields = {'schema', 'scope', 'source_version', 'equation_source_version', 'representation_id',
                  'solver_mode', 'start', 'stop', 'start_hex', 'stop_hex', 'input_epoch', 'state_names',
                  'state_sizes', 'dtype', 'anchor', 'segments', 'metrics'}
        if (not isinstance(payload, dict) or set(payload) != fields or payload['schema'] != INTERVAL_SCHEMA
                or payload['scope'] != 'native_accepted_interval_not_complete_model_checkpoint'
                or payload['source_version'] != backend.identity
                or payload['equation_source_version'] != backend.native.source_version
                or payload['solver_mode'] != 'candidate' or payload['dtype'] != 'float64'
                or payload['state_names'] != list(STATE_NAMES) or payload['state_sizes'] != list(backend.native.sizes)
                or type(payload['input_epoch']) is not int or payload['input_epoch'] < 0):
            raise ContractError('Native Radau accepted interval schema/source/layout differs')

        def exact_time(value, encoded):
            return (type(value) in (int, float) and math.isfinite(value)
                    and encoded == float(value).hex())

        start, stop = payload['start'], payload['stop']
        if not (exact_time(start, payload['start_hex']) and exact_time(stop, payload['stop_hex']) and start < stop):
            raise ContractError('Native Radau accepted interval exact time differs')

        def array(value, shape):
            if hasattr(value, 'detach'):
                if (backend.ops.torch is None or value.dtype != backend.ops.torch.float64
                        or value.device != backend.ops.device):
                    raise ContractError('Native Radau restored tensor device/FP64 differs')
            elif np.asarray(value).dtype != np.float64:
                raise ContractError('Native Radau restored arrays must remain FP64')
            result = backend.native._array(value)
            if tuple(result.shape) != shape:
                raise ContractError('Native Radau restored anchor shape differs')
            backend.ops.finite(result)
            return result

        raw = payload['anchor']
        if (not isinstance(raw, dict) or set(raw) != {'time_seconds', 'R', 'E', 'light',
                'input_stop_seconds', 'input_epoch'} or raw['input_epoch'] != payload['input_epoch']):
            raise ContractError('Native Radau restored analytic anchor schema/epoch differs')
        anchor = AnalyticREAnchor(raw['time_seconds'], array(raw['R'], (backend.native.nc,)),
            array(raw['E'], (backend.native.nc,)), array(raw['light'], (backend.native.nc,)),
            raw['input_stop_seconds'], raw['input_epoch'])
        if not anchor.time_seconds <= start < stop <= anchor.input_stop_seconds:
            raise ContractError('Native Radau restored interval crosses its input epoch')
        records, metrics = payload['segments'], payload['metrics']
        if (not isinstance(records, list) or not records or not isinstance(metrics, dict)
                or metrics.get('solver_mode') != 'candidate'):
            raise ContractError('Native Radau restored segments or arm metadata differ')
        segments = []
        expected = {'kind', 't0', 't1', 't0_hex', 't1_hex', *ANCHORS}
        previous = start
        for record in records:
            if not isinstance(record, dict) or set(record) != expected or record['kind'] != 'quintic_hermite':
                raise ContractError('Native Radau restored representation kind/schema differs')
            lo, hi = record['t0'], record['t1']
            if not (exact_time(lo, record['t0_hex']) and exact_time(hi, record['t1_hex'])
                    and lo == previous and start <= lo < hi <= stop and hi <= lo+MAX_STEP):
                raise ContractError('Native Radau restored macro-step exact times differ')
            values = tuple(array(record[name], (backend.native.nstate,)) for name in ANCHORS)
            segment = NativeRadauSegment(lo, hi, *values, anchor, backend.native)
            for time, state in ((lo, segment.y0), (lo+(hi-lo)/2, segment.ym), (hi, segment.y1)):
                backend.native._validate(state, anchor, time)
            segments.append(segment)
            previous = hi
        if previous != stop:
            raise ContractError('Native Radau restored interval lost its final macro-step')
        result = cls(backend, anchor, PiecewiseNativeRadauDense(tuple(segments)), metrics)
        if result.representation_id != payload['representation_id']:
            raise ContractError('Native Radau restored representation identity differs')
        return result


class GroupNativeRadauBackend:
    """The existing native lifecycle protocol with an independent B-method ID."""
    def __init__(self, native_physics, **options):
        self.native = native_physics
        self.ops = _Ops(native_physics.torch, native_physics.device)
        self.core = GroupNativeRadau(native_physics, **options)
        self.equation_source_version = native_physics.source_version
        sources = ('group_native_radau.py', 'group_native_radau_adapter.py', 'group_radau_math.py',
                   'group_scc_schedule.py')
        self.identity = canonical_hash(dict(method=METHOD, core=self.core.identity,
            source=native_physics.source_version, native_window_seconds=WINDOW_SECONDS,
            implementation={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in sources}))
        self.reference_identity = canonical_hash(dict(method='joint-native-reference-DOP853-fp64-100us-analyticRE-1',
            source=native_physics.source_version, rtol=self.core.rtol,
            atol=np.asarray(self.core.atol, dtype=np.float64), max_step_seconds=MAX_STEP,
            implementation_sha256=hashlib.sha256(Path(__file__).with_name('native_multirate.py').read_bytes()).hexdigest()))
        self.metrics = {}
        self._closed = False

    def initial_state(self, y, anchor):
        if self._closed:
            raise ContractError('Native Radau backend is closed')
        y = self.native._array(y)
        self.native._validate(y, anchor, anchor.time_seconds)
        return NativeMultirateState(float(anchor.time_seconds), y, anchor, self.identity, WINDOW_SECONDS)

    def _guard(self, state):
        if (self._closed or not isinstance(state, NativeMultirateState) or state.backend_id != self.identity
                or self.native.source_version != self.equation_source_version
                or state.next_window_seconds != WINDOW_SECONDS):
            raise ContractError('Native Radau state/backend equation identity differs')
        state.check_owned()
        self.native._validate(state.y, state.anchor, state.time_seconds)

    def refresh_input(self, state, light, input_stop_seconds, input_epoch):
        self._guard(state)
        if (state.time_seconds != state.anchor.input_stop_seconds
                or type(input_epoch) is not int or input_epoch != state.anchor.input_epoch+1):
            raise ContractError('Native Radau input refresh requires the exact real boundary and next epoch')
        R, E = state.anchor.at(state.time_seconds)
        anchor = AnalyticREAnchor(state.time_seconds, R, E, self.native._array(light),
                                  input_stop_seconds, input_epoch)
        self.native._validate(state.y, anchor, state.time_seconds)
        return NativeMultirateState(state.time_seconds, state.y, anchor, self.identity, WINDOW_SECONDS)

    def advance_one(self, state, stop_limit):
        self._guard(state)
        if not math.isfinite(stop_limit) or stop_limit <= state.time_seconds:
            raise ContractError('Native Radau one-window limit must be finite and in the future')
        stop = float(min(state.time_seconds+WINDOW_SECONDS, stop_limit, state.anchor.input_stop_seconds))
        if stop <= state.time_seconds:
            raise ContractError('Native Radau real input boundary requires an explicit refresh')
        began = perf_counter()
        result = self.core.advance(state.y, state.time_seconds, stop, state.anchor)
        metrics = dict(result.metrics, wall_seconds=perf_counter()-began,
            execution='numpy_cpu_reference' if self.ops.torch is None else 'explicit_torch_device_native',
            complete_model_advanced=False, D2H_full_vectors_per_stage=0,
            timing_scope='host wall; CUDA event timing belongs to the paired runner')
        metrics['provider_domain_checks'] = metrics.get('provider_domain_checks', 0)+1
        if self.ops.torch is not None:
            metrics['host_scalar_reads'] = metrics.get('host_scalar_reads', 0)+1
        interval = AcceptedNativeRadauInterval(self, state.anchor, result.dense, metrics)
        final = NativeMultirateState(stop, result.y, state.anchor, self.identity, WINDOW_SECONDS)
        self.metrics = metrics
        return NativeAdvanceResult(final, (interval,), dict(metrics))

    def interval_from_payload(self, payload):
        if self._closed:
            raise ContractError('Native Radau backend is closed')
        return AcceptedNativeRadauInterval.from_payload(self, payload)

    def close(self):
        if not self._closed and hasattr(self.core._diagonal_plan, 'close'):
            self.core._diagonal_plan.close()
        self._closed = True
