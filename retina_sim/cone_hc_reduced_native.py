"""Opt-in FP64 Cone--HC aggregation and fixed exponential-midpoint candidate.

R/E retain analytic memory; w is HC-sized and is never labelled cone z1.
No torch import, CUDA initialization, graph construction or execution on import.
"""
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
import hashlib
import math
import numbers

import numpy as np
from scipy import sparse

from .contracts import ContractError, canonical_hash
from .vh2005 import SOURCE_VALUES


METHOD = "cone_hc_reduced_exponential_midpoint_fp64_100us_quadratic_v1"
BASE_STEP_SECONDS = 0.0001
STATE_NAMES = ("X", "C", "Vis", "g_i", "Vbar", "w", "z2", "Vh", "z1_aux")


def _tensor(value):
    return hasattr(value, "data_ptr") and hasattr(value, "_version")


def _owned(value):
    if _tensor(value):
        return value.detach().clone()
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _signature(value):
    if _tensor(value):
        return value.data_ptr(), value._version, tuple(value.shape), str(value.dtype)
    a = np.asarray(value)
    return a.__array_interface__["data"][0], a.shape, a.dtype.str, a.flags.writeable


def _same_shape(*values):
    shape = tuple(values[0].shape)
    return all(tuple(v.shape) == shape for v in values)


def quadratic_at(y0, f0, y1, start, stop, t):
    """Second-order continuous extension; no extrapolation or epsilon snapping."""
    if (not all(math.isfinite(float(x)) for x in (start, stop, t)) or
            not start < stop or not start <= t <= stop):
        raise ContractError("Reduced native query outside its accepted interval")
    if t == start:
        return _owned(y0)
    if t == stop:
        return _owned(y1)
    h = stop - start
    theta = (t - start) / h
    return y0 + theta * h * f0 + theta * theta * (y1 - y0 - h * f0)


def frozen_exponential(x, a, b, delta, *, torch=None):
    """Phi(x;a,b,delta), including the exact b=0 limit, with one expm1."""
    if not math.isfinite(float(delta)) or delta < 0:
        raise ContractError("Invalid frozen exponential interval")
    if isinstance(b, numbers.Real):
        u = math.expm1(-float(b) * delta)
        ratio = delta if b == 0 else -u / b
    elif torch is None:
        b = np.asarray(b, dtype=np.float64)
        u = np.expm1(-b * delta)
        ratio = np.divide(-u, b, out=np.full_like(b, delta), where=b != 0)
    else:
        u = torch.expm1(-b * delta)
        zero = b == 0
        denominator = torch.where(zero, torch.ones_like(b), b)
        ratio = torch.where(zero, torch.full_like(b, delta), -u / denominator)
    return x * (1 + u) + a * ratio


@dataclass(frozen=True)
class AnalyticREAnchor:
    time_seconds: float
    R: object
    E: object
    light: object
    input_stop_seconds: float
    input_epoch: int = 0
    _versions: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (not math.isfinite(float(self.time_seconds)) or
                not math.isfinite(float(self.input_stop_seconds)) or
                self.input_stop_seconds <= self.time_seconds or
                type(self.input_epoch) is not int or self.input_epoch < 0):
            raise ContractError("Invalid analytic input anchor/epoch")
        for name in ("R", "E", "light"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        if not _same_shape(self.R, self.E, self.light) or len(self.R.shape) != 1:
            raise ContractError("Analytic input-anchor layout differs")
        object.__setattr__(self, "_versions", tuple(_signature(v) for v in (self.R, self.E, self.light)))

    def check_owned(self):
        if self._versions != tuple(_signature(v) for v in (self.R, self.E, self.light)):
            raise ContractError("Analytic R/E anchor storage was mutated")

    def at(self, t):
        self.check_owned()
        if not math.isfinite(float(t)) or not self.time_seconds <= t <= self.input_stop_seconds:
            raise ContractError("Analytic R/E query crosses an unknown input boundary")
        if t == self.time_seconds:
            return _owned(self.R), _owned(self.E)
        dt_ms = (t - self.time_seconds) * 1000.
        tr, te = SOURCE_VALUES["tau_R"], SOURCE_VALUES["tau_E"]
        er, ee = math.expm1(-dt_ms/tr), math.expm1(-dt_ms/te)
        R = self.R + (self.light-self.R)*(-er)
        E = self.E + (self.light-self.E)*(-ee) + (self.R-self.light)*(tr/(tr-te))*(er-ee)
        return R, E

    def rhs(self, t):
        R, E = self.at(t)
        return {"R": 1000.*((self.light-R)/SOURCE_VALUES["tau_R"]),
                "E": 1000.*((R-E)/SOURCE_VALUES["tau_E"])}


@dataclass(frozen=True)
class ReducedNativeState:
    time_seconds: float
    y: object
    anchor: AnalyticREAnchor
    source_version: str
    _copy_array: bool = field(default=True, repr=False, compare=False)
    _version: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (not math.isfinite(float(self.time_seconds)) or not self.source_version or
                not self.anchor.time_seconds <= self.time_seconds <= self.anchor.input_stop_seconds):
            raise ContractError("Reduced state has invalid time/source/input anchor")
        if self._copy_array:
            object.__setattr__(self, "y", _owned(self.y))
        elif not _tensor(self.y):
            self.y.setflags(write=False)
        object.__setattr__(self, "_version", _signature(self.y))

    def check_owned(self):
        self.anchor.check_owned()
        if self._version != _signature(self.y):
            raise ContractError("Reduced native state storage was mutated")

    def fork(self):
        self.check_owned()
        return type(self)(self.time_seconds, self.y, self.anchor, self.source_version, False)


@dataclass(frozen=True)
class QuadraticNativeSegment:
    start: float
    stop: float
    y0: object
    f0: object
    y1: object
    input_epoch: int
    anchor: object = None
    source_version: str = METHOD
    _copy_arrays: bool = field(default=True, repr=False, compare=False)
    _versions: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (not math.isfinite(float(self.start)) or not math.isfinite(float(self.stop)) or
                not self.start < self.stop or type(self.input_epoch) is not int or self.input_epoch < 0):
            raise ContractError("Invalid reduced accepted interval/epoch")
        for name in ("y0", "f0", "y1"):
            value = getattr(self, name)
            if self._copy_arrays:
                object.__setattr__(self, name, _owned(value))
            elif not _tensor(value):
                value.setflags(write=False)
        if not _same_shape(self.y0, self.f0, self.y1):
            raise ContractError("Reduced accepted interval layout differs")
        if self.anchor is not None and (
                self.anchor.input_epoch != self.input_epoch or
                not self.anchor.time_seconds <= self.start < self.stop <= self.anchor.input_stop_seconds):
            raise ContractError("Reduced segment crosses its input epoch")
        object.__setattr__(self, "_versions", tuple(_signature(v) for v in (self.y0, self.f0, self.y1)))

    def check_owned(self):
        if self._versions != tuple(_signature(v) for v in (self.y0, self.f0, self.y1)):
            raise ContractError("Accepted reduced native storage was mutated")
        if self.anchor is not None:
            self.anchor.check_owned()

    def at(self, t):
        self.check_owned()
        return quadratic_at(self.y0, self.f0, self.y1, self.start, self.stop, t)

    def project_rows(self, indices):
        self.check_owned()
        if _tensor(self.y0):
            arrays = tuple(v.index_select(0, indices) for v in (self.y0, self.f0, self.y1))
        else:
            arrays = tuple(v[np.asarray(indices, dtype=np.int64)] for v in (self.y0, self.f0, self.y1))
        return type(self)(self.start, self.stop, *arrays, self.input_epoch,
                          source_version=self.source_version)


@dataclass(frozen=True)
class NativeEvaluation:
    rhs: object
    outputs: dict
    analytic_rhs: dict


class ReducedNative:
    """Exact source parameters/topology with the separately identified B method.

    A/B must be retained NativeCSRPlan objects for GPU execution. Passing no torch
    enables only the CPU reference used by the small mathematical unit tests.
    """

    def __init__(self, native_model, *, torch=None, device=None, A=None, B=None, aux_indices=()):
        if dict(native_model.p) != SOURCE_VALUES:
            raise ContractError("Reduced native requires unchanged complete source parameters")
        self.model, self.p = native_model, MappingProxyType(dict(SOURCE_VALUES))
        self.nc, self.nh = int(native_model.nc), int(native_model.nh)
        if self.nc <= 0 or self.nh <= 0:
            raise ContractError("Reduced native requires the actual nonempty cone/HC populations")
        chosen = tuple(aux_indices)
        if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) for i in chosen):
            raise ContractError("Auxiliary z1 requires explicit integer cone indices")
        self.aux_indices = tuple(int(i) for i in chosen)
        if len(set(chosen)) != len(chosen) or any(i < 0 or i >= self.nc for i in chosen):
            raise ContractError("Auxiliary z1 IDs must be distinct existing cones")
        self.aux_cell_ids = tuple(native_model.cones.cells.cell_id[i] for i in chosen)
        self.sizes = (self.nc,)*5 + (self.nh,)*3 + (len(chosen),)
        self.nstate = sum(self.sizes)
        self.slices, offset = {}, 0
        for name, n in zip(STATE_NAMES, self.sizes):
            self.slices[name] = slice(offset, offset+n)
            offset += n
        self.torch = torch
        if torch is None:
            self.device = None
            self.A = self._cpu_matrix(native_model.A if A is None else A)
            self.B = self._cpu_matrix(native_model.B if B is None else B)
            self._aux = np.asarray(self.aux_indices, dtype=np.int64)
        else:
            from .resident_cusparse import NativeCSRPlan
            if not isinstance(A, NativeCSRPlan) or not isinstance(B, NativeCSRPlan):
                raise ContractError("Reduced GPU native requires retained explicit CSR ALG2 plans")
            self.device = A.device if device is None else torch.device(device)
            if A.device != self.device or B.device != self.device:
                raise ContractError("Reduced native CSR device differs")
            self.A, self.B = A, B
            self._aux = torch.as_tensor(self.aux_indices, dtype=torch.int64, device=self.device)
        if tuple(self.A.shape) != (self.nh, self.nc) or tuple(self.B.shape) != (self.nc, self.nh):
            raise ContractError("Reduced native A/B topology layout differs")
        self._topology = self._topology_signature()
        self.identity = dict(method=METHOD, source_model_hash=str(native_model.model_hash),
                             implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             state_names=STATE_NAMES, state_sizes=self.sizes,
                             aux_indices=self.aux_indices, aux_cell_ids=self.aux_cell_ids,
                             parameters=dict(self.p), base_step_seconds=BASE_STEP_SECONDS,
                             analytic_RE="source two-stage filter; source-ms converted to seconds",
                             dense="x0+theta*h*f0+theta^2*(x1-x0-h*f0)",
                             execution="numpy_cpu_reference" if torch is None else "cuda_retained_cusparse_ALG2",
                             original_full_z1_available=False)
        self.source_version = canonical_hash(self.identity)
        self.statistics = dict(steps=0, coefficient_evaluations=0)

    @staticmethod
    def _cpu_matrix(matrix):
        result = sparse.csr_matrix(matrix, copy=True)
        if result.dtype != np.float64:
            raise ContractError("Reduced native matrices must remain FP64")
        for v in (result.data, result.indices, result.indptr):
            v.setflags(write=False)
        return result

    def _topology_signature(self):
        if self.torch is None:
            return tuple((id(m), tuple(_signature(v) for v in (m.data, m.indices, m.indptr)))
                         for m in (self.A, self.B))
        return tuple((id(m), id(m.matrix), tuple(_signature(v) for v in
                     (m.matrix.values(), m.matrix.col_indices(), m.matrix.crow_indices())))
                     for m in (self.A, self.B))

    def require_aux_indices(self, chosen):
        if tuple(chosen) != self.aux_indices:
            raise ContractError("Auxiliary z1 selection/order cannot change during a run")

    def _array(self, value):
        if self.torch is None:
            if _tensor(value):
                raise ContractError("No implicit GPU-to-CPU state migration")
            a = np.asarray(value)
            if a.dtype != np.float64:
                raise ContractError("Reduced native state/input must remain FP64")
            return a
        if not _tensor(value):
            return self.torch.as_tensor(value, dtype=self.torch.float64, device=self.device)
        if value.dtype != self.torch.float64 or value.device != self.device:
            raise ContractError("Reduced native state/input device or FP64 layout differs")
        return value

    def _cat(self, values):
        return np.concatenate(values) if self.torch is None else self.torch.cat(values)

    def split(self, y):
        if tuple(y.shape) != (self.nstate,):
            raise ContractError("Reduced native state layout differs")
        return tuple(y[sl] for sl in self.slices.values())

    def blocks(self, y):
        return dict(zip(STATE_NAMES, self.split(y)))

    def _require_finite(self, *values):
        if self.torch is None:
            finite = all(np.all(np.isfinite(v)) for v in values)
        else:
            flag = self.torch.isfinite(values[0]).all()
            for value in values[1:]:
                flag = flag & self.torch.isfinite(value).all()
            finite = flag.item()
        if not finite:
            raise ContractError("Nonfinite reduced native equation RHS/readout")

    def _mv(self, matrix, value):
        return matrix @ value if self.torch is None else matrix.mv(value.contiguous())

    def _validate(self, y, anchor, t):
        if self._topology_signature() != self._topology:
            raise ContractError("Reduced native immutable A/B storage was changed")
        y = self._array(y)
        X, C, V, gi, Vbar, *_ = self.split(y)
        R, E = anchor.at(t)
        if tuple(R.shape) != (self.nc,) or tuple(anchor.light.shape) != (self.nc,):
            raise ContractError("Reduced native light/analytic filter layout differs")
        if self.torch is None:
            invalid = (not all(np.all(np.isfinite(a)) for a in (y, R, E, anchor.light)) or
                       np.any(anchor.light < 0) or np.any(X < 0) or np.any(C < 0) or
                       np.any(V < 0) or np.any(gi <= 0) or np.any(Vbar <= 0))
        else:
            torch = self.torch
            flag = (~torch.isfinite(y)).any() | (~torch.isfinite(R)).any() | (~torch.isfinite(E)).any()
            flag = flag | (~torch.isfinite(anchor.light)).any() | (anchor.light < 0).any()
            flag = flag | (X < 0).any() | (C < 0).any() | (V < 0).any() | (gi <= 0).any() | (Vbar <= 0).any()
            invalid = flag.item()
        if invalid:
            raise ContractError("Reduced native finite/positive/nonnegative domain violated; no clipping")
        return R, E

    def migrate(self, old_native, *, time_seconds, light, input_stop_seconds, input_epoch=0):
        old = self._array(old_native)
        if tuple(old.shape) != (8*self.nc+2*self.nh,):
            raise ContractError("One-way migration requires the complete legal old native state")
        sizes = (self.nc,)*8 + (self.nh,)*2
        cuts = np.cumsum((0,)+sizes)
        R, E, X, C, V, gi, Vbar, z1, z2, h = tuple(old[cuts[i]:cuts[i+1]] for i in range(10))
        w = self._mv(self.A, z1)
        aux = z1[self._aux]
        y = self._cat((X, C, V, gi, Vbar, w, z2, h, aux))
        anchor = AnalyticREAnchor(float(time_seconds), R, E, self._array(light),
                                  float(input_stop_seconds), input_epoch)
        self._validate(y, anchor, float(time_seconds))
        return ReducedNativeState(float(time_seconds), y, anchor, self.source_version, False)

    def refresh_input(self, state, light, input_stop_seconds, input_epoch):
        self._state_guard(state)
        if state.time_seconds != state.anchor.input_stop_seconds:
            raise ContractError("Analytic input may refresh only at its actual input boundary")
        if input_epoch != state.anchor.input_epoch+1:
            raise ContractError("Analytic input epoch must advance exactly once at a boundary")
        R, E = state.anchor.at(state.time_seconds)
        anchor = AnalyticREAnchor(state.time_seconds, R, E, self._array(light),
                                  float(input_stop_seconds), input_epoch)
        self._validate(state.y, anchor, state.time_seconds)
        return ReducedNativeState(state.time_seconds, state.y, anchor, self.source_version, False)

    def _state_guard(self, state):
        state.check_owned()
        if state.source_version != self.source_version:
            raise ContractError("Reduced state source/auxiliary selection identity differs")

    def _coefficients(self, y, t, anchor):
        R, E = self._validate(y, anchor, t)
        X, C, V, gi, Vbar, w, z2, h, aux = self.split(y)
        p = self.p
        bh = self._mv(self.B, h)
        q = 1./(1.+(np.exp(-((V-bh-p['V_k'])/p['V_n'])) if self.torch is None else
                    self.torch.exp(-((V-bh-p['V_k'])/p['V_n']))))
        aI = (Vbar/p['V_I'])**p['mu']
        release = q/aI
        pooled = (self._mv(self.A, Vbar)/p['V_I'])**p['mu']
        ar = self._mv(self.A, release)
        a = (1000./(1.+(p['a_C']*C)**p['n_C']), 1000.*X/p['tau_C'],
             1000.*X/(gi*p['tau_m']), 1000.*p['a_is']*V**p['gamma']/p['tau_is'],
             1000.*V/p['tau_a'], 1000.*p['g_t']*ar/p['tau_1'],
             1000.*w/(p['tau_2']*pooled), 1000.*z2/(p['tau_h']*pooled),
             1000.*p['g_t']*release[self._aux]/p['tau_1'])
        b = (1000.*(p['c_beta']+p['k_beta']*E), 1000./p['tau_C'],
             1000./p['tau_m'], 1000./p['tau_is'], 1000./p['tau_a'],
             1000./p['tau_1'], 1000./(p['tau_2']*pooled),
             1000./(p['tau_h']*pooled), 1000./p['tau_1'])
        outputs = dict(zip(STATE_NAMES, (X, C, V, gi, Vbar, w, z2, h, aux)))
        outputs.update(R=R, E=E, q=q, release=release, Ios=X, Bh=bh, aI=aI, pooled_aI=pooled)
        self.statistics['coefficient_evaluations'] += 1
        return a, b, outputs

    def evaluate(self, y, t, anchor):
        """True reduced-equation RHS, never the quadratic polynomial derivative."""
        a, b, outputs = self._coefficients(y, t, anchor)
        rhs = self._cat(tuple(ai-bi*xi for ai, bi, xi in zip(a, b, self.split(y))))
        self._require_finite(rhs, outputs['release'])
        return NativeEvaluation(rhs, outputs, anchor.rhs(t))

    def rhs(self, y, t, anchor):
        return self.evaluate(y, t, anchor).rhs

    def step(self, state):
        """Advance once, bounded only by fixed100us and the actual input end."""
        self._state_guard(state)
        t, y0 = state.time_seconds, state.y
        stop = min(t+BASE_STEP_SECONDS, state.anchor.input_stop_seconds)
        if stop <= t:
            raise ContractError("Native input boundary requires an explicit right-side refresh")
        delta = stop-t
        a0, b0, outputs0 = self._coefficients(y0, t, state.anchor)
        blocks = self.split(y0)
        f0 = self._cat(tuple(ai-bi*xi for xi, ai, bi in zip(blocks, a0, b0)))
        self._require_finite(f0, outputs0['release'])
        midpoint = self._cat(tuple(frozen_exponential(xi, ai, bi, delta/2, torch=self.torch)
                                   for xi, ai, bi in zip(blocks, a0, b0)))
        am, bm, _ = self._coefficients(midpoint, t+delta/2, state.anchor)
        y1 = self._cat(tuple(frozen_exponential(xi, ai, bi, delta, torch=self.torch)
                            for xi, ai, bi in zip(blocks, am, bm)))
        self._validate(y1, state.anchor, stop)
        segment = QuadraticNativeSegment(t, stop, y0, f0, y1, state.anchor.input_epoch,
                                          state.anchor, self.source_version, False)
        result = ReducedNativeState(stop, y1, state.anchor, self.source_version, False)
        self.statistics['steps'] += 1
        return result, segment
