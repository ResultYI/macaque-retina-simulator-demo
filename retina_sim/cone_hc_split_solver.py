"""Persistent native-only FP64 DOP853 and owned accepted dense intervals.

New B-class numerical method, NOT RUN. Imports perform no CUDA initialization.
Controller state survives inner events and caller chunk boundaries. A real
native input discontinuity invalidates its cached derivative/initial step.
No rejected attempt or workspace alias is exposed to a downstream consumer.
"""
from copy import copy
from dataclasses import dataclass
import math

from .contracts import ContractError


@dataclass(frozen=True)
class AcceptedNativeDense:
    start: float
    stop: float
    y0: object
    y1: object
    coefficients: object
    input_epoch: int

    def __post_init__(self):
        if not math.isfinite(self.start) or not math.isfinite(self.stop) or self.stop <= self.start:
            raise ContractError('Invalid accepted native dense interval')
        object.__setattr__(self, '_versions', tuple(
            (v.data_ptr(), v._version) for v in (self.y0, self.y1, self.coefficients)))

    def check_owned(self):
        now = tuple((v.data_ptr(), v._version) for v in (self.y0, self.y1, self.coefficients))
        if now != self._versions:
            raise ContractError('Accepted native dense storage was mutated')

    def at(self, t):
        """Owned state at an accepted time; endpoints use accepted RK states."""
        self.check_owned()
        if not math.isfinite(t) or not self.start <= t <= self.stop:
            raise ContractError('Native dense query would extrapolate or expose an unaccepted state')
        if t == self.start:
            return self.y0.clone()
        if t == self.stop:
            return self.y1.clone()
        x = (t-self.start)/(self.stop-self.start)
        value = self.y0.new_zeros(self.y0.shape)
        for j, row in enumerate(reversed(self.coefficients.unbind(0))):
            value += row
            value *= x if j % 2 == 0 else 1-x
        return value + self.y0

    def project_rows(self, indices):
        """Gather coefficients before evaluation; do not project nonlinear release."""
        self.check_owned()
        return type(self)(self.start, self.stop,
                          self.y0.index_select(0, indices).clone(),
                          self.y1.index_select(0, indices).clone(),
                          self.coefficients.index_select(1, indices).clone(),
                          self.input_epoch)

    def payload(self):
        self.check_owned()
        return dict(start=self.start, stop=self.stop, input_epoch=self.input_epoch,
                    y0=self.y0.detach().cpu().numpy().tolist(),
                    y1=self.y1.detach().cpu().numpy().tolist(),
                    coefficients=self.coefficients.detach().cpu().numpy().tolist())

    @classmethod
    def from_payload(cls, raw, torch, template):
        if type(raw['input_epoch']) is not int or raw['input_epoch'] < 0:
            raise ContractError('Native dense checkpoint input epoch invalid')
        def tensor(name, shape):
            v = torch.as_tensor(raw[name], dtype=torch.float64, device=template.device).clone()
            if tuple(v.shape) != shape or not torch.isfinite(v).all().item():
                raise ContractError('Invalid native dense checkpoint array: '+name)
            return v
        n = template.numel()
        return cls(float(raw['start']), float(raw['stop']), tensor('y0', (n,)),
                   tensor('y1', (n,)), tensor('coefficients', (7, n)), raw['input_epoch'])


class PersistentNativeDop853:
    """One native state/controller; scratch workspace belongs to the runner.

    ``rhs(t, y, left_limit=...)`` must evaluate ONLY native light/native RHS.
    ``bound`` is the next *native* input breakpoint or available-input endpoint,
    never an inner event/observation/caller chunk end. ``discontinuity`` controls
    the right-side derivative refresh at the next call. Each accepted step
    produces all three dense RHS stages, even if it covers several inner ticks.
    """

    def __init__(self, t, y, *, rtol, atol, max_step):
        import torch
        from scipy.integrate._ivp.common import validate_max_step, validate_tol

        if (y.dtype != torch.float64 or y.device.type != 'cuda' or y.ndim != 1
                or not y.numel() or not math.isfinite(t) or not torch.isfinite(y).all().item()):
            raise ContractError('Native persistent solver requires finite owned CUDA FP64 state')
        if not math.isfinite(rtol) or rtol <= 0 or not math.isfinite(atol) or atol <= 0:
            raise ContractError('Native persistent solver requires positive scalar FP64 tolerances')
        self.rtol, self.atol = validate_tol(rtol, atol, y.numel())
        self.max_step = validate_max_step(max_step)
        self.t, self.y = float(t), y.clone()
        self.f, self.h_abs = None, None
        self.needs_input_restart = False
        self.input_epoch = 0
        self.failed = False
        self.statistics = dict(nfev=0, attempts=0, rejected_attempts=0, accepted_steps=0,
                               dense_polynomials=0, initializations=0, input_restarts=0)
        self._seal_storage()

    def _seal_storage(self):
        self._owned_versions = tuple(None if v is None else (v.data_ptr(), v._version)
                                     for v in (self.y, self.f))

    def _check_storage(self):
        current = tuple(None if v is None else (v.data_ptr(), v._version)
                        for v in (self.y, self.f))
        if current != self._owned_versions:
            raise ContractError('Persistent native solver owned state/derivative was mutated')

    def fork(self):
        # y/f are never overwritten; only newly allocated accepted values replace
        # them. This shares immutable rows, not a mutable solver/controller.
        other = copy(self)
        other.statistics = dict(self.statistics)
        return other

    def accept_next(self, rhs, bound, workspace, *, discontinuity):
        import numpy as np
        import torch
        from scipy.integrate._ivp.rk import DOP853, SAFETY, MIN_FACTOR, MAX_FACTOR

        if self.failed:
            raise ContractError('Failed native solver cannot continue; restore an earlier checkpoint')
        self._check_storage()
        if not math.isfinite(bound) or bound <= self.t:
            raise ContractError('Native solve requires a strictly future input bound')
        try:
            with workspace.lease(self.y, self.atol) as work:
                n = self.y.numel()
                t, y = self.t, self.y

                def fun(at, value):
                    self.statistics['nfev'] += 1
                    out = rhs(float(at), value,
                              left_limit=bool(discontinuity and at >= bound))
                    if (out.dtype != torch.float64 or out.device != y.device or out.shape != y.shape
                            or not torch.isfinite(out).all().item()):
                        raise ContractError('Nonfinite/malformed persistent native derivative')
                    return out

                if self.f is None or self.needs_input_restart:
                    if self.needs_input_restart:
                        self.input_epoch += 1
                        self.statistics['input_restarts'] += 1
                    f = fun(t, y)  # right side after an input discontinuity
                    scale = work.at + y.abs()*self.rtol
                    rms = lambda v: torch.linalg.vector_norm(v).item()/n**.5
                    d0, d1 = rms(y/scale), rms(f/scale)
                    h0 = min(1e-6 if d0 < 1e-5 or d1 < 1e-5 else .01*d0/d1, bound-t)
                    f1 = fun(t+h0, y+h0*f)
                    d2 = rms((f1-f)/scale)/h0
                    h1 = (max(1e-6, h0*1e-3) if d1 <= 1e-15 and d2 <= 1e-15
                          else (.01/max(d1, d2))**(1/8))
                    h_abs = min(100*h0, h1, bound-t, self.max_step)
                    self.statistics['initializations'] += 1
                else:
                    f, h_abs = self.f, self.h_abs
                minimum = 10*abs(np.nextafter(t, np.inf)-t)
                h_abs = min(max(h_abs, minimum), self.max_step)
                rejected = False
                K = work.K
                while True:
                    if h_abs < minimum:
                        raise ContractError('Persistent native DOP853 step below floating point spacing')
                    self.statistics['attempts'] += 1
                    t_new = min(t+h_abs, bound)
                    h = t_new-t
                    h_abs = abs(h)
                    if h <= 0:
                        raise ContractError('Persistent native solver lost forward progress')
                    K[0].copy_(f)
                    for stage in range(1, 12):
                        delta = torch.mv(K[:stage].T, work.A[stage, :stage])*h
                        K[stage].copy_(fun(t+DOP853.C[stage]*h, y+delta))
                    y_new = y+h*torch.mv(K[:12].T, work.B)
                    f_new = fun(t_new, y_new)
                    K[12].copy_(f_new)
                    scale = work.at+torch.maximum(y.abs(), y_new.abs())*self.rtol
                    e5 = torch.mv(K[:13].T, work.E5)/scale
                    e3 = torch.mv(K[:13].T, work.E3)/scale
                    n5 = torch.linalg.vector_norm(e5).item()**2
                    n3 = torch.linalg.vector_norm(e3).item()**2
                    error = 0. if n5 == 0 and n3 == 0 else abs(h)*n5/np.sqrt((n5+.01*n3)*n)
                    if not np.isfinite(error):
                        raise ContractError('Nonfinite persistent native error estimate')
                    if error < 1:
                        factor = MAX_FACTOR if error == 0 else min(MAX_FACTOR, SAFETY*error**(-1/8))
                        h_next = h_abs*(min(1, factor) if rejected else factor)
                        break
                    h_abs *= max(MIN_FACTOR, SAFETY*error**(-1/8))
                    self.statistics['rejected_attempts'] += 1
                    rejected = True
                # No consumer can see K or an unaccepted attempt. Dense stages
                # use the same input side as the accepted step ending at bound.
                for stage in range(13, 16):
                    delta = torch.mv(K[:stage].T, work.AE[stage-13, :stage])*h
                    K[stage].copy_(fun(t+DOP853.C_EXTRA[stage-13]*h, y+delta))
                delta = y_new-y
                F = work.dense_storage()
                F[0].copy_(delta)
                F[1].copy_(h*K[0]-delta)
                F[2].copy_(2*delta-h*(f_new+K[0]))
                F[3:].copy_(h*torch.mm(work.DD, K))
                if not torch.isfinite(F).all().item():
                    raise ContractError('Nonfinite accepted native dense coefficients')
                segment = AcceptedNativeDense(t, t_new, y.clone(), y_new.clone(), F.clone(), self.input_epoch)
                self.t, self.y, self.f, self.h_abs = t_new, y_new, f_new, float(h_next)
                self.needs_input_restart = bool(discontinuity and t_new == bound)
                self._seal_storage()
                self.statistics['accepted_steps'] += 1
                self.statistics['dense_polynomials'] += 1
                return segment
        except Exception:
            self.failed = True
            raise

    def payload(self):
        if self.failed:
            raise ContractError('Failed native solver is not a resumable checkpoint')
        self._check_storage()
        return dict(t=self.t, y=self.y.detach().cpu().numpy().tolist(),
                    f=None if self.f is None else self.f.detach().cpu().numpy().tolist(),
                    h_abs=self.h_abs, rtol=float(self.rtol), atol=float(self.atol), max_step=self.max_step,
                    needs_input_restart=self.needs_input_restart, input_epoch=self.input_epoch,
                    statistics=dict(self.statistics))

    @classmethod
    def from_payload(cls, raw, torch, template, policy):
        if any(raw[k] != policy[k] for k in ('rtol', 'atol', 'max_step')):
            raise ContractError('Persistent native checkpoint numerical policy differs')
        y = torch.as_tensor(raw['y'], dtype=torch.float64, device=template.device).clone()
        if y.shape != template.shape:
            raise ContractError('Persistent native checkpoint state layout differs')
        obj = cls(raw['t'], y, **{k: raw[k] for k in ('rtol', 'atol', 'max_step')})
        if raw['f'] is not None:
            obj.f = torch.as_tensor(raw['f'], dtype=torch.float64, device=y.device).clone()
            if obj.f.shape != y.shape or not torch.isfinite(obj.f).all().item():
                raise ContractError('Persistent native checkpoint derivative invalid')
        h = raw['h_abs']
        if (h is None) != (obj.f is None) or (h is not None and (not math.isfinite(h) or h <= 0)):
            raise ContractError('Persistent native checkpoint controller is incomplete')
        obj.h_abs = h
        if type(raw['needs_input_restart']) is not bool or type(raw['input_epoch']) is not int or raw['input_epoch'] < 0:
            raise ContractError('Persistent native checkpoint input-side epoch invalid')
        obj.needs_input_restart, obj.input_epoch = raw['needs_input_restart'], raw['input_epoch']
        stats = raw['statistics']
        if set(stats) != set(obj.statistics) or any(type(v) is not int or v < 0 for v in stats.values()):
            raise ContractError('Persistent native checkpoint accounting invalid')
        obj.statistics = dict(stats)
        obj._seal_storage()
        return obj
