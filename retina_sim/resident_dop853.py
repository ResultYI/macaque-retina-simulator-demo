"""Opt-in resident FP64 DOP853 storage; no CUDA work occurs at import time.

This is an independent transport variant of ``gpu_dop853.integrate``.  It keeps
the installed SciPy coefficients, arithmetic expressions, error reductions,
initial step selection and dense polynomial.  A workspace reuses storage only:
no derivative, step size or controller history survives an interval.

The original compact backend and its source identity are deliberately untouched.
This implementation is UNVALIDATED.  Creating a workspace or calling the solver
requires separately authorized runtime validation; merely importing this module
does not initialize CUDA, compile kernels or construct a network.
"""
from contextlib import contextmanager
from threading import Lock, get_ident

from .contracts import ContractError


class Dop853Workspace:
    """Exclusive, single-device/stream coefficients and RK storage.

    ``y_template`` supplies dtype, device and total state count, not initial
    values.  ``atol`` is bound by value/shape and cannot change on reuse.  The
    dense workspace is allocated only when an interval actually requests dense
    output.  Returned samples and final states never alias this mutable storage.
    """

    def __init__(self, y_template, atol):
        import numpy as np
        import torch
        from scipy.integrate._ivp.rk import DOP853

        if (y_template.dtype != torch.float64 or y_template.ndim != 1
                or not y_template.numel() or y_template.device.type != 'cuda'):
            raise ContractError('Resident DOP853 workspace requires a nonempty CUDA float64 vector')
        host_atol = np.asarray(atol, dtype=np.float64)
        if (host_atol.ndim > 1 or (host_atol.ndim == 1 and host_atol.shape != y_template.shape)
                or not np.all(np.isfinite(host_atol)) or np.any(host_atol <= 0)):
            raise ContractError('Resident DOP853 workspace requires positive scalar or per-state atol')
        self.device = y_template.device
        self.n = y_template.numel()
        self._host_atol = host_atol.copy()
        self._host_atol.setflags(write=False)
        self._lock = Lock()
        self._lease_owner = None
        self._stream_id = torch.cuda.current_stream(self.device).cuda_stream
        self.A = torch.as_tensor(DOP853.A, dtype=torch.float64, device=self.device)
        self.B = torch.as_tensor(DOP853.B, dtype=torch.float64, device=self.device)
        self.E3 = torch.as_tensor(DOP853.E3, dtype=torch.float64, device=self.device)
        self.E5 = torch.as_tensor(DOP853.E5, dtype=torch.float64, device=self.device)
        self.AE = torch.as_tensor(DOP853.A_EXTRA, dtype=torch.float64, device=self.device)
        self.DD = torch.as_tensor(DOP853.D, dtype=torch.float64, device=self.device)
        self.at = torch.as_tensor(host_atol.copy(), dtype=torch.float64, device=self.device)
        self.K = torch.empty((16, self.n), dtype=torch.float64, device=self.device)
        self.F = None
        self.rhs_buffers = None
        self._constant_versions = tuple(
            (value.data_ptr(), value._version)
            for value in (self.A, self.B, self.E3, self.E5, self.AE, self.DD, self.at)
        )

    def _validate_binding(self, y, atol):
        import numpy as np
        import torch

        if y.dtype != torch.float64 or y.shape != (self.n,) or y.device != self.device:
            raise ContractError('Resident DOP853 workspace state binding differs')
        value = np.asarray(atol, dtype=np.float64)
        if value.shape != self._host_atol.shape or not np.array_equal(value, self._host_atol):
            raise ContractError('Resident DOP853 workspace atol binding differs')
        if torch.cuda.current_stream(self.device).cuda_stream != self._stream_id:
            raise ContractError('Resident DOP853 workspace cannot be shared across CUDA streams')
        current = tuple(
            (constant.data_ptr(), constant._version)
            for constant in (self.A, self.B, self.E3, self.E5, self.AE, self.DD, self.at)
        )
        if current != self._constant_versions:
            raise ContractError('Resident DOP853 immutable coefficient storage was modified')

    @contextmanager
    def lease(self, y, atol):
        """Reject reentrant/concurrent reuse instead of racing mutable RK rows."""
        if not self._lock.acquire(blocking=False):
            raise ContractError('Resident DOP853 workspace is already in use')
        try:
            self._validate_binding(y, atol)
            self._lease_owner = get_ident()
            yield self
        finally:
            self._lease_owner = None
            self._lock.release()

    def dense_storage(self):
        """For use only while holding the workspace lease."""
        import torch

        if not self._lock.locked() or self._lease_owner != get_ident():
            raise ContractError('Dense storage requires an active DOP853 workspace lease')
        if self.F is None:
            self.F = torch.empty((7, self.n), dtype=torch.float64, device=self.device)
        return self.F


def integrate_device(fun, t0, y0, t_bound, *, rtol, atol, max_step,
                     t_eval=None, workspace=None, dense_output=False, fun_into=None):
    """Run one original DOP853 interval and return owned device snapshots.

    Returns ``t`` (small host timestamp array), ``samples`` (a tuple of device
    tensors in the same order), ``final`` (the accepted endpoint tensor), and
    scalar accounting.  When samples include a dense endpoint the caller must
    preserve the original runner rule: persist that sample, rather than silently
    replacing it with ``final``.  All samples and final are mutually independent
    and survive future workspace reuse.  No full state is downloaded here.

    ``dense_output`` additionally retains an owned ``y_old`` and seven-row ``F``
    for every accepted step, including steps with no requested samples.
    ``fun_into(t, y, out)`` optionally writes each derivative into solver-owned
    storage and returns ``out``; this does not change the RHS schedule.
    """
    import numpy as np
    import torch
    from scipy.integrate._ivp.rk import DOP853, SAFETY, MIN_FACTOR, MAX_FACTOR
    from scipy.integrate._ivp.common import validate_tol, validate_max_step

    if (y0.dtype != torch.float64 or y0.ndim != 1 or not y0.numel()
            or y0.device.type != 'cuda' or not np.isfinite(t0)
            or not np.isfinite(t_bound) or t_bound <= t0):
        raise ContractError('Resident DOP853 requires a nonempty forward CUDA float64 interval')
    if (not np.isfinite(rtol) or rtol <= 0 or not np.all(np.isfinite(atol))
            or np.any(np.asarray(atol) <= 0) or max_step <= 0):
        raise ContractError('Invalid CUDA DOP853 numerical tolerances')
    n = y0.numel()
    rtol, atol = validate_tol(rtol, atol, n)
    max_step = validate_max_step(max_step)
    sample_times = np.asarray([] if t_eval is None else t_eval, dtype=float)
    if (sample_times.ndim != 1 or not np.all(np.isfinite(sample_times))
            or np.any(np.diff(sample_times) <= 0) or np.any(sample_times < t0)
            or np.any(sample_times > t_bound)):
        raise ContractError('CUDA DOP853 observations outside the increasing forward interval')
    selected = Dop853Workspace(y0, atol) if workspace is None else workspace
    if not isinstance(selected, Dop853Workspace):
        raise ContractError('Resident DOP853 requires its own exclusive workspace')
    with selected.lease(y0, atol) as work:
        # Clone the input: no accepted step or failed attempt mutates its owner.
        y = y0.clone()
        t = float(t0)
        nfev = 0
        steps = 0
        attempts = 0
        rejected_attempts = 0
        dense_polynomials = 0

        def rhs(at, state, output=None):
            nonlocal nfev
            nfev += 1
            result = (fun(float(at), state) if fun_into is None else
                      fun_into(float(at), state, output))
            if fun_into is not None and result is not output:
                raise ContractError('CUDA in-place RHS must return its supplied output')
            if (result.shape != state.shape or result.dtype != torch.float64
                    or result.device != state.device):
                raise ContractError('CUDA RHS execution layout differs')
            return result

        def rms(a):
            return torch.linalg.vector_norm(a).item() / n ** .5

        A, B, E3, E5 = work.A, work.B, work.E3, work.E5
        AE, DD, at, K = work.AE, work.DD, work.at, work.K
        # Deliberately repeat both original initial-step RHS calls per interval.
        if fun_into is not None:
            if work.rhs_buffers is None:
                work.rhs_buffers = (torch.empty_like(y), torch.empty_like(y))
            f_buffer, next_f_buffer = work.rhs_buffers
        else:
            f_buffer = next_f_buffer = None
        f = rhs(t, y, f_buffer)
        scale = at + y.abs() * rtol
        d0 = rms(y / scale)
        d1 = rms(f / scale)
        h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else .01 * d0 / d1
        h0 = min(h0, t_bound - t0)
        f1 = rhs(t0 + h0, y + h0 * f, next_f_buffer)
        d2 = rms((f1 - f) / scale) / h0
        h1 = max(1e-6, h0 * 1e-3) if d1 <= 1e-15 and d2 <= 1e-15 else (.01 / max(d1, d2)) ** (1 / 8)
        h_abs = min(100 * h0, h1, t_bound - t0, max_step)
        output = []
        dense_segments = []
        position = 0
        while position < len(sample_times) and sample_times[position] == t:
            output.append(y.clone())
            position += 1
        while t < t_bound:
            minimum = 10 * abs(np.nextafter(t, np.inf) - t)
            h_abs = min(max(h_abs, minimum), max_step)
            rejected = False
            while True:
                if h_abs < minimum:
                    raise ContractError('DOP853 required step size is below floating point spacing')
                attempts += 1
                t_new = min(t + h_abs, t_bound)
                h = t_new - t
                h_abs = abs(h)
                K[0].copy_(f)
                for stage in range(1, 12):
                    delta = torch.mv(K[:stage].T, A[stage, :stage]) * h
                    stage_rhs = rhs(t + DOP853.C[stage] * h, y + delta,
                                    K[stage] if fun_into is not None else None)
                    if fun_into is None:
                        K[stage].copy_(stage_rhs)
                y_new = y + h * torch.mv(K[:12].T, B)
                f_new = rhs(t_new, y_new, next_f_buffer)
                K[12].copy_(f_new)
                scale = at + torch.maximum(y.abs(), y_new.abs()) * rtol
                err5 = torch.mv(K[:13].T, E5) / scale
                err3 = torch.mv(K[:13].T, E3) / scale
                norm5 = torch.linalg.vector_norm(err5).item() ** 2
                norm3 = torch.linalg.vector_norm(err3).item() ** 2
                error = 0. if norm5 == 0 and norm3 == 0 else abs(h) * norm5 / np.sqrt((norm5 + .01 * norm3) * n)
                if not np.isfinite(error):
                    raise ContractError('Nonfinite DOP853 error estimate')
                if error < 1:
                    factor = MAX_FACTOR if error == 0 else min(MAX_FACTOR, SAFETY * error ** (-1 / 8))
                    if rejected:
                        factor = min(1, factor)
                    h_abs *= factor
                    break
                h_abs *= max(MIN_FACTOR, SAFETY * error ** (-1 / 8))
                rejected = True
                rejected_attempts += 1
            steps += 1
            end = np.searchsorted(sample_times, t_new, side='right')
            if dense_output or position < end:
                # Dense output is also built for requested accepted endpoints,
                # exactly as in the original transport and installed SciPy.
                for stage in range(13, 16):
                    delta = torch.mv(K[:stage].T, AE[stage - 13, :stage]) * h
                    stage_rhs = rhs(t + DOP853.C_EXTRA[stage - 13] * h, y + delta,
                                    K[stage] if fun_into is not None else None)
                    if fun_into is None:
                        K[stage].copy_(stage_rhs)
                delta = y_new - y
                F = work.dense_storage()
                F[0].copy_(delta)
                F[1].copy_(h * K[0] - delta)
                F[2].copy_(2 * delta - h * (f_new + K[0]))
                F[3:].copy_(h * torch.mm(DD, K))
                if dense_output:
                    dense_segments.append(dict(t_old=t, t=t_new,
                                               y_old=y.clone(), F=F.clone()))
                for sampled in sample_times[position:end]:
                    x = (sampled - t) / h
                    value = torch.zeros_like(y)
                    for j, row in enumerate(reversed(F.unbind(0))):
                        value += row
                        value *= x if j % 2 == 0 else 1 - x
                    value += y
                    output.append(value.clone())
                position = end
                dense_polynomials += 1
            t = t_new
            y = y_new
            f = f_new
            if fun_into is not None:
                f_buffer, next_f_buffer = next_f_buffer, f_buffer
        if position != len(sample_times):
            raise ContractError('Unreturned requested DOP853 observation')
        if not torch.isfinite(y).all().item():
            raise ContractError('Nonfinite DOP853 final state')
        return {
            't': sample_times.copy(), 'samples': tuple(output), 'final': y.clone(),
            'nfev': nfev, 'steps': steps, 'attempts': attempts,
            'rejected_attempts': rejected_attempts,
            'dense_polynomials': dense_polynomials,
            'dense_segments': tuple(dense_segments),
            'solver_scalar_D2H': 4 + 2 * attempts,
        }
