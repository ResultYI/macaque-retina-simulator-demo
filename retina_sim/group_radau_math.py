"""FP64 algebra for the SCC Radau backend; no model or time-controller policy.

All array work stays on its NumPy or Torch backend/device. Torch scalar status
reads synchronize and are counted when a counters mapping is supplied.
"""
import math
from contextlib import contextmanager
from numbers import Real
from threading import Lock

import numpy as np

from .contracts import ContractError


def _backend(value):
    if isinstance(value, np.ndarray):
        if value.dtype != np.float64:
            raise ContractError("Radau algebra requires FP64 arrays")
        return None
    import torch
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or value.requires_grad:
        raise ContractError("Radau algebra requires FP64 arrays without autograd")
    return torch


def _count(counters, name, amount=1):
    if counters is not None:
        counters[name] = counters.get(name, 0) + amount


def _scalar(value, counters=None):
    if not isinstance(value, (Real, np.generic)):
        _count(counters, "host_scalar_reads")
    return value.item() if hasattr(value, "item") else value


def _guard(guard, counters):
    if guard is not None:
        _count(counters, "guard_checks")
        status = guard()
        if status is not None and not _scalar(status, counters):
            raise ContractError("Radau algebra guard interrupted the solve")


def _finite(value, torch, counters=None):
    okay = np.isfinite(value).all() if torch is None else torch.isfinite(value).all()
    if not _scalar(okay, counters):
        raise ContractError("Nonfinite Radau algebra value")


def _like(value, reference):
    module = _backend(value)
    expected = _backend(reference)
    if (module is None) != (expected is None):
        raise ContractError("Radau arrays must use the same backend")
    if module is not None and value.device != reference.device:
        raise ContractError("Radau arrays must use the same device")
    return module


def _copy(value):
    return value.copy() if isinstance(value, np.ndarray) else value.clone()


def _zeros(shape, like, torch):
    return np.zeros(shape, dtype=np.float64) if torch is None else like.new_zeros(shape)


def _eye(n, like, torch):
    return np.eye(n, dtype=np.float64) if torch is None else torch.eye(n, dtype=torch.float64, device=like.device)


def _broadcast(value, shape, like, torch):
    if isinstance(value, Real):
        value = np.asarray(value, dtype=np.float64) if torch is None else like.new_tensor(value)
    else:
        _like(value, like)
    try:
        return np.broadcast_to(value, shape) if torch is None else torch.broadcast_to(value, shape)
    except (ValueError, RuntimeError) as exc:
        raise ContractError("Coefficient cannot broadcast to the Radau state shape") from exc


def _solve(matrix, rhs, torch, counters=None):
    _finite(matrix, torch, counters)
    _finite(rhs, torch, counters)
    try:
        if torch is None:
            out = np.linalg.solve(matrix, rhs[..., None])[..., 0]
        else:
            out, status = torch.linalg.solve_ex(matrix, rhs[..., None], check_errors=False)
            if not _scalar((status == 0).all(), counters):
                raise ContractError("Singular or failed Radau direct solve")
            out = out[..., 0]
    except (np.linalg.LinAlgError, RuntimeError) as exc:
        raise ContractError("Singular or failed Radau direct solve") from exc
    _finite(out, torch, counters)
    return out


def tableau(like=None):
    """Return (A, b, c, gamma, bhat); gamma is the separate embedded f0 weight.

    Arrays are FP64, on like's device for Torch. The embedded third-order rule
    uses gamma*f0 + bhat@F, not bhat@F alone.
    """
    r = math.sqrt(6.0)
    data = [[(88-7*r)/360, (296-169*r)/1800, (-2+3*r)/225],
            [(296+169*r)/1800, (88+7*r)/360, (-2-3*r)/225],
            [(16-r)/36, (16+r)/36, 1/9]]
    gamma = 1 / (3 + 3**(2/3) - 3**(1/3))
    torch = None if like is None else _backend(like)
    make = (lambda x: np.array(x, dtype=np.float64)) if torch is None else like.new_tensor
    a = make(data)
    b = _copy(a[-1])
    c = make([(4-r)/10, (4+r)/10, 1.0])
    bhat = b - gamma * make([(2+3*r)/6, (2-3*r)/6, 1/3])
    return a, b, c, gamma, bhat


def affine_stages(y0, q, k, dt, *, counters=None):
    """Solve y'=q-k*y at the three Radau nodes, returning [3, n] stages.

    y0 is [n]; q and k broadcast to [3,n]. A [3,1] coefficient varies with
    stage, and an [n] coefficient varies with state. No state is clipped.
    """
    torch = _backend(y0)
    if y0.ndim != 1 or not math.isfinite(dt) or dt < 0:
        raise ContractError("Affine Radau requires a vector and finite nonnegative dt")
    q = _broadcast(q, (3, len(y0)), y0, torch)
    k = _broadcast(k, (3, len(y0)), y0, torch)
    for value in (y0, q, k):
        _finite(value, torch, counters)
    if len(y0) == 0:
        return _copy(q)
    a = tableau(y0)[0]
    matrices = _eye(3, y0, torch)[:, :, None] + dt*a[:, :, None]*k[None, :, :]
    matrices = np.moveaxis(matrices, -1, 0) if torch is None else matrices.permute(2, 0, 1)
    rhs = y0[None, :] + dt*(a @ q)
    return _solve(matrices, rhs.T, torch, counters).T


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ContractError(name + " must be a bounded nonnegative integer")


def _tolerances(rtol, atol):
    if not math.isfinite(rtol) or rtol < 0 or not math.isfinite(atol) or atol < 0:
        raise ContractError("Finite nonnegative solver tolerances required")


class GMRESWorkspace:
    """Fixed-capacity scratch, exclusively leased by one GMRES call at a time.

    The shape/backend/device/restart are fixed at construction; CUDA workspaces
    also bind the current stream. Callbacks must preserve inputs and the current
    stream, and satisfy that stream's dependencies before returning. Banks are private scratch: arrays
    exposes them only for memory accounting, not as solver inputs or outputs.
    owned_bytes counts array payloads, excluding temporary operations, returned
    solution copies and Python object overhead. No workspace bank grows.
    """
    def __init__(self, like, *, restart=32):
        torch = _backend(like)
        _integer(restart, "restart", 1)
        if restart > 32:
            raise ContractError("GMRES workspace restart limit is 32")
        if torch is not None and like.device.type not in ("cpu", "cuda"):
            raise ContractError("GMRES workspaces require CPU or CUDA residency")
        self._shape = tuple(like.shape)
        self._restart = int(restart)
        self._torch = torch
        self._device = None if torch is None else like.device
        self._stream = (int(torch.cuda.current_stream(like.device).cuda_stream)
                        if torch is not None and like.device.type == "cuda" else None)
        n = math.prod(self._shape)
        m = min(restart, n)
        self._basis = _zeros((m+1, n), like, torch)
        self._h = _zeros((m+1, m), like, torch)
        self._x = _zeros((n,), like, torch)
        self._residual = _zeros((n,), like, torch)
        self._coefficient_rhs = _zeros((m,), like, torch)
        self._cosine = np.zeros(m, dtype=np.float64)
        self._sine = np.zeros(m, dtype=np.float64)
        self._g = np.zeros(m+1, dtype=np.float64)
        self._device_banks = (self._basis, self._h, self._x,
                              self._residual, self._coefficient_rhs)
        self._arrays = self._device_banks + (self._cosine, self._sine, self._g)
        self._signatures = tuple(self._signature(a) for a in self._arrays)
        device_payload = sum(a.size*8 if isinstance(a, np.ndarray) else a.numel()*a.element_size()
                             for a in self._device_banks)
        host_payload = self._cosine.nbytes + self._sine.nbytes + self._g.nbytes
        self._device_bytes = device_payload if self._stream is not None else 0
        self._host_bytes = host_payload + (device_payload if self._stream is None else 0)
        self._lock = Lock()
        self._uses = 0

    @staticmethod
    def _pointer(array):
        return array.ctypes.data if isinstance(array, np.ndarray) else array.untyped_storage().data_ptr()

    @classmethod
    def _signature(cls, array):
        if isinstance(array, np.ndarray):
            return (tuple(array.shape), array.strides, cls._pointer(array), array.nbytes)
        return (tuple(array.shape), tuple(array.stride()), array.storage_offset(),
                array.data_ptr(), array.untyped_storage().nbytes())

    def _check_stream(self):
        if self._stream is not None and int(self._torch.cuda.current_stream(self._device).cuda_stream) != self._stream:
            raise ContractError("GMRES workspace CUDA stream changed")

    @property
    def shape(self):
        return self._shape

    @property
    def restart(self):
        return self._restart

    @property
    def arrays(self):
        return self._arrays

    @property
    def owned_bytes(self):
        return self._device_bytes + self._host_bytes

    @property
    def device_bytes(self):
        return self._device_bytes

    @property
    def host_bytes(self):
        return self._host_bytes

    @property
    def uses(self):
        return self._uses

    def _validate(self, rhs, restart):
        torch = _like(rhs, self._x)
        if tuple(rhs.shape) != self._shape or restart != self._restart:
            raise ContractError("GMRES workspace shape/restart mismatch; growth is forbidden")
        self._check_stream()
        for bank, signature in zip(self._arrays, self._signatures):
            _backend(bank)
            if self._signature(bank) != signature:
                raise ContractError("GMRES workspace bank shape/storage/layout changed")
        for bank in self._device_banks:
            _like(bank, rhs)
        for bank in self._arrays:
            if torch is None:
                overlap = np.shares_memory(rhs, bank)
            elif isinstance(bank, np.ndarray):
                start = rhs.untyped_storage().data_ptr()
                stop = start + rhs.untyped_storage().nbytes()
                overlap = (rhs.device.type == "cpu" and rhs.numel() > 0 and bank.size > 0
                           and start < bank.ctypes.data+bank.nbytes and bank.ctypes.data < stop)
            else:
                overlap = rhs.numel() > 0 and bank.numel() > 0 and self._pointer(rhs) == self._pointer(bank)
            if overlap:
                raise ContractError("GMRES RHS may not alias workspace storage")

    @contextmanager
    def _lease(self, rhs, restart, counters):
        if not self._lock.acquire(blocking=False):
            raise ContractError("GMRES workspace is already leased")
        try:
            self._validate(rhs, restart)
            _count(counters, "gmres_workspace_uses")
            if counters is not None:
                counters["gmres_workspace_bytes_peak"] = max(
                    counters.get("gmres_workspace_bytes_peak", 0), self.owned_bytes)
            self._uses += 1
            yield
        finally:
            self._lock.release()


def gmres(matvec, rhs, precondition=None, *, rtol, atol, restart=32,
          maxiter=96, guard=None, counters=None, workspace=None):
    """Bounded standard GMRES with a fixed linear RIGHT preconditioner.

    Callbacks receive/return rhs-shaped FP64 arrays on the same device.
    precondition applies P^-1 and MUST remain linear and fixed for this call.
    Only V is stored: x += P^-1(V*y), without a flexible-GMRES second basis.
    restart <= 32 and maxiter <= 96 enforce the frozen B limits. maxiter
    counts Arnoldi iterations, not restart cycles. Convergence uses
    ||rhs-A*x||_2 <= max(atol, rtol*||rhs||_2), always recomputed explicitly.
    The caller MUST reject info['converged']==False; no fallback is attempted.
    An optional GMRESWorkspace reuses fixed banks; returned solutions never
    alias those banks. Operator and linear-solve temporaries still allocate.
    """
    if workspace is None:
        return _gmres_solve(matvec, rhs, precondition, rtol=rtol, atol=atol,
                            restart=restart, maxiter=maxiter, guard=guard, counters=counters)
    if not isinstance(workspace, GMRESWorkspace):
        raise ContractError("A GMRESWorkspace is required")
    with workspace._lease(rhs, restart, counters):
        return _gmres_solve(matvec, rhs, precondition, rtol=rtol, atol=atol,
                            restart=restart, maxiter=maxiter, guard=guard,
                            counters=counters, workspace=workspace)


def _gmres_solve(matvec, rhs, precondition, *, rtol, atol, restart, maxiter,
                 guard, counters, workspace=None):
    def checkpoint():
        _guard(guard, counters)
        if workspace is not None:
            workspace._check_stream()

    torch = _backend(rhs)
    _tolerances(rtol, atol)
    _integer(restart, "restart", 1)
    _integer(maxiter, "maxiter")
    if restart > 32 or maxiter > 96:
        raise ContractError("GMRES limits are restart <= 32 and maxiter <= 96")
    _count(counters, "gmres_calls")
    checkpoint()
    _finite(rhs, torch, counters)
    shape = rhs.shape
    flat = rhs.reshape(-1)
    n = flat.shape[0]
    if workspace is None:
        x = _zeros((n,), rhs, torch)
        residual = _copy(flat)
    else:
        x, residual = workspace._x, workspace._residual
        x[...] = 0
        residual[...] = flat
    norm = np.linalg.norm if torch is None else torch.linalg.vector_norm

    def checked(function, value, name):
        checkpoint()
        _count(counters, name)
        out = function(value.reshape(shape))
        if workspace is not None:
            workspace._check_stream()
        _like(out, rhs)
        if out.shape != shape:
            raise ContractError("GMRES callback changed the RHS shape")
        _finite(out, torch, counters)
        return _copy(out).reshape(-1)

    def apply(value):
        return checked(matvec, value, "gmres_matvecs")

    def pre(value):
        return value if precondition is None else checked(precondition, value, "gmres_preconditioner_applications")

    def length(value):
        result = float(_scalar(norm(value), counters))
        if not math.isfinite(result):
            raise ContractError("Nonfinite GMRES norm")
        return result

    beta = length(residual)
    tolerance = max(atol, rtol*beta)
    if not math.isfinite(tolerance):
        raise ContractError("Nonfinite GMRES stopping tolerance")
    iterations = 0
    cycles = 0

    def finish(converged, reason):
        if workspace is not None:
            workspace._check_stream()
        result = x if workspace is None else _copy(x)
        return result.reshape(shape), dict(converged=converged, residual=beta,
                                     iterations=iterations, reason=reason,
                                     tolerance=tolerance, cycles=cycles)

    if beta <= tolerance:
        return finish(True, "converged")
    if maxiter == 0:
        return finish(False, "iteration_limit")
    while iterations < maxiter:
        checkpoint()
        cycles += 1
        _count(counters, "gmres_cycles")
        if cycles > 1:
            _count(counters, "gmres_restarts")
        m = min(restart, maxiter-iterations, n)
        if workspace is None:
            basis = _zeros((m+1, n), rhs, torch)
            h = _zeros((m+1, m), rhs, torch)
            cosine, sine = [0.0]*m, [0.0]*m
            g = [0.0]*(m+1)
        else:
            basis = workspace._basis[:m+1]
            h = workspace._h[:m+1, :m]
            basis[...] = 0
            h[...] = 0
            cosine, sine, g = workspace._cosine, workspace._sine, workspace._g
            cosine.fill(0)
            sine.fill(0)
            g.fill(0)
        basis[0] = residual / beta
        g[0] = beta
        breakdown = False
        columns = 0
        for j in range(m):
            checkpoint()
            w = apply(pre(basis[j]))
            before = length(w)
            # Two-pass Gram-Schmidt preserves orthogonality for stiff blocks.
            projection = basis[:j+1] @ w
            w = w - projection @ basis[:j+1]
            correction = basis[:j+1] @ w
            w = w - correction @ basis[:j+1]
            h[:j+1, j] = projection + correction
            after = length(w)
            h[j+1, j] = after
            breakdown = after <= 16*np.finfo(np.float64).eps*before
            if not breakdown:
                basis[j+1] = w / after
            for i in range(j):
                first, second = _copy(h[i:i+1, j])[0], _copy(h[i+1:i+2, j])[0]
                h[i, j] = float(cosine[i])*first + float(sine[i])*second
                h[i+1, j] = -float(sine[i])*first + float(cosine[i])*second
            diagonal = float(_scalar(h[j, j], counters))
            radius = math.hypot(diagonal, after)
            iterations += 1
            _count(counters, "gmres_iterations")
            if not math.isfinite(radius):
                raise ContractError("Nonfinite GMRES rotation")
            if radius == 0:
                breakdown = True
                break
            cosine[j] = diagonal/radius
            sine[j] = after/radius
            h[j, j], h[j+1, j] = radius, 0.0
            g[j+1] = -float(sine[j])*float(g[j])
            g[j] = float(g[j])*float(cosine[j])
            columns = j+1
            if abs(g[j+1]) <= tolerance or breakdown:
                break
        if columns:
            if workspace is None:
                coefficients_rhs = np.asarray(g[:columns], dtype=np.float64) if torch is None else rhs.new_tensor(g[:columns])
            else:
                coefficients_rhs = workspace._coefficient_rhs[:columns]
                if torch is None:
                    coefficients_rhs[...] = g[:columns]
                else:
                    coefficients_rhs.copy_(torch.from_numpy(g[:columns]))
            checkpoint()
            coefficients = _solve(h[:columns, :columns], coefficients_rhs, torch, counters)
            x[...] = x + pre(coefficients @ basis[:columns])
        _finite(x, torch, counters)
        _count(counters, "gmres_true_residual_checks")
        residual[...] = flat - apply(x)
        beta = length(residual)
        if beta <= tolerance:
            return finish(True, "converged")
        if breakdown:
            return finish(False, "breakdown")
    return finish(False, "iteration_limit")


def batched_newton(initial, residual, jacobian, *, atol, rtol=0.0,
                   maxiter=8, guard=None, counters=None, domain=None):
    """Solve independent local systems [batch,width] with at most 8 updates.

    jacobian returns [batch,width,width]. The fixed per-component scale is
    atol + rtol*abs(initial); scaled row infinity residuals must be <= 1.
    domain returns a boolean scalar or [batch] mask (True means valid).
    Each row tries steps 1,1/2,1/4,1/8; invalid rows never reach residual().
    Converged rows are frozen. Failure is explicit in info['converged'].
    """
    torch = _backend(initial)
    if initial.ndim != 2 or initial.shape[1] == 0:
        raise ContractError("Batched Newton requires [batch,width] initial state")
    _integer(maxiter, "maxiter")
    if maxiter > 8 or not math.isfinite(rtol) or rtol < 0:
        raise ContractError("Batched Newton permits at most eight updates and finite rtol")
    _count(counters, "newton_calls")
    _guard(guard, counters)
    _finite(initial, torch, counters)
    absolute = np.abs if torch is None else torch.abs
    where = np.where if torch is None else torch.where
    absolute_tolerance = _broadcast(atol, initial.shape, initial, torch)
    _finite(absolute_tolerance, torch, counters)
    if not _scalar((absolute_tolerance >= 0).all(), counters):
        raise ContractError("Newton absolute tolerances must be nonnegative")
    scale = absolute_tolerance + rtol*absolute(initial)
    _finite(scale, torch, counters)
    if not _scalar((scale > 0).all(), counters):
        raise ContractError("Batched Newton needs strictly positive component scales")
    x = _copy(initial)
    batch, width = x.shape
    iterations = 0

    def all_rows(mask):
        return bool(_scalar(mask.all(), counters))

    def domain_mask(value):
        _guard(guard, counters)
        if domain is None:
            return np.ones(batch, dtype=bool) if torch is None else torch.ones(batch, dtype=torch.bool, device=x.device)
        mask = domain(value)
        if isinstance(mask, (bool, np.bool_)):
            return np.full(batch, mask, dtype=bool) if torch is None else torch.full((batch,), bool(mask), dtype=torch.bool, device=x.device)
        if torch is None:
            if not isinstance(mask, np.ndarray) or mask.dtype != bool:
                raise ContractError("Newton domain must return boolean values")
        elif not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.device != x.device:
            raise ContractError("Newton domain must return booleans on the state device")
        if mask.ndim == 0:
            return np.broadcast_to(mask, (batch,)) if torch is None else torch.broadcast_to(mask, (batch,))
        if mask.shape != (batch,):
            raise ContractError("Newton domain mask must have one value per local system")
        return mask

    def evaluate(value):
        _guard(guard, counters)
        _count(counters, "newton_residual_evaluations")
        result = residual(value)
        _like(result, initial)
        if result.shape != initial.shape:
            raise ContractError("Newton residual shape mismatch")
        _finite(result, torch, counters)
        result = _copy(result)
        norm = np.max(absolute(result)/scale, axis=1) if torch is None else torch.amax(absolute(result)/scale, dim=1)
        _finite(norm, torch, counters)
        return result, norm

    def finish(converged, reason, norms):
        worst = 0.0 if batch == 0 else float(_scalar(norms.max(), counters))
        return x, dict(converged=converged, residual=worst,
                       iterations=iterations, reason=reason)

    if batch == 0:
        return finish(True, "converged", None)
    if not all_rows(domain_mask(x)):
        raise ContractError("Initial Newton state outside its domain")
    values, norms = evaluate(x)
    while not all_rows(norms <= 1):
        if iterations == maxiter:
            return finish(False, "iteration_limit", norms)
        _guard(guard, counters)
        _count(counters, "newton_jacobian_evaluations")
        matrix = jacobian(x)
        _like(matrix, initial)
        if matrix.shape != (batch, width, width):
            raise ContractError("Newton Jacobian shape mismatch")
        _finite(matrix, torch, counters)
        active = norms > 1
        matrix = where(active[:, None, None], matrix, _eye(width, x, torch)[None, :, :])
        rhs = where(active[:, None], -values, _zeros(initial.shape, initial, torch))
        _guard(guard, counters)
        _count(counters, "newton_linear_solves")
        _count(counters, "newton_linear_systems", batch)
        step = _solve(matrix, rhs, torch, counters)
        iterations += 1
        _count(counters, "newton_iterations")
        accepted = ~active
        next_x, next_values, next_norms = _copy(x), _copy(values), _copy(norms)
        for fraction in (1.0, 0.5, 0.25, 0.125):
            _guard(guard, counters)
            _count(counters, "newton_line_search_trials")
            candidate = x + fraction*step
            finite = np.isfinite(candidate).all(axis=1) if torch is None else torch.isfinite(candidate).all(dim=1)
            domain_safe = where(finite[:, None], candidate, x)
            eligible = (~accepted) & finite & domain_mask(domain_safe)
            safe = where(eligible[:, None], candidate, x)
            trial_values, trial_norms = evaluate(safe)
            take = eligible & ((trial_norms < norms) | (trial_norms <= 1))
            next_x = where(take[:, None], candidate, next_x)
            next_values = where(take[:, None], trial_values, next_values)
            next_norms = where(take, trial_norms, next_norms)
            accepted = accepted | take
            if all_rows(accepted):
                break
        x, values, norms = next_x, next_values, next_norms
        if not all_rows(accepted):
            return finish(False, "line_search_failed", norms)
        if all_rows(norms <= 1):
            # Certify the final assembled state, including independent row steps.
            values, norms = evaluate(x)
    return finish(True, "converged", norms)

