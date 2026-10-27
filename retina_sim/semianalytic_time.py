"""FP64 continuous kernels for explicitly supplied, accepted driving waveforms.

These kernels neither initialize nor advance a RetiPath model. Polynomial forcing
is integrated exactly up to floating point roundoff; approximation to an actual
driver is separately estimated, never claimed to be a rigorous error bound.
"""
from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss

from .contracts import ContractError


_NODES = .5*(1.-np.cos(np.arange(8)*np.pi/7.))
_CHECK_NODES = .5*(1.+leggauss(9)[0])
_GAUSS = {n: leggauss(n) for n in (8, 16)}


def _finite(value, name, *, copy=False):
    result = np.array(value, dtype=np.float64, copy=True) if copy else np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ContractError('Finite FP64 '+name+' required')
    return result


def _interval(t0, t1):
    if not math.isfinite(t0) or not math.isfinite(t1) or t1 <= t0:
        raise ContractError('Finite increasing representation interval required')
    return float(t0), float(t1)


def _query(t, side, t0, t1):
    if (side not in ('left', 'right') or not np.isscalar(t) or not math.isfinite(t)
            or t < t0 or t > t1 or (t == t0 and side != 'right')
            or (t == t1 and side != 'left')):
        raise ContractError('Query outside accepted interval or on incorrect boundary side')
    return float(t)


def _ids(state_ids, count):
    if state_ids is None:
        return slice(None)
    rows = np.asarray(state_ids)
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or np.any(rows < 0) or np.any(rows >= count):
        raise ContractError('Valid explicit state IDs required')
    return rows


def phi_sequence(z, order=8):
    """Stable phi_1..phi_order for finite nonpositive real FP64 arguments.

    Small negative arguments use the defining series. For |z| >= 4 the
    recurrence has no harmful small-denominator amplification up to order 8.
    """
    z = _finite(z, 'phi argument')
    if type(order) is not int or not 1 <= order <= 8 or np.any(z > 0):
        raise ContractError('phi order 1..8 and nonpositive real argument required')
    flat = z.reshape(-1)
    result = np.empty((order, len(flat)), dtype=np.float64)
    small = flat > -4.
    if np.any(small):
        v = flat[small]
        for k in range(1, order+1):
            term = np.full(len(v), 1./math.factorial(k))
            total = term.copy()
            for n in range(1, 100):
                term = term*v/(n+k)
                total += term
                if np.all(np.abs(term) <= np.finfo(float).eps*np.abs(total)):
                    break
            else:
                raise ContractError('phi series did not converge')
            result[k-1, small] = total
    if np.any(~small):
        x = -flat[~small]
        value = -np.expm1(-x)/x
        result[0, ~small] = value
        for k in range(2, order+1):
            value = (1./math.factorial(k-1)-value)/x
            result[k-1, ~small] = value
    return result.reshape((order,)+z.shape)


def phi(k, z):
    """phi_k(z) = integral exp((1-u)z) u**(k-1)/(k-1)! du, 0<=u<=1."""
    return phi_sequence(z, k)[k-1]


def polynomial_value(coefficients, x):
    """Horner evaluation; coefficients multiply normalized time x**j."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    result = np.array(coefficients[-1], copy=True)
    for coefficient in coefficients[-2::-1]:
        result = result*x+coefficient
    return result


def _polynomial_absolute_bound(coefficients, lo, hi):
    """Bernstein convex-hull bound after mapping [lo,hi] to [0,1]."""
    power = np.zeros_like(coefficients)
    for k in range(8):
        for j in range(k, 8):
            power[k] += coefficients[j]*math.comb(j, k)*lo**(j-k)*(hi-lo)**k
    bound = np.zeros(coefficients.shape[1])
    for i in range(8):
        beta = sum(power[k]*math.comb(i, k)/math.comb(7, k) for k in range(i+1))
        bound = np.maximum(bound, np.abs(beta))
    return bound


def _coefficients(coefficients):
    a = _finite(coefficients, 'degree-seven coefficients', copy=True)
    if a.ndim != 2 or a.shape[0] != 8 or a.shape[1] < 1:
        raise ContractError('Degree-seven coefficient shape (8, state_count) required')
    a.flags.writeable = False
    return a


def exponential_forced_value(y0, decay, coefficients, H, s):
    """Same exponential-polynomial expression at endpoints and internal times.

    q(s)=sum(coefficients[j]*(s/H)**j). ``decay`` is in inverse seconds;
    coefficients have state-unit/second. No driver approximation is performed.
    """
    y0 = _finite(y0, 'initial state')
    q = _finite(coefficients, 'forcing coefficients')
    decay = _finite(decay, 'decay rate')
    if (y0.ndim != 1 or q.shape != (8, len(y0)) or np.any(decay < 0)
            or decay.shape not in ((), y0.shape) or not math.isfinite(H) or H <= 0
            or not math.isfinite(s) or s < 0 or s > H):
        raise ContractError('Compatible fixed-decay waveform, shape and interval required')
    if s == 0:
        return y0.copy()
    z = -decay*s
    if not np.isfinite(z).all():
        raise ContractError('Decay-time product outside finite FP64 domain')
    p = phi_sequence(z)
    result = np.exp(z)*y0
    xpower = 1.
    for j in range(8):
        result = result+q[j]*(math.factorial(j)*s*xpower)*p[j]
        xpower *= s/H
    if not np.isfinite(result).all():
        raise ContractError('Nonfinite exponential-polynomial state')
    return result


def exponential_error_bound(decay, s, drive_error, initial_error=0.):
    """Propagation of a supplied sup-norm driver bound; does not certify it."""
    decay, error, initial = np.broadcast_arrays(_finite(decay, 'decay'),
        _finite(drive_error, 'driver error'), _finite(initial_error, 'initial error'))
    if np.any(decay < 0) or np.any(error < 0) or np.any(initial < 0) or not math.isfinite(s) or s < 0:
        raise ContractError('Nonnegative decay, time and error bounds required')
    gain = np.full(decay.shape, s, dtype=np.float64)
    np.divide(-np.expm1(-decay*s), decay, out=gain, where=decay > 0)
    return np.exp(-decay*s)*initial+gain*error


@dataclass(frozen=True)
class PolynomialDrive:
    t0: float
    t1: float
    coefficients: np.ndarray
    unit: str

    def __post_init__(self):
        _interval(self.t0, self.t1)
        if not isinstance(self.unit, str) or not self.unit:
            raise ContractError('Explicit driving waveform unit required')
        object.__setattr__(self, 'coefficients', _coefficients(self.coefficients))

    @property
    def available_until(self):
        return self.t1

    def evaluate(self, t, side='right', state_ids=None):
        t = _query(t, side, self.t0, self.t1)
        return polynomial_value(self.coefficients[:, _ids(state_ids, self.coefficients.shape[1])],
                                (t-self.t0)/(self.t1-self.t0))

    def primitive(self, t, side='right', state_ids=None):
        """Integral from t0, with the original physical-time scale."""
        t = _query(t, side, self.t0, self.t1)
        q = self.coefficients[:, _ids(state_ids, self.coefficients.shape[1])]
        x = (t-self.t0)/(self.t1-self.t0)
        return (t-self.t0)*polynomial_value(q/np.arange(1, 9)[:, None], x)


@dataclass(frozen=True)
class ExponentialForced:
    t0: float
    t1: float
    y0: np.ndarray
    decay: np.ndarray
    drive: PolynomialDrive
    state_unit: str

    def __post_init__(self):
        _interval(self.t0, self.t1)
        y0 = _finite(self.y0, 'initial state', copy=True)
        decay = _finite(self.decay, 'decay', copy=True)
        if (y0.ndim != 1 or decay.shape not in ((), y0.shape) or np.any(decay < 0)
                or self.drive.coefficients.shape != (8, len(y0))
                or (self.t0, self.t1) != (self.drive.t0, self.drive.t1)
                or not isinstance(self.state_unit, str) or not self.state_unit):
            raise ContractError('Fixed-decay state and forcing interval/units differ')
        y0.flags.writeable = decay.flags.writeable = False
        object.__setattr__(self, 'y0', y0)
        object.__setattr__(self, 'decay', decay)

    @property
    def available_until(self):
        return self.t1

    def evaluate(self, t, side='right', state_ids=None):
        t = _query(t, side, self.t0, self.t1)
        rows = _ids(state_ids, len(self.y0))
        decay = self.decay if self.decay.ndim == 0 else self.decay[rows]
        return exponential_forced_value(self.y0[rows], decay, self.drive.coefficients[:, rows],
                                        self.t1-self.t0, t-self.t0)

    def represented_rhs_at(self, t, side='right', state_ids=None):
        """RHS of represented forcing; full-monitor callers use original RHS."""
        rows = _ids(state_ids, len(self.y0))
        decay = self.decay if self.decay.ndim == 0 else self.decay[rows]
        return self.drive.evaluate(t, side, state_ids)-decay*self.evaluate(t, side, state_ids)


@dataclass(frozen=True)
class DriveApproximation:
    segments: tuple
    estimated_max_error: np.ndarray
    source_evaluations: int
    finite_node_estimate_only: bool = True


def fit_drive(source, t0, t1, *, forcing_budget, unit, max_depth=20):
    """Chebyshev-Lobatto degree7 fit, checked against two half-window fits.

    ``source(t, side)`` must read actual accepted upstream values. The caller
    supplies its allocated forcing sub-budget in q units; no unused time,
    coupling or quadrature allowance is borrowed. Independent Gauss nodes also
    compare the polynomial with the actual source. Finite nodes are estimates.
    """
    t0, t1 = _interval(t0, t1)
    budget = _finite(forcing_budget, 'allocated forcing error')
    if np.any(budget <= 0) or type(max_depth) is not int or max_depth < 0:
        raise ContractError('Positive forcing allocation and nonnegative subdivision depth required')
    count, accepted, errors = 0, [], []

    def actual(t, lo, hi):
        nonlocal count
        count += 1
        value = _finite(source(t, 'left' if t == hi else 'right'), 'actual driving samples')
        if value.ndim != 1 or not len(value):
            raise ContractError('Actual driving source must return a nonempty state vector')
        return value

    def fit(lo, hi):
        times = lo+(hi-lo)*_NODES
        times[0], times[-1] = lo, hi
        if np.any(np.diff(times) <= 0):
            raise ContractError('Forcing interval too short for distinct FP64 fit nodes')
        values = np.stack([actual(float(t), lo, hi) for t in times])
        try:
            # Use represented FP64 times rather than ideal nodes: large t0
            # can shift an otherwise valid node by an appreciable H fraction.
            vandermonde = np.polynomial.polynomial.polyvander((times-lo)/(hi-lo), 7)
            coefficients = np.linalg.solve(vandermonde, values)
        except ValueError as exc:
            raise ContractError('Driver source shape changed within a window') from exc
        return PolynomialDrive(lo, hi, coefficients, unit)

    def visit(lo, hi, depth):
        whole = fit(lo, hi)
        middle = lo+(hi-lo)/2
        if not lo < middle < hi:
            raise ContractError('Forcing subdivision exhausted FP64 interval precision')
        halves = (fit(lo, middle), fit(middle, hi))
        error = np.zeros(whole.coefficients.shape[1])
        for half in halves:
            # Half-fit nodes test the full fit; independent nodes test both
            # against actual source samples, not another fitted derivative.
            for x in np.r_[_NODES, _CHECK_NODES]:
                t = float(half.t0+(half.t1-half.t0)*x)
                if x == 0: t = half.t0
                if x == 1: t = half.t1
                side = 'left' if t == half.t1 else 'right'
                y = polynomial_value(whole.coefficients, (t-lo)/(hi-lo))
                h = half.evaluate(t, side)
                truth = actual(t, lo, hi)
                error = np.maximum(error, np.maximum(np.abs(y-h), np.maximum(np.abs(y-truth), np.abs(h-truth))))
        try:
            meets = np.all(error <= np.broadcast_to(budget, error.shape))
        except ValueError as exc:
            raise ContractError('Forcing allocation shape differs from source states') from exc
        if meets:
            accepted.append(whole)
            errors.append(error)
        elif depth >= max_depth:
            raise ContractError('Driver approximation exceeds its allocated error; subdivision limit reached')
        else:
            visit(lo, middle, depth+1)
            visit(middle, hi, depth+1)

    visit(t0, t1, 0)
    maximum = np.max(errors, axis=0)
    maximum.flags.writeable = False
    return DriveApproximation(tuple(accepted), maximum, count)


@dataclass(frozen=True)
class AffineMembrane:
    """Variable-coefficient y'=a(t)-b(t)y with explicit quadrature allocation.

    a has state-unit/s, b has s^-1. No midpoint coefficient freezing is used;
    exact polynomial primitives of b enter every quadrature node.
    """
    t0: float
    t1: float
    y0: np.ndarray
    a: PolynomialDrive
    b: PolynomialDrive
    quadrature_budget: np.ndarray
    state_unit: str
    max_depth: int = 20

    def __post_init__(self):
        _interval(self.t0, self.t1)
        y0 = _finite(self.y0, 'membrane initial state', copy=True)
        budget = _finite(self.quadrature_budget, 'allocated quadrature error', copy=True)
        if (y0.ndim != 1 or self.a.coefficients.shape != (8, len(y0))
                or self.b.coefficients.shape != self.a.coefficients.shape
                or (self.a.t0, self.a.t1) != (self.t0, self.t1)
                or (self.b.t0, self.b.t1) != (self.t0, self.t1)
                or budget.shape not in ((), y0.shape) or np.any(budget <= 0)
                or type(self.max_depth) is not int or self.max_depth < 0
                or not isinstance(self.state_unit, str) or not self.state_unit):
            raise ContractError('Affine membrane intervals, shapes, units or quadrature allocation differ')
        y0.flags.writeable = budget.flags.writeable = False
        object.__setattr__(self, 'y0', y0)
        object.__setattr__(self, 'quadrature_budget', budget)

    @property
    def available_until(self):
        return self.t1

    def evaluate_with_error(self, t, side='right', state_ids=None):
        t = _query(t, side, self.t0, self.t1)
        rows = _ids(state_ids, len(self.y0))
        y0 = self.y0[rows]
        if t == self.t0:
            return y0.copy(), np.zeros_like(y0)
        H, s = self.t1-self.t0, t-self.t0
        a, b = self.a.coefficients[:, rows], self.b.coefficients[:, rows]
        primitive = b/np.arange(1, 9)[:, None]
        Bs = s*polynomial_value(primitive, s/H)
        allocated = self.quadrature_budget if self.quadrature_budget.ndim == 0 else self.quadrature_budget[rows]

        def rule(lo, hi, order):
            nodes, weights = _GAUSS[order]
            total = np.zeros_like(y0)
            for x, w in zip(nodes, weights):
                u = lo+(hi-lo)*.5*(1+x)
                Bu = u*polynomial_value(primitive, u/H)
                with np.errstate(over='raise', invalid='raise'):
                    try:
                        value = np.exp(-(Bs-Bu))*polynomial_value(a, u/H)
                    except FloatingPointError as exc:
                        raise ContractError('Affine integrating factor outside finite FP64 domain') from exc
                total += w*value
            return .5*(hi-lo)*total

        def integrate(lo, hi, budget, depth):
            # Embedded rules can both miss a stiff endpoint layer. Limiting
            # integrating-factor variation prevents a false zero agreement.
            variation = (hi-lo)*_polynomial_absolute_bound(b, lo/H, hi/H)
            if np.any(variation > 4.):
                middle = lo+(hi-lo)/2
                if depth >= self.max_depth or not lo < middle < hi:
                    raise ContractError('Affine coefficient variation requires unresolved quadrature subdivision')
                left, le = integrate(lo, middle, budget/2, depth+1)
                right, re = integrate(middle, hi, budget/2, depth+1)
                return left+right, le+re
            low, high = rule(lo, hi, 8), rule(lo, hi, 16)
            error = np.abs(high-low)
            if np.all(error <= budget):
                return high, error
            middle = lo+(hi-lo)/2
            if depth >= self.max_depth or not lo < middle < hi:
                raise ContractError('Affine quadrature exceeds its allocated error; subdivision limit reached')
            left, le = integrate(lo, middle, budget/2, depth+1)
            right, re = integrate(middle, hi, budget/2, depth+1)
            return left+right, le+re

        value, error = integrate(0., s, allocated, 0)
        with np.errstate(over='raise', invalid='raise'):
            try:
                result = np.exp(-Bs)*y0+value
            except FloatingPointError as exc:
                raise ContractError('Affine membrane state outside finite FP64 domain') from exc
        if not np.isfinite(result).all():
            raise ContractError('Nonfinite affine membrane state')
        return result, error

    def evaluate(self, t, side='right', state_ids=None):
        return self.evaluate_with_error(t, side, state_ids)[0]

    def represented_rhs_at(self, t, side='right', state_ids=None):
        return self.a.evaluate(t, side, state_ids)-self.b.evaluate(t, side, state_ids)*self.evaluate(t, side, state_ids)
