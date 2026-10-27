"""Static degree-at-most-twelve polynomial enclosures, without ODE execution.

These enclose the real polynomials defined by supplied FP64 power coefficients.
Every arithmetic operation in the restriction and Bernstein conversion rounds
outward. They do not include coefficient construction, waveform evaluation,
interpolation error, analytic rows, nonlinear RHS residuals or ODE solution error.
A candidate-reference difference budget is never an absolute ODE error budget.
"""
import math
from functools import lru_cache
from time import perf_counter
from numbers import Real

import numpy as np

from .contracts import ContractError

_DENSE_KERNELS = None
_DENSE_PREPARATION = None
_DENSE_THREADS = 16


def _coefficients(value):
    result = np.asarray(value)
    if (result.dtype != np.float64 or result.ndim != 2
            or not 1 <= result.shape[0] <= 13 or result.shape[1] < 1
            or not np.isfinite(result).all()):
        raise ContractError('Finite FP64 ascending power coefficients (1..13, state_count) required')
    return result



def _coefficient_interval(value):
    if isinstance(value, tuple) and len(value) == 2:
        lower, upper = map(_coefficients, value)
        if lower.shape != upper.shape or np.any(lower > upper):
            raise ContractError('Ordered matching lower/upper coefficient arrays required')
        return lower, upper
    coefficients = _coefficients(value)
    return coefficients, coefficients


def _interval(lo, hi):
    if (not isinstance(lo, Real) or not isinstance(hi, Real)
            or isinstance(lo, (bool, np.bool_)) or isinstance(hi, (bool, np.bool_))
            or not math.isfinite(lo) or not math.isfinite(hi) or not 0 <= lo < hi <= 1):
        raise ContractError('Finite normalized polynomial subinterval 0 <= lo < hi <= 1 required')
    return float(lo), float(hi)


def _outward(lower, upper):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        lower = np.nextafter(lower, -np.inf)
        upper = np.nextafter(upper, np.inf)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ContractError('Polynomial enclosure exceeds the finite FP64 domain')
    return lower, upper


def _add(a, b):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        lower, upper = _outward(a[0]+b[0], a[1]+b[1])
    azero, bzero = (a[0] == 0) & (a[1] == 0), (b[0] == 0) & (b[1] == 0)
    return np.where(azero, b[0], np.where(bzero, a[0], lower)), np.where(azero, b[1], np.where(bzero, a[1], upper))


def _subtract(a, b):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        lower, upper = _outward(a[0]-b[1], a[1]-b[0])
    equal_points = (a[0] == a[1]) & (b[0] == b[1]) & (a[0] == b[0])
    return np.where(equal_points, 0., lower), np.where(equal_points, 0., upper)


def _multiply(a, b):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        products = np.stack(np.broadcast_arrays(a[0]*b[0], a[0]*b[1],
                                                a[1]*b[0], a[1]*b[1]))
        lower, upper = _outward(np.min(products, axis=0), np.max(products, axis=0))
    zero = ((a[0] == 0) & (a[1] == 0)) | ((b[0] == 0) & (b[1] == 0))
    return np.where(zero, 0., lower), np.where(zero, 0., upper)


def _restrict(coefficients, lo, hi):
    lower, upper = coefficients
    degree = lower.shape[0]-1
    if degree == 0:
        return lower.copy(), upper.copy()
    if np.all(lower[1:] == 0) and np.all(upper[1:] == 0):
        one = (np.array(1.), np.array(1.))
        zero = (np.zeros(lower.shape[1]), np.zeros(lower.shape[1]))
        term = _multiply(_multiply((lower[0], upper[0]), one), one)
        total = _add(_add(zero, term), zero)
        result_lower, result_upper = np.zeros_like(lower), np.zeros_like(upper)
        result_lower[0], result_upper[0] = _multiply(total, one)
        return result_lower, result_upper
    if _DENSE_KERNELS is not None:
        powers, combinations = _restriction_constants(degree, lo, hi)
        try:
            return _run_dense_kernel(_DENSE_KERNELS[0], _kernel_array(lower), _kernel_array(upper), powers, combinations)
        except ValueError as exc:
            raise ContractError('Polynomial enclosure exceeds the finite FP64 domain') from exc
    lo_powers = [(np.array(1.), np.array(1.))]
    width_powers = [(np.array(1.), np.array(1.))]
    width = _subtract((np.array(hi), np.array(hi)), (np.array(lo), np.array(lo)))
    for _ in range(degree):
        lo_powers.append(_multiply(lo_powers[-1], (np.array(lo), np.array(lo))))
        width_powers.append(_multiply(width_powers[-1], width))
    result_lower, result_upper = np.zeros_like(lower), np.zeros_like(upper)
    for k in range(degree+1):
        total = (np.zeros(lower.shape[1]), np.zeros(lower.shape[1]))
        for j in range(k, degree+1):
            factor = np.array(float(math.comb(j, k)))
            term = _multiply((lower[j], upper[j]), (factor, factor))
            term = _multiply(term, lo_powers[j-k])
            total = _add(total, term)
        result_lower[k], result_upper[k] = _multiply(total, width_powers[k])
    return result_lower, result_upper


def restrict_power_coefficients(coefficients, *, lo=0., hi=1.):
    """Enclose coefficients of p(lo+(hi-lo)*x), x in [0,1].

    Input may be power coefficients or a (lower, upper) coefficient pair.
    Return owned lower/upper arrays. Bounds include transformation arithmetic
    but assume the supplied coefficient values or enclosures.
    """
    coefficients = _coefficient_interval(coefficients)
    lo, hi = _interval(lo, hi)
    return _restrict(coefficients, lo, hi)




def quintic_power_coefficient_enclosure(y0, f0, ym, fm, y1, f1, *, duration):
    """Enclose Hermite anchor-to-power arithmetic using the supplied FP64 h.

    The six inputs are FP64 state vectors, and f values are physical-time
    slopes. This still excludes arithmetic in subsequent waveform evaluation.
    """
    vectors = tuple(np.asarray(value) for value in (y0, f0, ym, fm, y1, f1))
    if (any(value.dtype != np.float64 or value.ndim != 1 or not len(value)
            or not np.isfinite(value).all() for value in vectors)
            or any(value.shape != vectors[0].shape for value in vectors)
            or not isinstance(duration, Real) or isinstance(duration, (bool, np.bool_))
            or not math.isfinite(duration) or duration <= 0):
        raise ContractError('Matching finite FP64 Hermite vectors and positive duration required')
    y0, f0, ym, fm, y1, f1 = tuple((value, value) for value in vectors)
    h = (np.array(float(duration)), np.array(float(duration)))
    half = (np.array(.5), np.array(.5))
    s0, sm, s1 = (_multiply(h, value) for value in (f0, fm, f1))
    a = _subtract(_subtract(ym, y0), _multiply(s0, half))
    b = _subtract(sm, s0)
    c = _subtract(_subtract(y1, y0), s0)
    d = _subtract(s1, s0)
    powers = [y0, s0]
    for weights in ((16., -8., 7., -1.), (-32., 32., -34., 5.),
                    (16., -40., 52., -8.), (0., 16., -24., 4.)):
        total = (np.zeros_like(vectors[0]), np.zeros_like(vectors[0]))
        for weight, value in zip(weights, (a, b, c, d)):
            if weight:
                factor = np.array(weight)
                total = _add(total, _multiply((factor, factor), value))
        powers.append(total)
    return tuple(np.stack([power[side] for power in powers]) for side in (0, 1))


def dop853_power_coefficient_enclosure(y0, coefficients):
    """Enclose ascending powers from the original alternating DOP853 F form."""
    coefficients = _coefficients(coefficients)
    y0 = np.asarray(y0)
    if (coefficients.shape[0] != 7 or y0.dtype != np.float64
            or y0.shape != (coefficients.shape[1],) or not np.isfinite(y0).all()):
        raise ContractError('DOP853 finite FP64 y0 and F shape (7, state_count) required')
    value = (np.zeros((1, len(y0))), np.zeros((1, len(y0))))
    for i, coefficient in enumerate(coefficients[::-1]):
        value[0][0], value[1][0] = _add((value[0][0], value[1][0]),
                                       (coefficient, coefficient))
        shifted = tuple(np.pad(bound, ((1, 0), (0, 0))) for bound in value)
        if i % 2 == 0:
            value = shifted
        else:
            value = _subtract(tuple(np.pad(bound, ((0, 1), (0, 0))) for bound in value), shifted)
    value[0][0], value[1][0] = _add((value[0][0], value[1][0]), (y0, y0))
    return value


def _bernstein_enclosure(coefficients):
    lower, upper = coefficients
    degree = lower.shape[0]-1
    if degree == 0:
        return lower[0].copy(), upper[0].copy()
    if np.all(lower[1:] == 0) and np.all(upper[1:] == 0):
        ratio = _outward(np.array(1.), np.array(1.))
        zero = (np.zeros(lower.shape[1]), np.zeros(lower.shape[1]))
        return _add(_add(zero, _multiply((lower[0], upper[0]), ratio)), zero)
    if _DENSE_KERNELS is not None:
        try:
            return _run_dense_kernel(_DENSE_KERNELS[1], _kernel_array(lower), _kernel_array(upper), _bernstein_constants(degree))
        except ValueError as exc:
            raise ContractError('Polynomial enclosure exceeds the finite FP64 domain') from exc
    beta_lower, beta_upper = np.empty_like(lower), np.empty_like(upper)
    for i in range(degree+1):
        total = (np.zeros(lower.shape[1]), np.zeros(lower.shape[1]))
        for k in range(i+1):
            ratio = np.array(math.comb(i, k)/math.comb(degree, k))
            ratio_bounds = _outward(ratio, ratio)
            total = _add(total, _multiply((lower[k], upper[k]), ratio_bounds))
        beta_lower[i], beta_upper[i] = total
    return np.min(beta_lower, axis=0), np.max(beta_upper, axis=0)



def polynomial_range_enclosure(coefficients, *, lo=0., hi=1.):
    """Enclose every represented value on one normalized subinterval."""
    coefficients = _coefficient_interval(coefficients)
    lo, hi = _interval(lo, hi)
    return _bernstein_enclosure(_restrict(coefficients, lo, hi))


def polynomial_difference_enclosure(candidate_coefficients, reference_coefficients, *,
        lo=0., hi=1., reference_lo=None, reference_hi=None):
    """Bound candidate minus reference at every point of a common subinterval.

    Each input is ascending normalized power coefficients or a (lower, upper)
    coefficient interval pair, not SciPy DOP853 F coefficients. Both have the same ordered state rows; degrees may differ.
    With reference_lo/reference_hi, compare candidate(lo+(hi-lo)*x) against
    reference(reference_lo+(reference_hi-reference_lo)*x) for x in [0,1].
    The caller establishes that these maps cover the same physical interval.
    """
    candidate = _coefficient_interval(candidate_coefficients)
    reference = _coefficient_interval(reference_coefficients)
    lo, hi = _interval(lo, hi)
    if candidate[0].shape[1] != reference[0].shape[1]:
        raise ContractError('Polynomial difference requires identical ordered state rows')
    if (reference_lo is None) != (reference_hi is None):
        raise ContractError('Both reference subinterval endpoints are required')
    degree_count = max(len(candidate[0]), len(reference[0]))
    candidate = tuple(np.pad(bound, ((0, degree_count-len(bound)), (0, 0))) for bound in candidate)
    reference = tuple(np.pad(bound, ((0, degree_count-len(bound)), (0, 0))) for bound in reference)
    if reference_lo is None:
        difference = _subtract(candidate, reference)
        restricted = _restrict(difference, lo, hi)
    else:
        reference_lo, reference_hi = _interval(reference_lo, reference_hi)
        restricted = _subtract(_restrict(candidate, lo, hi),
                               _restrict(reference, reference_lo, reference_hi))
    return _bernstein_enclosure(restricted)



def polynomial_pair_enclosures(candidate_coefficients, reference_coefficients, *,
        lo=0., hi=1., reference_lo=0., reference_hi=1.):
    """Return the same difference and two domain hulls with shared restrictions.

    Zero padding leaves each original-degree prefix unchanged. Each domain's
    Bernstein conversion still uses its original degree and operation order.
    """
    candidate = _coefficient_interval(candidate_coefficients)
    reference = _coefficient_interval(reference_coefficients)
    lo, hi = _interval(lo, hi)
    reference_lo, reference_hi = _interval(reference_lo, reference_hi)
    if candidate[0].shape[1] != reference[0].shape[1]:
        raise ContractError('Polynomial difference requires identical ordered state rows')
    nc, nr = len(candidate[0]), len(reference[0])
    degree_count = max(nc, nr)
    pad = lambda bounds: tuple(np.pad(bound, ((0, degree_count-len(bound)), (0, 0))) for bound in bounds)
    cand_restricted = _restrict(pad(candidate), lo, hi)
    ref_restricted = _restrict(pad(reference), reference_lo, reference_hi)
    difference = _bernstein_enclosure(_subtract(cand_restricted, ref_restricted))
    cand_range = _bernstein_enclosure(candidate if nc == 1 else tuple(bound[:nc] for bound in cand_restricted))
    ref_range = _bernstein_enclosure(reference if nr == 1 else tuple(bound[:nr] for bound in ref_restricted))
    return difference, cand_range, ref_range


def polynomial_difference_bound(candidate_coefficients, reference_coefficients, **interval):
    """Per-state absolute convex-hull bound; no budget or ODE-error assertion."""
    lower, upper = polynomial_difference_enclosure(candidate_coefficients,
                                                   reference_coefficients, **interval)
    return np.maximum(np.abs(lower), np.abs(upper))


def _kernel_array(value):
    return value if value.flags.c_contiguous and value.flags.writeable else np.array(value, order='C', copy=True)


@lru_cache(maxsize=256)
def _restriction_constants(degree, lo, hi):
    powers = np.empty((4, degree+1))
    powers[:, 0] = 1.
    width = _subtract((np.array(hi), np.array(hi)), (np.array(lo), np.array(lo)))
    for k in range(1, degree+1):
        powers[0, k], powers[1, k] = _multiply((powers[0, k-1], powers[1, k-1]), (lo, lo))
        powers[2, k], powers[3, k] = _multiply((powers[2, k-1], powers[3, k-1]), width)
    combinations = np.zeros((degree+1, degree+1))
    for j in range(degree+1):
        for k in range(j+1):
            combinations[j, k] = float(math.comb(j, k))
    powers.flags.writeable = combinations.flags.writeable = False
    return powers, combinations


@lru_cache(maxsize=13)
def _bernstein_constants(degree):
    weights = np.zeros((2, degree+1, degree+1))
    for i in range(degree+1):
        for k in range(i+1):
            ratio = np.array(math.comb(i, k)/math.comb(degree, k))
            weights[0, i, k], weights[1, i, k] = _outward(ratio, ratio)
    weights.flags.writeable = False
    return weights


def _run_dense_kernel(kernel, *args):
    import numba
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(_DENSE_THREADS)
        if numba.get_num_threads() != _DENSE_THREADS:
            raise ContractError('Dense certificate fixed thread mask was not applied')
        return kernel(*args)
    finally:
        numba.set_num_threads(previous)


def prepare_dense_certificate():
    """Compile and check strict CPU loops; the caller charges this to cold QA.

    Until explicitly prepared, all queries use the unchanged NumPy loops.
    Workqueue runs independent coordinates with a scoped fixed thread mask. No disk
    cache, device initialization or model operations are used.
    """
    global _DENSE_KERNELS, _DENSE_PREPARATION
    if _DENSE_PREPARATION is not None:
        return dict(_DENSE_PREPARATION, already_prepared=True, preparation_seconds=0.)
    started = perf_counter()
    from numba import njit, prange
    import numba
    try:
        initialized_layer = numba.threading_layer()
    except ValueError:
        initialized_layer = None
    if initialized_layer not in (None, 'workqueue'):
        raise ContractError('Dense certificate requires the workqueue Numba threading layer')
    if numba.config.NUMBA_NUM_THREADS < _DENSE_THREADS:
        raise ContractError('Dense certificate fixed threads exceed the process Numba pool')
    numba.config.THREADING_LAYER = 'workqueue'
    previous_thread_count = numba.get_num_threads()
    if numba.threading_layer() != 'workqueue':
        raise ContractError('Dense certificate workqueue threading layer was not selected')
    options = dict(cache=False, fastmath=False, boundscheck=True, error_model='numpy')

    @njit(inline='always', **options)
    def adjacent(value, up):
        if value == 0.:
            return 5e-324 if up else -5e-324
        if not np.isfinite(value):
            return np.nextafter(value, np.inf if up else -np.inf)
        bits = np.float64(value).view(np.uint64)
        increment = np.uint64(1) if (value > 0.) == up else np.uint64(18446744073709551615)
        result = np.uint64(bits+increment)
        return result.view(np.float64)

    @njit(inline='always', **options)
    def outward(lo, hi):
        lo, hi = adjacent(lo, False), adjacent(hi, True)
        invalid = not np.isfinite(lo) or not np.isfinite(hi)
        return lo, hi, invalid

    @njit(inline='always', **options)
    def add(a0, a1, b0, b1):
        lo, hi, invalid = outward(a0+b0, a1+b1)
        if a0 == 0. and a1 == 0.:
            return b0, b1, invalid
        if b0 == 0. and b1 == 0.:
            return a0, a1, invalid
        return lo, hi, invalid

    @njit(inline='always', **options)
    def multiply(a0, a1, b0, b1):
        p0, p1, p2, p3 = a0*b0, a0*b1, a1*b0, a1*b1
        lo, hi, invalid = outward(min(p0, p1, p2, p3), max(p0, p1, p2, p3))
        if (a0 == 0. and a1 == 0.) or (b0 == 0. and b1 == 0.):
            return 0., 0., invalid
        return lo, hi, invalid

    @njit(parallel=True, **options)
    def restrict_kernel(lower, upper, powers, combinations):
        degree = lower.shape[0]-1
        out_lo, out_hi = np.empty_like(lower), np.empty_like(upper)
        errors = np.empty(lower.shape[1], dtype=np.bool_)
        for state in prange(lower.shape[1]):
            invalid = False
            for k in range(degree+1):
                lo, hi = 0., 0.
                for j in range(k, degree+1):
                    factor = combinations[j, k]
                    a, b, failed = multiply(lower[j, state], upper[j, state], factor, factor)
                    invalid = invalid or failed
                    a, b, failed = multiply(a, b, powers[0, j-k], powers[1, j-k])
                    invalid = invalid or failed
                    lo, hi, failed = add(lo, hi, a, b)
                    invalid = invalid or failed
                out_lo[k, state], out_hi[k, state], failed = multiply(lo, hi, powers[2, k], powers[3, k])
                invalid = invalid or failed
            errors[state] = invalid
        for state in range(lower.shape[1]):
            if errors[state]:
                raise ValueError('nonfinite interval')
        return out_lo, out_hi

    @njit(parallel=True, **options)
    def bernstein_kernel(lower, upper, weights):
        degree = lower.shape[0]-1
        out_lo, out_hi = np.empty(lower.shape[1]), np.empty(lower.shape[1])
        errors = np.empty(lower.shape[1], dtype=np.bool_)
        for state in prange(lower.shape[1]):
            invalid = False
            minimum, maximum = np.inf, -np.inf
            for i in range(degree+1):
                lo, hi = 0., 0.
                for k in range(i+1):
                    a, b, failed = multiply(lower[k, state], upper[k, state], weights[0, i, k], weights[1, i, k])
                    invalid = invalid or failed
                    lo, hi, failed = add(lo, hi, a, b)
                    invalid = invalid or failed
                minimum, maximum = min(minimum, lo), max(maximum, hi)
            out_lo[state], out_hi[state] = minimum, maximum
            errors[state] = invalid
        for state in range(lower.shape[1]):
            if errors[state]:
                raise ValueError('nonfinite interval')
        return out_lo, out_hi

    values = np.arange(-24., 24.).reshape(8, 6)/29
    expected_restriction = _restrict((values, values), .125, .625)
    powers, combinations = _restriction_constants(7, .125, .625)
    actual_restriction = _run_dense_kernel(restrict_kernel, values, values, powers, combinations)
    expected_hull = _bernstein_enclosure(expected_restriction)
    actual_hull = _run_dense_kernel(bernstein_kernel, *actual_restriction, _bernstein_constants(7))
    if any(not np.array_equal(a.view(np.uint64), b.view(np.uint64))
            for actual, expected in ((actual_restriction, expected_restriction), (actual_hull, expected_hull))
            for a, b in zip(actual, expected)):
        raise ContractError('Compiled interval arithmetic differs from its original NumPy order')
    _DENSE_KERNELS = restrict_kernel, bernstein_kernel
    _DENSE_PREPARATION = dict(implementation='strict_numba_parallel_coordinate_interval_loops_v2',
        numba_version=numba.__version__, fastmath=False, parallel=True, disk_cache=False,
        parallel_scope='independent state coordinates; scalar operation order unchanged',
        error_scan='ascending state index after parallel loop',
        threading_layer=numba.threading_layer(), thread_count=_DENSE_THREADS,
        thread_mask_policy='save/set/restore for each kernel call',
        thread_count_check='get_num_threads before every kernel call',
        fixed_threads=_DENSE_THREADS, pool_limit=int(numba.config.NUMBA_NUM_THREADS),
        previous_thread_count=previous_thread_count,
        caller_thread_count_after_prepare=numba.get_num_threads(), boundscheck=True,
        compiled_against_numpy_bits=True, preparation_seconds=perf_counter()-started)
    return dict(_DENSE_PREPARATION)
