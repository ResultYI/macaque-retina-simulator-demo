"""Bitwise regressions against the pre-optimization interval arithmetic.

Only synthetic arrays and accepted packet fixtures are used. No model advances.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from retina_sim import group_dense_certificate as certificate
from retina_sim import group_scc_dense_qa as qa
from retina_sim.continuous_state import AcceptedSeries, DiscreteHold, QuinticHermite
from retina_sim.contracts import ContractError


@pytest.fixture(scope='module')
def prepared_dense_backend():
    previous = certificate._DENSE_KERNELS, certificate._DENSE_PREPARATION, certificate._DENSE_THREADS
    previous_threads = None
    if previous[1] is not None:
        import numba
        previous_threads = numba.get_num_threads()
    metadata = certificate.prepare_dense_certificate()
    import numba
    if previous_threads is None:
        previous_threads = metadata['previous_thread_count']
    assert certificate._DENSE_KERNELS is not None
    try:
        yield certificate._DENSE_KERNELS, certificate._DENSE_PREPARATION, metadata
    finally:
        numba.set_num_threads(previous_threads)
        certificate._DENSE_KERNELS, certificate._DENSE_PREPARATION, certificate._DENSE_THREADS = previous


@pytest.fixture(scope='module', autouse=True, params=(None, 1, 2, 8, 16),
                ids=('numpy', 'compiled_1_thread', 'compiled_2_threads', 'compiled_8_threads', 'compiled_16_threads'))
def dense_execution_backend(request):
    previous = certificate._DENSE_KERNELS, certificate._DENSE_PREPARATION, certificate._DENSE_THREADS
    previous_threads = None
    try:
        if request.param is None:
            certificate._DENSE_KERNELS = None
        else:
            kernels, preparation, _ = request.getfixturevalue('prepared_dense_backend')
            import numba
            previous_threads = numba.get_num_threads()
            numba.set_num_threads(3)
            certificate._DENSE_KERNELS, certificate._DENSE_PREPARATION = kernels, preparation
            certificate._DENSE_THREADS = request.param
            assert numba.get_num_threads() == 3
        yield request.param
    finally:
        if previous_threads is not None:
            numba.set_num_threads(previous_threads)
        certificate._DENSE_KERNELS, certificate._DENSE_PREPARATION, certificate._DENSE_THREADS = previous


@pytest.fixture(autouse=True)
def dense_caller_thread_mask_is_restored(dense_execution_backend):
    if dense_execution_backend is None:
        yield
        return
    import numba
    caller_threads = numba.get_num_threads()
    yield
    assert numba.get_num_threads() == caller_threads == 3


def test_compiled_backend_uses_strict_workqueue_policy(dense_execution_backend):
    if dense_execution_backend is None:
        assert certificate._DENSE_KERNELS is None
        return
    import numba
    metadata = certificate._DENSE_PREPARATION
    assert metadata['implementation'] == 'strict_numba_parallel_coordinate_interval_loops_v2'
    assert metadata['numba_version'] == numba.__version__
    assert metadata['parallel'] is True
    assert metadata['parallel_scope'] == 'independent state coordinates; scalar operation order unchanged'
    assert metadata['error_scan'] == 'ascending state index after parallel loop'
    assert metadata['fastmath'] is False
    assert metadata['boundscheck'] is True
    assert metadata['disk_cache'] is False
    assert metadata['compiled_against_numpy_bits'] is True
    assert metadata['threading_layer'] == numba.threading_layer() == 'workqueue'
    assert metadata['thread_count'] == metadata['fixed_threads']
    assert 8 <= metadata['fixed_threads'] <= metadata['pool_limit']
    assert 1 <= metadata['previous_thread_count'] <= metadata['pool_limit']
    assert metadata['thread_mask_policy'] == 'save/set/restore for each kernel call'
    assert metadata['caller_thread_count_after_prepare'] == metadata['previous_thread_count']
    assert certificate._DENSE_THREADS == dense_execution_backend
    assert numba.get_num_threads() == 3
    for kernel in certificate._DENSE_KERNELS:
        assert kernel.nopython_signatures
        assert kernel.targetoptions['parallel'] is True
        assert kernel.targetoptions['fastmath'] is False
        assert kernel.targetoptions['boundscheck'] is True


# These are the original loops before the QA performance correction.
def _reference_restrict(coefficients, lo, hi):
    lower, upper = coefficients
    degree = lower.shape[0]-1
    if degree == 0:
        return lower.copy(), upper.copy()
    lo_powers = [(np.array(1.), np.array(1.))]
    width_powers = [(np.array(1.), np.array(1.))]
    width = certificate._subtract((np.array(hi), np.array(hi)), (np.array(lo), np.array(lo)))
    for _ in range(degree):
        lo_powers.append(certificate._multiply(lo_powers[-1], (np.array(lo), np.array(lo))))
        width_powers.append(certificate._multiply(width_powers[-1], width))
    result_lower, result_upper = np.zeros_like(lower), np.zeros_like(upper)
    for k in range(degree+1):
        total = (np.zeros(lower.shape[1]), np.zeros(lower.shape[1]))
        for j in range(k, degree+1):
            factor = np.array(float(math.comb(j, k)))
            term = certificate._multiply((lower[j], upper[j]), (factor, factor))
            term = certificate._multiply(term, lo_powers[j-k])
            total = certificate._add(total, term)
        result_lower[k], result_upper[k] = certificate._multiply(total, width_powers[k])
    return result_lower, result_upper


def _reference_bernstein(coefficients):
    lower, upper = coefficients
    degree = lower.shape[0]-1
    if degree == 0:
        return lower[0].copy(), upper[0].copy()
    beta_lower, beta_upper = np.empty_like(lower), np.empty_like(upper)
    for i in range(degree+1):
        total = (np.zeros(lower.shape[1]), np.zeros(lower.shape[1]))
        for k in range(i+1):
            ratio = np.array(math.comb(i, k)/math.comb(degree, k))
            ratio_bounds = certificate._outward(ratio, ratio)
            total = certificate._add(total, certificate._multiply((lower[k], upper[k]), ratio_bounds))
        beta_lower[i], beta_upper[i] = total
    return np.min(beta_lower, axis=0), np.max(beta_upper, axis=0)


def _reference_difference(candidate, reference, *, lo=0., hi=1., reference_lo=None, reference_hi=None):
    candidate = certificate._coefficient_interval(candidate)
    reference = certificate._coefficient_interval(reference)
    count = max(len(candidate[0]), len(reference[0]))
    candidate = tuple(np.pad(bound, ((0, count-len(bound)), (0, 0))) for bound in candidate)
    reference = tuple(np.pad(bound, ((0, count-len(bound)), (0, 0))) for bound in reference)
    if reference_lo is None:
        difference = certificate._subtract(candidate, reference)
        restricted = _reference_restrict(difference, lo, hi)
    else:
        restricted = certificate._subtract(_reference_restrict(candidate, lo, hi),
                                           _reference_restrict(reference, reference_lo, reference_hi))
    return _reference_bernstein(restricted)


def _same_bits(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a.dtype == b.dtype == np.float64
        assert a.shape == b.shape
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


def _coefficients(degree, kind):
    tiny = np.nextafter(0., 1.)
    held = np.array([-0., 0., tiny, -tiny, 3*tiny, .1, -.1,
                     np.nextafter(1., 0.), np.nextafter(1., np.inf),
                     np.finfo(float).tiny, np.finfo(float).max/256])
    lower = np.zeros((degree+1, len(held)))
    lower[0] = held
    if kind.startswith('mixed') and degree:
        random = np.random.default_rng(8192+degree)
        lower[1:] = random.uniform(-.25, .25, (degree, len(held)))
        lower[:, :4] *= tiny
        lower[0, -1] = 1.
    upper = lower.copy()
    if kind.endswith('interval'):
        active = np.ones(lower.shape, bool) if kind.startswith('mixed') else np.indices(lower.shape)[0] == 0
        lower[active] = np.nextafter(lower[active], -np.inf)
        upper[active] = np.nextafter(upper[active], np.inf)
    return lower, upper


@pytest.mark.parametrize('degree', [0, 1, 5, 7, 12])
@pytest.mark.parametrize('kind', ['constant_point', 'constant_interval', 'mixed_point', 'mixed_interval'])
def test_restriction_and_hull_match_original_bits(degree, kind):
    coefficients = _coefficients(degree, kind)
    for lo, hi in ((0., 1.), (.125, .75), (np.nextafter(.5, 0.), np.nextafter(.5, 1.))):
        expected = _reference_restrict(coefficients, lo, hi)
        actual = certificate._restrict(coefficients, lo, hi)
        _same_bits(actual, expected)
        _same_bits(certificate._bernstein_enclosure(actual), _reference_bernstein(expected))


@pytest.mark.parametrize('candidate_degree,reference_degree', [(0,7), (5,7), (7,5), (7,7), (12,5), (5,12)])
def test_padded_mapped_difference_matches_original_bits(candidate_degree, reference_degree):
    for kind in ('constant_point', 'constant_interval', 'mixed_interval'):
        candidate = _coefficients(candidate_degree, kind)
        reference = _coefficients(reference_degree, kind)
        reference[0][0, 5] = np.nextafter(reference[0][0, 5], -np.inf)
        for intervals in (dict(lo=.125, hi=.75),
                          dict(lo=0., hi=1., reference_lo=.25, reference_hi=.5)):
            _same_bits(certificate.polynomial_difference_enclosure(candidate, reference, **intervals),
                       _reference_difference(candidate, reference, **intervals))


def test_constant_fast_paths_preserve_original_overflow_rejection():
    coefficients = np.zeros((8, 1)); coefficients[0, 0] = np.finfo(float).max
    for implementation in (_reference_restrict, certificate._restrict):
        with pytest.raises(ContractError, match='finite FP64 domain'):
            implementation((coefficients, coefficients), 0., 1.)
    for implementation in (_reference_bernstein, certificate._bernstein_enclosure):
        with pytest.raises(ContractError, match='finite FP64 domain'):
            implementation((coefficients, coefficients))


def test_held_dop_preparation_and_range_match_original_bits():
    y0 = np.array([-0., 0., np.nextafter(0., 1.), -.1, .2, np.nextafter(1., 0.)])
    f = np.zeros((7, len(y0)))
    source = dict(t_old=0., t=2.**-14, y_old=y0, F=f)
    counters = dict(host_readback_bytes=0, polynomial_conversions=0)
    actual = qa._prepare(qa._piece(source, 'dop853_dense'), len(y0), counters)
    expected = certificate.dop853_power_coefficient_enclosure(y0, f)
    expected[0][:] = 0.; expected[1][:] = 0.
    expected[0][0] = y0; expected[1][0] = y0
    _same_bits(actual.coefficients, expected)
    _same_bits((actual.roundoff, actual.remainder), (np.zeros_like(y0), np.zeros_like(y0)))
    for lo, hi in ((0., 1.), (.125, .75)):
        old_range = _reference_bernstein(_reference_restrict(expected, lo, hi))
        old_range[0][:] = expected[0][0]; old_range[1][:] = expected[1][0]
        old_range = qa._expand(old_range, qa._sum_upper([actual.remainder, actual.roundoff]))
        _same_bits(qa._range(actual, lo, hi), old_range)


def _wrapper_fixture(candidate_kind, offset):
    key = ('synthetic', 'dense_equivalence', 'V')
    stop = 2.**-14
    initial = np.array([.2, .6, .8])
    slope = np.zeros(3) if candidate_kind == 'hold' else np.array([.25, -.5, 0.])
    model = SimpleNamespace(state_keys=(key,), slices={key:slice(0, 3)},
                            parameters={'populations':{}})
    arm = lambda: SimpleNamespace(inner=model, native_physics=SimpleNamespace(slices={}), native_intervals=[])
    def dop(a, b, extra=0.):
        f = np.zeros((7, 3)); f[0] = slope*(b-a)
        return dict(t_old=a, t=b, y_old=initial+slope*a+extra, F=f)
    reference = SimpleNamespace(t0=0., t1=stop, dense_segments=(dop(0., stop),), representations={})
    if candidate_kind == 'dop':
        candidate = SimpleNamespace(t0=0., t1=stop,
            dense_segments=(dop(0., stop/2, offset), dop(stop/2, stop, offset)), representations={})
    else:
        segments = []
        for a, b in ((0., stop/2), (stop/2, stop)):
            common = dict(t0=a,t1=b,state_ids=np.arange(3),y0=initial+slope*a+offset)
            segments.append(DiscreteHold(**common) if candidate_kind == 'hold' else
                QuinticHermite(**common,ym=initial+slope*(a+(b-a)/2)+offset,
                    y1=initial+slope*b+offset,f0=slope,fm=slope,f1=slope))
        candidate = SimpleNamespace(t0=0., t1=stop,dense_segments=(),
            representations={key:[(np.arange(3),AcceptedSeries(segments))]})
    row = dict(spec=SimpleNamespace(count=3,unit='V',meaning='voltage'),initial=initial,
               scale=np.ones(3),budget=np.full(3, 1e-8))
    return arm(), arm(), reference, candidate, 0., stop, {key:row}


@pytest.mark.parametrize('candidate_kind', ['hold', 'dop', 'quintic'])
@pytest.mark.parametrize('offset', [0., .001])
def test_complete_wrapper_status_and_witnesses_match_original_loops(monkeypatch, candidate_kind, offset):
    actual = qa.compare_dense_interval(*_wrapper_fixture(candidate_kind, offset))
    with monkeypatch.context() as patch:
        patch.setattr(certificate, '_restrict', _reference_restrict)
        patch.setattr(certificate, '_bernstein_enclosure', _reference_bernstein)
        expected = qa.compare_dense_interval(*_wrapper_fixture(candidate_kind, offset))
    actual.pop('wall_seconds'); expected.pop('wall_seconds')
    assert actual == expected


@pytest.mark.parametrize('candidate_degree,reference_degree', [(0,7), (5,7), (12,7), (7,12)])
def test_shared_pair_enclosures_match_three_original_calls(candidate_degree, reference_degree):
    for kind in ('constant_point', 'constant_interval', 'mixed_interval'):
        candidate = _coefficients(candidate_degree, kind)
        reference = _coefficients(reference_degree, kind)
        for lo, hi, rlo, rhi in ((0.,1.,0.,1.), (.125,.75,.25,.625)):
            actual = certificate.polynomial_pair_enclosures(candidate, reference,
                lo=lo,hi=hi,reference_lo=rlo,reference_hi=rhi)
            expected = (_reference_difference(candidate, reference,lo=lo,hi=hi,reference_lo=rlo,reference_hi=rhi),
                _reference_bernstein(_reference_restrict(candidate,lo,hi)),
                _reference_bernstein(_reference_restrict(reference,rlo,rhi)))
            for a, b in zip(actual, expected):
                _same_bits(a, b)


@pytest.mark.parametrize('value', [np.finfo(float).max, -np.finfo(float).max,
                                  np.nextafter(np.finfo(float).max, 0.),
                                  np.nextafter(-np.finfo(float).max, 0.)])
def test_held_dop_shortcut_preserves_near_limit_exception_order(value):
    y0, f = np.array([value]), np.zeros((7, 1))
    def original():
        bounds = certificate.dop853_power_coefficient_enclosure(y0, f)
        roundoff = qa._roundoff([abs(y0), *abs(f)], 64)
        roundoff[:] = 0.
        bounds[0][:] = 0.; bounds[1][:] = 0.
        bounds[0][0] = y0; bounds[1][0] = y0
        return bounds, roundoff
    try:
        expected, expected_roundoff = original()
    except ContractError as exc:
        with pytest.raises(ContractError, match=str(exc)):
            qa._prepare(qa._piece(dict(t_old=0.,t=2.**-14,y_old=y0,F=f), 'dop853_dense'),
                        1, dict(host_readback_bytes=0,polynomial_conversions=0))
    else:
        actual = qa._prepare(qa._piece(dict(t_old=0.,t=2.**-14,y_old=y0,F=f), 'dop853_dense'),
                             1, dict(host_readback_bytes=0,polynomial_conversions=0))
        _same_bits(actual.coefficients, expected)
        _same_bits((actual.roundoff,), (expected_roundoff,))


def test_domain_failure_witness_matches_original_loops(monkeypatch):
    def fixture():
        result = _wrapper_fixture('quintic', 1.2)
        row = next(iter(result[-1].values()))
        row['spec'].unit = 'dimensionless'
        row['spec'].meaning = 'release_probability'
        return result
    actual = qa.compare_dense_interval(*fixture())
    with monkeypatch.context() as patch:
        patch.setattr(certificate, '_restrict', _reference_restrict)
        patch.setattr(certificate, '_bernstein_enclosure', _reference_bernstein)
        expected = qa.compare_dense_interval(*fixture())
    assert actual['status'] == expected['status'] == 'DOMAIN_FAIL'
    actual.pop('wall_seconds'); expected.pop('wall_seconds')
    assert actual == expected


@pytest.mark.parametrize('degree', [5,7,12])
@pytest.mark.parametrize('layout', ['readonly', 'strided'])
def test_nonconstant_subnormal_and_normal_boundaries_match_original_bits(degree, layout):
    tiny = np.nextafter(0.,1.)
    c0 = np.array([-0.,0.,tiny,-tiny,np.finfo(float).tiny,
        np.nextafter(np.finfo(float).tiny,0.),-np.finfo(float).tiny,
        np.nextafter(-np.finfo(float).tiny,0.),.5,-.5])
    values = np.zeros((degree+1,len(c0)))
    values[0] = c0
    values[1] = np.array([tiny,-tiny,2*tiny,-2*tiny,3*tiny,-3*tiny,tiny,-tiny,tiny,-tiny])
    values[-1] = -values[1]
    assert np.any(values[1:] != 0)
    if layout == 'strided':
        backing = np.zeros((2*(degree+1),2*len(c0)))
        backing[::2,::2] = values
        values = backing[::2,::2]
        assert not values.flags.c_contiguous
    values.flags.writeable = False
    before = values.copy()
    for lo,hi in ((0.,1.),(.125,.875)):
        actual = certificate.restrict_power_coefficients(values,lo=lo,hi=hi)
        expected = _reference_restrict((values,values),lo,hi)
        _same_bits(actual,expected)
        _same_bits(certificate.polynomial_range_enclosure(values,lo=lo,hi=hi),
                   _reference_bernstein(expected))
    _same_bits((values,),(before,))
    assert not values.flags.writeable


@pytest.mark.parametrize('sign', [-1.,1.])
def test_nonconstant_maximum_inputs_preserve_compiled_overflow_guard(sign):
    values = np.zeros((8,1))
    values[0,0] = sign*np.finfo(float).max
    values[1,0] = np.nextafter(0.,sign)
    for implementation in (_reference_restrict,certificate._restrict):
        with pytest.raises(ContractError,match='finite FP64 domain'):
            implementation((values,values),0.,1.)


@pytest.mark.parametrize('degree', [2, 7, 12])
@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('bad_states', [(0,), (20,), (40,), (0, 20, 40)])
def test_multicoordinate_overflow_survives_later_zero_shortcut(degree, sign, bad_states):
    values = np.zeros((degree+1, 41))
    values[0] = np.linspace(-.1, .1, 41)
    values[-1] = .125
    values[-1, list(bad_states)] = sign*(np.finfo(float).max/2)
    before = values.copy()
    # At lo=0 an overflowing intermediate binomial term is later multiplied by
    # zero; the error must survive even when all final coefficient bounds fit.
    for implementation in (_reference_restrict, certificate._restrict):
        with pytest.raises(ContractError, match='finite FP64 domain'):
            implementation((values, values), 0., .125)
    _same_bits((values,), (before,))


@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('bad_states', [(0,), (20,), (40,), (0, 20, 40)])
def test_multicoordinate_bernstein_overflow_matches_original_guard(sign, bad_states):
    values = np.zeros((8, 41))
    values[0] = np.linspace(-.1, .1, 41)
    values[1] = np.nextafter(0., 1.)
    values[0, list(bad_states)] = sign*np.finfo(float).max
    before = values.copy()
    for implementation in (_reference_bernstein, certificate._bernstein_enclosure):
        with pytest.raises(ContractError, match='finite FP64 domain'):
            implementation((values, values))
    _same_bits((values,), (before,))


def test_compiled_kernel_calls_switch_and_restore_thread_mask(monkeypatch, dense_execution_backend):
    if dense_execution_backend is None:
        return
    import numba
    observed = []
    def observe(kernel):
        def call(*args):
            observed.append(numba.get_num_threads())
            return kernel(*args)
        return call
    with monkeypatch.context() as patch:
        patch.setattr(certificate, '_DENSE_KERNELS', tuple(map(observe, certificate._DENSE_KERNELS)))
        coefficients = _coefficients(7, 'mixed_interval')
        restricted = certificate._restrict(coefficients, .125, .75)
        assert numba.get_num_threads() == 3
        certificate._bernstein_enclosure(restricted)
        assert numba.get_num_threads() == 3
    assert observed == [dense_execution_backend, dense_execution_backend]
