"""Synthetic waveform/packet arithmetic only; no retina trajectory or GPU."""
from fractions import Fraction

import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.continuous_state import (Dop853Dense, ExponentialForced,
    QuinticHermite, from_packet, load_representation, save_representation)
from retina_sim.group_dense_certificate import (polynomial_difference_bound,
    polynomial_difference_enclosure, restrict_power_coefficients,
    dop853_power_coefficient_enclosure, quintic_power_coefficient_enclosure)


def _poly(coefficients, x):
    return np.polynomial.polynomial.polyval(x, coefficients)


def _representation(coefficients, *, t0=3., h=2.**-7, **metadata):
    derivative = np.arange(1, len(coefficients))[:, None]*coefficients[1:]
    return QuinticHermite(t0=t0, t1=t0+h,
        state_ids=np.arange(coefficients.shape[1])[::-1],
        y0=_poly(coefficients, 0.), f0=_poly(derivative, 0.)/h,
        ym=_poly(coefficients, .5), fm=_poly(derivative, .5)/h,
        y1=_poly(coefficients, 1.), f1=_poly(derivative, 1.)/h, **metadata)


@pytest.mark.parametrize('degree', range(1, 6))
def test_quintic_reproduces_polynomials_with_physical_time_slopes(degree):
    coefficients = np.arange(1., 2*(degree+1)+1).reshape(degree+1, 2)/16
    coefficients[1::2, 0] *= -1
    representation = _representation(coefficients)
    h = representation.t1-representation.t0
    derivative = np.arange(1, degree+1)[:, None]*coefficients[1:]
    for x in np.linspace(0., 1., 33):
        t = representation.t0+h*x
        side = 'left' if x == 1 else 'right'
        np.testing.assert_allclose(representation.evaluate(t, side), _poly(coefficients, x),
                                   rtol=0, atol=3e-14)
        np.testing.assert_allclose(representation.representation_derivative(t, side),
                                   _poly(derivative, x)/h, rtol=0, atol=3e-11)
    padded = np.pad(coefficients, ((0, 6-len(coefficients)), (0, 0)))
    np.testing.assert_allclose(representation.coefficients, padded, rtol=0, atol=2e-14)
    assert representation.kind == 'quintic_hermite'


def test_large_constant_and_linear_baselines_do_not_create_spurious_curvature():
    coefficients = np.array([[2.**40, -2.**35], [.125, 0.]])
    representation = _representation(coefficients)
    np.testing.assert_array_equal(representation.coefficients[2:], np.zeros((4, 2)))
    for x in (.125, .5, .875):
        np.testing.assert_array_equal(representation.evaluate(3.+x/128, 'right'), _poly(coefficients, x))
        np.testing.assert_array_equal(representation.representation_derivative(3.+x/128, 'right'), [16., 0.])


def test_quintic_owns_anchors_and_observes_boundaries_ids_and_original_rhs(tmp_path):
    arrays = dict(y0=np.array([1., 2.]), f0=np.array([3., 4.]),
        ym=np.array([5., 6.]), fm=np.array([7., 8.]),
        y1=np.array([9., 10.]), f1=np.array([11., 12.]))
    original = QuinticHermite(t0=0., t1=.125, state_ids=['b', 'a'], **arrays,
        original_rhs=lambda t, side, y: np.full_like(y, 123.), original_rhs_id='original-rhs')
    identity = original.representation_id
    for array in arrays.values():
        array[:] = -999.
    for array in original.packet()['arrays'].values():
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(ValueError):
        original.coefficients.flags.writeable = True
    output = original.evaluate(.0625, 'right', ['a', 'b'])
    np.testing.assert_allclose(output, [6., 5.], rtol=0, atol=1e-14)
    output[:] = 999.
    np.testing.assert_allclose(original.evaluate(.0625, 'right'), [5., 6.], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(original.rhs_at(.0625, 'left', ['a']), [123.])
    assert original.representation_derivative(.0625, 'right')[0] != 123.
    for t, side in ((0., 'left'), (.125, 'right'), (np.nextafter(.125, np.inf), 'left')):
        with pytest.raises(ContractError):
            original.evaluate(t, side)
    with pytest.raises(ContractError):
        original.evaluate(.03, 'right', ['missing'])
    directory = tmp_path/'quintic'
    save_representation(directory, original)
    restored = load_representation(directory)
    assert restored.representation_id == identity
    for t, side in ((0., 'right'), (.037, 'right'), (.0625, 'left'), (.125, 'left')):
        np.testing.assert_array_equal(restored.evaluate(t, side), original.evaluate(t, side))
    with pytest.raises(ContractError, match='original-equation'):
        restored.rhs_at(.03, 'right')
    with pytest.raises(ContractError, match='identity'):
        from_packet(original.packet(), original_rhs=lambda t, side, y: y, original_rhs_id='wrong')
    tampered = original.packet()
    tampered['arrays'] = dict(tampered['arrays'], fm=np.array([0., 0.]))
    with pytest.raises(ContractError, match='identity'):
        from_packet(tampered)
    dop = Dop853Dense(t0=0., t1=.125, state_ids=['b', 'a'], y0=np.array([1., 2.]),
                     coefficients=np.zeros((7, 2)))
    assert dop.representation_id != identity


def test_quintic_rejects_wrong_shapes_dtype_nonfinite_and_unaccepted_queries():
    common = dict(t0=0., t1=1., state_ids=['a'], **{name: np.array([1.])
        for name in ('y0', 'f0', 'ym', 'fm', 'y1', 'f1')})
    for bad in (np.array([1.], dtype=np.float32), np.array([np.nan]), np.array([1., 2.])):
        with pytest.raises(ContractError, match='FP64'):
            QuinticHermite(**dict(common, fm=bad))
    for t0, t1 in ((1., np.nextafter(1., np.inf)), (-1e308, 1e308)):
        with pytest.raises(ContractError, match='duration'):
            QuinticHermite(**dict(common, t0=t0, t1=t1))
    with pytest.raises(ContractError, match='finite FP64 domain'):
        QuinticHermite(**dict(common, f1=np.array([1e308])))
    prediction = QuinticHermite(**common, accepted=False)
    with pytest.raises(ContractError, match='Unaccepted'):
        prediction.evaluate(.5, 'right')


def test_constant_forcing_factory_reuses_exact_exponential_packet():
    y0, forcing = np.array([-.05, .3, 2.]), np.array([1., -.2, 5.])
    decay = np.array([100., 1e-12, 0.])
    representation = ExponentialForced.from_constant_forcing(t0=0., t1=.125,
        state_ids=['rod', 'slow', 'zero'], y0=y0, forcing=forcing, decay=decay)
    for t in (0., 1e-12, .003, .125):
        gain = np.full(3, t)
        np.divide(-np.expm1(-decay*t), decay, out=gain, where=decay > 0)
        expected = np.exp(-decay*t)*y0+gain*forcing
        np.testing.assert_allclose(representation.evaluate(t, 'left' if t == .125 else 'right'),
                                   expected, rtol=3e-14, atol=1e-15)
    assert representation.kind == 'exponential_forced'
    restored = from_packet(representation.packet())
    np.testing.assert_array_equal(restored.evaluate(.03, 'right'), representation.evaluate(.03, 'right'))
    forcing[:] = 999.
    np.testing.assert_array_equal(representation.coefficients[0], [1., -.2, 5.])


def _fraction_poly(coefficients, x):
    result = Fraction(0)
    for coefficient in coefficients[::-1]:
        result = result*x+Fraction.from_float(float(coefficient))
    return result


def test_bernstein_encloses_exact_rational_values_and_interior_extrema():
    candidate = np.array([[1e16, 0.], [-1e16, 4.], [2., -4.], [0., .125],
                          [0., -.125], [0., .25], [0., -.25], [0., .5]])
    reference = np.array([[np.nextafter(1e16, np.inf), 0.]])
    for lo, hi in ((0., 1.), (.25, .75), (.125, .375)):
        lower, upper = polynomial_difference_enclosure(candidate, reference, lo=lo, hi=hi)
        bound = polynomial_difference_bound(candidate, reference, lo=lo, hi=hi)
        np.testing.assert_array_equal(bound, np.maximum(abs(lower), abs(upper)))
        for j in range(65):
            x = Fraction.from_float(lo)+(Fraction.from_float(hi)-Fraction.from_float(lo))*j/64
            for state in range(2):
                exact = _fraction_poly(candidate[:, state], x)-_fraction_poly(reference[:, state], x)
                assert Fraction.from_float(lower[state]) <= exact <= Fraction.from_float(upper[state])
    hump = np.array([[0.], [4.], [-4.]])
    bound = polynomial_difference_bound(hump, np.zeros((1, 1)))
    assert 2. <= bound[0] < 2.000000000001
    restricted = polynomial_difference_bound(hump, np.zeros((1, 1)), lo=.25, hi=.75)
    assert 1.25 <= restricted[0] < 1.250000000001


def test_restriction_and_distinct_partition_maps_include_the_exact_coefficients():
    coefficients = np.array([[0.], [0.], [1.]])
    lower, upper = restrict_power_coefficients(coefficients, lo=.25, hi=.75)
    exact = np.array([[.0625], [.25], [.25]])
    assert np.all(lower <= exact) and np.all(exact <= upper)
    # x maps physical [0,1], while reference x maps physical [0,2].
    bound = polynomial_difference_bound(coefficients, np.array([[0.], [0.], [4.]]),
        lo=.25, hi=.75, reference_lo=.125, reference_hi=.375)
    assert bound[0] < 1e-12


def test_polynomial_enclosure_validation_and_subnormal_roundoff():
    tiny = np.array([[np.nextafter(0., 1.)]])
    lower, upper = polynomial_difference_enclosure(tiny, np.zeros((1, 1)))
    assert lower[0] <= tiny[0, 0] <= upper[0]
    with pytest.raises(ContractError, match='finite FP64 domain'):
        polynomial_difference_bound(np.array([[np.finfo(float).max]]), np.array([[-np.finfo(float).max]]))
    for bad in (np.ones((14, 1)), np.ones((2, 1), dtype=np.float32), np.array([[np.inf]])):
        with pytest.raises(ContractError, match='power coefficients'):
            polynomial_difference_bound(bad, np.zeros((1, 1)))
    for lo, hi in ((-.1, 1.), (.5, .5), (0., 1.1), (False, 1.)):
        with pytest.raises(ContractError, match='subinterval'):
            polynomial_difference_bound(np.zeros((1, 1)), np.zeros((1, 1)), lo=lo, hi=hi)
    with pytest.raises(ContractError, match='Both reference'):
        polynomial_difference_bound(np.zeros((1, 1)), np.zeros((1, 1)), reference_lo=0.)
    with pytest.raises(ContractError, match='ordered state'):
        polynomial_difference_bound(np.zeros((1, 2)), np.zeros((1, 1)))


def test_dop853_conversion_encloses_exact_original_alternating_recurrence():
    y0 = np.array([.3, -2.])
    coefficients = np.arange(-7., 7.).reshape(7, 2)/13
    bounds = dop853_power_coefficient_enclosure(y0, coefficients)
    for state in range(2):
        power = [Fraction(0)]
        for i, coefficient in enumerate(coefficients[::-1, state]):
            power[0] += Fraction.from_float(float(coefficient))
            if i % 2 == 0:
                power = [Fraction(0)]+power
            else:
                power = [a-b for a, b in zip(power+[Fraction(0)], [Fraction(0)]+power)]
        power[0] += Fraction.from_float(float(y0[state]))
        for i, exact in enumerate(power):
            assert Fraction.from_float(bounds[0][i, state]) <= exact <= Fraction.from_float(bounds[1][i, state])
    midpoint = (bounds[0]+bounds[1])/2
    difference = polynomial_difference_bound(midpoint, bounds)
    assert np.all(difference < 1e-10)
    representation = Dop853Dense(t0=0., t1=1., state_ids=['a', 'b'], y0=y0,
                                 coefficients=coefficients)
    for x in np.linspace(0., 1., 17):
        np.testing.assert_allclose(_poly(midpoint, x),
            representation.evaluate(x, 'left' if x == 1 else 'right'), rtol=0, atol=3e-14)
    with pytest.raises(ContractError, match='DOP853'):
        dop853_power_coefficient_enclosure(y0, coefficients[:6])
    with pytest.raises(ContractError, match='Ordered matching'):
        polynomial_difference_bound((np.ones((1, 1)), np.zeros((1, 1))), np.zeros((1, 1)))


def test_quintic_conversion_encloses_exact_fraction_anchor_polynomial():
    representation = _representation(np.arange(-6., 6.).reshape(6, 2)/13)
    arrays = [getattr(representation, name) for name in ('y0', 'f0', 'ym', 'fm', 'y1', 'f1')]
    bounds = quintic_power_coefficient_enclosure(*arrays, duration=representation.t1-representation.t0)
    h = Fraction.from_float(representation.t1-representation.t0)
    for state in range(2):
        y0, f0, ym, fm, y1, f1 = (Fraction.from_float(float(value[state])) for value in arrays)
        s0 = h*f0
        a, b, c, d = ym-y0-s0/2, h*fm-s0, y1-y0-s0, h*f1-s0
        exact = [y0, s0, 16*a-8*b+7*c-d, -32*a+32*b-34*c+5*d,
                 16*a-40*b+52*c-8*d, 16*b-24*c+4*d]
        for i, coefficient in enumerate(exact):
            assert Fraction.from_float(bounds[0][i, state]) <= coefficient <= Fraction.from_float(bounds[1][i, state])
    assert np.all(polynomial_difference_bound(representation.coefficients, bounds) < 1e-10)
    with pytest.raises(ContractError, match='Hermite vectors'):
        quintic_power_coefficient_enclosure(*arrays, duration=0.)


def test_bernstein_bounds_include_exact_all_interval_control_points():
    candidate = np.arange(-8., 8.).reshape(8, 2)/7
    reference = np.arange(6.).reshape(3, 2)/11
    lo, hi = .125, .625
    lower, upper = polynomial_difference_enclosure(candidate, reference, lo=lo, hi=hi)
    from math import comb
    a, width = Fraction.from_float(lo), Fraction.from_float(hi)-Fraction.from_float(lo)
    for state in range(2):
        delta = [Fraction.from_float(float(candidate[k, state]))
                 -(Fraction.from_float(float(reference[k, state])) if k < 3 else 0) for k in range(8)]
        restricted = [width**k*sum(delta[j]*comb(j, k)*a**(j-k) for j in range(k, 8)) for k in range(8)]
        beta = [sum(restricted[k]*Fraction(comb(i, k), comb(7, k)) for k in range(i+1)) for i in range(8)]
        assert Fraction.from_float(lower[state]) <= min(beta)
        assert max(beta) <= Fraction.from_float(upper[state])


def test_tiny_duration_derivative_combines_normalized_terms_before_division():
    representation = QuinticHermite(t0=0., t1=1e-308, state_ids=['linear'],
        y0=np.array([0.]), ym=np.array([.5]), y1=np.array([1.]),
        f0=np.array([1e308]), fm=np.array([1e308]), f1=np.array([1e308]))
    derivative = representation.representation_derivative(1.25e-309, 'right')
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative, [1e308], rtol=3e-15)
