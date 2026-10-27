"""Synthetic static exponential enclosures; no retina trajectory or GPU."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.group_dense_analytic import constant_forcing_power_enclosure


def test_all_coefficients_and_remainder_enclose_exact_fraction_arithmetic():
    y0 = np.array([-.05, .3, 2., 1e-300])
    forcing = np.array([1., -.2, 5., 1e-300])
    decay = np.array([200., 1e-12, 0., 1.])
    duration = .0001
    (lower, upper), remainder = constant_forcing_power_enclosure(y0, forcing, decay, duration=duration)
    h = Fraction.from_float(duration)
    for row in range(len(y0)):
        initial, drive, rate = (Fraction.from_float(float(v[row])) for v in (y0, forcing, decay))
        d, z = drive-rate*initial, rate*h
        exact = [initial]+[h*d*(-z)**(k-1)/math.factorial(k) for k in range(1, 13)]
        for k, value in enumerate(exact):
            assert Fraction.from_float(lower[k, row]) <= value <= Fraction.from_float(upper[k, row])
        assert abs(d)*h*z**12/math.factorial(13) <= Fraction.from_float(remainder[row])
    assert np.all(remainder >= 0)
    np.testing.assert_array_equal(lower[2:, 2], np.zeros(11))
    np.testing.assert_array_equal(upper[2:, 2], np.zeros(11))
    assert remainder[2] == 0.
    y0[:] = 999.
    assert lower[0, 0] == -.05 and upper[0, 0] == -.05


@pytest.mark.parametrize('rate', [0., 2.**-20, .02, 1., 20.])
def test_uniform_remainder_matches_high_precision_exponential(rate):
    y0, forcing, decay = np.array([-.25]), np.array([.75]), np.array([rate])
    (_, _), remainder = constant_forcing_power_enclosure(y0, forcing, decay, duration=1.)
    with localcontext() as context:
        context.prec = 120
        initial, drive, a = map(Decimal.from_float, (-.25, .75, rate))
        d = drive-a*initial
        for j in range(17):
            x = Decimal(j)/16
            exact = initial+drive*x if rate == 0 else initial+d*(-(-a*x).exp()+1)/a
            polynomial = initial+sum(d*(Decimal(1) if k == 1 else (-a)**(k-1))*x**k/Decimal(math.factorial(k)) for k in range(1, 13))
            assert abs(exact-polynomial) <= Decimal.from_float(float(remainder[0]))


def test_scalar_decay_and_selected_vectors_match_and_return_owned_arrays():
    y0 = np.array([1., 2., 3.]); forcing = np.array([2., 3., 4.])
    all_bounds, all_remainder = constant_forcing_power_enclosure(y0, forcing, np.array(.2), duration=.125)
    bounds, remainder = constant_forcing_power_enclosure(y0[[2, 0]], forcing[[2, 0]], np.array([.2, .2]), duration=.125)
    for full, selected in zip(all_bounds, bounds):
        np.testing.assert_array_equal(selected, full[:, [2, 0]])
    np.testing.assert_array_equal(remainder, all_remainder[[2, 0]])
    bounds[0][:] = 123.
    assert all_bounds[0][0, 0] == 1.


def test_invalid_inputs_and_overflow_fail_closed():
    common = dict(y0=np.array([1.]), forcing=np.array([2.]), decay=np.array([.5]), duration=.1)
    for patch in (dict(y0=np.array([], dtype=float)), dict(y0=np.array([1.], dtype=np.float32)),
                  dict(forcing=np.array([1., 2.])), dict(decay=np.array([-1.])),
                  dict(decay=np.array([np.nan])), dict(duration=0.), dict(duration=True),
                  dict(duration=np.inf)):
        with pytest.raises(ContractError, match='constant-forcing'):
            constant_forcing_power_enclosure(**dict(common, **patch))
    with pytest.raises(ContractError, match='finite FP64 domain'):
        constant_forcing_power_enclosure(**dict(common, decay=np.array([1e308]), duration=2.))
