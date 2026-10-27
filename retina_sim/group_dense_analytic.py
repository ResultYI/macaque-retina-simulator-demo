"""Static constant-forcing exponential enclosures; no ODE advancement.

The represented analytic waveform, its Taylor coefficients and its truncation
remainder are enclosed. This does not bound model, initial-state, nonlinear RHS,
or floating-point waveform-evaluation error.
"""
import math
from numbers import Real

import numpy as np

from .contracts import ContractError
from .group_dense_certificate import _multiply, _outward, _subtract


def constant_forcing_power_enclosure(y0, forcing, decay, *, duration):
    """Return degree-12 coefficient bounds and a uniform absolute remainder.

    The waveform solves y'=forcing-decay*y for the supplied constant FP64
    data; x=(t-t0)/duration is in [0,1]. All coefficient arithmetic is outward
    rounded. With d=forcing-decay*y0 and z=decay*duration, integration of the
    degree-11 exponential Taylor remainder gives |R| <= |d|*duration*z**12/13!
    for every z>=0, since exp(-v)<=1. No small-z assumption is needed.

    Return ((lower, upper), remainder_upper), with coefficient shape (13,n)
    and remainder shape (n,). Outputs own their data; rows may be preselected.
    """
    y0, forcing, decay = (np.asarray(value) for value in (y0, forcing, decay))
    if (y0.dtype != np.float64 or y0.ndim != 1 or not len(y0)
            or forcing.dtype != np.float64 or forcing.shape != y0.shape
            or decay.dtype != np.float64 or decay.shape not in ((), y0.shape)
            or not all(np.isfinite(value).all() for value in (y0, forcing, decay))
            or np.any(decay < 0) or not isinstance(duration, Real)
            or isinstance(duration, (bool, np.bool_))
            or not math.isfinite(duration) or duration <= 0):
        raise ContractError('Finite FP64 constant-forcing vectors, nonnegative decay and positive duration required')
    decay = np.broadcast_to(decay, y0.shape)
    h = (np.array(float(duration)), np.array(float(duration)))
    initial, drive, rate = (y0, y0), (forcing, forcing), (decay, decay)
    d = _subtract(drive, _multiply(rate, initial))
    z = _multiply(rate, h)
    negative_z = (-z[1], -z[0])
    powers = [initial, _multiply(h, d)]
    for k in range(2, 13):
        reciprocal = np.array(1./k)
        powers.append(_multiply(_multiply(powers[-1], _outward(reciprocal, reciprocal)), negative_z))
    lower = np.stack([power[0] for power in powers])
    upper = np.stack([power[1] for power in powers])
    magnitude = np.maximum(np.abs(lower[-1]), np.abs(upper[-1]))
    reciprocal = np.array(1./13)
    remainder = _multiply(_multiply((magnitude, magnitude),
        _outward(reciprocal, reciprocal)), z)[1].copy()
    # Zero decay is exactly linear, so later Taylor terms and its tail vanish.
    zero_decay = decay == 0
    if np.any(zero_decay):
        linear = _multiply(h, (forcing[zero_decay], forcing[zero_decay]))
        lower[1, zero_decay], upper[1, zero_decay] = linear
        lower[2:, zero_decay] = upper[2:, zero_decay] = 0.
        remainder[zero_decay] = 0.
    return (lower, upper), remainder
