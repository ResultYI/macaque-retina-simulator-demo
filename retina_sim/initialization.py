"""Deterministic local reference balances, distinct from coupled-network equilibrium."""
import numpy as np
from scipy.special import expit

def initial_event_rate(profile):
    def a(k,u,m):return profile.quantity(k,u,m).values
    r=a('rmax','events/s','release_attempt_rate')*expit((a('V0','V','initial_membrane_voltage')-
        a('r_half','V','release_half_voltage'))/a('r_slope','V','release_voltage_slope'))
    U=a('U','dimensionless','release_probability')
    x=1/(1+a('tau_recovery','s','release_recovery_time_constant')*U*r)
    return a('Npool','dimensionless','release_pool_size')*U*r*x

def initial_release(profile):
    return expit((profile.quantity('V0','V','initial_membrane_voltage').values-
        profile.quantity('release_half','V','activation_half_voltage').values)/
        profile.quantity('release_slope','V','activation_voltage_slope').values)
