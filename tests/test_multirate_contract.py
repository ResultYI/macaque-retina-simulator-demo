import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.multirate_contract import (MultiratePolicy, LocalErrorAllocation, CalendarEvent,
                                           BlockEventCalendar, PCG64Transaction, forcing_allowance)


def test_original_budget_is_shared_not_four_copies_and_no_rollover():
    b = LocalErrorAllocation([0., 1.], [0., 2.], rtol=1e-6, atol=1e-9, active=np.array([True, False]))
    good = {k: np.array([.25e-9, 999.]) for k in ('time', 'forcing', 'coupling', 'quadrature')}
    assert b.assess(good)['accepted']
    good['forcing'][0] = .26e-9
    good['time'][0] = 0.
    assert not b.assess(good)['accepted']
    with pytest.raises(ContractError, match='All four'):
        b.assess({'time': np.zeros(2)})
    with pytest.raises(ContractError):
        MultiratePolicy(fast_max_step_seconds=.0002)


def test_event_calendar_keeps_ULPs_and_does_not_restart_unrelated_blocks():
    t = 1/150
    above = np.nextafter(t, np.inf)
    c = BlockEventCalendar([CalendarEvent(t, 'input', ('cone', 'rod'), 'frame1'),
        CalendarEvent(.0001, 'rod', ('rod', 'rod_membrane'), 'rod1'),
        CalendarEvent(above, 'intervention', ('HC',), 'clamp1')])
    assert c.next_boundary('cone', 0., .01) == t
    assert c.next_boundary('rod', 0., .01) == .0001
    assert c.next_boundary('HC', 0., .01) == above
    assert c.at(t)[0].event_id == 'frame1'
    assert c.at(np.nextafter(t, 0.)) == ()


def test_rejected_attempt_does_not_consume_rng_or_duplicate_pending_events():
    a, reference = (np.random.Generator(np.random.PCG64(139)) for _ in range(2))
    rejected = PCG64Transaction(a)
    rejected.stage({'u': rejected.rng.random(4).tolist()})
    rejected.reject()
    accepted = PCG64Transaction(a)
    draws = accepted.rng.random(4)
    accepted.stage({'u': draws.tolist()})
    events = accepted.commit()
    np.testing.assert_array_equal(draws, reference.random(4))
    np.testing.assert_array_equal(a.random(4), reference.random(4))
    assert len(events) == 1
    with pytest.raises(ContractError, match='closed'):
        accepted.commit()


def test_concurrent_original_rng_consumption_refuses_commit():
    rng = np.random.Generator(np.random.PCG64(1))
    transaction = PCG64Transaction(rng)
    rng.random()
    with pytest.raises(ContractError, match='concurrently'):
        transaction.commit()


def test_state_to_driver_units_preserves_allocated_share_at_internal_times():
    from retina_sim.semianalytic_time import exponential_error_bound
    decay = np.array([0., 10., 1000.])
    allowed = np.array([1e-9, 2e-9, 3e-9])
    initial = allowed*.2
    q_allowance = forcing_allowance(allowed, decay, .001, initial_error=initial)
    for t in np.linspace(0, .001, 19):
        propagated = exponential_error_bound(decay, t, q_allowance, initial)
        assert np.all(propagated <= np.nextafter(allowed, np.inf))
    with pytest.raises(ContractError, match='initial'):
        forcing_allowance(allowed, decay, .001, initial_error=allowed*2)


def test_finite_input_overflow_cannot_accept_arbitrary_error():
    with pytest.raises(ContractError, match='Computed original tolerance scale'):
        LocalErrorAllocation([1e308], [1e308], rtol=2., atol=1., active=np.array([True]))


def test_integer_time_units_and_small_decay_initial_error_limit():
    result = forcing_allowance(1., 2., 1)
    np.testing.assert_allclose(result, 2./(-np.expm1(-2.)), rtol=1e-15)
    result = forcing_allowance(1., 1e-20, 1, initial_error=1.)
    np.testing.assert_allclose(result, 1e-20, rtol=1e-15)
