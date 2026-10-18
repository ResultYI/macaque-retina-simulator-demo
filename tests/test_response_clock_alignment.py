"""Regression for real post-preparation absolute clock/delay boundaries."""
from types import SimpleNamespace
import numpy as np
import pytest
from retina_sim.contracts import ContractError,Quantity
from retina_sim.recording import TimestampedDelay
from tools.disk3mm_response_check import aligned_response_times


def test_absolute_prepared_origin_keeps_requested_endpoints():
    runner=SimpleNamespace(periods={'rod':.0001,'BC':.001,'widefield':.00073},
        observation_dt=.0001,delayed_step=None)
    origin=2.099999999999879
    times,audit=aligned_response_times(runner,0.,origin,.02)
    assert len(times)==201 and times[0]==origin and times[-1]==origin+.02
    assert np.all(np.diff(times)>0)
    assert audit['maximum_adjustment_seconds']<1e-12
    assert not audit['state_time_or_RNG_rebased']
    assert times[23]==21023*.0001


def test_unaligned_readout_hits_real_pruning_boundary_but_absolute_grid_does_not():
    q=Quantity([1.],'vH_native_voltage','native_relative_voltage','synthetic_fixture')
    # The34-row real checkpoint includes the widefield event at2.09948s.
    # After the2.1023s observation, its delayed timestamp is a pruning boundary.
    history=[2.0993,2.0994,2.09948,2.0995,2.0996,2.0997,2.0998,2.0999,2.1]
    delay=TimestampedDelay(.00282,history,[[1.]]*len(history),q,'linear',0.)
    for tick in range(21001,21024):delay.append(tick*.0001,q)
    raw_time=2.099999999999879+23*.0001
    with pytest.raises(ContractError,match='pruned'):
        delay.at(raw_time)
    runner=SimpleNamespace(periods={'rod':.0001,'BC':.001,'widefield':.00073},
        observation_dt=.0001,delayed_step=None)
    times,_=aligned_response_times(runner,0.,2.099999999999879,.02)
    np.testing.assert_array_equal(delay.at(times[23]).values,[1.])
