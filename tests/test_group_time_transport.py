import numpy as np
import pytest
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.group_time_transport import PinnedContinuousTransport, OwnedReadoutWindow
from retina_sim.continuous_bridge import ContinuousBridgeSelection, ContinuousReleasePacket


def make_packet():
    selection = ContinuousBridgeSelection(sparse.csr_matrix([[1.]]), np.array([0]))
    values = np.array([20., 30., 2.])
    declarations = [(name, dict(kind='discrete_hold', t0=0., t1=.001, epoch=0,
        state_ids=np.array([0]), fields=[('y0', i, i+1, (1,))])) for i, name in enumerate(('Vis', 'Vbar', 'Vh'))]
    representations = PinnedContinuousTransport.materialize(values, declarations)
    packet = ContinuousReleasePacket(selection, representations, source_version='unit').wait()
    return packet, values


def test_materialized_packets_survive_reusable_host_overwrite():
    packet, host = make_packet()
    original = packet.release_at(.0005, 'right')
    host[:] = np.nan
    np.testing.assert_array_equal(packet.release_at(.0005, 'right'), original)


def test_all_four_owned_consumers_must_finish_before_discard():
    packet, _ = make_packet()
    window = OwnedReadoutWindow([packet])
    window.acknowledge('inner', window.stop)
    with pytest.raises(ContractError, match='Unconsumed'):
        window.require_consumed()
    for who in ('monitor', 'recording', 'observation'):
        window.acknowledge(who, window.stop)
    window.require_consumed()
    with pytest.raises(ContractError, match='exact boundary'):
        window.at(window.stop, 'right')
