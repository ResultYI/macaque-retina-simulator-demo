"""CPU protocol tests with fake readiness; none validate a CUDA transfer."""
from copy import deepcopy

import numpy as np
import pytest
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.continuous_bridge import (CONSUMERS, ContinuousBridgeSelection,
    ContinuousPacketBridge, ContinuousReleasePacket)
from retina_sim.continuous_state import DiscreteHold, Dop853Dense, ExponentialForced
from retina_sim.vh2005 import SOURCE_VALUES


def selection():
    return ContinuousBridgeSelection(sparse.csr_matrix(
        [[.2, 0., .8, 0.], [0., 0., 0., 0.], [0., .5, .5, 0.]], dtype=np.float64),
        np.array([2, 0, 1]))


def representations(chosen, start=0., stop=.001, epoch=0):
    f = np.zeros((7, len(chosen.cone_rows))); f[0] = .3
    q = np.zeros((8, len(chosen.cone_rows))); q[0] = .2
    hq = np.zeros((8, len(chosen.hc_rows))); hq[0] = .01
    return dict(Vis=Dop853Dense(t0=start, t1=stop, state_ids=chosen.cone_rows,
                               y0=np.array([3., 1., 2.]), coefficients=f, epoch=epoch),
        Vbar=ExponentialForced(t0=start, t1=stop, state_ids=chosen.cone_rows,
            y0=np.array([20., 21., 22.]), coefficients=q, decay=np.zeros(3), epoch=epoch),
        Vh=ExponentialForced(t0=start, t1=stop, state_ids=chosen.hc_rows,
            y0=np.array([2., 4., 8.]), coefficients=hq, decay=np.array([2., 4., 7.]), epoch=epoch))


def test_reconstruct_each_current_waveform_then_original_B_and_nonlinearity():
    chosen = selection(); signals = representations(chosen)
    packet = ContinuousReleasePacket(chosen, signals, source_version='candidate-method')
    assert chosen.cone_rows.tolist() == [2, 0, 1]
    assert chosen.hc_rows.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(chosen.B.toarray()[2], [0., 0., 0.])
    for t in (0., .00037, .001):
        side = 'left' if t == .001 else 'right'
        v, vbar, h = [signals[n].evaluate(t, side) for n in ('Vis', 'Vbar', 'Vh')]
        bh = chosen.B@h
        p = SOURCE_VALUES
        q = 1./(1.+np.exp(-((v-bh-p['V_k'])/p['V_n'])))
        expected = q/(vbar/p['V_I'])**p['mu']
        result = packet.signals_at(t, side)
        np.testing.assert_array_equal(result['Bh'], bh)
        np.testing.assert_array_equal(result['release'], expected)
    with pytest.raises(ContractError, match='side'):
        packet.release_at(.001, 'right')


def test_all_zero_B_readout_transfers_no_HC_array():
    chosen = ContinuousBridgeSelection(sparse.csr_matrix((1, 3), dtype=np.float64), np.array([0]))
    reps = dict(Vis=DiscreteHold(t0=0., t1=1., state_ids=np.array([0]), y0=np.array([2.])),
        Vbar=DiscreteHold(t0=0., t1=1., state_ids=np.array([0]), y0=np.array([20.])), Vh=None)
    result = ContinuousReleasePacket(chosen, reps, source_version='candidate-method').signals_at(.5, 'right')
    assert result['Vh'].size == 0 and result['Bh'][0] == 0.
    assert np.isfinite(result['release']).all()


def test_delayed_fake_copy_is_waited_before_query_payload_and_release():
    chosen = selection(); truth = representations(chosen)
    pending, expected = {}, {}
    for name, representation in truth.items():
        expected[name] = representation.packet()
        pending[name] = dict(metadata=dict(expected[name]['metadata']), arrays={
            k: v.copy() if k == 'state_ids' else np.zeros_like(v) for k, v in expected[name]['arrays'].items()})
    calls = []
    def fake_copy_complete():
        calls.append('copy-complete')
        for name, representation in expected.items():
            for key, value in representation['arrays'].items():
                np.copyto(pending[name]['arrays'][key], value)
    bridge = ContinuousPacketBridge(chosen)
    packet = bridge.submit(pending, source_version='candidate-method', ready_waiter=fake_copy_complete)
    assert calls == []
    result = packet.release_at(.0003, 'right')
    assert calls == ['copy-complete']
    np.testing.assert_array_equal(result, ContinuousReleasePacket(chosen, truth,
        source_version='candidate-method').release_at(.0003, 'right'))
    owned = packet.payload()
    for values in pending.values():
        values['arrays']['y0'][:] = 999.
    restored = ContinuousReleasePacket.from_payload(chosen, owned)
    np.testing.assert_array_equal(restored.release_at(.0003, 'right'), result)
    for consumer in CONSUMERS:
        bridge.acknowledge(consumer, .001)
    assert calls == ['copy-complete']
    with pytest.raises(ContractError, match='released'):
        packet.release_at(.0003, 'right')
    release_only_waits = []
    other = ContinuousPacketBridge(chosen)
    other.submit(truth, source_version='candidate-method', ready_waiter=lambda: release_only_waits.append(True))
    for consumer in CONSUMERS:
        other.acknowledge(consumer, .001)
    assert release_only_waits == [True]


def test_two_slots_require_all_four_consumers_and_preserve_generations():
    chosen = selection(); bridge = ContinuousPacketBridge(chosen)
    first = bridge.submit(representations(chosen), source_version='candidate-method', segment_id=0)
    second = bridge.submit(representations(chosen, .001, .002), source_version='candidate-method', segment_id=1)
    bridge.acknowledge('inner', .002)
    with pytest.raises(ContractError, match='Both'):
        bridge.submit(representations(chosen, .002, .003), source_version='candidate-method')
    with pytest.raises(ContractError, match='four'):
        bridge.release(first)
    for consumer in ('monitor', 'observation', 'recording'):
        bridge.acknowledge(consumer, .001)
    assert bridge.common_watermark == .001
    third = bridge.submit(representations(chosen, .002, .003, epoch=1), source_version='candidate-method', segment_id=2)
    assert bridge._slots[0].generation == 2
    with pytest.raises(ContractError, match='released'):
        first.release_at(.0004, 'right')
    assert np.isfinite(second.release_at(.0014, 'right')).all()
    assert np.isfinite(third.release_at(.0024, 'right')).all()
    with pytest.raises(ContractError, match='watermark'):
        bridge.acknowledge('inner', .001)


def test_checkpoint_waits_and_copies_pending_slots_watermarks_and_generations():
    chosen = selection(); bridge = ContinuousPacketBridge(chosen)
    waits = []
    original = bridge.submit(representations(chosen), source_version='candidate-method',
                             ready_waiter=lambda: waits.append(True), segment_id=4)
    bridge.acknowledge('inner', .001)
    bridge.acknowledge('monitor', .0005)
    payload = bridge.checkpoint_payload()
    assert waits == [True]
    assert payload['full_runner_restore_supported'] is False
    restored = ContinuousPacketBridge.from_checkpoint_payload(chosen, payload)
    assert restored.consumed == bridge.consumed
    assert restored._slots[0].generation == bridge._slots[0].generation
    np.testing.assert_array_equal(restored._slots[0].packet.release_at(.0007, 'right'),
                                  original.release_at(.0007, 'right'))
    # Caller-owned checkpoint buffers do not alias either live bridge.
    payload['slots'][0]['packet']['representations']['Vis']['arrays']['y0'][:] = 123.
    np.testing.assert_array_equal(restored._slots[0].packet.release_at(.0007, 'right'),
                                  original.release_at(.0007, 'right'))
    for consumer in CONSUMERS:
        restored.acknowledge(consumer, .001)
    empty = restored.checkpoint_payload()
    assert all(s['packet'] is None for s in empty['slots'])
    assert ContinuousPacketBridge.from_checkpoint_payload(chosen, empty).common_watermark == .001


def test_checkpoint_generation_causality_or_selection_tampering_fails():
    chosen = selection(); bridge = ContinuousPacketBridge(chosen)
    bridge.submit(representations(chosen), source_version='candidate-method')
    payload = bridge.checkpoint_payload()
    for mutate in (lambda p: p.update(selection_sha256='wrong'),
                   lambda p: p['slots'][0].update(generation=0),
                   lambda p: p.update(frontier=.002),
                   lambda p: p['consumed'].update(recording=.002)):
        altered = deepcopy(payload); mutate(altered)
        with pytest.raises(ContractError):
            ContinuousPacketBridge.from_checkpoint_payload(chosen, altered)


def test_submit_rejects_predictions_extra_states_epoch_or_domain_errors():
    chosen = selection()
    for mutate in (lambda p: p['Vis']['metadata'].update(accepted=False),
                   lambda p: p['Vh']['metadata'].update(epoch=1),
                   lambda p: p['Vbar']['arrays'].update(state_ids=np.array([0, 1, 2])),
                   lambda p: p['Vis']['arrays'].update(y0=np.ones(4))):
        raw = {name: deepcopy(rep.packet()) for name, rep in representations(chosen).items()}
        mutate(raw)
        with pytest.raises(ContractError):
            ContinuousPacketBridge(chosen).submit(raw, source_version='candidate-method')
    reps = representations(chosen)
    reps['Vbar'] = DiscreteHold(t0=0., t1=.001, state_ids=chosen.cone_rows, y0=np.array([20., 0., 21.]))
    with pytest.raises(ContractError, match='domain'):
        ContinuousReleasePacket(chosen, reps, source_version='candidate-method').release_at(.0005, 'right')
    with pytest.raises(ContractError, match='source parameters'):
        ContinuousBridgeSelection(sparse.eye(2, format='csr', dtype=np.float64), [0], parameters={})
