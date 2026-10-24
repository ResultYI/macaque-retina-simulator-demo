"""CPU-only math/ownership checks; no torch import or CUDA initialization."""
from contextlib import nullcontext
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.cone_hc_reduced_bridge import BridgeSelection, PinnedSegmentBridge
from retina_sim.vh2005 import SOURCE_VALUES


def selection():
    B = sparse.csr_matrix((np.array([.25, .75, 1., .4, .6]),
                           np.array([2, 0, 1, 2, 1]), np.array([0, 2, 3, 5])), shape=(3, 3))
    return BridgeSelection(B, np.array([2, 0]),
                           cone_ids=('c0', 'c1', 'c2'), hc_ids=('h0', 'h1', 'h2'))


def segment(start=0., stop=.0001, version='fixture-v1'):
    y0 = np.ones(24, dtype=np.float64)*20.
    y1 = y0.copy()
    f0 = np.zeros_like(y0)
    y0[6:9], y1[6:9] = [10., 20., 30.], [13., 24., 35.]
    y0[12:15], y1[12:15] = [21., 22., 23.], [25., 26., 27.]
    y0[21:24], y1[21:24] = [4., 5., 6.], [7., 9., 11.]
    f0[:] = 1000.
    return NS(start=start, stop=stop, y0=y0, f0=f0, y1=y1,
              input_epoch=3, source_version=version)


def test_selection_preserves_consumer_order_weights_and_unsorted_row_order():
    chosen = selection()
    np.testing.assert_array_equal(chosen.cone_rows, [2, 0])
    np.testing.assert_array_equal(chosen.hc_rows, [0, 1, 2])
    np.testing.assert_array_equal(chosen.B.indices, [2, 1, 2, 0])
    np.testing.assert_array_equal(chosen.B.data, [.4, .6, .25, .75])
    np.testing.assert_array_equal(chosen.rows, [8, 6, 14, 12, 21, 22, 23])
    assert chosen.packet_bytes == 8*(6*2+3*3)
    assert not chosen.B.data.flags.writeable
    with pytest.raises(TypeError):
        chosen.parameters['mu'] = 1.


def test_quadratic_endpoints_internal_stages_and_owned_projection():
    chosen, source = selection(), segment()
    packet = chosen.project(source, segment_id=7)
    for t in (source.start, .000017, .00005, .000083, source.stop):
        theta = t/source.stop
        expected = source.y0+theta*source.stop*source.f0+theta**2*(
            source.y1-source.y0-source.stop*source.f0)
        actual = np.concatenate(packet.state_at(t))
        np.testing.assert_array_equal(actual, expected[chosen.rows])
    before = packet.values.copy()
    source.y0[:] = 900.
    np.testing.assert_array_equal(packet.values, before)
    assert packet.nbytes == chosen.packet_bytes
    assert packet.metadata['segment_id'] == 7
    assert packet.metadata['selected_ids_hash'] == chosen.sha256


def test_no_epsilon_rounding_or_extrapolation_at_adjacent_float_times():
    chosen = selection()
    a = 6.5610999999999997
    b = np.nextafter(a, np.inf)
    source = segment(a, b)
    packet = chosen.project(source)
    np.testing.assert_array_equal(np.concatenate(packet.state_at(a)), source.y0[chosen.rows])
    np.testing.assert_array_equal(np.concatenate(packet.state_at(b)), source.y1[chosen.rows])
    for t in (np.nextafter(a, -np.inf), np.nextafter(b, np.inf), np.nan):
        with pytest.raises(ContractError, match='outside'):
            packet.state_at(t)


def test_cpu_release_uses_interpolated_h_before_original_B_and_nonlinearity():
    chosen, source = selection(), segment()
    packet = chosen.project(source)
    t = .000037
    v, vbar, h = packet.state_at(t)
    p = SOURCE_VALUES
    expected_bh = np.array([.4*h[2]+.6*h[1], .25*h[2]+.75*h[0]])
    expected = (1./(1.+np.exp(-((v-expected_bh-p['V_k'])/p['V_n']))))/(
        vbar/p['V_I'])**p['mu']
    signals = packet.signals_at(t)
    np.testing.assert_array_equal(signals['Bh'], expected_bh)
    np.testing.assert_array_equal(signals['release'], expected)
    endpoint_linear = (1-t/source.stop)*packet.release_at(0.)+(t/source.stop)*packet.release_at(source.stop)
    assert np.max(np.abs(expected-endpoint_linear)) > 1e-5


@pytest.mark.parametrize('bad', ('negative_voltage', 'zero_adaptation', 'nonfinite'))
def test_original_domains_not_clipped(bad):
    chosen, source = selection(), segment()
    if bad == 'negative_voltage':
        source.y0[6] = -1.
    elif bad == 'zero_adaptation':
        source.y0[12] = 0.
    else:
        source.f0[6] = np.inf
    if bad == 'nonfinite':
        with pytest.raises(ContractError, match='nonfinite'):
            chosen.project(source)
    else:
        with pytest.raises(ContractError, match='domain'):
            chosen.project(source).release_at(0.)


def test_packet_checkpoint_roundtrip_binds_layout_and_owns_values():
    chosen = selection()
    packet = chosen.project(segment(), segment_id=2, input_side='left')
    raw = packet.payload()
    restored = chosen.from_payload(raw)
    np.testing.assert_array_equal(restored.release_at(.000027), packet.release_at(.000027))
    raw['values'][:] = 0.
    np.testing.assert_array_equal(restored.values, packet.values)
    raw = packet.payload()
    raw['metadata']['selected_ids_hash'] = 'different'
    with pytest.raises(ContractError, match='selection'):
        chosen.from_payload(raw)
    raw = packet.payload()
    raw['values'] = raw['values'].astype(np.float32)
    with pytest.raises(ContractError, match='FP64'):
        chosen.from_payload(raw)


def test_invalid_selection_or_identity_is_rejected():
    chosen = selection()
    B = sparse.eye(3, dtype=np.float64, format='csr')
    with pytest.raises(ContractError, match='duplicate'):
        BridgeSelection(B, np.array([0, 0]))
    with pytest.raises(ContractError, match='FP64'):
        BridgeSelection(B.astype(np.float32), np.array([0]))
    with pytest.raises(ContractError, match='parameters'):
        BridgeSelection(B, np.array([0]), parameters=dict(SOURCE_VALUES, tau_1=5.))
    raw = chosen.project(segment()).payload()
    other = BridgeSelection(B, np.array([0, 2]))
    with pytest.raises(ContractError, match='selection'):
        other.from_payload(raw)


class FakeDevice:
    def __init__(self, name):
        self.type = str(name).split(':')[0]

    def __eq__(self, other):
        return isinstance(other, FakeDevice) and self.type == other.type


class FakeTensor:
    def __init__(self, array, device, cuda):
        self.array, self.device, self.cuda = array, FakeDevice(device), cuda
        self.dtype, self.ndim, self.shape = array.dtype, array.ndim, array.shape

    def numel(self):
        return self.array.size

    def __getitem__(self, item):
        return FakeTensor(self.array[item], self.device.type, self.cuda)

    def numpy(self):
        return self.array.view()

    def copy_(self, source, non_blocking):
        assert non_blocking
        self.cuda.pending.append(lambda: np.copyto(self.array, source.array))


class FakeEvent:
    def __init__(self, cuda):
        self.cuda = cuda

    def record(self, stream):
        self.operations, self.cuda.pending = self.cuda.pending, []

    def synchronize(self):
        for operation in self.operations:
            operation()
        self.operations = []
        self.cuda.waits += 1

    def elapsed_time(self,other):
        return 0.  # Fake CPU ownership test, never a hardware timing claim.


class FakeTorch:
    """Defer CPU visibility until event.wait, without importing torch/CUDA."""
    float64, int64 = np.dtype('float64'), np.dtype('int64')

    def __init__(self):
        self.cuda = NS(pending=[], waits=0)
        self.cuda.Event = lambda **kwargs: FakeEvent(self.cuda)
        self.cuda.current_stream = lambda device: None
        self.select_calls = 0

    def as_tensor(self, values, dtype, device):
        return FakeTensor(np.array(values, dtype=dtype), device, self.cuda)

    def empty(self, shape, dtype, device, pin_memory=False):
        assert pin_memory == (str(device) == 'cpu')
        name = device.type if isinstance(device, FakeDevice) else device
        return FakeTensor(np.full(shape, np.nan, dtype=dtype), name, self.cuda)

    def no_grad(self):
        return nullcontext()

    def index_select(self, source, axis, rows, out):
        self.select_calls += 1
        np.take(source.array, rows.array, axis=axis, out=out.array)


def gpu_source(source, fake):
    return NS(**{**vars(source), **{k: FakeTensor(getattr(source, k), 'cuda', fake.cuda)
                                  for k in ('y0', 'f0', 'y1')}})


def test_double_pinned_lease_waits_and_rejects_overwrite_or_stale_reads():
    chosen, fake = selection(), FakeTorch()
    bridge = PinnedSegmentBridge(chosen, fake, 'cuda')
    source = gpu_source(segment(), fake)
    first = bridge.submit(source, segment_id=1)
    second = bridge.submit(source, segment_id=2)
    assert fake.cuda.waits == 0
    with pytest.raises(ContractError, match='Both'):
        bridge.submit(source, segment_id=3)
    expected = chosen.project(segment()).release_at(.00003)
    np.testing.assert_array_equal(first.release_at(.00003), expected)
    assert fake.cuda.waits == 1
    calls = fake.select_calls
    first.release_at(.00007)
    assert fake.select_calls == calls
    raw = first.payload()
    bridge.release(first)
    third = bridge.submit(source, segment_id=3)
    with pytest.raises(ContractError, match='released'):
        first.release_at(0.)
    with pytest.raises(ContractError, match='lease'):
        bridge.release(first)
    saved = bridge.payloads()
    assert sorted(r['metadata']['segment_id'] for r in saved) == [2, 3]
    assert bridge.metrics['D2H_bytes'] == 3*chosen.packet_bytes
    bridge.release(second)
    bridge.release(third)
    np.testing.assert_array_equal(chosen.from_payload(raw).release_at(.00003), expected)

